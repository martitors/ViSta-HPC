"""
Does the GPU kernel work, and does it give what the CPU kernel gives?

Three checks, each printed with PASS or FAIL, and one RESULT line at the end:

1. **the build**: the kernel was built with CUDA and sees a GPU;
2. **the kernel**: blocks of random visibilities (1, 2 and 4 correlations,
   sources up to 60" from the phase centre, flags, NaN, unequal and
   decreasing channels) go through the CPU kernel and through the GPU kernel,
   all the blocks in one launch; flags and UVW must be identical, the
   visibilities equal to the last digits of single precision;
3. **a stack**: a small simulated sample is stacked and measured twice,
   ``device="cpu"`` and ``device="gpu"``; every column of the two stacks is
   compared, and the measured fluxes.

::

    python benchmarks/gpu_check.py --work $SCRATCH/gpu_check
    python benchmarks/gpu_check.py --work /tmp/gpu_check --dry-run   # no GPU: tests the script

The times of the two stacks are printed too, but this sample is small: it
says that the GPU works, not how fast it is.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "examples")]

from vista import ms_ops                                    # noqa: E402

LINE = 345.79599
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""), flush=True)
    return bool(ok)


# ---------------------------------------------------------------------------
def check_build(dry_run):
    print("1. The build")
    built = ms_ops.built_with_cuda()
    check("kernel built with CUDA (cmake -DWITH_CUDA=ON)", built or dry_run,
          "" if built else "no: this is a CPU build")
    seen = built and ms_ops.gpu_available()
    detail = ""
    if seen:
        free, total = ms_ops.gpu_memory_info()
        detail = f"{total / 2**30:.0f} GB, {free / 2**30:.0f} GB free"
    elif built:
        detail = (ms_ops.gpu_last_error() or "no GPU visible")
        detail += f" (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r})"
    check("a GPU is visible", seen or dry_run, detail)
    print(f"        OpenMP threads of the CPU kernel: {ms_ops.omp_max_threads()}")
    return seen


# ---------------------------------------------------------------------------
def kernel_blocks(rng):
    """Blocks of rows of different shapes, as chunks of different datasets."""
    blocks = []
    for ncorr, offset, nchan_old, nchan_new, kind in (
            (2, 0.0, 120, 90, "equal"), (2, 5.0, 96, 128, "equal"), (1, 20.0, 64, 40, "decreasing"),
            (4, 60.0, 200, 150, "unequal"), (2, 20.0, 480, 100, "widths")):
        nrow = int(rng.integers(300, 900))
        z = float(rng.uniform(0.5, 3.0))
        f0 = rng.uniform(90e9, 250e9)
        step = rng.uniform(1e6, 8e6)
        freq = f0 + np.arange(nchan_old) * step
        narrow = 0.8 if kind == "widths" else 1.0       # channels narrower than their spacing
        width = np.full(nchan_old, narrow * step) * (1 + z)
        if kind == "decreasing":
            freq = freq[::-1].copy()
        elif kind == "unequal":                         # two spectral windows with a gap
            freq[nchan_old // 2:] += 37 * step
        freq_old = freq * (1 + z)
        lo, hi = np.sort(freq_old)[[nchan_old // 8, -nchan_old // 8]]
        freq_new = np.linspace(lo, hi, nchan_new)
        vis = (rng.normal(size=(nrow, nchan_old, ncorr))
               + 1j * rng.normal(size=(nrow, nchan_old, ncorr))).astype(np.complex64)
        flag = rng.random((nrow, nchan_old, ncorr)) < 0.05
        vis[rng.integers(nrow), rng.integers(nchan_old), 0] = np.nan
        vis[rng.integers(nrow), rng.integers(nchan_old), -1] = np.inf
        uvw = rng.normal(0, 1500, (nrow, 3))
        ra0, dec0 = np.radians(rng.uniform(0, 360)), np.radians(rng.uniform(-70, 20))
        ra1 = ra0 + np.radians(offset / 3600) / np.cos(dec0)
        dec1 = dec0 + np.radians(0.7 * offset / 3600)
        blocks.append(dict(vis=vis, flag=flag, uvw=uvw, z=z, freq_old=freq_old, ra0=ra0,
                           dec0=dec0, ra1=ra1, dec1=dec1, freq_new=freq_new, width=width,
                           label=f"{ncorr} corr, {offset:.0f}\" off-centre, {kind} channels"))
    return blocks


def check_kernel(dry_run):
    print("2. The kernel: CPU against GPU, five blocks in one launch")
    blocks = kernel_blocks(np.random.default_rng(7))
    cpu = [ms_ops.full_pipeline(b["vis"], b["flag"], b["uvw"], b["z"], b["freq_old"], b["ra0"],
                                b["dec0"], b["ra1"], b["dec1"], b["freq_new"], b["width"])
           for b in blocks]
    outs = [(np.empty_like(c[0]), np.empty_like(c[1]), np.empty_like(c[2])) for c in cpu]
    t = time.perf_counter()
    ms_ops.full_pipeline_batch_into(
        [b["vis"] for b in blocks], [b["flag"] for b in blocks], [b["uvw"] for b in blocks],
        [b["z"] for b in blocks], [b["freq_old"] for b in blocks],
        [b["ra0"] for b in blocks], [b["dec0"] for b in blocks],
        [b["ra1"] for b in blocks], [b["dec1"] for b in blocks],
        [b["freq_new"] for b in blocks],
        [o[0] for o in outs], [o[1] for o in outs], [o[2] for o in outs],
        width_old_list=[b["width"] for b in blocks])
    dt = time.perf_counter() - t
    on_gpu = ms_ops.built_with_cuda() and ms_ops.gpu_available()
    check("the launch ran on the GPU", on_gpu or dry_run,
          f"{dt:.2f} s" if on_gpu else (ms_ops.gpu_last_error() or "it ran on the CPU"))
    for b, c, g in zip(blocks, cpu, outs):
        same_flags = np.array_equal(c[1], g[1])
        good = ~c[1]
        d_vis = float(np.abs(c[0][good] - g[0][good]).max()) if good.any() else 0.0
        rms = float(np.abs(c[0][good]).std()) if good.any() else 1.0
        d_uvw = float(np.abs(c[2] - g[2]).max())
        finite = bool(np.isfinite(g[0][~g[1]]).all())
        check(b["label"], same_flags and finite and d_vis < 2e-5 * rms and d_uvw < 1e-9,
              f"flags {'equal' if same_flags else 'DIFFERENT'}, visibilities differ by at most "
              f"{d_vis / rms:.1e} of their rms, UVW by {d_uvw:.1e} m")


# ---------------------------------------------------------------------------
def check_stack(work, dry_run, keep):
    print("3. A stack: device='cpu' against device='gpu'")
    import simulate
    from vista import ViSta
    from vista.stackio import open_stack

    sample = os.path.join(work, "sample")
    if not os.path.exists(os.path.join(sample, "input_list.txt")):
        rng = np.random.default_rng(3)
        simulate.make_sample(
            sample, n_sources=6, z_range=(2.0, 3.0), rest_freq_ghz=LINE, line_flux=0.4,
            fwhm_kms=300, cont_flux=2e-4, size_arcsec=0.6, noise=0.05, rmax=600, seed=5,
            offset_arcsec=lambda _r: tuple(rng.uniform(-20, 20, 2)),
            band_offset_kms=lambda _r: float(rng.uniform(-200, 200)))
    runs = {}
    for device in ("cpu", "gpu"):
        if device == "gpu" and dry_run:
            device_used = "cpu"
        else:
            device_used = device
        out = os.path.join(work, f"stack_{device}")
        v = ViSta(os.path.join(sample, "input_list.txt"), output_name=out, device=device_used,
                  verbose=False)
        v.stack(line_freq=LINE, line_window=500, velocity_range=900)
        v.extract(figures_draw=False, n_bootstrap=100, n_null=100)
        t = time.perf_counter()
        result = v.run()
        runs[device] = (out + ".zarr", result, time.perf_counter() - t)
        print(f"        device={device}: {runs[device][2]:.1f} s", flush=True)
    failed = ms_ops.built_with_cuda() and not ms_ops.gpu_available()
    check("the GPU stayed on for the whole run", (not failed) or dry_run,
          ms_ops.gpu_last_error() if failed else "")

    a, b = open_stack(runs["cpu"][0]), open_stack(runs["gpu"][0])
    same_dd = a.dd_ids() == b.dd_ids()
    check("the two stacks have the same datasets", same_dd)
    worst = {}
    equal = {"FLAG": True, "rows": True}
    for dd in (a.dd_ids() if same_dd else []):
        cols = [c for c in ("DATA", "CORRECTED_DATA", "MODEL_DATA", "FLAG", "UVW", "WEIGHT")
                if c in a.columns(dd)]
        pa, pb = list(a.pieces(dd)), list(b.pieces(dd))
        if pa != pb:
            equal["rows"] = False
            continue
        for row0, nrow in pa:
            x, y = a.read(dd, row0, nrow, cols), b.read(dd, row0, nrow, cols)
            equal["FLAG"] &= bool(np.array_equal(x["FLAG"], y["FLAG"]))
            good = ~x["FLAG"]
            for c in cols:
                if c == "FLAG":
                    continue
                if x[c].shape == good.shape:
                    d = np.abs(x[c][good] - y[c][good])
                    scale = float(np.abs(x[c][good]).std()) or 1.0
                else:
                    d = np.abs(x[c] - y[c])
                    scale = float(np.abs(x[c]).std()) or 1.0
                worst[c] = max(worst.get(c, 0.0), float(d.max()) / scale if d.size else 0.0)
    check("same rows", equal["rows"])
    check("FLAG identical", equal["FLAG"])
    for c, d in worst.items():
        limit = 1e-9 if c in ("UVW", "WEIGHT") else 2e-5
        check(f"{c}", d < limit, f"largest difference {d:.1e} of its rms")
    fa, fb = (runs[k][1] for k in ("cpu", "gpu"))
    for key in ("flux", "continuum"):
        va, vb, err = fa[key]["value"], fb[key]["value"], fa[key]["error"]
        check(f"measured {key}", abs(va - vb) < 1e-3 * err,
              f"cpu {va:.6g}, gpu {vb:.6g} (error {err:.2g})")
    if not keep:
        for device in ("cpu", "gpu"):
            shutil.rmtree(runs[device][0], ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True, help="directory for the sample and the stacks")
    ap.add_argument("--dry-run", action="store_true",
                    help="no GPU needed: the GPU calls fall back to the CPU; tests this script")
    ap.add_argument("--keep", action="store_true", help="keep the two stacks")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    seen = check_build(args.dry_run)
    if seen or args.dry_run:
        check_kernel(args.dry_run)
        check_stack(args.work, args.dry_run, args.keep)
    ok = all(RESULTS)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} ({sum(RESULTS)} of {len(RESULTS)} checks)"
          + ("  [dry run: no GPU was used]" if args.dry_run else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
