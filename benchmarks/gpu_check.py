"""
Does the GPU kernel work, and does it give what the CPU kernel gives?

In this version the kernel uses the GPU by itself whenever it was built with
CUDA and a GPU is visible, so the CPU results come from a second process
that hides the GPU (``CUDA_VISIBLE_DEVICES=""``).  After a check of the build
(1), three checks, each printed with PASS or FAIL, and one RESULT line at
the end:

2. **the kernel**: blocks of random visibilities (1, 2 and 4 correlations,
   sources up to 60" from the phase centre, flags, unequal and decreasing
   channels) through ``ms_ops.full_pipeline`` on the CPU and on the GPU;
3. **the batched launch**: blocks of the same shape through
   ``ms_ops.full_pipeline_batch_same_shape`` (one launch) against the same
   blocks one at a time on the CPU and on the GPU; and the grouping of
   ``pipeline.full_pipeline_batch``, with a block of another shape and one
   with irregular channels, which must go alone;
4. **a stack**: the datasets of an input list stacked on the CPU and on the
   GPU, with ``--gpu-batch`` datasets per launch (the number of batched
   launches is reported); every column of the two Measurement Sets is
   compared.

::

    python benchmarks/gpu_check.py --input sample/input_list.txt --line-freq 345.79599 \
                                   --work $SCRATCH/gpu_check_extract
    python benchmarks/gpu_check.py --kernel-only --work /tmp/gpu_check   # 1-3, no data

The input list can be the sample that the same check of the ``extract_opt``
branch writes (``<its work directory>/sample/input_list.txt``).
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""), flush=True)


def kernel_blocks(rng):
    blocks = []
    for ncorr, offset, nchan_old, nchan_new, kind in (
            (2, 0.0, 120, 90, "equal"), (2, 5.0, 96, 128, "equal"), (1, 20.0, 64, 40, "decreasing"),
            (4, 60.0, 200, 150, "unequal"), (2, 20.0, 480, 100, "widths")):
        nrow = int(rng.integers(300, 900))
        z = float(rng.uniform(0.5, 3.0))
        f0, step = rng.uniform(90e9, 250e9), rng.uniform(1e6, 8e6)
        freq = f0 + np.arange(nchan_old) * step
        width = np.full(nchan_old, (0.8 if kind == "widths" else 1.0) * step) * (1 + z)
        if kind == "decreasing":
            freq = freq[::-1].copy()
        elif kind == "unequal":
            freq[nchan_old // 2:] += 37 * step
        freq_old = freq * (1 + z)
        lo, hi = np.sort(freq_old)[[nchan_old // 8, -nchan_old // 8]]
        vis = (rng.normal(size=(nrow, nchan_old, ncorr))
               + 1j * rng.normal(size=(nrow, nchan_old, ncorr))).astype(np.complex64)
        ra0, dec0 = np.radians(rng.uniform(0, 360)), np.radians(rng.uniform(-70, 20))
        blocks.append(dict(
            vis=vis, flag=rng.random((nrow, nchan_old, ncorr)) < 0.05,
            uvw=rng.normal(0, 1500, (nrow, 3)), z=z, freq_old=freq_old, ra0=ra0, dec0=dec0,
            ra1=ra0 + np.radians(offset / 3600) / np.cos(dec0),
            dec1=dec0 + np.radians(0.7 * offset / 3600),
            freq_new=np.linspace(lo, hi, nchan_new), width=width,
            label=f"{ncorr} corr, {offset:.0f}\" off-centre, {kind} channels"))
    return blocks


def batch_blocks(rng, n=4, nrow=500, nchan_old=128, nchan_new=96, ncorr=2):
    """Blocks of the same shape with contiguous channels: each its own
    redshift, band, source and output grid."""
    blocks = []
    for k in range(n):
        z = float(rng.uniform(2.0, 3.0))
        f0, step = rng.uniform(85e9, 115e9), rng.uniform(1.5e6, 2.5e6)
        freq_old = (f0 + np.arange(nchan_old) * step) * (1 + z)
        lo, hi = freq_old[nchan_old // 8], freq_old[-nchan_old // 8]
        ra0, dec0 = np.radians(rng.uniform(0, 360)), np.radians(rng.uniform(-60, 10))
        offset = rng.uniform(0, 5)
        blocks.append(dict(
            vis=(rng.normal(size=(nrow, nchan_old, ncorr))
                 + 1j * rng.normal(size=(nrow, nchan_old, ncorr))).astype(np.complex64),
            flag=rng.random((nrow, nchan_old, ncorr)) < 0.03,
            uvw=rng.normal(0, 1500, (nrow, 3)), z=z, freq_old=freq_old, ra0=ra0, dec0=dec0,
            ra1=ra0 + np.radians(offset / 3600) / np.cos(dec0),
            dec1=dec0 + np.radians(0.6 * offset / 3600),
            freq_new=np.linspace(lo, hi, nchan_new), width=np.full(nchan_old, step * (1 + z))))
    return blocks


def mixed_blocks(rng):
    """For the grouping: three blocks of one shape, one of another shape, and
    one of the first shape with a gap in its channels (irregular widths)."""
    same = batch_blocks(rng, n=4)
    other = batch_blocks(rng, n=1, nrow=300)
    gap = same.pop()
    gap["freq_old"] = gap["freq_old"].copy()
    gap["freq_old"][64:] += 40 * (gap["freq_old"][1] - gap["freq_old"][0])
    return same + other + [gap]


def single(ms_ops, b):
    v, f, u = ms_ops.full_pipeline(b["vis"], b["flag"], b["uvw"], b["z"], b["freq_old"],
                                   b["ra0"], b["dec0"], b["ra1"], b["dec1"], b["freq_new"],
                                   b["width"])
    return np.asarray(v), np.asarray(f), np.asarray(u)


def worker(args):
    """One device: the kernel on the random blocks, then the stack."""
    from vista import ms_ops
    from vista import pipeline
    print(f"[{args.worker}] the kernel sees a GPU: {pipeline._gpu_available}", flush=True)
    out = {}
    for k, b in enumerate(kernel_blocks(np.random.default_rng(7))):
        v, f, u = ms_ops.full_pipeline(b["vis"], b["flag"], b["uvw"], b["z"], b["freq_old"],
                                       b["ra0"], b["dec0"], b["ra1"], b["dec1"], b["freq_new"],
                                       b["width"])
        out[f"vis{k}"], out[f"flag{k}"], out[f"uvw{k}"] = np.asarray(v), np.asarray(f), np.asarray(u)
    out["gpu"] = np.array(bool(pipeline._gpu_available))
    # the batched launch: the same blocks one at a time, and (GPU) in one launch
    for k, b in enumerate(batch_blocks(np.random.default_rng(11))):
        out[f"bvis{k}"], out[f"bflag{k}"], out[f"buvw{k}"] = single(ms_ops, b)
    out["has_batch"] = np.array(bool(pipeline._gpu_batch))
    if pipeline._gpu_batch:
        bs = batch_blocks(np.random.default_rng(11))
        v, f, u = ms_ops.full_pipeline_batch_same_shape(
            np.stack([b["vis"] for b in bs]), np.stack([b["flag"] for b in bs]),
            np.stack([b["uvw"] for b in bs]), np.array([b["z"] for b in bs]),
            np.stack([b["freq_old"] for b in bs]), np.array([b["ra0"] for b in bs]),
            np.array([b["dec0"] for b in bs]), np.array([b["ra1"] for b in bs]),
            np.array([b["dec1"] for b in bs]), np.stack([b["freq_new"] for b in bs]))
        out["Bvis"], out["Bflag"], out["Buvw"] = np.asarray(v), np.asarray(f), np.asarray(u)
        # the grouping: count the launches
        launches = []
        real = ms_ops.full_pipeline_batch_same_shape

        def counted(*a, **kw):
            launches.append(len(a[0]))
            return real(*a, **kw)
        ms_ops.full_pipeline_batch_same_shape = counted
        ms = mixed_blocks(np.random.default_rng(13))
        res = pipeline.full_pipeline_batch(
            [b["vis"] for b in ms], [b["flag"] for b in ms], [b["uvw"] for b in ms],
            [b["z"] for b in ms], [b["freq_old"] for b in ms], [b["ra0"] for b in ms],
            [b["dec0"] for b in ms], [b["ra1"] for b in ms], [b["dec1"] for b in ms],
            [b["freq_new"] for b in ms], [b["width"] for b in ms])
        worst = 0.0
        for b, (v, f, u) in zip(ms, res):
            v1, f1, _ = single(ms_ops, b)
            good = ~f1
            same = np.array_equal(np.asarray(f), f1)
            d = float(np.abs(np.asarray(v)[good] - v1[good]).max()) / float(np.abs(v1[good]).std())
            worst = max(worst, d if same else np.inf)
        out["mixed_launches"] = np.array(launches, dtype=int)
        out["mixed_worst"] = np.array(worst)
        ms_ops.full_pipeline_batch_same_shape = counted       # stays counted for the stack
        launches.clear()
    np.savez(os.path.join(args.work, f"kernel_{args.worker}.npz"), **out)
    if args.kernel_only:
        return
    os.environ["VISTA_GPU_BATCH"] = str(args.gpu_batch)
    t = time.perf_counter()
    stack = pipeline.ViSta(input_file=args.input, verbose=False)
    stack.run(ms_out=os.path.join(args.work, f"stack_{args.worker}.ms"),
              central_freq=args.line_freq * 1e9, velocity_range_kms=args.velocity_range)
    print(f"[{args.worker}] stack: {time.perf_counter() - t:.1f} s", flush=True)
    if pipeline._gpu_batch:
        np.save(os.path.join(args.work, "stack_launches.npy"), np.array(launches, dtype=int))


def compare_tables(a, b):
    from casacore.tables import table
    ta, tb = table(a, ack=False), table(b, ack=False)
    check("same number of rows", ta.nrows() == tb.nrows(), f"{ta.nrows()} and {tb.nrows()}")
    if ta.nrows() != tb.nrows():
        return
    step = 200000
    worst, flags = {}, True
    for r0 in range(0, ta.nrows(), step):
        n = min(step, ta.nrows() - r0)
        fa, fb = ta.getcol("FLAG", r0, n), tb.getcol("FLAG", r0, n)
        flags &= bool(np.array_equal(fa, fb))
        good = ~fa
        for c in ("DATA", "UVW", "WEIGHT"):
            x, y = ta.getcol(c, r0, n), tb.getcol(c, r0, n)
            if x.shape == good.shape:
                x, y = x[good], y[good]
            scale = float(np.abs(x).std()) or 1.0
            worst[c] = max(worst.get(c, 0.0), float(np.abs(x - y).max()) / scale if x.size else 0.0)
    check("FLAG identical", flags)
    for c, d in worst.items():
        check(c, d < (1e-9 if c != "DATA" else 2e-5), f"largest difference {d:.1e} of its rms")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", help="input list of the datasets to stack")
    ap.add_argument("--line-freq", type=float, help="rest frequency, GHz")
    ap.add_argument("--kernel-only", action="store_true",
                    help="checks 1-3 only: random blocks, no datasets")
    ap.add_argument("--gpu-batch", type=int, default=3,
                    help="datasets per GPU launch in the stack (small GPUs: 2-3)")
    ap.add_argument("--velocity-range", type=float, default=900.0, help="km/s")
    ap.add_argument("--work", required=True)
    ap.add_argument("--worker", choices=["cpu", "gpu"], help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not args.kernel_only and not (args.input and args.line_freq):
        ap.error("--input and --line-freq are needed (or --kernel-only)")
    os.makedirs(args.work, exist_ok=True)
    if args.worker:
        return worker(args)

    for f in ("stack_launches.npy",):
        if os.path.exists(os.path.join(args.work, f)):
            os.remove(os.path.join(args.work, f))
    for device in ("cpu", "gpu"):
        env = dict(os.environ)
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        cmd = [sys.executable, os.path.abspath(__file__), "--work", args.work, "--worker", device,
               "--gpu-batch", str(args.gpu_batch)]
        if args.kernel_only:
            cmd += ["--kernel-only"]
        else:
            cmd += ["--input", args.input, "--line-freq", str(args.line_freq),
                    "--velocity-range", str(args.velocity_range)]
        subprocess.run(cmd, env=env, check=True)
    cpu = np.load(os.path.join(args.work, "kernel_cpu.npz"))
    gpu = np.load(os.path.join(args.work, "kernel_gpu.npz"))
    print("1. The build")
    check("the CPU process did not see the GPU", not bool(cpu["gpu"]))
    check("the GPU process saw a GPU (kernel built with -DWITH_CUDA=ON)", bool(gpu["gpu"]))
    print("2. The kernel: CPU against GPU")
    for k, b in enumerate(kernel_blocks(np.random.default_rng(7))):
        same = np.array_equal(cpu[f"flag{k}"], gpu[f"flag{k}"])
        good = ~cpu[f"flag{k}"]
        rms = float(np.abs(cpu[f"vis{k}"][good]).std())
        d = float(np.abs(cpu[f"vis{k}"][good] - gpu[f"vis{k}"][good]).max()) / rms
        du = float(np.abs(cpu[f"uvw{k}"] - gpu[f"uvw{k}"]).max())
        check(b["label"], same and d < 2e-5 and du < 1e-9,
              f"flags {'equal' if same else 'DIFFERENT'}, visibilities differ by at most "
              f"{d:.1e} of their rms, UVW by {du:.1e} m")
    print("3. The batched launch")
    check("the kernel has the batched launch (full_pipeline_batch_same_shape)",
          bool(gpu["has_batch"]))
    if bool(gpu["has_batch"]):
        for k in range(len(batch_blocks(np.random.default_rng(11)))):
            fb, vb, ub = gpu["Bflag"][k], gpu["Bvis"][k], gpu["Buvw"][k]
            for ref, name in ((cpu, "CPU"), (gpu, "GPU")):
                f1, v1, u1 = ref[f"bflag{k}"], ref[f"bvis{k}"], ref[f"buvw{k}"]
                same = np.array_equal(fb, f1)
                good = ~f1
                d = float(np.abs(vb[good] - v1[good]).max()) / float(np.abs(v1[good]).std())
                du = float(np.abs(ub - u1).max())
                check(f"block {k} in one launch = alone on the {name}",
                      same and d < 2e-5 and du < 1e-9,
                      f"flags {'equal' if same else 'DIFFERENT'}, visibilities {d:.1e} of "
                      f"their rms, UVW {du:.1e} m")
        launches = [int(x) for x in gpu["mixed_launches"]]
        check("grouping: 3 blocks of one shape in one launch, the other 2 alone",
              launches == [3], f"launches with {launches} blocks")
        worst = float(gpu["mixed_worst"])
        check("grouping: every block = alone", worst < 2e-5,
              f"visibilities differ by at most {worst:.1e} of their rms")
    if not args.kernel_only:
        print("4. A stack: CPU against GPU")
        path = os.path.join(args.work, "stack_launches.npy")
        if os.path.exists(path):
            n = np.load(path)
            check("the GPU stack used batched launches", len(n) > 0,
                  f"{len(n)} launches, {int(n.sum()) if len(n) else 0} datasets"
                  f" (batch of {args.gpu_batch})")
        compare_tables(os.path.join(args.work, "stack_cpu.ms"),
                       os.path.join(args.work, "stack_gpu.ms"))
    ok = all(RESULTS)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} ({sum(RESULTS)} of {len(RESULTS)} checks)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
