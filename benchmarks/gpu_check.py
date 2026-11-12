"""
Does the GPU kernel work, and does it give what the CPU kernel gives?

In this version the kernel uses the GPU by itself whenever it was built with
CUDA and a GPU is visible, so the CPU results come from a second process
that hides the GPU (``CUDA_VISIBLE_DEVICES=""``).  Two checks, each printed
with PASS or FAIL, and one RESULT line at the end:

1. **the kernel**: blocks of random visibilities (1, 2 and 4 correlations,
   sources up to 60" from the phase centre, flags, unequal and decreasing
   channels) through ``ms_ops.full_pipeline`` on the CPU and on the GPU;
2. **a stack**: the datasets of an input list stacked on the CPU and on the
   GPU; every column of the two Measurement Sets is compared.

::

    python benchmarks/gpu_check.py --input sample/input_list.txt --line-freq 345.79599 \
                                   --work $SCRATCH/gpu_check_extract

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
    np.savez(os.path.join(args.work, f"kernel_{args.worker}.npz"), **out)
    t = time.perf_counter()
    stack = pipeline.ViSta(input_file=args.input, verbose=False)
    stack.run(ms_out=os.path.join(args.work, f"stack_{args.worker}.ms"),
              central_freq=args.line_freq * 1e9, velocity_range_kms=args.velocity_range)
    print(f"[{args.worker}] stack: {time.perf_counter() - t:.1f} s", flush=True)


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
    ap.add_argument("--input", required=True, help="input list of the datasets to stack")
    ap.add_argument("--line-freq", type=float, required=True, help="rest frequency, GHz")
    ap.add_argument("--velocity-range", type=float, default=900.0, help="km/s")
    ap.add_argument("--work", required=True)
    ap.add_argument("--worker", choices=["cpu", "gpu"], help=argparse.SUPPRESS)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    if args.worker:
        return worker(args)

    for device in ("cpu", "gpu"):
        env = dict(os.environ)
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run([sys.executable, os.path.abspath(__file__), "--input", args.input,
                        "--line-freq", str(args.line_freq), "--velocity-range",
                        str(args.velocity_range), "--work", args.work, "--worker", device],
                       env=env, check=True)
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
    print("3. A stack: CPU against GPU")
    compare_tables(os.path.join(args.work, "stack_cpu.ms"), os.path.join(args.work, "stack_gpu.ms"))
    ok = all(RESULTS)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} ({sum(RESULTS)} of {len(RESULTS)} checks)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
