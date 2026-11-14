"""The compute kernel called from Python, to compare with the same kernel
called from C++ (``bench_binding.cpp``): what the pybind11 interface costs.

``ms_ops.full_pipeline`` checks the arrays, allocates the three output arrays,
releases the GIL and calls ``cpu_full_pipeline`` -- the very function the C++
program calls directly.  The difference between the two times is the price of
going through Python, and the C++ side measures it both with the output
buffers allocated once and allocated at every call, which is what the binding
has to do.

The data are built by the same formulas on both sides and both print the
checksum of the result: they must agree, or the two are not doing the same
work.  One JSON object per line.

    python bench_binding.py --threads 1
"""
import argparse
import json
import os
import sys
import time

Z, RA_OLD, DEC_OLD = 2.5, 1.0, 0.3
RA_NEW, DEC_NEW = 1.0 + 1e-5, 0.3 + 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--nchan", type=int, default=1000)
    ap.add_argument("--nchan-new", type=int, default=800)
    ap.add_argument("--ncorr", type=int, default=2)
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 8, 64, 512, 5000])
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))

    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    from vista import ms_ops
    ms_ops.omp_set_threads(a.threads)

    freq_old = (340e9 + 1.875e9 * np.arange(a.nchan) / (a.nchan - 1.0)) * (1.0 + Z)
    freq_new = freq_old[0] + (freq_old[-1] - freq_old[0]) * (
        0.05 + 0.90 * np.arange(a.nchan_new) / (a.nchan_new - 1.0))

    for nrow in a.rows:
        r = np.arange(nrow)[:, None, None]
        c = np.arange(a.nchan)[None, :, None]
        p = np.arange(a.ncorr)[None, None, :]
        phase = 0.001 * (r + 3.0 * c + 7.0 * p)
        vis = (np.sin(phase) + 1j * np.cos(phase)).astype(np.complex64)
        flag = ((r + c + p) % 97 == 0) & np.ones((nrow, a.nchan, a.ncorr), bool)
        uvw = np.empty((nrow, 3))
        uvw[:, 0] = 100.0 + 0.5 * np.arange(nrow)
        uvw[:, 1] = -50.0 + 0.25 * np.arange(nrow)
        uvw[:, 2] = 10.0 + 0.1 * np.arange(nrow)

        def call():
            return ms_ops.full_pipeline(vis, flag, uvw, Z, freq_old,
                                        RA_OLD, DEC_OLD, RA_NEW, DEC_NEW, freq_new)

        n = nrow * a.nchan * a.ncorr
        reps = int(max(3.0, min(200.0, 2e8 / (n + 1))))
        vo, fo, uo = call()                                   # warm-up
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            call()
            times.append(time.perf_counter() - t0)
        times.sort()
        print(json.dumps({
            "side": "python", "threads": a.threads, "nrow": nrow,
            "nchan_old": a.nchan, "nchan_new": a.nchan_new, "ncorr": a.ncorr,
            "reps": reps, "python": times[len(times) // 2], "python_min": times[0],
            "sum_vis": float(vo.real.astype(np.float64).sum() + vo.imag.astype(np.float64).sum()),
            "n_flag": int(fo.sum()), "sum_uvw": float(uo.sum())}), flush=True)


if __name__ == "__main__":
    main()
