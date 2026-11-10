"""What a perfect measurement gives on this sample: the reference of the
noiseless runs.

In a heterogeneous sample the galaxies do not share their uv coverage: the
short baselines come mostly from the compact arrays, the long ones from the
extended ones, so each annulus of the stack averages a different subset of
galaxies.  The stacked visibility function is then not ``F g(b)`` of the
mean galaxy, and a one-Gaussian model fitted to it returns a flux at zero
baseline that depends on which galaxies sit at which baselines -- in any
code.  This script separates that from the errors of a pipeline:

1. it stacks the noiseless sample (ViSta-HPC, one pass) and checks, galaxy by
   galaxy, that every cell of the statistics holds the true visibility of
   that galaxy (stacked / true, which must be 1);
2. it builds the true cells of the stack: each annulus and channel averages
   the true visibilities of the galaxies with the weights they have *there*;
3. it fits the one-Gaussian model (line: F phi(v) g(b); continuum: C g(b))
   to these true cells: the result is what an exact pipeline with that
   model measures on this sample.

The weights are those of ViSta-HPC (weights x A^2) or of ViSta 1.0 (weights
as they are, H0 = 70).  Writes ``WORK/results/ideal.json``::

    python ideal.py --work WORK --extract-opt /path/to/ViSta-HPC
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
C_KMS = 299792.458

STACK = """
import sys
from vista import ViSta
v = ViSta(sys.argv[1], output_name=sys.argv[3] + "/stack")
v.stack(line_freq=float(sys.argv[2]), line_window=550, velocity_range=900)
v.extract()                  # the statistics are written with the stack
v.run(only="stack")
"""


def phi(v, v0, fwhm):
    s = fwhm / 2.3548200450309493
    return np.exp(-0.5 * ((v - v0) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def main():
    import simulate
    from scipy.optimize import least_squares
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--extract-opt", required=True, help="built checkout of ViSta-HPC extract_opt")
    ap.add_argument("--rest-freq", type=float, default=345.79599)
    a = ap.parse_args()
    data = os.path.join(a.work, "data", "noiseless")
    tmp = os.path.join(a.work, "runs", "ideal")
    os.makedirs(tmp, exist_ok=True)
    if not os.path.exists(os.path.join(tmp, "stack_line_stats.npy")):
        subprocess.run([sys.executable, "-c", STACK, os.path.join(data, "input_list.txt"),
                        str(a.rest_freq), tmp], check=True,
                       env=dict(os.environ, PYTHONPATH=a.extract_opt))
    sys.path.insert(0, a.extract_opt)
    from vista.extract.statistics import SufficientStatistics

    truth = json.load(open(os.path.join(data, "truth.json")))
    src, z_ref = truth["sources"], truth["z_ref"]
    rest = a.rest_freq * 1e9
    A = np.array([s["primary_beam"] for s in src])
    out = {}
    for kind in ("line", "cont"):
        st = SufficientStatistics.load(os.path.join(tmp, f"stack_{kind}_stats.npy"))
        v = C_KMS * (rest - st.freq_grid) / rest
        cells, check = [], []
        for i, s in enumerate(src):
            S = st.sums[i]
            W = S[..., 2]
            ok = W > 0
            b = np.where(ok, S[..., 3] / np.where(ok, W, 1), 0.0)
            re = np.where(ok, S[..., 0] / np.where(ok, W, 1), 0.0)
            if kind == "line":
                spec = s["line_flux"] * phi(v, s["v0_kms"], s["fwhm_kms"])
                g = simulate.shape_visibility(b, s["size_arcsec"])
                core = np.abs(v - s["v0_kms"]) < 0.5 * s["fwhm_kms"]
            else:
                nu = st.freq_grid / (1 + s["z"])
                spec = s["cont_flux"] * (nu / (rest / (1 + s["z"]))) ** s["cont_index"]
                g = simulate.shape_visibility(b, s["cont_size_arcsec"])
                core = np.abs(v) < 100
            true = A[i] * spec[:, None] * g                    # as observed (attenuated)
            m = ok & core[:, None]
            check.append(float(np.sum(re[m] * W[m]) / np.sum(true[m] * W[m])))
            cells.append((W, b, true))
        res = {"stacked_over_true": check}
        for scheme, (H0, Om0, pw) in {"vista_hpc": (67.4, 0.315, 2), "vista_casa": (70.0, 0.3, 0)}.items():
            alpha = np.array([simulate.alpha_factor(s["z"], z_ref, H0, Om0) for s in src])
            num = sum(A[i] ** pw * W * alpha[i] * true / A[i] for i, (W, _, true) in enumerate(cells))
            den = sum(A[i] ** pw * W for i, (W, _, _) in enumerate(cells))
            wb = sum(W * b for W, b, _ in cells)
            wsum = sum(W for W, _, _ in cells)
            ok = den > 0
            T = np.where(ok, num / np.where(ok, den, 1), 0.0)
            bm = np.where(wsum > 0, wb / np.where(wsum > 0, wsum, 1), 0.0)
            if kind == "line":
                def resid(p):
                    mod = p[0] * phi(v, p[1], abs(p[2]))[:, None] * simulate.shape_visibility(bm, abs(p[3]))
                    return (np.sqrt(den) * (T - mod))[ok]
                p = least_squares(resid, [0.3, 0.0, 300.0, 0.5], x_scale=[0.1, 10, 50, 0.1]).x
                res[scheme] = {"line_flux": p[0], "v0_kms": p[1], "fwhm_kms": abs(p[2]),
                               "size_arcsec": abs(p[3])}
            else:
                sel = ok & (np.abs(v) < 100)[:, None]

                def resid(p):
                    return (np.sqrt(den) * (T - p[0] * simulate.shape_visibility(bm, abs(p[1]))))[sel]
                p = least_squares(resid, [2e-4, 0.3], x_scale=[1e-4, 0.1]).x
                res[scheme] = {"cont_flux": p[0], "cont_size_arcsec": abs(p[1])}
        out[kind] = res
    ideal = {}
    for scheme in ("vista_hpc", "vista_casa"):
        ideal[scheme] = {**out["line"][scheme], **out["cont"][scheme]}
    ideal["stacked_over_true"] = {"line": out["line"]["stacked_over_true"],
                                  "continuum": out["cont"]["stacked_over_true"]}
    os.makedirs(os.path.join(a.work, "results"), exist_ok=True)
    json.dump(ideal, open(os.path.join(a.work, "results", "ideal.json"), "w"), indent=1, default=float)
    for k in ("vista_hpc", "vista_casa"):
        print(k, {q: round(float(x), 5) for q, x in ideal[k].items()})
    for k, r in ideal["stacked_over_true"].items():
        print(f"stacked / true, {k}: median {np.median(r):.4f}, range {min(r):.4f}-{max(r):.4f}")


if __name__ == "__main__":
    main()
