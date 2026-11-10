"""Monte Carlo of the tutorial: the same twelve galaxies, many noise realisations.

The tutorial notebook (examples/ViSta_tutorial.ipynb) measures one noise
realisation, so every number in it carries the same fluctuation.  Here the
sample of the notebook (same galaxies, same weights, same settings) is
observed again with ``--n`` new realisations of the noise, and the stacked
line is measured each time in ``detect`` (width known) and ``search`` (free)
mode.  Over the realisations: the mean of measured / true (the bias), the
scatter, the mean and rms of the pulls (measured - true) / error, and the
fraction of realisations whose ±1 sigma contains the truth (0.68 if the errors
are right).

Usage::

    python mc_tutorial.py --work mc_tutorial --n 50
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))

REST_GHZ = 345.79599
# the simulated sample of the tutorial (section 1)
SIM = dict(n_sources=12, z_range=(2.0, 3.0), line_flux=0.4, fwhm_kms=300.0, size_arcsec=0.8,
           cont_flux=2e-4, noise=0.1, kappa=1.3, hanning=True, depth_spread_dex=0.2,
           v_jitter_kms=0.0, flux_scatter_dex=0.0, same_luminosity=True, rmax=600.0, seed=0)

QUANTITIES = [("flux", "flux", "flux_error"), ("peak", "peak", "peak_error"),
              ("fwhm", "fwhm_kms", "fwhm_error_kms"), ("v0", "v0_kms", "v0_error_kms"),
              ("size", "theta_fwhm_arcsec", "theta_error_arcsec")]


def realisation(k, work, n_boot):
    import simulate
    from vista import ViSta
    d = os.path.join(work, "run")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    counter = itertools.count()
    noise_seed = (lambda rng: 100000 * k + next(counter)) if k > 0 else None
    kw = dict(SIM)
    if noise_seed is not None:
        kw["noise_seed"] = noise_seed
    truth = simulate.make_sample(os.path.join(d, "sim"), rest_freq_ghz=REST_GHZ, **kw)
    v = ViSta(truth["input_list"], output_name=os.path.join(d, "stack"))
    v.stack(line_freq=REST_GHZ, line_window=500, velocity_range=900)
    common = dict(weighting_scheme="natural", z_norm=True, pb_corr=True, flux_measure="gauss",
                  n_null=0, n_null_phase=0, n_bootstrap=n_boot, seed=1, figures_draw=False)
    v.extract(**common)
    v.run(only="stack")                       # the stack and its statistics
    out = {"realisation": k}
    # 'detect': centre and width fixed to the true ones; 'search': free in the window
    for mode, options in (("detect", dict(v0_range=(0, 0), fwhm_range=(SIM["fwhm_kms"],) * 2)),
                          ("search", dict())):
        v.extract(**common, **options)
        r = v.run(only="measure").line
        ln = r.summary["line"]
        out[mode] = {q: [ln.get(key), ln.get(err) or (r.summary.get(err) if q == "size" else None)]
                     for q, key, err in QUANTITIES}
        out[mode]["snr"] = r.summary["detection"]["snr"]
    out["truth"] = {"flux": truth["line_flux_stack"], "peak": truth["line_peak_stack"],
                    "fwhm": truth["fwhm_kms_stack"], "v0": truth["v0_kms_stack"],
                    "size": SIM["size_arcsec"]}
    shutil.rmtree(d, ignore_errors=True)
    return out


def summary(rows):
    lines = []
    t = rows[0]["truth"]
    for mode in ("detect", "search"):
        lines.append(f"\n{mode} ({len(rows)} realisations, mean S/N "
                     f"{np.mean([r[mode]['snr'] for r in rows]):.1f})")
        lines.append(f"  {'':6s} {'true':>8s} {'mean':>9s} {'bias':>17s} {'scatter':>9s} "
                     f"{'mean err':>9s} {'pull mean':>10s} {'pull rms':>9s} {'in 1 sigma':>10s}")
        for q, _, _ in QUANTITIES:
            if mode == "detect" and q in ("fwhm", "v0"):
                continue
            m = np.array([r[mode][q][0] for r in rows], float)
            e = np.array([r[mode][q][1] if r[mode][q][1] is not None else np.nan for r in rows], float)
            tv = t[q]
            pull = (m - tv) / e
            n = len(m)
            bias = m.mean() - tv
            rel = f"{100 * bias / tv:+.1f}% ± {100 * m.std(ddof=1) / np.sqrt(n) / tv:.1f}%" if tv \
                else f"{bias:+.2f} ± {m.std(ddof=1) / np.sqrt(n):.2f}"
            lines.append(f"  {q:6s} {tv:8.4g} {m.mean():9.4g} {rel:>17s} {m.std(ddof=1):9.3g} "
                         f"{np.nanmean(e):9.3g} {np.nanmean(pull):+10.2f} {np.sqrt(np.nanmean(pull ** 2)):9.2f} "
                         f"{np.nanmean(np.abs(pull) <= 1):10.2f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--first", type=int, default=0, help="0 is the realisation of the notebook")
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    path = os.path.join(a.work, "mc_tutorial.json")
    rows = json.load(open(path)) if os.path.exists(path) else []
    done = {r["realisation"] for r in rows}
    for k in range(a.first, a.first + a.n):
        if k in done:
            continue
        t0 = time.time()
        rows.append(realisation(k, a.work, a.n_boot))
        json.dump(rows, open(path, "w"), indent=1, default=float)
        r = rows[-1]
        print(f"[{k:3d}] {time.time() - t0:4.0f} s  detect F = {r['detect']['flux'][0]:.3f} "
              f"± {r['detect']['flux'][1]:.3f}  search F = {r['search']['flux'][0]:.3f}, "
              f"peak = {r['search']['peak'][0] * 1e3:.3f} mJy, FWHM = {r['search']['fwhm'][0]:.0f}",
              flush=True)
    text = summary(rows)
    print(text)
    open(os.path.join(a.work, "summary.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
