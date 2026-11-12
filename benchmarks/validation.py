"""
Validation of the measurement on synthetic stacks, where the truth is known.

Many noise realisations of the same sample are measured, and the script
counts how well the quoted numbers hold: the *bias* of the flux, its
*scatter*, and the *coverage* of the quoted error (the fraction of
realisations whose flux +- 1 sigma holds the truth; 0.68 for a right error).

The stacks are built directly as the compressed statistics that ViSta
measures (:mod:`vista.extract.statistics`), cell by cell, so a realisation
takes milliseconds to make.  What is simulated:

* an ALMA-like array for every source (random antennas, so every source has
  its own baselines), a line that is a Gaussian in velocity, a continuum,
  a source that is a point or a round Gaussian;
* sources at different redshifts and observed to different depths, so they
  do not weigh the same;
* noise on every visibility, larger than the weights say (``kappa``) and
  correlated between adjacent channels (``rho``);
* the continuum subtracted visibility by visibility on the channels outside
  the line window, as the stacking does, with the noise this adds.

::

    python benchmarks/validation.py --realisations 100 --out validation
    python benchmarks/validation.py --only small_n --realisations 200

writes ``validation.json`` (every number) and ``validation.md`` (the tables
of ``docs/extraction/validation.md``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from vista.extract.config import (BootstrapConfig, LineFitConfig, Lines, UVFitConfig,   # noqa: E402
                                  WeightingConfig)
from vista.extract.linefit import uv_gaussian                                           # noqa: E402
from vista.extract.measure import measure_line                                          # noqa: E402
from vista.extract.sources import Entry, amplitude_factors, group_by_position           # noqa: E402
from vista.extract.stack import measure_continuum                                       # noqa: E402
from vista.extract.statistics import SufficientStatistics                               # noqa: E402

C_KMS = 299792.458
LINE_GHZ = 345.79599
WINDOW = 500.0                        # the line window, km/s
EDGES = np.geomspace(3e3, 3e6, 19)    # 18 annuli of baseline length, wavelengths


# ---------------------------------------------------------------------------
# a synthetic stack
# ---------------------------------------------------------------------------
def make_stack(rng, n_sources=12, line_flux=0.4, fwhm=300.0, size=0.0, cont=2e-4,
               cont_size=None, noise=0.07, depth_spread=0.15, z_range=(2.0, 3.0),
               kappa=1.3, rho=0.3, nchan=90, dv=20.0, flux_scatter=0.0, mixed=False):
    """Statistics of a stack of ``n_sources`` sources: ``(line, continuum,
    entries, truth)``.

    ``line_flux`` [Jy km/s] and ``cont`` [Jy] are those of every source at the
    reference redshift (the median); ``size`` [arcsec, FWHM] of the source, 0
    for a point; ``mixed``: half of the sources are points whatever ``size``;
    ``flux_scatter`` [dex]: intrinsic scatter of the fluxes; ``noise`` [Jy]
    per visibility as the weights say, ``kappa`` times larger in truth, and
    ``depth_spread`` [dex] its scatter from one source to another.
    """
    cont_size = size if cont_size is None else cont_size
    z = np.sort(rng.uniform(*z_range, n_sources))
    entries = [Entry(dd=k, ms=f"synthetic_{k}.ms", z=float(z[k]),
                     ra=math.radians(10.0 + k), dec=math.radians(-20.0)) for k in range(n_sources)]
    factors, z_ref, _, _ = amplitude_factors(group_by_position(entries)[0], WeightingConfig(),
                                             verbose=False)
    vel = (np.arange(nchan) - (nchan - 1) / 2) * dv
    freq = LINE_GHZ * 1e9 * (1 - vel / C_KMS)
    sig_v = fwhm / 2.3548200450309493
    phi = np.exp(-0.5 * (vel / sig_v) ** 2) / (sig_v * math.sqrt(2 * math.pi))   # 1 / (km/s)
    free = np.abs(vel) > WINDOW
    a = (1 - math.sqrt(max(1 - 4 * rho * rho, 0))) / (2 * rho) if rho else 0.0   # MA(1): rho at lag 1
    line, model = {}, {}
    intrinsic = 10 ** rng.normal(0, flux_scatter, n_sources) if flux_scatter else np.ones(n_sources)
    if flux_scatter:
        intrinsic /= intrinsic.mean()
    for k in range(n_sources):
        # the array: 43 antennas within 600 m, 40 integrations, 2 correlations
        r, t = 600.0 * np.sqrt(rng.uniform(0, 1, 43)), rng.uniform(0, 2 * math.pi, 43)
        x, y = r * np.cos(t), r * np.sin(t)
        i, j = np.triu_indices(43, 1)
        b = np.hypot(x[i] - x[j], y[i] - y[j]) * (LINE_GHZ * 1e9 / (1 + z[k])) / (C_KMS * 1e3)
        ann = np.digitize(b, EDGES) - 1
        ok = (ann >= 0) & (ann < len(EDGES) - 1)
        n_vis = np.bincount(ann[ok], minlength=len(EDGES) - 1) * 80.0
        theta = 0.0 if (mixed and k % 2 == 0) else size
        ctheta = 0.0 if (mixed and k % 2 == 0) else cont_size
        g1 = np.bincount(ann[ok], uv_gaussian(b[ok], theta), len(EDGES) - 1) * 80.0
        g2 = np.bincount(ann[ok], uv_gaussian(b[ok], theta) ** 2, len(EDGES) - 1) * 80.0
        c1 = np.bincount(ann[ok], uv_gaussian(b[ok], ctheta), len(EDGES) - 1) * 80.0
        sb = np.bincount(ann[ok], b[ok], len(EDGES) - 1) * 80.0
        sigma_w = noise * 10 ** rng.normal(0, depth_spread)          # what the weights say
        w, sigma = 1.0 / sigma_w ** 2, sigma_w * kappa
        s_line = line_flux * intrinsic[k] / factors[k] * phi[:, None]        # per unit g, (nchan, 1)
        s_cont = cont * intrinsic[k] / factors[k]
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.where(n_vis > 0, sigma / np.sqrt(np.maximum(n_vis, 1)), 0.0)
        blocks = {}
        for part in ("re", "im"):
            e = rng.normal(size=(nchan + 1, len(n_vis)))
            m = (e[:-1] + a * e[1:]) / math.sqrt(1 + a * a) * sd     # noise of the cell means
            m_free = m[free].mean(axis=0)                            # what the subtraction takes
            blocks[part] = (m - m_free, m_free)
        dof = np.maximum(n_vis - 1, 0)
        n_free = free.sum()
        # scatter inside a cell: the noise of a visibility, plus that of the level subtracted
        var_line = sigma ** 2 * (1 + (1 + 2 * rho) / n_free)
        var_model = sigma ** 2 * (1 + 2 * rho) / n_free
        out_line = np.zeros((nchan, len(n_vis), 7))
        out_model = np.zeros((nchan, len(n_vis), 7))
        (m_re, mf_re), (m_im, mf_im) = blocks["re"], blocks["im"]
        sig1 = s_line * g1                                            # sum of the signal
        out_line[..., 0] = w * (sig1 + n_vis * m_re)
        out_line[..., 1] = w * (n_vis * m_im)
        out_line[..., 2] = w * n_vis
        out_line[..., 3] = w * sb
        out_line[..., 4] = n_vis
        out_line[..., 5] = w * (s_line ** 2 * g2 + 2 * m_re * sig1 + n_vis * m_re ** 2
                                + var_line * rng.chisquare(np.maximum(dof, 1), (nchan, len(n_vis)))
                                * (dof > 0))
        out_line[..., 6] = w * (n_vis * m_im ** 2
                                + var_line * rng.chisquare(np.maximum(dof, 1), (nchan, len(n_vis)))
                                * (dof > 0))
        out_model[..., 0] = w * (s_cont * c1 + n_vis * mf_re)
        out_model[..., 1] = w * (n_vis * mf_im)
        out_model[..., 2:5] = out_line[..., 2:5]
        mean_re = np.where(n_vis > 0, (s_cont * c1 + n_vis * mf_re) / np.maximum(n_vis, 1), 0.0)
        chi = var_model * rng.chisquare(np.maximum(dof, 1), len(n_vis)) * (dof > 0)
        out_model[..., 5] = w * (n_vis * mean_re ** 2 + chi)
        out_model[..., 6] = w * (n_vis * mf_im ** 2 + chi)
        line[k], model[k] = out_line, out_model
    meta = {"line_column": "CORRECTED_DATA"}
    truth = {"line_flux": line_flux, "cont": cont, "z_ref": z_ref, "fwhm": fwhm, "size": size}
    return (SufficientStatistics(line, freq, EDGES, meta=dict(meta)),
            SufficientStatistics(model, freq, EDGES, meta=dict(meta)), entries, truth)


# ---------------------------------------------------------------------------
# one realisation, measured
# ---------------------------------------------------------------------------
def one(args):
    """Measure one realisation of a case; returns the numbers kept."""
    case, seed, n_bootstrap = args
    rng = np.random.default_rng(seed)
    sim = {k: v for k, v in case.items() if k not in ("name", "extract", "what")}
    line, model, entries, truth = make_stack(rng, **sim)
    lines = Lines(LINE_GHZ, WINDOW)
    weighting = WeightingConfig(primary_beam=False)
    boot = BootstrapConfig(n_realisations=n_bootstrap, seed=seed)
    ex = dict(case.get("extract", {}))
    out = {}
    if case.get("what", "line") in ("line", "both"):
        cfg = LineFitConfig(n_null=0, n_null_phase=0, **ex.get("line", {}))
        r = measure_line(line, entries, lines, weighting, cfg, boot, verbose=False)
        b = r.summary["line"]
        out.update(flux=b["flux"], flux_error=b["flux_error"], method=b["flux_method"],
                   flux_fit=b["flux_gauss"], flux_fit_thermal=b["flux_gauss_error_thermal"],
                   flux_fit_bootstrap=b["flux_gauss_error_bootstrap"],
                   flux_fit_error=b["flux_gauss_error"],
                   flux_window=b["flux_area"], flux_window_error=b["flux_area_error"],
                   theta=b.get("theta_fwhm_arcsec") or 0.0, snr=b["detection"]["snr"],
                   model=(r.summary.get("size_test") or {}).get("adopted"))
    if case.get("what", "line") in ("continuum", "both"):
        c = measure_continuum(model, entries, lines, weighting,
                              UVFitConfig(**ex.get("continuum", {})), boot, verbose=False)
        out.update(cont=c.summary["flux_total"], cont_error=c.summary["flux_total_error"],
                   cont_theta=c.summary.get("theta_fwhm_arcsec") or 0.0)
    return out


def statistics(values, errors, truth):
    """Bias, scatter and coverage of a set of measurements of ``truth``."""
    v, e = np.asarray(values, float), np.asarray(errors, float)
    ok = np.isfinite(v) & np.isfinite(e) & (e > 0)
    v, e = v[ok], e[ok]
    pull = (v - truth) / e
    n = len(v)
    cover = float(np.mean(np.abs(pull) < 1))
    return {"n": int(n), "bias": float(v.mean() / truth - 1),
            "bias_error": float(v.std(ddof=1) / truth / math.sqrt(n)),
            "scatter": float(v.std(ddof=1) / truth),
            "quoted": float(np.median(e) / truth),
            "coverage": cover, "coverage_error": float(math.sqrt(cover * (1 - cover) / n)),
            "beyond_2sigma": float(np.mean(np.abs(pull) > 2)),
            "pull_rms": float(np.sqrt(np.mean(pull ** 2)))}


# ---------------------------------------------------------------------------
# the cases
# ---------------------------------------------------------------------------
BASE = dict(line_flux=0.4, fwhm=300.0, cont=2e-4, noise=0.07)

SMALL_N = [dict(BASE, name=f"{label}, {n} sources", n_sources=n, size=size, what="both",
                # the same depth of the whole stack for every N: the S/N does not change
                noise=0.07 * math.sqrt(n / 12.0),
                extract={"continuum": {"model": "gauss" if size else "point"}})
           for label, size in (("extended 0.8\"", 0.8), ("point", 0.0))
           for n in (5, 10, 20, 50)]

CASES = [
    dict(BASE, name="point sources, S/N ~ 30", n_sources=20, size=0.0),
    dict(BASE, name="point sources, S/N ~ 6", n_sources=20, size=0.0, line_flux=0.08),
    dict(BASE, name="point sources, S/N ~ 3", n_sources=20, size=0.0, line_flux=0.04),
    dict(BASE, name="extended 0.8\", S/N ~ 15", n_sources=20, size=0.8),
    dict(BASE, name="extended 0.8\", S/N ~ 5", n_sources=20, size=0.8, line_flux=0.13),
    dict(BASE, name="extended 0.8\", as a point (model=\"point\")", n_sources=20, size=0.8,
         extract={"line": {"size": "point"}}),
    dict(BASE, name="half points, half 0.8\"", n_sources=20, size=0.8, mixed=True),
    dict(BASE, name="extended 0.8\", fluxes scattered by 0.3 dex", n_sources=20, size=0.8,
         flux_scatter=0.3),
    dict(BASE, name="extended 0.8\", centre and width fixed", n_sources=20, size=0.8,
         line_flux=0.13, extract={"line": {"v0_range_kms": (0, 0), "fwhm_range_kms": (300, 300)}}),
]


def run(cases, n_real, n_bootstrap, processes, label):
    """Measure ``n_real`` realisations of every case; one row of numbers each."""
    rows = []
    with Pool(processes) as pool:
        for case in cases:
            t = time.time()
            got = pool.map(one, [(case, 1000 + i, n_bootstrap) for i in range(n_real)])
            row = {"name": case["name"], "n_sources": case["n_sources"]}
            if "flux" in got[0]:
                f = case["line_flux"]
                for key, err in (("flux", "flux_error"), ("flux_fit", "flux_fit_error"),
                                 ("flux_window", "flux_window_error"),
                                 ("flux_fit_thermal_only", "flux_fit_thermal"),
                                 ("flux_fit_bootstrap_only", "flux_fit_bootstrap")):
                    values = [g[key.replace("_thermal_only", "").replace("_bootstrap_only", "")]
                              for g in got]
                    row[key] = statistics(values, [g[err] for g in got], f)
                row["snr"] = float(np.median([g["snr"] for g in got]))
                row["gauss_chosen"] = float(np.mean([g["model"] == "gauss" for g in got]))
                row["window_quoted"] = float(np.mean([g["method"] == "area" for g in got]))
                row["size"] = float(np.median([g["theta"] for g in got]))
            if "cont" in got[0]:
                row["continuum"] = statistics([g["cont"] for g in got],
                                              [g["cont_error"] for g in got], case["cont"])
            row["seconds"] = time.time() - t
            rows.append(row)
            print(f"[{label}] {case['name']}: {row['seconds']:.0f} s  "
                  + "  ".join(f"{k} bias {row[k]['bias']:+.1%} cov {row[k]['coverage']:.2f}"
                              for k in ("flux", "continuum") if k in row), flush=True)
    return rows


def _cell(st, what=("bias", "scatter", "coverage")):
    out = []
    if "bias" in what:
        out.append(f"{st['bias']:+.0%}")
    if "scatter" in what:
        out.append(f"{st['scatter']:.2f}")
    if "quoted" in what:
        out.append(f"{st['quoted']:.2f}")
    if "coverage" in what:
        out.append(f"{st['coverage']:.2f}")
    return " | ".join(out)


def markdown(out):
    """The tables of ``docs/extraction/validation.md``."""
    rows = []
    if "small_n" in out:
        rows += ["### The errors, from 5 to 50 sources", "",
                 "| Sample | Line: bias | scatter | quoted | coverage | bootstrap alone | "
                 "Continuum: bias | scatter | quoted | coverage |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for r in out["small_n"]:
            rows.append(f"| {r['name']} | {_cell(r['flux'], ('bias', 'scatter', 'quoted', 'coverage'))}"
                        f" | {r['flux_fit_bootstrap_only']['coverage']:.2f}"
                        f" | {_cell(r['continuum'], ('bias', 'scatter', 'quoted', 'coverage'))} |")
        rows.append("")
    if "cases" in out:
        rows += ["### Point and extended sources", "",
                 "| Case | S/N | Flux quoted: bias | scatter | coverage | Fit: bias | coverage | "
                 "Window: bias | coverage | Gaussian chosen | size |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in out["cases"]:
            rows.append(f"| {r['name']} | {r['snr']:.0f} | {_cell(r['flux'])}"
                        f" | {_cell(r['flux_fit'], ('bias', 'coverage'))}"
                        f" | {_cell(r['flux_window'], ('bias', 'coverage'))}"
                        f" | {r['gauss_chosen']:.0%} | {r['size']:.2f}\" |")
        rows.append("")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--realisations", type=int, default=100)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--processes", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--only", choices=["small_n", "cases"], default=None)
    ap.add_argument("--out", default="validation")
    args = ap.parse_args()
    out = {"realisations": args.realisations, "bootstrap": args.bootstrap}
    if os.path.exists(args.out + ".json"):
        out.update({k: v for k, v in json.load(open(args.out + ".json")).items()
                    if k in ("small_n", "cases")})
    for label, cases in (("small_n", SMALL_N), ("cases", CASES)):
        if args.only in (None, label):
            out[label] = run(cases, args.realisations, args.bootstrap, args.processes, label)
            with open(args.out + ".json", "w") as fh:
                json.dump(out, fh, indent=1)
    with open(args.out + ".md", "w") as fh:
        fh.write(markdown(out))
    print("written", args.out + ".json,", args.out + ".md")


if __name__ == "__main__":
    main()
