"""The benchmark sample: heterogeneous ALMA Band 3 observations of CO(3-2)
at z = 2-3, and its true values.

Every galaxy differs in what a real archival sample mixes: redshift,
luminosity, line width and centre (redshift error), size, continuum level and
slope, position in the primary beam, array configuration (maximum baseline,
number of antennas), date and duration of the observation (uv tracks), channel
width, bandwidth, position of the line in the band, depth, how wrong the
weights are, and Hanning smoothing.  The parameters are drawn once from
``SampleConfig.seed``; the noise of every realisation from its own seed, so
that the same galaxies and weights can be observed again with new noise
(Monte Carlo) or without noise.

Usage::

    python sample.py OUTDIR [--noiseless] [--realisation K]

writes ``OUTDIR/galaxy_XX.ms``, ``OUTDIR/input_list.txt`` and
``OUTDIR/truth.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
import simulate  # noqa: E402

C_KMS = 299792.458


@dataclass
class SampleConfig:
    n_galaxies: int = 24
    z_range: tuple = (2.0, 3.0)
    rest_freq_ghz: float = 345.79599          # CO(3-2)
    line_flux_ref: float = 0.35               # Jy km/s at z_ref, median galaxy
    luminosity_scatter_dex: float = 0.25      # log-normal spread of the luminosities
    fwhm_range_kms: tuple = (150.0, 500.0)    # log-uniform
    redshift_error_kms: float = 60.0          # rms offset of the line from its catalogue z
    point_fraction: float = 0.25              # unresolved galaxies
    size_range_arcsec: tuple = (0.3, 1.2)     # FWHM of the others
    cont_range_mjy: tuple = (0.0, 0.4)        # continuum at the line frequency
    cont_index_range: tuple = (2.5, 4.0)
    offset_max_arcsec: float = 12.0           # from the phase centre (primary beam)
    rmax_choices_m: tuple = (150.0, 300.0, 600.0, 1000.0)   # array configurations
    nant_range: tuple = (36, 46)
    ntime_choices: tuple = (30, 40, 50)       # x 30 s integrations
    dv_choices_kms: tuple = (15.0, 20.0, 25.0, 30.0)
    band_kms: float = 2300.0                  # bandwidth of every dataset
    band_offset_max_kms: float = 200.0        # line away from the band centre
    noise_20kms: float = 0.13                 # Jy per visibility at 20 km/s, median dataset
    depth_spread_dex: float = 0.25
    kappa_range: tuple = (1.0, 1.5)           # weights wrong by this factor
    hanning_fraction: float = 0.7
    seed: int = 2026


def draw(cfg: SampleConfig):
    """The galaxies and their observations, one dict of make_ms keywords each."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_galaxies
    z = np.sort(rng.uniform(*cfg.z_range, n))
    z_ref = float(np.median(z))
    gal = []
    for i in range(n):
        alpha = simulate.alpha_factor(z[i], z_ref)
        f_ref = cfg.line_flux_ref * 10 ** rng.normal(0.0, cfg.luminosity_scatter_dex)
        size = 0.0 if rng.uniform() < cfg.point_fraction else rng.uniform(*cfg.size_range_arcsec)
        dv = float(rng.choice(cfg.dv_choices_kms))
        r_off, a_off = cfg.offset_max_arcsec * math.sqrt(rng.uniform()), rng.uniform(0, 2 * math.pi)
        gal.append(dict(
            z=float(z[i]),
            line_flux=float(f_ref / alpha),               # observed: the luminosity at z
            fwhm_kms=float(10 ** rng.uniform(*np.log10(cfg.fwhm_range_kms))),
            v0_kms=float(rng.normal(0.0, cfg.redshift_error_kms)),
            size_arcsec=float(size),
            cont_size_arcsec=float(size * rng.uniform(0.5, 1.0)),
            cont_flux=float(rng.uniform(*cfg.cont_range_mjy) * 1e-3),
            cont_index=float(rng.uniform(*cfg.cont_index_range)),
            phase_centre=(float(rng.uniform(0, 360)), float(rng.uniform(-40, 10))),
            offset_arcsec=(float(r_off * math.cos(a_off)), float(r_off * math.sin(a_off))),
            rmax=float(rng.choice(cfg.rmax_choices_m)),
            nant=int(rng.integers(cfg.nant_range[0], cfg.nant_range[1] + 1)),
            ntime=int(rng.choice(cfg.ntime_choices)),
            dt=30.0,
            t0=float(5.06e9 + rng.uniform(0, 400) * 86400.0),
            layout_seed=int(rng.integers(1, 10 ** 6)),
            dv_kms=dv,
            nchan=int(math.ceil(cfg.band_kms / dv)),
            band_offset_kms=float(rng.uniform(-1, 1) * cfg.band_offset_max_kms),
            noise=float(cfg.noise_20kms * math.sqrt(20.0 / dv)
                        * 10 ** rng.uniform(-cfg.depth_spread_dex, cfg.depth_spread_dex)),
            kappa=float(rng.uniform(*cfg.kappa_range)),
            hanning=bool(rng.uniform() < cfg.hanning_fraction),
            seed=int(rng.integers(1, 10 ** 6)),
            alpha=float(alpha), line_flux_ref=float(f_ref),
        ))
    return gal, z_ref


def write(outdir, cfg: SampleConfig, noiseless=False, realisation=0, verbose=True):
    """Write the Measurement Sets, the input list and truth.json."""
    from casacore import tables as ct
    gal, z_ref = draw(cfg)
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    lines = ["# <ms_path>  <redshift>  <RA>  <Dec>"]
    recs = []
    for i, g in enumerate(gal):
        path = os.path.join(outdir, f"galaxy_{i:02d}.ms")
        kw = {k: v for k, v in g.items() if k not in ("alpha", "line_flux_ref")}
        rec = simulate.make_ms(path, rest_freq_ghz=cfg.rest_freq_ghz, noiseless=noiseless,
                               noise_seed=None if realisation == 0 else 7919 * realisation + i,
                               **kw)
        with ct.table(path, ack=False) as t:
            w = t.getcol("WEIGHT")
        radius = rec["offset_arcsec"]
        rec.update(g)
        rec["offset_radius_arcsec"] = radius
        rec["weight_sum"] = float(w.sum())                 # per channel, both correlations
        rec["weight_per_kms"] = rec["weight_sum"] / g["dv_kms"]
        recs.append(rec)
        lines.append(f"{os.path.abspath(path)}  {g['z']:.6f}  {simulate.fmt_ra(rec['ra_deg'])}  "
                     f"{simulate.fmt_dec(rec['dec_deg'])}")
    with open(os.path.join(outdir, "input_list.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    truth = {"config": asdict(cfg), "z_ref": z_ref, "noiseless": noiseless,
             "realisation": realisation, "sources": recs}
    truth["targets"] = {name: targets(recs, z_ref, **kw) for name, kw in PIPELINE_TRUTH.items()}
    with open(os.path.join(outdir, "truth.json"), "w") as fh:
        json.dump(truth, fh, indent=1, default=float)
    if verbose:
        print(f"{len(recs)} datasets in {outdir} in {time.time() - t0:.0f} s "
              f"({'noiseless' if noiseless else f'realisation {realisation}'})")
    return truth


# ---------------------------------------------------------------------------
# the true values of the stack
# ---------------------------------------------------------------------------
def _gauss(v, area, v0, fwhm):
    s = fwhm / 2.3548200450309493
    return area * np.exp(-0.5 * ((v - v0) / s) ** 2) / (s * math.sqrt(2 * math.pi))


# how each pipeline combines the galaxies (see targets)
PIPELINE_TRUTH = {
    "vista_casa": dict(weights="instrumental", H0=70.0, Om0=0.3),
    "extract": dict(weights="pb_corrected", H0=67.4, Om0=0.315),
    "extract_opt": dict(weights="pb_corrected", H0=67.4, Om0=0.315),
}


def targets(recs, z_ref, weights="pb_corrected", H0=67.4, Om0=0.315):
    """What a stack of these galaxies, transported to z_ref and corrected for
    the primary beam, should measure.

    Every galaxy enters with its luminosity transported to z_ref (flux
    alpha * F) and a weight: its weight per unit velocity in the data
    (natural weighting), times A^2 when the weights are those of the
    primary-beam-corrected visibilities (``pb_corrected``: ViSta-HPC) or as
    they are (``instrumental``: the visibilities are multiplied by 1/A and the
    weights left alone, as in the CASA pipeline), with the cosmology of the
    pipeline for alpha (ViSta 1.0: H0 = 70, Om0 = 0.3; ViSta-HPC: 67.4,
    0.315).  The line of the stack is a
    weighted sum of Gaussians of different centres and widths, and the source
    a weighted sum of Gaussians of different sizes: the effective centre,
    FWHM and size are those of the single Gaussian closest to them.
    """
    from scipy.optimize import curve_fit
    w = np.array([r["weight_per_kms"] for r in recs])
    if weights == "pb_corrected":
        w = w * np.array([r["primary_beam"] for r in recs]) ** 2
    w = w / w.sum()
    a = np.array([simulate.alpha_factor(r["z"], z_ref, H0, Om0) for r in recs])
    flux = float(np.sum(w * a * np.array([r["line_flux"] for r in recs])))
    cont = float(np.sum(w * a * np.array([r["cont_flux"] for r in recs])))
    v = np.linspace(-1500, 1500, 6001)
    prof = sum(wi * ai * _gauss(v, r["line_flux"], r["v0_kms"], r["fwhm_kms"])
               for wi, ai, r in zip(w, a, recs))
    p, _ = curve_fit(_gauss, v, prof, p0=[flux, 0.0, 300.0])
    # the source: weighted mean of F g(b; size), fitted by one Gaussian over the
    # baselines the arrays sample (log-spaced, 5-500 klambda)
    b = np.geomspace(5e3, 5e5, 200)
    vis = sum(wi * ai * r["line_flux"] * simulate.shape_visibility(b, r["size_arcsec"])
              for wi, ai, r in zip(w, a, recs))
    q, _ = curve_fit(lambda bb, f, th: f * simulate.shape_visibility(bb, abs(th)), b, vis,
                     p0=[flux, 0.5])
    return {"line_flux": flux, "line_flux_gauss": float(p[0]), "v0_kms": float(p[1]),
            "fwhm_kms": float(abs(p[2])), "peak": float(prof.max()), "size_arcsec": float(abs(q[1])),
            "line_flux_uv_gauss": float(q[0]), "cont_flux": cont}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--noiseless", action="store_true")
    ap.add_argument("--realisation", type=int, default=0,
                    help="noise realisation (0: the reference one)")
    ap.add_argument("--n-galaxies", type=int, default=SampleConfig.n_galaxies)
    ap.add_argument("--seed", type=int, default=SampleConfig.seed)
    a = ap.parse_args()
    write(a.outdir, SampleConfig(n_galaxies=a.n_galaxies, seed=a.seed),
          noiseless=a.noiseless, realisation=a.realisation)


if __name__ == "__main__":
    main()
