"""
Full ViSta workflow: stack, post-process, extract.

Nothing here is specific to a line or a sample: the rest frequency, the
velocity window and the input list are the only things you set, at the top or
on the command line::

    python run_extraction.py --input input_list.txt --rest-freq 345.7959899
    python run_extraction.py --input input_list.txt --rest-freq 1900.5369 \\
                             --v-window -800 800 --continuum joint

The three ways of handling the continuum are shown in turn; pick the one that
matches your data with ``--continuum``. The shell driver ``run_vista.sh`` does
the same thing from the command line, with more options.

Stacking and the two MS-reading steps need ``casatools``/``ms_ops``; the
extraction itself only needs numpy and scipy, so it can be repeated cheaply.
"""

import argparse

from vista import ViSta
from vista.extract import (BinningConfig, BootstrapConfig, ContinuumConfig,
                           LineConfig, ProfileConfig, UVFitConfig,
                           WeightingConfig, compress_visibilities,
                           extract_flux, subtract_continuum)
from vista.extract.plots import plot_all

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", default="input_list.txt",
                    help="ViSta input list")
parser.add_argument("--rest-freq", type=float, required=True, metavar="GHZ",
                    help="rest frequency of the line being stacked")
parser.add_argument("--v-window", nargs=2, type=float, default=(-600.0, 600.0),
                    metavar=("LO", "HI"), help="velocity window, km/s")
parser.add_argument("--continuum", default="subtract",
                    choices=["subtract", "joint", "none"])
parser.add_argument("--weighting", default="democratic",
                    choices=["democratic", "natural"])
parser.add_argument("--tag", default="stack", help="prefix of every output")
parser.add_argument("--skip-stack", action="store_true")
args = parser.parse_args()

MS = f"stacked_{args.tag}.ms"

# ---------------------------------------------------------------------------
# spectral setup: the one thing the code cannot guess
# ---------------------------------------------------------------------------
line = LineConfig(rest_freq_ghz=args.rest_freq,
                  v_window_kms=tuple(args.v_window))

weighting = WeightingConfig(scheme=args.weighting, redshift_rescaling=True)
bootstrap = BootstrapConfig(n_realisations=500, seed=42)
binning = BinningConfig(b_min_klambda=3.0, b_max_klambda=3000.0, n_bins=18)

# ---------------------------------------------------------------------------
# 1. stacking
# ---------------------------------------------------------------------------
if not args.skip_stack:
    ViSta(input_file=args.input, chunk_rows=50_000, verbose=True).run(
        ms_out=MS,
        central_freq=line.rest_freq_hz,
        velocity_range_kms=2000.0)

# ===========================================================================
# 2-4.  the continuum decides the rest
# ===========================================================================
if args.continuum == "subtract":
    # ---- remove it first, then two independent fits -----------------------
    subtract_continuum(MS, line, ContinuumConfig(order=0))   # -1 for the BIC

    compress_visibilities(MS, args.input, line, binning=binning,
                          line_column="CORRECTED_DATA",
                          continuum_column="MODEL_DATA",
                          out_line=f"{args.tag}_line_stats.npy",
                          out_continuum=f"{args.tag}_cont_stats.npy",
                          max_amplitude=20.0)

    # the continuum first: its size then serves as a prior on the line size
    continuum = extract_flux(
        f"{args.tag}_cont_stats.npy", args.input, line,
        weighting=weighting, profile=ProfileConfig(method="continuum"),
        bootstrap=bootstrap, output_prefix=f"{args.tag}_continuum")
    plot_all(continuum, line, f"{args.tag}_continuum", unit="Jy")

    result = extract_flux(
        f"{args.tag}_line_stats.npy", args.input, line,
        weighting=weighting, profile=ProfileConfig(method="template"),
        uvfit=UVFitConfig(model="gauss", min_sources_per_bin=3,
                          theta_prior=(continuum.summary["theta_fwhm_arcsec"],
                                       0.15)),
        bootstrap=bootstrap, output_prefix=f"{args.tag}_line")
    plot_all(result, line, f"{args.tag}_line", unit="Jy")

else:
    # ---- nothing subtracted: read DATA ------------------------------------
    compress_visibilities(MS, args.input, line, binning=binning,
                          line_column="DATA",
                          out_line=f"{args.tag}_line_stats.npy",
                          max_amplitude=20.0)

    profile = ProfileConfig(method="template",
                            joint_continuum=args.continuum == "joint")
    result = extract_flux(
        f"{args.tag}_line_stats.npy", args.input, line,
        weighting=weighting, profile=profile,
        uvfit=UVFitConfig(model="gauss", min_sources_per_bin=3),
        bootstrap=bootstrap,
        output_prefix=f"{args.tag}_joint" if profile.joint_continuum
        else f"{args.tag}_line")
    plot_all(result, line,
             f"{args.tag}_joint" if profile.joint_continuum
             else f"{args.tag}_line", unit="Jy")

summary = result.summary
print(f"F = {summary['flux_total']:.4g} +- {summary['flux_total_error']:.3g}   "
      f"theta = {summary['theta_fwhm_arcsec']:.3f}\"")
if "continuum" in summary:
    block = summary["continuum"]
    print(f"continuum S = {block['flux_total']:.4g} "
          f"+- {block['flux_total_error']:.3g}   "
          f"theta = {block['theta_fwhm_arcsec']:.3f}\"")

# ---------------------------------------------------------------------------
# tests worth running before believing any of the above
# ---------------------------------------------------------------------------
# is the stack actually resolved?  compare with a point-source model
extract_flux(f"{args.tag}_line_stats.npy", args.input, line,
             weighting=weighting, uvfit=UVFitConfig(model="point"),
             output_prefix=f"{args.tag}_point")

# does the other weighting scheme give the same answer?
other = "natural" if args.weighting == "democratic" else "democratic"
extract_flux(f"{args.tag}_line_stats.npy", args.input, line,
             weighting=WeightingConfig(scheme=other),
             output_prefix=f"{args.tag}_{other}")
