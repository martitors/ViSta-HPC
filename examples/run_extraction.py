"""
Full ViSta workflow: stack, post-process, extract.

Nothing here is specific to a line or a sample: the rest frequency, the
velocity window and the input list are the only things you set, at the top or
on the command line::

    python run_extraction.py --input input_list.txt --rest-freq 345.7959899
    python run_extraction.py --input input_list.txt --rest-freq 1900.5369 \\
                             --v-window -800 800 --continuum joint

The three ways of handling the continuum are shown in turn; pick the one that
matches your data with ``--continuum``.  With ``--skip-stack`` the continuum
subtraction and the compression are redone on the existing stack, e.g. after
widening ``--v-window`` because the profile showed a line in the continuum
channels. The shell driver ``run_vista.sh`` does
the same thing from the command line, with more options.

Stacking and the two steps that read the stack need the compiled kernel
``ms_ops`` (and ``zarr`` / ``python-casacore``); the extraction itself only
needs numpy and scipy, so it can be repeated cheaply.
"""

import argparse

from vista import ViSta
from vista.extract import (BinningConfig, BootstrapConfig, ContinuumConfig,
                           LineConfig, LineFitConfig, ProfileConfig, UVFitConfig,
                           WeightingConfig, compress_visibilities, extract_flux,
                           measure_line, subtract_continuum)
from vista.extract.plots import plot_all
from vista.extract.plots_line import plot_line_result

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
parser.add_argument("--fwhm", type=float, default=None,
                    help="line FWHM known in advance [km/s]: the line is then "
                         "measured with a matched filter (mode='detect')")
parser.add_argument("--tag", default="stack", help="prefix of every output")
parser.add_argument("--skip-stack", action="store_true",
                    help="the stack exists: redo the subtraction (with the "
                         "current window) and the compression on it")
args = parser.parse_args()

MS = f"stacked_{args.tag}.zarr"   # the stack (vista.export_ms() for an MS)
STATS_LINE, STATS_CONT = f"{args.tag}_line_stats.npy", f"{args.tag}_cont_stats.npy"

# ---------------------------------------------------------------------------
# spectral setup: the one thing the code cannot guess
# ---------------------------------------------------------------------------
line = LineConfig(rest_freq_ghz=args.rest_freq,
                  v_window_kms=tuple(args.v_window))

weighting = WeightingConfig(scheme=args.weighting, redshift_rescaling=True)
bootstrap = BootstrapConfig(n_realisations=500, seed=42)
binning = BinningConfig(b_min_klambda=3.0, b_max_klambda=3000.0, n_bins=18)
subtract = args.continuum == "subtract"
contsub = ContinuumConfig(order=0,                 # -1 for the BIC
                          exclude_kms=max(abs(v) for v in args.v_window))

# ---------------------------------------------------------------------------
# 1-3. stacking, with the continuum subtraction and the compression in the
#      same pass (the stack keeps DATA, so 2-3 can be redone later on it)
# ---------------------------------------------------------------------------
if not args.skip_stack:
    ViSta(input_file=args.input, chunk_rows=5000, verbose=True).run(
        output=MS, central_freq=line.rest_freq_hz, velocity_range_kms=2000.0,
        line=line, continuum=contsub if subtract else None,
        stats_line=STATS_LINE, stats_continuum=STATS_CONT if subtract else None,
        binning=binning, max_amplitude=20.0)
else:
    if subtract:
        subtract_continuum(MS, line, contsub)
    compress_visibilities(MS, args.input, line, binning=binning,
                          line_column="CORRECTED_DATA" if subtract else "DATA",
                          out_line=STATS_LINE,
                          out_continuum=STATS_CONT if subtract else None,
                          max_amplitude=20.0, resume=False)

# ---------------------------------------------------------------------------
# 4. the line: one fit of all channels and annuli, noise from the data
# ---------------------------------------------------------------------------
fit = (LineFitConfig(mode="detect", v0_kms=0.0, fwhm_kms=args.fwhm)
       if args.fwhm else LineFitConfig(mode="search"))
fit.continuum = args.continuum == "joint"
result = measure_line(STATS_LINE, args.input, line, weighting, fit, bootstrap,
                      output_prefix=f"{args.tag}_line")
plot_line_result(result, line, f"{args.tag}_line", unit="Jy", rebin=1)
# the same figures in your own style, from the files just written:
#   python examples/plot_line.py <tag>_line --unit Jy --rebin 2 --format pdf

s = result.summary
print(f"F = {s['line']['flux']:.4g} +- {s['line']['flux_error']:.3g} Jy km/s   "
      f"peak {s['line']['peak']:.3g} Jy   FWHM {s['line']['fwhm_kms']:.0f} km/s   "
      f"size {s['line']['size_model']}")
print(f"detection: {s['detection']['significance_sigma']:.1f} sigma   noise: "
      f"kappa {s['noise']['kappa_scatter_over_weights']:.2f}, imaginary "
      f"{s['noise'].get('imaginary_rms_over_noise', float('nan')):.2f} x the noise")

# the continuum, from MODEL_DATA, with the chain of Sec. 5.3
if subtract:
    continuum = extract_flux(
        STATS_CONT, args.input, line,
        weighting=weighting, profile=ProfileConfig(method="continuum"),
        bootstrap=bootstrap, output_prefix=f"{args.tag}_continuum")
    plot_all(continuum, line, f"{args.tag}_continuum", unit="Jy")

# ---------------------------------------------------------------------------
# tests worth running before believing any of the above
# ---------------------------------------------------------------------------
# the chain of Sec. 5.3 on the same statistics
extract_flux(STATS_LINE, args.input, line, weighting=weighting,
             profile=ProfileConfig(method="template", joint_continuum=fit.continuum),
             uvfit=UVFitConfig(model="point"), output_prefix=f"{args.tag}_template")

# does the other weighting scheme give the same answer?
other = "natural" if args.weighting == "democratic" else "democratic"
measure_line(STATS_LINE, args.input, line, WeightingConfig(scheme=other), fit,
             bootstrap, output_prefix=f"{args.tag}_{other}")
