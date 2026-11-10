"""
The measurement repeated with different options on the same stack.

The stack and its statistics are written once; every variant below only
re-reads the statistics (``run(only="measure")``) and takes seconds.  Each
variant rewrites ``stacked_results.json`` and the figures: the table at the
end keeps the numbers of all of them.

    python run_extraction.py --input input_list.txt --line-freq 345.79599
    python run_extraction.py --input input_list.txt --line-freq 345.79599 --skip-stack
"""

import argparse

from vista import ViSta

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--input", default="input_list.txt")
ap.add_argument("--line-freq", type=float, nargs="+", required=True, metavar="GHZ")
ap.add_argument("--line-window", type=float, default=500.0, metavar="KMS")
ap.add_argument("--skip-stack", action="store_true", help="the stack exists already")
args = ap.parse_args()

v = ViSta(args.input, output_name="stacked")
v.stack(line_freq=args.line_freq, line_window=args.line_window)
v.extract(n_bootstrap=200)
if not args.skip_stack:
    v.run(only="stack")                     # the stack and its statistics

variants = {
    "reference": dict(),
    "width fixed at 300 km/s": dict(fwhm_range=(300, 300)),
    "centre and width fixed": dict(v0_range=(0, 0), fwhm_range=(300, 300)),
    "point source": dict(model="point"),
    "size fixed at 0.8\"": dict(model=0.8),
    "area flux quoted": dict(flux_measure="area"),
}
rows = []
for scheme in ("natural", "democratic"):
    for name, options in variants.items():
        v.extract(weighting_scheme=scheme, n_bootstrap=200, **options)
        ln = v.run(only="measure").line.summary["line"]
        rows.append((f"{scheme}, {name}", ln))

print(f"\n{'':40s} {'flux':>16s} {'Gaussian':>9s} {'area':>9s} {'FWHM':>6s} {'size':>6s}")
for name, ln in rows:
    print(f"{name:40s} {ln['flux']:8.4f} ± {ln['flux_error']:.4f} {ln['flux_gauss']:9.4f} "
          f"{ln['flux_area']:9.4f} {ln['fwhm_kms']:6.0f} {ln['theta_fwhm_arcsec']:6.2f}")
