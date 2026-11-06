"""
vista.extract.cli
=================
Command-line front end of the extraction, one subcommand per step::

    python -m vista.extract contsub   stack.zarr      --rest-freq 345.7959899
    python -m vista.extract compress  stack.zarr      --input input_list.txt \
                                                      --rest-freq 345.7959899
    python -m vista.extract flux      line_stats.npy  --input input_list.txt \
                                                      --rest-freq 345.7959899 \
                                                      [--method joint]
    python -m vista.extract export-ms stack.zarr      stack.ms

The stack is the Zarr store written by the stacking, or a Measurement Set.

Every option maps one to one onto a field of the configuration objects in
:mod:`vista.extract.config`, which remain the reference for the defaults and
for the meaning of each parameter.
"""

from __future__ import annotations

import argparse
import json
import sys

from .. import _say, _warn
from .config import (BinningConfig, BootstrapConfig, ContinuumConfig,
                     LineConfig, ProfileConfig, UVFitConfig, WeightingConfig)


def _add_line_options(parser):
    parser.add_argument("--rest-freq", type=float, required=True,
                        metavar="GHz",
                        help="rest frequency of the target line [GHz]")
    parser.add_argument("--v-window", nargs=2, type=float,
                        default=(-600.0, 600.0), metavar=("LO", "HI"),
                        help="velocity window of the line [km/s]")
    parser.add_argument("--second-rest-freq", type=float, default=None,
                        metavar="GHz",
                        help="rest frequency of a second line in the band")
    parser.add_argument("--second-v-window", nargs=2, type=float,
                        default=(-500.0, 500.0), metavar=("LO", "HI"))
    parser.add_argument("--exclude-v", nargs="*", type=float, default=None,
                        metavar="V",
                        help="pairs LO HI of extra velocity intervals to keep "
                             "out of the line-free channels")


def _line_from_args(args) -> LineConfig:
    extra = []
    if args.exclude_v:
        values = list(args.exclude_v)
        if len(values) % 2:
            raise SystemExit("--exclude-v needs pairs LO HI")
        extra = [(values[i], values[i + 1]) for i in range(0, len(values), 2)]
    return LineConfig(rest_freq_ghz=args.rest_freq,
                      v_window_kms=tuple(args.v_window),
                      second_rest_freq_ghz=args.second_rest_freq,
                      second_v_window_kms=tuple(args.second_v_window),
                      exclude_v_kms=extra)


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vista.extract",
        description="ViSta post-processing and uv-domain flux extraction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- continuum subtraction -------------------------------------------
    p = subparsers.add_parser("contsub",
                              help="subtract the continuum from a stack (Zarr or MS)")
    p.add_argument("ms_path", help="the stack: Zarr store or Measurement Set")
    p.add_argument("--io-threads", type=int, default=4,
                   help="threads reading and writing a Zarr store (default 4)")
    _add_line_options(p)
    p.add_argument("--order", type=int, default=0, choices=[0, 1, -1],
                   help="0 constant (default), 1 linear, -1 chosen per "
                        "spectral window by BIC")
    p.add_argument("--delta-bic", type=float, default=2.0)
    p.add_argument("--free-slope-sign", action="store_true",
                   help="with --order -1, accept a negative slope too")
    p.add_argument("--exclude-kms", type=float, default=None,
                   help="half-width of the excluded window [km/s]; defaults "
                        "to the line window")
    p.add_argument("--exclude-kms-lo", type=float, default=None)
    p.add_argument("--exclude-kms-hi", type=float, default=None)
    p.add_argument("--data-column", default="DATA")
    p.add_argument("--model-column", default="MODEL_DATA")
    p.add_argument("--line-column", default="CORRECTED_DATA")
    p.add_argument("--chunk-rows", type=int, default=50_000)
    p.add_argument("--only-dd", default=None,
                   help="comma separated data descriptor ids")
    p.add_argument("--start-dd", type=int, default=0)

    # ---- compression -----------------------------------------------------
    p = subparsers.add_parser(
        "compress", help="compress a stack (Zarr or MS) into sufficient statistics")
    p.add_argument("ms_path", help="the stack: Zarr store or Measurement Set")
    p.add_argument("--io-threads", type=int, default=4,
                   help="threads reading a Zarr store (default 4)")
    p.add_argument("--input", required=True, help="ViSta input list")
    _add_line_options(p)
    p.add_argument("--bins", nargs=3, type=float, default=(3.0, 3000.0, 18),
                   metavar=("B_MIN_KL", "B_MAX_KL", "N_BINS"))
    p.add_argument("--out-line", default="stack_line_stats.npy")
    p.add_argument("--out-continuum", default=None,
                   help="also compress the continuum column into this file "
                        "(only after a continuum subtraction)")
    p.add_argument("--line-column", default=None,
                   help="default: CORRECTED_DATA if present, else DATA")
    p.add_argument("--continuum-column", default="MODEL_DATA")
    p.add_argument("--max-amplitude", type=float, default=None,
                   help="discard visibilities above this amplitude; NaN and "
                        "infinities are always discarded")
    p.add_argument("--chunk-mb", type=float, default=512.0)
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   default=True)

    # ---- extraction ------------------------------------------------------
    p = subparsers.add_parser("flux",
                              help="extract the stacked flux from statistics")
    p.add_argument("statistics")
    p.add_argument("--input", required=True, help="ViSta input list")
    _add_line_options(p)
    p.add_argument("--weighting", default="natural",
                   choices=["natural", "democratic"],
                   help="natural (default): native weights, the deepest "
                        "observations dominate; democratic: every source "
                        "contributes with total weight 1")
    p.add_argument("--no-redshift-rescaling", dest="redshift_rescaling",
                   action="store_false", default=True,
                   help="do not transport the sources to a common redshift")
    p.add_argument("--z-ref", type=float, default=None,
                   help="reference redshift; default is the sample median")
    p.add_argument("--no-primary-beam", dest="primary_beam",
                   action="store_false", default=True,
                   help="do not correct for the primary beam attenuation")
    p.add_argument("--flux-normalisation", action="store_true", default=False,
                   help="rescale the amplitudes by norm_ref/norm, using the "
                        "last column of the input list")
    p.add_argument("--norm-ref", type=float, default=None,
                   help="reference value of the normalisation; default is "
                        "the sample median")
    p.add_argument("--already-normalised", dest="already_applied",
                   action="store_true", default=False,
                   help="the amplitudes on disk already carry alpha")
    p.add_argument("--match-tol-arcsec", type=float, default=2.0,
                   help="tolerance to recognise two entries as the same "
                        "physical source")
    p.add_argument("--method", default="template",
                   choices=["joint", "template", "window", "continuum"],
                   help="joint: one fit of all channels and annuli, with the "
                        "noise measured from the data (vista.extract.measure); "
                        "template, window, continuum: the three-step chain of "
                        "Sec. 5.3")
    j = p.add_argument_group("joint method (--method joint)")
    j.add_argument("--mode", default="search", choices=["search", "detect"],
                   help="search: centre and width free, significance from the same "
                        "search on noise; detect: centre and width fixed (--v0, "
                        "--fwhm), matched filter")
    j.add_argument("--v0", type=float, default=0.0, metavar="KMS",
                   help="line centre: fixed (detect) or start (search)")
    j.add_argument("--fwhm", type=float, default=None, metavar="KMS",
                   help="line FWHM: fixed (detect, required) or start (search)")
    j.add_argument("--v0-range", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                   help="allowed centres in search mode (default: the line window)")
    j.add_argument("--fwhm-range", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                   help="allowed FWHM in search mode (default: 2 channels .. window)")
    j.add_argument("--size", default="auto", choices=["auto", "point", "gauss"],
                   help="source size: auto (Gaussian only if it improves chi2 by "
                        "--resolved-dchi2), point, gauss")
    j.add_argument("--resolved-dchi2", type=float, default=9.0,
                   help="chi2 improvement needed to adopt a Gaussian source (default 9: 3 sigma)")
    j.add_argument("--noise", default="scatter", choices=["scatter", "weights"],
                   help="noise of the cells from the scatter of the visibilities "
                        "(default) or from the weights")
    j.add_argument("--no-channel-correlation", dest="channel_correlation",
                   action="store_false", default=True)
    j.add_argument("--n-null", type=int, default=1000,
                   help="realisations of the search on noise (significance)")
    j.add_argument("--n-null-phase", type=int, default=200,
                   help="random-phase stacks (data-driven null); 0 to skip")
    j.add_argument("--window", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                   help="window of the summed flux (default: --v-window)")
    j.add_argument("--plot-rebin", type=int, default=1,
                   help="channels per bin in the profile figure")
    p.add_argument("--joint-continuum", action="store_true",
                   help="fit the line on top of a constant continuum and "
                        "measure both: use it when the stack was NOT "
                        "continuum subtracted")
    p.add_argument("--fit-span-kms", type=float, default=1500.0)
    p.add_argument("--shape-tie", default="free",
                   choices=["free", "centroid", "centroid+width"])
    p.add_argument("--fixed-shape", nargs=2, type=float, default=None,
                   metavar=("CENTROID", "SIGMA"))
    p.add_argument("--continuum-overlap-frac", type=float, default=1.0)
    p.add_argument("--model", default="gauss",
                   choices=["gauss", "point", "gauss2"])
    p.add_argument("--theta-fixed", type=float, default=None, metavar="ARCSEC")
    p.add_argument("--theta-max", type=float, default=15.0, metavar="ARCSEC")
    p.add_argument("--theta-prior", nargs=2, type=float, default=None,
                   metavar=("THETA_REF", "SIGMA_DEX"))
    p.add_argument("--theta-prior-from", default=None,
                   help="results JSON of a previous fit, typically the "
                        "continuum of the same band, to take THETA_REF from")
    p.add_argument("--theta-prior-sigma", type=float, default=0.15)
    p.add_argument("--b-max-kl", type=float, default=None)
    p.add_argument("--min-sources-per-bin", type=int, default=2)
    p.add_argument("--error-floor-frac", type=float, default=0.1)
    p.add_argument("--second-line-tie-dex", type=float, default=0.2)
    p.add_argument("--no-second-line-tie", dest="second_line_tie",
                   action="store_false", default=True)
    p.add_argument("--continuum-tie-dex", type=float, default=None,
                   help="soft tie of the continuum size to the line size "
                        "(joint mode); default: free")
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="stack")
    p.add_argument("--plot", action="store_true",
                   help="write the figures that fit the run: <out>_joint.png "
                        "for a joint fit, otherwise <out>_uvamp.png and "
                        "<out>_spectrum.png")
    p.add_argument("--unit", default=None,
                   help="name of the amplitude unit for the axis labels, "
                        "e.g. Jy (default: 'data units')")
    p.add_argument("--v-limits", nargs=2, type=float, default=None,
                   metavar=("LO", "HI"),
                   help="velocity range of the spectrum plot [km/s]")

    # ---- export ----------------------------------------------------------
    p = subparsers.add_parser("export-ms",
                              help="write a Zarr stack as a Measurement Set")
    p.add_argument("store", help="Zarr store written by the stacking")
    p.add_argument("ms_path", help="Measurement Set to write (replaced if it exists)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export-ms":
        from ..stackio import export_ms
        export_ms(args.store, args.ms_path)
        return 0
    line = _line_from_args(args)

    if args.command == "contsub":
        from .contsub import subtract_continuum
        config = ContinuumConfig(
            order=args.order, delta_bic=args.delta_bic,
            require_positive_slope=not args.free_slope_sign,
            exclude_kms=(args.exclude_kms if args.exclude_kms is not None
                         else max(abs(line.v_window_kms[0]),
                                  abs(line.v_window_kms[1]))),
            exclude_kms_lo=args.exclude_kms_lo,
            exclude_kms_hi=args.exclude_kms_hi,
            data_column=args.data_column, model_column=args.model_column,
            line_column=args.line_column, chunk_rows=args.chunk_rows)
        only = ({int(x) for x in args.only_dd.split(",") if x.strip()}
                if args.only_dd else None)
        subtract_continuum(args.ms_path, line, config, only_dd=only,
                           start_dd=args.start_dd, n_io=args.io_threads)
        return 0

    if args.command == "compress":
        from .statistics import compress_visibilities
        binning = BinningConfig(b_min_klambda=args.bins[0],
                                b_max_klambda=args.bins[1],
                                n_bins=int(args.bins[2]))
        compress_visibilities(args.ms_path, args.input, line, binning,
                              out_line=args.out_line,
                              out_continuum=args.out_continuum,
                              line_column=args.line_column,
                              continuum_column=args.continuum_column,
                              max_amplitude=args.max_amplitude,
                              chunk_mb=args.chunk_mb, resume=args.resume,
                              n_io=args.io_threads)
        return 0

    # ---- flux ------------------------------------------------------------
    from .stack import extract_flux

    theta_prior = tuple(args.theta_prior) if args.theta_prior else None
    if args.theta_prior_from:
        with open(args.theta_prior_from) as fh:
            previous = json.load(fh)
        reference = previous.get("theta_fwhm_arcsec")
        if isinstance(reference, (int, float)) and reference > 0.05:
            theta_prior = (float(reference), args.theta_prior_sigma)
            _say(f"Size prior from {args.theta_prior_from}: "
                 f"theta_ref={reference:.3f}\" +- "
                 f"{args.theta_prior_sigma:.2f} dex")
        else:
            _warn(f"no usable source size in {args.theta_prior_from}: "
                  f"size prior off")
    if theta_prior is not None and args.method == "continuum":
        _say("Size prior ignored: the continuum is the reference")
        theta_prior = None

    weighting = WeightingConfig(
        scheme=args.weighting, redshift_rescaling=args.redshift_rescaling,
        z_ref=args.z_ref, flux_normalisation=args.flux_normalisation,
        norm_ref=args.norm_ref, already_applied=args.already_applied,
        primary_beam=args.primary_beam)
    if args.method == "joint":
        from .config import LineFitConfig
        from .measure import measure_line
        fitcfg = LineFitConfig(
            mode=args.mode, v0_kms=args.v0, fwhm_kms=args.fwhm,
            v0_range_kms=tuple(args.v0_range) if args.v0_range else None,
            fwhm_range_kms=tuple(args.fwhm_range) if args.fwhm_range else None,
            size=args.size, theta_fixed_arcsec=args.theta_fixed,
            theta_max_arcsec=args.theta_max, resolved_delta_chi2=args.resolved_dchi2,
            theta_prior_arcsec=theta_prior,
            continuum=args.joint_continuum, shape_tie=args.shape_tie,
            fit_span_kms=args.fit_span_kms, min_sources_per_bin=args.min_sources_per_bin,
            noise=args.noise, channel_correlation=args.channel_correlation,
            n_null=args.n_null, n_null_phase=args.n_null_phase,
            window_kms=tuple(args.window) if args.window else None)
        result = measure_line(args.statistics, args.input, line, weighting, fitcfg,
                              BootstrapConfig(n_realisations=args.n_bootstrap,
                                              seed=args.seed),
                              match_tol_arcsec=args.match_tol_arcsec,
                              output_prefix=args.out)
        if args.plot:
            from .plots_line import plot_line_result
            plot_line_result(result, line, args.out, unit=args.unit,
                             v_limits=tuple(args.v_limits) if args.v_limits else None,
                             rebin=args.plot_rebin)
        return 0

    profile = ProfileConfig(
        method=args.method, fit_span_kms=args.fit_span_kms,
        shape_tie=args.shape_tie,
        fixed_shape_kms=(tuple(args.fixed_shape) if args.fixed_shape
                         else None),
        joint_continuum=args.joint_continuum,
        continuum_overlap_frac=args.continuum_overlap_frac)
    uvfit = UVFitConfig(
        model=args.model, theta_fixed_arcsec=args.theta_fixed,
        theta_max_arcsec=args.theta_max, theta_prior=theta_prior,
        b_max_klambda=args.b_max_kl,
        min_sources_per_bin=args.min_sources_per_bin,
        error_floor_frac=args.error_floor_frac,
        second_line_theta_tie_dex=(args.second_line_tie_dex
                                   if args.second_line_tie else None),
        continuum_theta_tie_dex=args.continuum_tie_dex)
    bootstrap = BootstrapConfig(n_realisations=args.n_bootstrap,
                                seed=args.seed)

    result = extract_flux(args.statistics, args.input, line,
                          weighting=weighting, profile=profile, uvfit=uvfit,
                          bootstrap=bootstrap,
                          match_tol_arcsec=args.match_tol_arcsec,
                          output_prefix=args.out)
    if args.plot:
        from .plots import plot_all
        plot_all(result, line, args.out, unit=args.unit,
                 v_limits=tuple(args.v_limits) if args.v_limits else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
