"""
vista.cli
=========
The command line: the same options as :class:`vista.ViSta` (``ViSta()``,
``stack()``, ``extract()``), one flag each.  Without ``--extract`` only the
stack is written, as without ``v.extract()``.

::

    python -m vista --input input_list.txt --line-freq 345.796 --line-window 500 --extract
    python -m vista --input input_list.txt --line-freq 345.796 --only measure \\
                    --extract --fwhm-range 100:600 --flux-measure area
    python -m vista export stacked.zarr stacked.ms

(``run_vista.sh`` calls it.)  Windows and ranges: a number is a half-width
(``500``: -500..+500 km/s), a pair is ``lo:hi`` or ``"lo hi"`` in quotes;
with several lines give one per line, in the order of ``--line-freq``.  A
pair whose low end is negative needs the ``=`` form or the quotes:
``--line-window=-400:600`` or ``--line-window "-400 600"``.
"""

from __future__ import annotations

import argparse
import sys


def _pair(text):
    """'500' -> 500.0 ; '-400:600' or '-400 600' -> (-400.0, 600.0)."""
    t = str(text).replace(",", " ").replace(":", " ").split()
    try:
        vals = [float(x) for x in t]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number or a pair lo:hi: {text!r}") from None
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 2:
        return tuple(vals)
    raise argparse.ArgumentTypeError(f"a number or a pair lo:hi, not {text!r}")


def _one_or_list(values):
    if values is None:
        return None
    return values[0] if len(values) == 1 else list(values)


def _range_arg(values):
    """Ranges: one pair for all the lines, or one per line."""
    if values is None:
        return None
    out = []
    for v in values:
        if not isinstance(v, tuple):
            out.append((v, v))                  # a single number: fixed
        else:
            out.append(v)
    return out[0] if len(out) == 1 else out


def _freq(text):
    """A rest frequency: GHz, or with its unit ('345.796GHz', '345796MHz')."""
    from .extract.config import parse_freq_ghz
    try:
        parse_freq_ghz(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    try:
        return float(text)
    except ValueError:
        return text


def _fit_span(text):
    if text is None:
        return 1500.0
    if str(text).lower() == "none":
        return None
    try:
        return float(text)
    except ValueError:
        sys.exit(f"vista: --fit-span: km/s or 'none', not {text!r}")


def _size(text):
    if text in ("auto", "point", "gauss"):
        return text
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"size: auto, point, gauss or a size in arcsec, not {text!r}") from None


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m vista",
        description="Stack interferometric observations in the uv plane and measure the "
                    "stacked lines and continuum (the options of vista.ViSta: ViSta(), "
                    "stack(), extract()).",
        epilog="python -m vista export STORE MS   writes a Zarr stack as a Measurement Set.")
    g = p.add_argument_group("input, output, how it runs (ViSta)")
    g.add_argument("--input", required=True, help="input list (required)")
    g.add_argument("--output-name", default="stacked", metavar="NAME",
                   help="name and path of every file written (default 'stacked')")
    g.add_argument("--output-format", choices=["zarr", "ms"], default=None,
                   help="the stack: zarr (default) or ms")
    g.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    g.add_argument("--readers", type=int, default=None, help="reader processes (default: "
                   "from the cores)")
    g.add_argument("--writers", type=int, default=None, help="writer threads, zarr only")
    g.add_argument("--threads", type=int, default=None, help="kernel threads")
    g.add_argument("--gpu-batch", type=int, default=None, help="GPU: chunks per launch")
    g.add_argument("--scratch", default=None, metavar="DIR",
                   help="write the stack on this disk, then move it")
    g.add_argument("--verbose", action="store_true", help="progress and results")
    g.add_argument("--debug", action="store_true",
                   help="also the details and the time of every stage")
    g.add_argument("--log-file", default=None, help="also write the log here")
    g.add_argument("--only", choices=["stack", "continuum", "measure"], nargs="+",
                   help="run only these steps (default: all)")
    g.add_argument("--tag", default=None, help="name of the measurement: files <name>_<tag>_*")

    s = p.add_argument_group("the stacking (stack)")
    s.add_argument("--line-freq", type=_freq, nargs="+", required=True, metavar="FREQ",
                   help="rest frequency of the line (required): GHz, or with its unit "
                        "(345.796GHz, 345796MHz); several for several lines, the first is "
                        "the one of the stack")
    s.add_argument("--line-window", type=_pair, nargs="+", default=[600.0], metavar="W",
                   help="window of every line, km/s: 500 (= -500..500) or lo:hi; one per "
                        "line or one for all (default 600); lines whose windows overlap "
                        "are fitted together")
    s.add_argument("--velocity-range", type=_pair, default=None, metavar="KMS",
                   help="band of the stack around the first line (default: the widest band "
                        "of one dataset)")
    s.add_argument("--nchan", type=int, default=None, help="number of channels instead")
    s.add_argument("--channel-width", type=float, default=None, metavar="HZ",
                   help="rest-frame channel width (default: the widest of the sample)")
    s.add_argument("--frame", default="LSRK", help="frame of the redshifts (default LSRK)")
    s.add_argument("--data-column", default="auto",
                   help="column of the inputs (default auto: CORRECTED_DATA, else DATA)")
    s.add_argument("--continuum", choices=["subtract", "fit", "none"], default="subtract",
                   help="subtracted while stacking (default), fitted with the lines, or "
                        "left as it is")
    s.add_argument("--continuum-order", default="0", choices=["0", "1", "bic"],
                   help="subtract: constant, straight line, or chosen by the BIC (default 0)")
    s.add_argument("--exclude-between", action="store_true",
                   help="also leave out of the continuum fit the channels between the lines")

    m = p.add_argument_group("the measurement (extract): only with --extract")
    m.add_argument("--extract", action="store_true",
                   help="measure the stack (without it, only the stack is written)")
    m.add_argument("--annuli", type=float, nargs=3, default=None, metavar=("BMIN", "BMAX", "N"),
                   help="annuli of baseline length, kλ (default 3 3000 18)")
    m.add_argument("--weighting-scheme", choices=["natural", "democratic"], default="natural")
    m.add_argument("--no-z-norm", action="store_true", help="no transport to z_ref")
    m.add_argument("--z-ref", type=float, default=None, help="default: the median")
    m.add_argument("--no-pb-corr", action="store_true", help="no primary-beam correction")
    m.add_argument("--flux-norm", action="store_true",
                   help="divide by the NORM column of the input list")
    m.add_argument("--flux-ref", type=float, default=None, help="default: the median")
    m.add_argument("--v0-range", type=_pair, nargs="+", default=None, metavar="LO:HI",
                   help="allowed centres, km/s; one for all or one per line; lo = hi fixes it")
    m.add_argument("--fwhm-range", type=_pair, nargs="+", default=None, metavar="LO:HI",
                   help="allowed FWHM, km/s; as above")
    m.add_argument("--model", type=_size, default="auto",
                   help="auto, point, gauss or a fixed FWHM in arcsec (default auto)")
    m.add_argument("--model-prior", type=float, nargs=2, default=None,
                   metavar=("ARCSEC", "DEX"), help="soft tie of the size")
    m.add_argument("--theta-range", type=_pair, default=None, metavar="MIN:MAX",
                   help="allowed size of the Gaussian, arcsec (default 0:15); min > 0 "
                        "leaves the point source out")
    m.add_argument("--flux-measure", choices=["auto", "gauss", "area"], default="auto",
                   help="the flux quoted: Gaussian, sum over the window, or auto (default)")
    m.add_argument("--shape-nsigma", type=float, default=None, metavar="X",
                   help="threshold of the shape test of auto, sigma (default 2)")
    m.add_argument("--window", type=_pair, nargs="+", default=None, metavar="W",
                   help="window of the area flux (default: the line windows)")
    m.add_argument("--fit-span", default=None, metavar="KMS",
                   help="channels farther than this from every line are not fitted "
                        "(default 1500; 'none': every channel)")
    m.add_argument("--min-sources-per-annulus", type=int, default=None, metavar="N",
                   help="annuli with fewer sources are left out (default 2)")
    m.add_argument("--n-bootstrap", type=int, default=500)
    m.add_argument("--n-null", type=int, default=1000)
    m.add_argument("--seed", type=int, default=42)
    m.add_argument("--figures-unit", default="Jy", help="unit of the data, for the figures")
    m.add_argument("--no-figures", action="store_true")
    return p


_EXTRACT_FLAGS = ("annuli", "no_z_norm", "z_ref", "no_pb_corr", "flux_norm", "flux_ref",
                  "v0_range", "fwhm_range", "model_prior", "theta_range", "shape_nsigma",
                  "window", "fit_span", "min_sources_per_annulus", "no_figures")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv[:1] == ["export"]:
        if len(argv) != 3:
            sys.exit("usage: python -m vista export STORE MS")
        from .stackio import export_ms
        export_ms(argv[1], argv[2])
        return 0
    args = build_parser().parse_args(argv)
    from .api import ViSta

    verbose = "debug" if args.debug else bool(args.verbose)
    given = [f for f in _EXTRACT_FLAGS if getattr(args, f) not in (None, False)]
    if given and not args.extract:
        sys.exit(f"vista: --{given[0].replace('_', '-')} is an option of the measurement: "
                 f"add --extract")
    try:
        v = ViSta(args.input, output_name=args.output_name, output_format=args.output_format,
                  device=args.device, n_readers=args.readers, n_writers=args.writers,
                  n_threads=args.threads, gpu_batch=args.gpu_batch, scratch_dir=args.scratch,
                  verbose=verbose, log_file=args.log_file)
        v.stack(line_freq=args.line_freq, line_window=_one_or_list(args.line_window),
                velocity_range=args.velocity_range, nchan=args.nchan,
                channel_width=args.channel_width, frame=args.frame,
                data_column=args.data_column, continuum=args.continuum,
                continuum_order="bic" if args.continuum_order == "bic"
                else int(args.continuum_order),
                exclude_between=args.exclude_between)
        if args.extract:
            v.extract(annuli=tuple(args.annuli) if args.annuli else (3.0, 3000.0, 18),
                      weighting_scheme=args.weighting_scheme, z_norm=not args.no_z_norm,
                      z_ref=args.z_ref, pb_corr=not args.no_pb_corr,
                      flux_norm=args.flux_norm, flux_ref=args.flux_ref,
                      v0_range=_range_arg(args.v0_range),
                      fwhm_range=_range_arg(args.fwhm_range), model=args.model,
                      model_prior=args.model_prior,
                      theta_range=(args.theta_range if isinstance(args.theta_range, tuple)
                                   else (0.0, args.theta_range or 15.0)),
                      flux_measure=args.flux_measure,
                      shape_nsigma=2.0 if args.shape_nsigma is None else args.shape_nsigma,
                      window=_one_or_list(args.window),
                      fit_span=_fit_span(args.fit_span),
                      min_sources_per_annulus=args.min_sources_per_annulus or 2,
                      n_bootstrap=args.n_bootstrap, n_null=args.n_null, seed=args.seed,
                      figures_draw=not args.no_figures, figures_unit=args.figures_unit)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        sys.exit(f"vista: {exc}")
    try:
        result = v.run(only=args.only, tag=args.tag)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        sys.exit(f"vista: {exc}")
    print(result)
    for kind, path in result.files.items():
        print(f"  {kind:22s} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
