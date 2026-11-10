"""
vista.api
=========
:class:`ViSta`: the whole method behind one object.

::

    from vista import ViSta

    v = ViSta("input_list.txt", output_name="co32")      # input, output, how it runs
    v.stack(line_freq=345.796, line_window=500)           # the stacking: lines, band, continuum
    v.extract(fwhm_range=(100, 600))                      # the measurement (skip it: stack only)
    v.help()                                              # every option: value now, what it can be
    result = v.run()                                      # stack, statistics, measurement
    result = v.run(only="measure")                        # the measurement again, in seconds

Three calls, three groups of options:

* ``ViSta(...)``: what is read and written, on which device and with which
  resources, and the log;
* ``v.stack(...)``: what makes the stack -- the lines and their windows, the
  band, the frame, the continuum (subtracted while the stack is written,
  fitted with the lines, or left as it is);
* ``v.extract(...)``: what is measured on it -- the annuli of baseline
  length, the weighting, the line model, the fluxes, the figures.  Without
  this call, ``run()`` only writes the stack.

Only the input list and the rest frequency are needed; the options of
``stack()`` and ``extract()`` are given by name.  Anything that does not make
sense is refused when it is set, before any data is read.
Every file starts with ``output_name``: the stack (``co32.zarr`` or
``co32.ms``), the statistics (``co32_line_stats.npy``,
``co32_cont_stats.npy``), the results (``co32_results.json``,
``co32_continuum_results.json``) and the figures (``co32_profile.png``,
``co32_uv.png``, ``co32_noise.png``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np

from . import _notice, _say, set_verbosity

_STEPS = ("stack", "continuum", "measure")
_CONTINUUM = ("subtract", "fit", "none")


def _check_count(name, value):
    if value is None:
        return None
    if isinstance(value, bool) or not float(value).is_integer() or int(value) < 1:
        raise ValueError(f"{name} must be a whole number >= 1 (or None: automatic), "
                         f"not {value!r}")
    return int(value)


def _check_bool(name, value):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be True or False, not {value!r}")
    return value


# ---------------------------------------------------------------------------
# help: every option, what it can be, what it does
# ---------------------------------------------------------------------------
_HELP_STEPS = {
    "vista": ("ViSta(input_list, ...)",
              "input, output, how it runs (input_list is required)"),
    "stack": ("v.stack(line_freq=..., ...)",
              "what makes the stack (line_freq is required)"),
    "extract": ("v.extract(...)",
                "what is measured; without this call run() only writes the stack"),
    "run": ("v.run(...)", "what runs"),
}

_HELP = {
    "vista": [
        ("input_list", "path", "one dataset per line: <ms> <z> <RA> <Dec> [FIELD_ID] "
                               "[SPW_IDS] [NORM]"),
        ("output_name", "name or path", "name of everything written: <name>.zarr (or .ms), "
                                        "<name>_*"),
        ("output_format", "'zarr' | 'ms' | None", "the stack: Zarr (parallel, compressed) or a "
                                                  "Measurement Set; None: zarr (ms if the name "
                                                  "ends in .ms)"),
        ("device", "'cpu' | 'gpu'", "where the kernel runs ('gpu': a CUDA build)"),
        ("n_readers", "int | None", "reader processes; None: from the cores"),
        ("n_writers", "int | None", "writer threads (zarr only); None: from the cores"),
        ("n_threads", "int | None", "kernel threads; None: the cores left"),
        ("gpu_batch", "int | None", "GPU: chunks per launch; None: as many as fit in memory"),
        ("scratch_dir", "path | None", "write the stack on this disk, then move it"),
        ("verbose", "False | True | 'debug'", "warnings only | progress and results | also "
                                              "the details and the timing"),
        ("log_file", "path | None", "also write the log here"),
    ],
    "stack": [
        ("line_freq", "GHz | '345.8GHz' | list", "rest frequency (a number: GHz; a string: "
                                                 "with its unit); several lines: the first is "
                                                 "the one of the stack"),
        ("line_window", "W | (lo, hi) | list", "km/s, window of every line (500: -500..500): "
                                               "out of the continuum fit; lines whose windows "
                                               "overlap are fitted together"),
        ("velocity_range", "km/s | (lo, hi) | None", "band of the stack around the first line; "
                                                     "None: the widest coverage of one dataset"),
        ("nchan", "int | None", "a number of channels instead of velocity_range"),
        ("channel_width", "Hz | None", "rest-frame channel width; None: the coarsest channel "
                                       "of the sample"),
        ("frame", "'LSRK' | 'BARY' | ...", "frame of the redshifts: the inputs are brought to it"),
        ("data_column", "'auto' | column", "column read; auto: CORRECTED_DATA if present, "
                                           "else DATA"),
        ("continuum", "'subtract' | 'fit' | 'none'", "subtracted while stacking (then measured "
                                                     "on its own) | left, fitted with the lines "
                                                     "| left as it is"),
        ("continuum_order", "0 | 1 | 'bic'", "subtract: constant, straight line, or chosen by "
                                             "the BIC per spectral window"),
        ("exclude_between", "True | False", "several lines: also keep the channels between "
                                            "them out of the continuum fit"),
    ],
    "extract": [
        ("annuli", "(kλ min, kλ max, n)", "annuli of baseline length, logarithmic"),
        ("weighting_scheme", "'natural' | 'democratic'", "weights of the data (deepest count "
                                                         "most) | every source the same"),
        ("z_norm", "True | False", "bring every source to the redshift z_ref"),
        ("z_ref", "float | None", "that redshift; None: the median of the sample (reported)"),
        ("pb_corr", "True | False", "correct every source for the primary beam"),
        ("flux_norm", "True | False", "divide every source by the NORM column of the list"),
        ("flux_ref", "float | None", "reference of that normalisation; None: the median "
                                     "(reported)"),
        ("v0_range", "(lo, hi) | list | None", "km/s, allowed centre of every line; lo = hi "
                                              "fixes it; None: in the window"),
        ("fwhm_range", "(lo, hi) | list | None", "km/s, allowed FWHM; lo = hi fixes it; None: "
                                                "two channels to the window"),
        ("model", "'auto' | 'point' | 'gauss' | arcsec", "the source: Gaussian if the data "
                                                         "prefer it | point | Gaussian | "
                                                         "Gaussian of this FWHM"),
        ("model_prior", "(arcsec, dex) | None", "soft tie of the size of the Gaussian"),
        ("flux_measure", "'auto' | 'gauss' | 'area'", "flux quoted (both are given): Gaussian "
                                                      "unless not Gaussian | Gaussian | sum over "
                                                      "the window"),
        ("window", "W | (lo, hi) | list | None", "window of the area flux; None: the line "
                                                 "windows"),
        ("n_bootstrap", "int", "resamplings of the sources, for the errors (0: none)"),
        ("n_null", "int", "searches on noise, for the significance (0: Bonferroni bound)"),
        ("seed", "int", "seed of the random numbers of bootstrap and null searches: the same "
                        "seed gives the same numbers"),
        ("figures_draw", "True | False", "draw the figures (profile, uv, noise, continuum)"),
        ("figures_unit", "str", "unit of the data, for the labels"),
        ("(advanced)", "keywords", "noise, channel_correlation, n_null_phase, "
                                   "resolved_delta_chi2, theta_max_arcsec, fit_span_kms, "
                                   "min_sources_per_bin, shape_nsigma, residual_nsigma, "
                                   "H0, Om0, already_applied"),
    ],
    "run": [
        ("only", "None | 'stack' | 'continuum' | 'measure' | list",
         "None: everything; 'continuum': the subtraction again on the stack; 'measure': "
         "the measurement again (seconds)"),
        ("tag", "str | None", "name of a measurement: its files are <output_name>_<tag>_*"),
    ],
}

def _fmt(value):
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(x, (int, float)) for x in value):
            return "(" + ", ".join(f"{x:g}" for x in value) + ")"
        return "[" + ", ".join(_fmt(x) for x in value) + "]"
    return repr(value)


@dataclass
class RunResult:
    """What :meth:`ViSta.run` produced.

    Attributes
    ----------
    line : vista.extract.LineResult or None
        The lines: ``line.summary`` holds every number (also in
        ``<output_name>_results.json``).
    continuum : vista.extract.ExtractionResult or None
        The continuum, when it was subtracted and measured.
    stack : dict or None
        Summary of the stacking (when it ran).
    files : dict
        The files of the run, by kind.
    """

    line: object = None
    continuum: object = None
    stack: Optional[dict] = None
    files: dict = field(default_factory=dict)

    def __repr__(self):
        rows = []
        if self.stack is not None:
            s = self.stack
            rows.append(f"stack: {s['output']} ({s['n_rows']:,} rows, {s['nchan']} channels, "
                        f"{s['seconds']:.0f} s)")
        if self.line is not None:
            sm = self.line.summary
            for k, b in enumerate(sm["lines"]):
                rows.append(f"line {k + 1} ({b['rest_freq_ghz']:.6g} GHz): "
                            f"F = {b['flux']:.4g} ± {b['flux_error']:.2g} "
                            f"({'area' if b['flux_method'] == 'area' else 'Gaussian'}), "
                            f"v0 = {b['v0_kms']:+.0f} km/s, FWHM = {b['fwhm_kms']:.0f} km/s, "
                            f"{b['detection']['significance_sigma']:.1f} sigma")
            if sm.get("z_ref") is not None:
                rows.append(f"z_ref = {sm['z_ref']:.4f} "
                            f"({'given' if sm.get('z_ref_given') else 'median of the sample'})")
            if sm.get("flux_ref") is not None:
                rows.append(f"flux_ref = {sm['flux_ref']:.4g} "
                            f"({'given' if sm.get('flux_ref_given') else 'median of the sample'})")
        if self.continuum is not None:
            c = self.continuum.summary
            rows.append(f"continuum: {c['flux_total']:.4g} ± {c['flux_total_error']:.2g}, "
                        f"size {c.get('theta_fwhm_arcsec', 0):.2f}\"")
        elif self.line is not None and self.line.summary.get("continuum_fit"):
            c = self.line.summary["continuum_fit"]
            rows.append(f"continuum (fitted with the lines): {c['flux_density']:.4g} ± "
                        f"{c['flux_density_error']:.2g}")
        return "RunResult(\n  " + "\n  ".join(rows or ["nothing run"]) + "\n)"


class ViSta:
    """Stack interferometric observations of many sources in the uv plane
    and measure the stacked lines and continuum.

    ``ViSta(...)`` takes the input, the output and how the run uses the
    machine; :meth:`stack` what makes the stack (the lines are required);
    :meth:`extract` what is measured on it (without it, :meth:`run` only
    writes the stack); :meth:`help` lists every option with its value now and
    what it can be.

    Parameters
    ----------
    input_list : str
        Text file, one dataset per line::

            <ms_path>  <redshift>  <RA>  <Dec>  [FIELD_ID]  [SPW_IDS]  [NORM]

    output_name : str, default 'stacked'
        Name (and path) of everything written: the stack is
        ``<output_name>.zarr`` (or ``.ms``), the other files start with it.
        For ``run(only="measure")`` it says where the stack and its
        statistics are.
    output_format : {'zarr', 'ms'} or None
        The stack: a Zarr store (written in parallel; ``vista.export_ms``
        converts it) or a Measurement Set.  None: ``'zarr'``, or ``'ms'``
        when *output_name* ends in ``.ms``.
    device : {'cpu', 'gpu'}, default 'cpu'
        Where the kernel runs (``'gpu'``: a CUDA build and a GPU).
    n_readers, n_writers, n_threads : int or None
        Reader processes, writer threads (Zarr only) and kernel threads.
        None: from the cores this process may use; the ones you fix are
        taken out first, the others share what is left; more than the cores
        is refused.
    gpu_batch : int or None
        GPU: chunks per launch.  None: as many as fit in the memory.
    scratch_dir : str or None
        Write the stack on this (fast, local) disk and move it at the end.
    verbose : bool or 'debug', default False
        False: warnings and errors only; True: the progress and the results;
        'debug': also the details and the time of every stage.
    log_file : str or None
        Also write the log here.
    """

    def __init__(self, input_list, output_name="stacked", output_format=None, device="cpu",
                 n_readers=None, n_writers=None, n_threads=None, gpu_batch=None,
                 scratch_dir=None, verbose=False, log_file=None):
        from .extract.sources import read_input_list
        from .resources import plan_workers

        if verbose not in (False, True, "debug"):
            raise ValueError(f"verbose must be False, True or 'debug', not {verbose!r}")
        if not os.path.isfile(str(input_list)):
            raise FileNotFoundError(f"input list {input_list!r} not found (cwd {os.getcwd()})")
        self.input_list = str(input_list)
        self._entries = [e for e in read_input_list(self.input_list) if e is not None]
        if not self._entries:
            raise ValueError(f"{input_list}: no dataset in the input list")
        missing = [e.ms for e in self._entries if not os.path.isdir(e.ms)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} dataset(s) of {input_list} not found: "
                                    + ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else ""))
        self.n_entries = len(self._entries)

        name = str(output_name or "").rstrip("/")
        if not name:
            raise ValueError("output_name must not be empty")
        ext = os.path.splitext(name)[1].lower()
        if output_format is None:
            output_format = "ms" if ext == ".ms" else "zarr"
        if output_format not in ("zarr", "ms"):
            raise ValueError(f"output_format must be 'zarr' or 'ms', not {output_format!r}")
        if ext in (".ms", ".zarr"):
            if ext[1:] != output_format:
                raise ValueError(f"output_name {output_name!r} ends in {ext} but "
                                 f"output_format is {output_format!r}")
            name = name[:-len(ext)]
        if device not in ("cpu", "gpu"):
            raise ValueError(f"device must be 'cpu' or 'gpu', not {device!r}")
        counts = {k: _check_count(k, v) for k, v in (("n_readers", n_readers),
                                                     ("n_writers", n_writers),
                                                     ("n_threads", n_threads),
                                                     ("gpu_batch", gpu_batch))}
        plan_workers(self.n_entries, output_format == "zarr", counts["n_readers"],
                     counts["n_writers"], counts["n_threads"])        # refuses now
        if scratch_dir is not None and not os.path.isdir(scratch_dir):
            raise FileNotFoundError(f"scratch_dir {scratch_dir!r} does not exist")
        self.output_name, self.output_format = name, output_format
        self.run_options = dict(device=device, scratch_dir=scratch_dir, **counts)
        self.verbose = verbose
        self.log_file = log_file
        self.lines = None
        self.stack_options = None
        self.extract_options = None

    # ---- the stacking -------------------------------------------------------
    def stack(self, *, line_freq, line_window=600.0, velocity_range=None, nchan=None,
              channel_width=None, frame="LSRK", data_column="auto", continuum="subtract",
              continuum_order=0, exclude_between=False):
        """What makes the stack.  Every option is given by name:
        ``v.stack(line_freq=345.796, line_window=500)``.

        Parameters
        ----------
        line_freq : float, str or a list of them
            Rest frequency of the line: a number in GHz, or a string with its
            unit (``"345.796GHz"``, ``"345796 MHz"``; Hz, kHz, MHz, GHz,
            THz).  Several lines: a list; the first is the one the stack is
            centred on, the velocities refer to it.
        line_window : float, (lo, hi) or list, default 600
            Window of every line in its own velocity, km/s: ``500`` means
            -500..+500.  With several lines, one entry per line (or one for
            all).  Its channels are kept out of the continuum fit, and the area
            flux of the line is the sum over it.  Lines whose windows overlap
            are fitted together; the others each on its own.
        velocity_range : float, (lo, hi) or None, default None
            Band of the stack around the first line, km/s (2000: -2000 to
            +2000).  None: the widest band a single dataset covers.
        nchan : int or None
            A number of channels instead of *velocity_range*.
        channel_width : float or None
            Rest-frame channel width, Hz; None: the widest channel of the
            sample (the finest grid every dataset supports).
        frame : str, default 'LSRK'
            Frame of the redshifts ('LSRK', 'BARY', ...): the frequencies of
            every dataset are brought to it, row by row.
        data_column : str, default 'auto'
            Column of the inputs: 'auto' takes CORRECTED_DATA when present,
            otherwise DATA.
        continuum : {'subtract', 'fit', 'none'}, default 'subtract'
            ``'subtract'``: fitted row by row on the channels outside the line
            windows and subtracted while the stack is written (``MODEL_DATA``,
            ``CORRECTED_DATA``; ``DATA`` is kept), then measured on its own.
            ``'fit'``: left in the data and fitted together with the lines (a
            constant with their size).  ``'none'``: left as it is (data
            already subtracted, or no continuum); a continuum still present is
            flagged by the measurement.
        continuum_order : {0, 1, 'bic'}, default 0
            With ``'subtract'``: a constant, a straight line, or the one of
            the two the BIC prefers in every spectral window.
        exclude_between : bool, default False
            With several lines, also keep the channels between them out of
            the continuum fit.
        """
        from .extract.config import Lines
        from .frames import TARGET_FRAMES

        lines = Lines(line_freq, line_window, exclude_between=_check_bool(
            "exclude_between", exclude_between))
        fr = str(frame).upper()
        if fr not in TARGET_FRAMES:
            raise ValueError(f"frame must be one of {TARGET_FRAMES}, not {frame!r}")
        if continuum is True:
            continuum = "subtract"
        elif continuum in (False, None, "no"):
            continuum = "none"
        if continuum not in _CONTINUUM:
            raise ValueError(f"continuum must be 'subtract', 'fit' or 'none', not {continuum!r}")
        order = {"bic": -1, "BIC": -1}.get(continuum_order, continuum_order)
        if order not in (0, 1, -1):
            raise ValueError(f"continuum_order must be 0, 1 or 'bic', not {continuum_order!r}")
        if nchan is not None:
            nchan = _check_count("nchan", nchan)
            if velocity_range is not None:
                raise ValueError("give velocity_range or nchan, not both")
        if velocity_range is not None:
            lo, hi = ((-abs(float(velocity_range)), abs(float(velocity_range)))
                      if np.ndim(velocity_range) == 0
                      else tuple(float(x) for x in velocity_range))
            if not lo < hi:
                raise ValueError(f"velocity_range: an empty band {velocity_range!r}")
            for k in range(lines.n):                  # every window inside the band
                f_lo, f_hi = lines.window_freq_hz(k)
                v = lines.velocity(np.array([f_hi, f_lo]))
                if v[0] < lo or v[1] > hi:
                    raise ValueError(
                        f"the window of line {k + 1} ({v[0]:+.0f}..{v[1]:+.0f} km/s from the "
                        f"first line) is not inside the band of the stack ({lo:+.0f}..{hi:+.0f} "
                        f"km/s): widen velocity_range")
        if channel_width is not None and not float(channel_width) > 0:
            raise ValueError("channel_width must be > 0 Hz")
        self.lines = lines
        self.stack_options = dict(
            velocity_range_kms=velocity_range, nchan_out=nchan, channel_width_hz=channel_width,
            frame=fr, data_column=data_column, continuum=continuum, continuum_order=int(order))
        if self.extract_options is not None:
            self._check_extract(self.extract_options)
        return self

    # ---- the measurement ----------------------------------------------------
    def extract(self, *, annuli=(3.0, 3000.0, 18), weighting_scheme="natural", z_norm=True,
                z_ref=None, pb_corr=True, flux_norm=False, flux_ref=None, v0_range=None,
                fwhm_range=None, model="auto", model_prior=None, flux_measure="auto",
                window=None, n_bootstrap=500, n_null=1000, seed=42,
                figures_draw=True, figures_unit="Jy", **advanced):
        """What is measured on the stack.  Without this call, :meth:`run`
        only writes the stack.  Every option is given by name.

        Every line is fitted on every channel and annulus of baseline length,
        its centre and width searched within their ranges.  Lines whose
        windows (``line_window`` of :meth:`stack`) overlap, directly or
        through another line, are fitted together, each with its own flux,
        centre and width and sharing the size, so the fit separates them;
        every other line is measured on its own (its own size and model, on
        the channels near it and outside the windows of the other lines).

        Parameters
        ----------
        annuli : (b_min, b_max, n), default (3, 3000, 18)
            Annuli of rest-frame baseline length, kλ, logarithmic: the
            statistics are summed in them (changing them compresses the stack
            again).
        weighting_scheme : {'natural', 'democratic'}, default 'natural'
            The weights of the data (the deepest observations count most), or
            every source the same.
        z_norm : bool, default True
            Bring every source to the same redshift ``z_ref`` (luminosity
            distance and (1 + z) factors).
        z_ref : float or None
            That redshift; None: the median of the sample, reported in the
            results.
        pb_corr : bool, default True
            Correct every source for the primary beam at its position.
        flux_norm : bool, default False
            Divide every source by the last column of the input list (NORM: a
            luminosity, a continuum flux...), times ``flux_ref``; every line
            of the list must have it.
        flux_ref : float or None
            Reference of that normalisation; None: the median, reported.
        v0_range, fwhm_range : (lo, hi), a list of them (one per line), or None
            Allowed centre and FWHM, km/s, in the velocity of each line;
            ``lo == hi`` fixes the value.  None: the centre anywhere in the
            window, the FWHM from two channels to the width of the window.
        model : 'auto', 'point', 'gauss' or a number, default 'auto'
            The source: a Gaussian only if the data prefer it (Δχ² ≥ 9),
            always a point, always a Gaussian, or a Gaussian of this FWHM
            (arcsec).
        model_prior : (theta_arcsec, sigma_dex) or None
            Soft tie of the size of the Gaussian (e.g. to the continuum size).
        flux_measure : {'auto', 'gauss', 'area'}, default 'auto'
            Which flux is quoted for every line (both are always given and
            drawn): the integral of the Gaussian, the sum of the spectrum
            over the window, or the Gaussian unless the line is not Gaussian
            (the two differ by more than 2 sigma).
        window : number, (lo, hi), a list of them, or None
            Window of the area flux of every line; None: the line windows.
            It can be changed and the measurement repeated without stacking.
        n_bootstrap : int, default 500
            Resamplings of the sources, for the population errors (0: none).
        n_null : int, default 1000
            Searches on noise, for the significance (0: a Bonferroni bound).
        seed : int, default 42
            Seed of the random numbers of the bootstrap and of the null
            searches: the same seed gives the same numbers.
        figures_draw : bool, default True
            Draw the figures (profile, uv, noise, continuum).
        figures_unit : str, default 'Jy'
            Unit of the data, for the labels (scaled to mJy, μJy as needed).
        advanced
            ``noise`` ('scatter' or 'weights'), ``channel_correlation``,
            ``n_null_phase``, ``resolved_delta_chi2``, ``theta_max_arcsec``,
            ``fit_span_kms``, ``min_sources_per_bin``, ``shape_nsigma``,
            ``residual_nsigma``; for the weighting ``H0``, ``Om0``,
            ``already_applied``.
        """
        from .extract.config import (BinningConfig, BootstrapConfig, LineFitConfig,
                                     WeightingConfig)

        line_keys = {f.name for f in fields(LineFitConfig)}
        weight_keys = {"H0", "Om0", "already_applied"}
        own = {"v0_range_kms", "fwhm_range_kms", "size", "theta_fixed_arcsec",
               "theta_prior_arcsec", "flux", "window_kms", "n_null", "continuum"}
        bad = sorted(set(advanced) - (line_keys | weight_keys) | (set(advanced) & own))
        if bad:
            raise TypeError(f"extract: unknown option(s) {', '.join(bad)} "
                            f"(v.help('extract') lists them)")
        # the annuli
        try:
            b_min, b_max, n_bins = annuli
        except (TypeError, ValueError):
            raise ValueError(f"annuli: (b_min, b_max, n) in kλ, not {annuli!r}") from None
        if not 0 < float(b_min) < float(b_max):
            raise ValueError(f"annuli: need 0 < b_min < b_max (kλ), not {b_min}, {b_max}")
        binning = BinningConfig(float(b_min), float(b_max), _check_count("n", n_bins))
        # the weighting
        for name, value in (("z_norm", z_norm), ("pb_corr", pb_corr), ("flux_norm", flux_norm),
                            ("figures_draw", figures_draw)):
            _check_bool(name, value)
        if flux_norm:
            no_norm = [e for e in self._entries if e.norm is None]
            if no_norm:
                raise ValueError(
                    f"flux_norm=True, but {len(no_norm)} line(s) of {self.input_list} have no "
                    f"NORM column (the last one): "
                    + ", ".join(os.path.basename(e.ms) for e in no_norm[:5])
                    + (" ..." if len(no_norm) > 5 else ""))
        weighting = WeightingConfig(
            scheme=weighting_scheme, redshift_rescaling=z_norm, z_ref=z_ref,
            primary_beam=pb_corr, flux_normalisation=flux_norm, norm_ref=flux_ref,
            **{k: v for k, v in advanced.items() if k in weight_keys})
        # the lines
        theta_fixed = None
        size = model
        if not isinstance(model, str):
            theta_fixed = float(model)
            if theta_fixed < 0:
                raise ValueError(f"model: a fixed size must be >= 0 arcsec, not {model}")
            size = "point" if theta_fixed == 0 else "gauss"
        elif model not in ("auto", "point", "gauss"):
            raise ValueError(f"model must be 'auto', 'point', 'gauss' or a size in arcsec, "
                             f"not {model!r}")
        if model_prior is not None:
            model_prior = tuple(float(x) for x in model_prior)
            if len(model_prior) != 2 or model_prior[0] <= 0 or model_prior[1] <= 0:
                raise ValueError("model_prior: (theta_arcsec > 0, sigma_dex > 0)")
        if flux_measure not in ("auto", "gauss", "area"):
            raise ValueError(f"flux_measure must be 'auto', 'gauss' or 'area', "
                             f"not {flux_measure!r}")
        n_boot = int(n_bootstrap)
        if n_boot < 0 or int(n_null) < 0:
            raise ValueError("n_bootstrap and n_null must be >= 0")
        line_cfg = LineFitConfig(
            v0_range_kms=v0_range, fwhm_range_kms=fwhm_range, size=size,
            theta_fixed_arcsec=theta_fixed if theta_fixed else None,
            theta_prior_arcsec=model_prior, flux=flux_measure, window_kms=window,
            n_null=int(n_null),
            **{k: v for k, v in advanced.items() if k in line_keys})
        opts = dict(binning=binning, weighting=weighting, line=line_cfg,
                    bootstrap=BootstrapConfig(n_realisations=n_boot, seed=int(seed)),
                    draw=figures_draw, unit=str(figures_unit),
                    values=dict(annuli=(float(b_min), float(b_max), int(n_bins)),
                                weighting_scheme=weighting_scheme, z_norm=z_norm, z_ref=z_ref,
                                pb_corr=pb_corr, flux_norm=flux_norm, flux_ref=flux_ref,
                                v0_range=v0_range, fwhm_range=fwhm_range, model=model,
                                model_prior=model_prior, flux_measure=flux_measure,
                                window=window, n_bootstrap=n_boot,
                                n_null=int(n_null), seed=int(seed), figures_draw=figures_draw,
                                figures_unit=str(figures_unit)))
        if self.lines is not None:
            self._check_extract(opts)
        self.extract_options = opts
        return self

    def _check_extract(self, opts):
        """The options of the measurement that depend on the number of lines."""
        from .extract.config import per_line, ranges_per_line
        n, c = self.lines.n, opts["line"]
        ranges_per_line(c.v0_range_kms, n, "v0_range")
        ranges_per_line(c.fwhm_range_kms, n, "fwhm_range")
        if c.window_kms is not None:
            per_line(c.window_kms, n, "window")

    # ---- help -----------------------------------------------------------------
    def _values(self, step):
        """The value now of every option of a step."""
        if step == "vista":
            o = self.run_options
            return {"input_list": self.input_list, "output_name": self.output_name,
                    "output_format": self.output_format, "device": o["device"],
                    "n_readers": o["n_readers"], "n_writers": o["n_writers"],
                    "n_threads": o["n_threads"], "gpu_batch": o["gpu_batch"],
                    "scratch_dir": o["scratch_dir"], "verbose": self.verbose,
                    "log_file": self.log_file}
        if step == "stack":
            if self.stack_options is None:
                return {"line_freq": "(not set)"}
            o, ln = self.stack_options, self.lines
            order = o["continuum_order"]
            return {"line_freq": ln.freqs_ghz[0] if ln.n == 1 else ln.freqs_ghz,
                    "line_window": ln.windows_kms[0] if ln.n == 1 else ln.windows_kms,
                    "velocity_range": o["velocity_range_kms"], "nchan": o["nchan_out"],
                    "channel_width": o["channel_width_hz"], "frame": o["frame"],
                    "data_column": o["data_column"], "continuum": o["continuum"],
                    "continuum_order": "bic" if order == -1 else order,
                    "exclude_between": ln.exclude_between}
        if step == "extract":
            if self.extract_options is None:
                return {"annuli": "(not set: run() only stacks)"}
            return dict(self.extract_options["values"], **{"(advanced)": ""})
        return {}

    def help(self, *steps):
        """Print the options, with their values now and what they can be.

        Parameters
        ----------
        steps : str
            Some of ``'vista'``, ``'stack'``, ``'extract'``, ``'run'``; none: all.
        """
        import shutil
        import textwrap

        steps = steps or tuple(_HELP_STEPS)
        bad = [x for x in steps if x not in _HELP_STEPS]
        if bad:
            raise ValueError(f"help: {', '.join(map(repr, bad))} is not a step: "
                             f"{', '.join(_HELP_STEPS)}")
        width = max(90, min(shutil.get_terminal_size((110, 24)).columns, 140))
        out = []
        for step in steps:
            call, what = _HELP_STEPS[step]
            out.append(f"{call}  —  {what}")
            values = self._values(step)
            rows = []
            for name, allowed, text in _HELP[step]:
                now = values.get(name, "")
                rows.append((name, now if isinstance(now, str) and now.startswith("(not")
                             else "" if now == "" else _fmt(now), allowed, text))
            w0 = max(len("option"), *(len(r[0]) for r in rows)) + 2
            w1 = min(max(len("now"), *(len(r[1]) for r in rows)) + 2, 30)
            w2 = min(max(len("can be"), *(len(r[2]) for r in rows)) + 2, 40)
            w3 = max(30, width - w0 - w1 - w2 - 2)
            out.append(f"  {'option':{w0}}{'now':{w1}}{'can be':{w2}}what it does")
            for name, now, allowed, text in rows:
                first, *rest = textwrap.wrap(text, w3) or [""]
                if len(now) >= w1:                    # a long value: on its own row
                    out.append(f"  {name:{w0}}{now}")
                    out.append(f"  {'':{w0 + w1}}{allowed}")
                    out.append(f"  {'':{w0 + w1 + w2}}{first}")
                elif len(allowed) >= w2:
                    out.append(f"  {name:{w0}}{now:{w1}}{allowed}")
                    out.append(f"  {'':{w0 + w1 + w2}}{first}")
                else:
                    out.append(f"  {name:{w0}}{now:{w1}}{allowed:{w2}}{first}")
                out += [f"  {'':{w0 + w1 + w2}}{t}" for t in rest]
            out.append("")
        out.append("help(ViSta), help(v.stack), help(v.extract): the full description.")
        print("\n".join(out))

    # ---- files ----------------------------------------------------------------
    @property
    def files(self) -> dict:
        """The files of a run (the measured ones appear once written)."""
        return self._files(None)

    def _files(self, tag):
        base = self.output_name
        res = base if not tag else f"{base}_{tag}"
        return {"stack": f"{base}.{self.output_format}",
                "line_statistics": f"{base}_line_stats.npy",
                "continuum_statistics": f"{base}_cont_stats.npy",
                "results": f"{res}_results.json",
                "continuum_results": f"{res}_continuum_results.json"}

    # ---- the run --------------------------------------------------------------
    def run(self, only=None, tag=None):
        """Stack and, if :meth:`extract` was called, measure.

        Parameters
        ----------
        only : None, 'stack', 'continuum', 'measure' or a list of them
            None: everything (the stack; with :meth:`extract`, its statistics
            and the measurement).  'stack': the stack (and its statistics).
            'continuum': the subtraction again on the existing stack (e.g.
            with other line windows; DATA is always kept).  'measure': the
            measurement again from the statistics, in seconds; the stack is
            compressed again first if the annuli or the continuum changed.
        tag : str or None
            Name of this measurement: its results and figures are written as
            ``<output_name>_<tag>_*`` instead of ``<output_name>_*``, so that
            variants on the same statistics do not overwrite each other.

        Returns
        -------
        RunResult
        """
        if self.stack_options is None:
            raise RuntimeError("call v.stack(line_freq=..., ...) first: the lines are needed "
                               "(also to measure an existing stack)")
        ex = self.extract_options
        steps = ((["stack", "measure"] if ex is not None else ["stack"]) if only is None
                 else [only] if isinstance(only, str) else list(only))
        bad = [s for s in steps if s not in _STEPS]
        if bad:
            raise ValueError(f"only: {bad} not among {_STEPS}")
        if "measure" in steps and ex is None:
            raise RuntimeError("nothing to measure: call v.extract(...) first")
        if "stack" in steps and "continuum" in steps:
            steps.remove("continuum")              # done while the stack is written
        if "continuum" in steps and self.stack_options["continuum"] != "subtract":
            raise ValueError("only='continuum' redoes the subtraction: it needs "
                             "v.stack(..., continuum='subtract')")
        if tag is not None and (not str(tag) or "/" in str(tag)):
            raise ValueError(f"tag must be a plain name, not {tag!r}")
        set_verbosity(self.verbose, self.log_file)
        out = RunResult(files=dict(self._files(tag)))
        d = os.path.dirname(os.path.abspath(self.output_name))
        os.makedirs(d, exist_ok=True)
        if "stack" in steps:
            out.stack = self._stack()
        if "continuum" in steps:
            self._continuum_again()
        if "measure" in steps:
            out.line, out.continuum = self._measure(tag)
        subtract = self.stack_options["continuum"] == "subtract"
        if out.continuum is None:
            out.files.pop("continuum_results", None)
        if ex is None:
            for k in ("line_statistics", "continuum_statistics", "results"):
                out.files.pop(k, None)
        elif not subtract:
            out.files.pop("continuum_statistics", None)
        return out

    def _stack(self):
        from .extract.config import ContinuumConfig
        from .pipeline import Stacker

        o = dict(self.stack_options)
        cont, order = o.pop("continuum"), o.pop("continuum_order")
        ex = self.extract_options
        f = self.files
        subtract = cont == "subtract"
        return Stacker(self.input_list, verbose=self.verbose).run(
            output=f["stack"], central_freq=self.lines.rest_freq_hz,
            output_format=self.output_format, line=self.lines,
            continuum=ContinuumConfig(order=order) if subtract else None,
            stats_line=f["line_statistics"] if ex is not None else None,
            stats_continuum=f["continuum_statistics"] if ex is not None and subtract else None,
            binning=ex["binning"] if ex is not None else None, **o, **self.run_options)

    def _compress(self):
        """The statistics again from the stack (annuli or continuum changed)."""
        from .extract.statistics import compress_visibilities

        f, ex = self.files, self.extract_options
        subtract = self.stack_options["continuum"] == "subtract"
        compress_visibilities(f["stack"], self.input_list, self.lines,
                              binning=ex["binning"], out_line=f["line_statistics"],
                              out_continuum=f["continuum_statistics"] if subtract else None,
                              line_column="CORRECTED_DATA" if subtract else "DATA",
                              resume=False, verbose=bool(self.verbose))

    def _continuum_again(self):
        from .extract.config import ContinuumConfig
        from .extract.contsub import subtract_continuum

        f = self.files
        if not os.path.isdir(f["stack"]):
            raise FileNotFoundError(f"{f['stack']} not found: run the stacking first "
                                    f"(run() or run(only='stack'))")
        subtract_continuum(f["stack"], self.lines,
                           ContinuumConfig(order=self.stack_options["continuum_order"]),
                           verbose=bool(self.verbose))
        if self.extract_options is not None:
            self._compress()

    def _statistics_ready(self):
        """Compress the stack again when its statistics are missing, or were
        made with other annuli or from another column."""
        from .extract.statistics import SufficientStatistics

        f, ex = self.files, self.extract_options
        subtract = self.stack_options["continuum"] == "subtract"
        column = "CORRECTED_DATA" if subtract else "DATA"
        why = None
        if not os.path.isfile(f["line_statistics"]):
            why = "there are no statistics"
        else:
            st = SufficientStatistics.load(f["line_statistics"])
            edges = np.asarray(ex["binning"].edges_lambda(), float)
            if (st.bin_edges_lambda.shape != edges.shape
                    or not np.allclose(st.bin_edges_lambda, edges, rtol=1e-9)):
                why = "the annuli changed"
            elif (st.meta or {}).get("line_column", column) != column:
                if subtract:
                    raise RuntimeError(
                        "the statistics are from DATA (a stack without the continuum "
                        "subtraction): subtract it first, run(only=['continuum', 'measure'])")
                why = "the continuum is no longer subtracted: the line from DATA"
        if why is None:
            return
        if not os.path.isdir(f["stack"]):
            raise FileNotFoundError(f"{f['stack']} not found: run the stacking first (run())")
        _say(f"Compressing the stack again ({why})", bool(self.verbose))
        self._compress()

    def _measure(self, tag=None):
        from dataclasses import replace

        from .extract.config import ProfileConfig, UVFitConfig
        from .extract.measure import measure_line
        from .extract.plots_line import plot_line_result
        from .extract.stack import extract_flux

        self._statistics_ready()
        f, ex = self._files(tag), self.extract_options
        prefix = f["results"][:-len("_results.json")]
        verbose = bool(self.verbose)
        mode = self.stack_options["continuum"]
        cfg = replace(ex["line"], continuum=mode == "fit")
        line = measure_line(f["line_statistics"], self.input_list, self.lines,
                            ex["weighting"], cfg, ex["bootstrap"],
                            output_prefix=prefix, verbose=verbose)
        sm = line.summary
        refs = []
        if sm.get("z_ref") is not None:
            refs.append(f"z_ref = {sm['z_ref']:.4f} "
                        f"({'given' if sm['z_ref_given'] else 'median of the sample'})")
        if sm.get("flux_ref") is not None:
            refs.append(f"flux_ref = {sm['flux_ref']:.4g} "
                        f"({'given' if sm['flux_ref_given'] else 'median of the sample'})")
        _notice("; ".join(refs) if refs else "no redshift rescaling, no flux normalisation")
        if ex["draw"]:
            plot_line_result(line, self.lines, prefix, unit=ex["unit"], verbose=verbose)
        cont = None
        has_cont = mode == "subtract" and os.path.isfile(f["continuum_statistics"])
        if has_cont and (self.n_entries < 2 or ex["bootstrap"].n_realisations < 2):
            _say("Continuum not measured: its errors come from the bootstrap over the "
                 "sources (at least 2 sources and n_bootstrap >= 2)", verbose)
        elif has_cont:
            _say("Continuum, from the channels outside the line windows (MODEL_DATA):",
                 verbose)
            cont = extract_flux(f["continuum_statistics"], self.input_list, self.lines,
                                ex["weighting"], profile=ProfileConfig(method="continuum"),
                                uvfit=UVFitConfig(model="gauss"), bootstrap=ex["bootstrap"],
                                output_prefix=f"{prefix}_continuum", verbose=verbose)
            if ex["draw"]:
                from .extract.plots import plot_all
                plot_all(cont, self.lines, f"{prefix}_continuum", unit=ex["unit"])
        return line, cont

    def __repr__(self):
        lines = "no lines yet (v.stack)" if self.lines is None else str(self.lines)
        return (f"ViSta({self.input_list!r}: {self.n_entries} datasets, {lines}, "
                f"→ {self.files['stack']}"
                + (", measured" if self.extract_options is not None else ", stack only") + ")")


__all__ = ["ViSta", "RunResult"]
