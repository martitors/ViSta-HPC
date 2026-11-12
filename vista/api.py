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
from dataclasses import dataclass, field
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


def _check_min_sources(value):
    if isinstance(value, bool) or not float(value).is_integer() or int(value) < 1:
        raise ValueError(f"min_sources_per_annulus must be a whole number >= 1, not {value!r}")
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
              "what makes the stack (without line_freq: the continuum only)"),
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
        ("line_freq", "GHz | '345.8GHz' | list | None", "rest frequency (a number: GHz; a "
                                                        "string: with its unit); several "
                                                        "lines: the first is the one of the "
                                                        "stack; None: no line, the stack of "
                                                        "the continuum"),
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
        ("model", "'auto' | 'point' | 'gauss' | 'continuum' | arcsec",
         "the source of the lines: Gaussian if the data prefer it | point | Gaussian | the one "
         "measured on the continuum | Gaussian of this FWHM"),
        ("continuum_model", "'auto' | 'point' | 'gauss' | 'line' | arcsec",
         "the source of the continuum: its own Gaussian if the data prefer it, else the size "
         "of the line, else a point | point | Gaussian | the one of the line | this FWHM"),
        ("model_prior", "(arcsec, dex) | None", "soft tie of the size of the Gaussian"),
        ("theta_range", "(min, max)", "arcsec, allowed size of the Gaussian (lines and "
                                      "continuum); min > 0 leaves the point source out"),
        ("flux_measure", "'auto' | 'gauss' | 'area'", "flux quoted (both are given): Gaussian "
                                                      "unless not Gaussian | Gaussian | sum over "
                                                      "the window"),
        ("shape_nsigma", "float", "threshold of that test: area and Gaussian differ by more "
                                  "than this many sigma"),
        ("window", "W | (lo, hi) | list | None", "window of the area flux; None: the line "
                                                 "windows"),
        ("measure", "None | 'line' | 'continuum'", "measure only the lines or only the "
                                                   "continuum; None: both"),
        ("fit_span", "km/s | None", "channels farther than this from every line are not "
                                    "fitted; None: every channel"),
        ("min_sources_per_annulus", "int", "annuli with fewer sources are left out"),
        ("n_bootstrap", "int", "resamplings of the sources, for the errors (0: none)"),
        ("n_null", "int", "searches on noise, for the significance (0: Bonferroni bound)"),
        ("seed", "int", "seed of the random numbers of bootstrap and null searches: the same "
                        "seed gives the same numbers"),
        ("figures_draw", "True | False", "draw the figures (profile, uv, noise, continuum)"),
        ("figures_unit", "str", "unit of the data, for the labels"),
        ("(advanced)", "keywords", "noise, channel_correlation, n_null_phase, "
                                   "residual_nsigma, H0, Om0"),
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
    continuum : vista.extract.ContinuumResult or None
        The continuum, when it was subtracted and measured.
    stack : dict or None
        Summary of the stacking (when it ran).
    files : dict
        The files of the run, by kind.

    ``plot()`` draws the figures of the measurement (in a notebook, right
    away); :func:`vista.compare` draws several results side by side.
    """

    line: object = None
    continuum: object = None
    stack: Optional[dict] = None
    files: dict = field(default_factory=dict)
    unit: Optional[str] = None

    def plot(self, what=None, unit=None, rebin=1, v_limits=None, save=None, show=True):
        """The figures of the measurement, the same as the files of the run.

        Parameters
        ----------
        what : str or list, optional
            ``"line"``: the stacked spectrum of the line, as it was measured
            (after the continuum subtraction, or above the continuum fitted
            with it), with the two fluxes and the residuals;
            ``"line_uv"``: the line flux against baseline length, with the
            Gaussian and point source models;
            ``"continuum"``, ``"continuum_uv"``: the same for the continuum,
            when it was subtracted and measured on its own;
            ``"noise"``: the noise checks and the null searches;
            ``"image"``: the dirty image of the stack over the window of the
            first line (see :func:`vista.dirty_image`).
            A list draws several.  Default: the line and, when there is one,
            the continuum, spectrum and baselines of each.
        unit : str
            Unit of the data (default: ``figures_unit`` of the run).
        rebin : int
            Channels averaged in the spectra.
        save : str
            Also write the figure(s) here: a path, or a prefix for several
            (``<save>_<what>.png``).
        show : bool
            In a notebook the figures are drawn and nothing is returned;
            ``show=False`` returns them (a list for several).
        """
        from .extract import plots_line

        if self.line is None and self.continuum is None:
            raise RuntimeError("nothing measured: run() with v.extract(...) set")
        if what is None:
            what = ((["line", "line_uv"] if self.line is not None else [])
                    + ((["continuum"] if self.line is not None else []) + ["continuum_uv"]
                       if self.continuum is not None else []))
        kinds = [what] if isinstance(what, str) else list(what)
        unit = unit if unit is not None else (self.unit or "Jy")
        out = []
        for kind in kinds:
            if kind.startswith("continuum") and self.continuum is None:
                raise RuntimeError("no continuum measured on its own (it is, with "
                                   "stack(..., continuum='subtract'))")
            if kind != "continuum_uv" and self.line is None:
                raise RuntimeError(f"plot({kind!r}): no line was measured; the continuum "
                                   f"alone has plot('continuum_uv')")
            if kind in ("line", "continuum"):
                fig = plots_line.plot_profile(self.line, unit=unit, v_limits=v_limits,
                                              rebin=rebin, continuum=self.continuum, show=kind)
            elif kind in ("line_uv", "continuum_uv"):
                fig = plots_line.plot_uv(self.line, unit=unit, continuum=self.continuum,
                                         show=kind[:-3])
            elif kind == "noise":
                fig = plots_line.plot_noise(self.line, unit=unit)
            elif kind == "image":
                fig = self._image(unit)
            else:
                raise ValueError(f"plot: 'line', 'line_uv', 'continuum', 'continuum_uv', "
                                 f"'noise' or 'image', not {kind!r}")
            path = (None if not save else save if len(kinds) == 1
                    else f"{os.path.splitext(save)[0]}_{kind}.png")
            out.append(plots_line.display(fig, path, show))
        if show and all(f is None for f in out):
            return None
        return out[0] if len(out) == 1 else out

    def _sizes(self):
        """The size of the source measured on the line and on the continuum,
        each on its own, and how well they agree."""
        out = {}
        if self.line is not None:
            sm = self.line.summary
            st = sm.get("size_own") or sm.get("size_test") or {}
            own = sm.get("size_from") != "continuum" and (sm["line"].get("theta_fwhm_arcsec") or 0)
            if own:
                out["line"] = (float(sm["line"]["theta_fwhm_arcsec"]),
                               sm["line"].get("theta_error_arcsec", sm.get("theta_error_arcsec")))
            elif st.get("theta_gauss_arcsec"):
                out["line"] = (float(st["theta_gauss_arcsec"]), None)
        if self.continuum is not None:
            c = self.continuum.summary
            ct = c.get("size_test") or {}
            # only of a continuum that is detected: otherwise its size is noise
            if ct.get("theta_gauss_arcsec") and c["flux_total"] > 2 * c["flux_total_error"]:
                out["continuum"] = (float(ct["theta_gauss_arcsec"]),
                                    ct.get("theta_gauss_error_arcsec"))
        if len(out) == 2 and all(v[1] for v in out.values()):
            (a, ea), (b, eb) = out["line"], out["continuum"]
            out["difference_sigma"] = float(abs(a - b) / np.hypot(ea, eb))
        return out

    def __getitem__(self, key):
        """``result["flux"]``: one of the quantities of :meth:`as_dict`."""
        return self.as_dict()[key]

    def __contains__(self, key):
        return key in self.as_dict()

    def keys(self):
        """The quantities that ``result[...]`` gives."""
        return self.as_dict().keys()

    def as_dict(self):
        """Everything measured, in one dictionary; every quantity with its
        ``value``, ``error`` and ``unit``.  ``result[key]`` reads one of
        them::

            result["flux"]            # the line flux quoted, and how it was measured
            result["flux_fit"]        # area of the Gaussian fit
            result["flux_window"]     # area under the spectrum in the window
            result["centre"], result["fwhm"], result["peak"]
            result["snr"]             # S/N of the line
            result["significance"]    # in sigma, after the search over centre and width
            result["size"]            # FWHM of the source used for the line (0: a point)
            result["model"]           # source model chosen, with its probability
            result["window"]          # km/s
            result["noise"]           # per channel
            result["continuum"]       # flux density, size, model, how it was treated
            result["sizes"]           # line and continuum, each on its own, and their agreement
            result["summary"]         # all of it in words (the same as print(result))

        With several lines these are of the first; ``result["lines"]`` has the
        same for every line.  Without a line (the continuum only) the keys
        of the line are missing.
        """
        if self.line is None and self.continuum is None:
            raise RuntimeError("nothing measured: run() with v.extract(...) set")
        u = self.unit or "data units"

        def q(value, error=None, unit=u, **more):
            out = {"value": None if value is None else float(value), "unit": unit}
            if error is not None:
                out["error"] = float(error)
            out.update(more)
            return out

        def model_of(test, theta, source=None):
            out = {"chosen": "gauss" if theta else "point", "size_from": source or "own fit",
                   "probability": None, "delta_chi2": None, "delta_chi2_needed": None}
            if test and test.get("delta_chi2_point_minus_gauss") is not None:
                d = test["delta_chi2_point_minus_gauss"]
                p_gauss = test.get("weight_gauss")
                if p_gauss is None:
                    from .extract.measure import akaike_weight
                    p_gauss = akaike_weight(d)
                out.update(probability=float(p_gauss if theta else 1 - p_gauss),
                           delta_chi2=float(d), delta_chi2_needed=float(test["threshold"]))
            return out

        res = {}
        if self.line is not None:
            sm = self.line.summary
            lines = []
            for k, b in enumerate(sm["lines"]):
                d, sc = b["detection"], b.get("shape_check", {})
                area = b.get("flux_method") == "area"
                theta = b.get("theta_fwhm_arcsec", sm.get("theta_fwhm_arcsec")) or 0.0
                lines.append({
                    "rest_freq_ghz": b["rest_freq_ghz"],
                    "flux": q(b["flux"], b["flux_error"], f"{u} km/s",
                              method="window" if area else "fit",
                              reason=sc.get("reason", "asked")),
                    "flux_fit": q(b["flux_gauss"], b["flux_gauss_error"], f"{u} km/s"),
                    "flux_window": q(b["flux_area"], b["flux_area_error"], f"{u} km/s"),
                    "centre": q(b["v0_kms"], b.get("v0_error_kms"), "km/s"),
                    "fwhm": q(b["fwhm_kms"], b.get("fwhm_error_kms"), "km/s"),
                    "peak": q(b["peak"], b.get("peak_error"), u),
                    "snr": float(d["snr"]),
                    "significance": (None if d.get("fixed_shape")
                                     else float(d["significance_sigma"])),
                    "size": q(theta, b.get("theta_error_arcsec", sm.get("theta_error_arcsec")),
                              "arcsec"),
                    "model": model_of(b.get("size_test")
                                      or (sm.get("size_test") if k == 0 else None), theta,
                                      "continuum" if sm.get("size_from") == "continuum" else None),
                    "window": {"low": float(b["window_kms"][0]),
                               "high": float(b["window_kms"][1]), "unit": "km/s"},
                    "chi2_reduced": b.get("chi2_reduced"),
                })
            res.update(lines[0])
            res["lines"] = lines
            res["noise"] = q(float(np.nanmedian(self.line.profile_error)), unit=u,
                             per="channel", channel_width_kms=sm.get("channel_width_kms"))
        else:
            sm = self.continuum.summary
        cf = sm.get("continuum_fit") if self.line is not None else None
        rc = (sm.get("continuum_residual") or {}) if self.line is not None else {}
        if cf:
            res["continuum"] = q(cf["flux_density"], cf["flux_density_error"], u,
                                 treatment="fitted with the line")
        elif self.continuum is not None:
            c = self.continuum.summary
            theta = c.get("theta_fwhm_arcsec") or 0.0
            ct = c.get("size_test") or {}
            res["continuum"] = q(c["flux_total"], c["flux_total_error"], u,
                                 treatment=("measured on every channel (no line)"
                                            if c.get("channels") == "every channel"
                                            else "subtracted, measured on its own"),
                                 snr=float(c["flux_total"] / c["flux_total_error"]),
                                 size=q(theta, (c.get("theta_error_arcsec") or 0) > 1e-6
                                        and c["theta_error_arcsec"] or None, "arcsec"),
                                 model=model_of(ct, theta, {"line": "line", "fixed": "given"}
                                                .get(ct.get("adopted"))))
        else:
            res["continuum"] = q(None, unit=u, treatment="subtracted" if
                                 rc.get("continuum_subtracted") else "none")
        if rc:
            res["continuum"]["left_outside_window"] = q(rc["offset"], rc["offset_error"], u,
                                                        detected=bool(rc["detected"]))
        sizes = self._sizes()
        res["sizes"] = {k: (q(v[0], v[1], "arcsec") if isinstance(v, tuple) else v)
                        for k, v in sizes.items()}
        res["n_sources"] = sm["n_sources"]
        res["weighting"] = sm["weighting_scheme"]
        res["z_ref"] = sm.get("z_ref")
        res["flux_ref"] = sm.get("flux_ref")
        res["summary"] = self.report()
        return res

    def _image(self, unit):
        """Dirty image of the stack over the window of the first line."""
        from . import quicklook
        b = self.line.summary["line"]
        img = quicklook.dirty_image(self.files["stack"], line_freq=b["rest_freq_ghz"],
                                    velocity=b["window_kms"])
        ax = quicklook.plot_image(img, unit=unit, title="Stacked line, image")
        return ax.figure

    def report(self):
        """What was measured and the choices made, in words: for every line
        the detection, the flux quoted and why, the Gaussian, the source
        model and why; the continuum; the references.  ``print(result)``."""
        u = self.unit or "data units"
        # flux densities of data in Jy are written in mJy
        fd, ud = (1e3, "mJy") if u.lower() == "jy" else (1.0, u)
        rows = []
        if self.stack is not None:
            s = self.stack
            rows.append(f"Stack      {s['output']}: {s['n_rows']:,} rows, {s['nchan']} channels, "
                        f"{s['seconds']:.0f} s")
        if self.line is not None:
            sm = self.line.summary
            several = len(sm["lines"]) > 1
            for k, b in enumerate(sm["lines"]):
                d, sc = b["detection"], b.get("shape_check", {})
                rows.append(f"{f'Line {k + 1}' if several else 'Line':10s} "
                            f"{b['rest_freq_ghz']:.6g} GHz, {sm['n_sources']} sources: "
                            + (f"S/N {d['snr']:.1f} with the centre and the width fixed"
                               if d.get("fixed_shape") else
                               f"detected at {d['significance_sigma']:.1f} sigma (S/N "
                               f"{d['snr']:.1f}, searched over centre and width)"))
                area = b.get("flux_method") == "area"
                rows.append(f"  flux     {b['flux']:.4g} ± {b['flux_error']:.2g} {u} km/s, "
                            + ("the area under the spectrum in the window"
                               if area else "the area of the Gaussian fit"))
                rows.append(f"           because {sc.get('reason', 'asked')}")
                rows.append(f"           Gaussian fit {b['flux_gauss']:.4g} ± "
                            f"{b['flux_gauss_error']:.2g}; area in the window "
                            f"({b['window_kms'][0]:+.0f} to {b['window_kms'][1]:+.0f} km/s) "
                            f"{b['flux_area']:.4g} ± {b['flux_area_error']:.2g}")
                rows.append(f"  Gaussian centre {b['v0_kms']:+.0f} ± {b['v0_error_kms']:.0f} km/s, "
                            f"FWHM {b['fwhm_kms']:.0f} ± {b['fwhm_error_kms']:.0f} km/s, "
                            f"peak {b['peak'] * fd:.3g} ± "
                            f"{b.get('peak_error', float('nan')) * fd:.2g} {ud}")
                first_of_group = any(g and g[0] == k + 1 for g in sm.get("line_groups") or [[1]])
                if first_of_group:
                    st = b.get("size_test") or (sm.get("size_test") if k == 0 else None)
                    theta = b.get("theta_fwhm_arcsec", sm.get("theta_fwhm_arcsec"))
                    err = b.get("theta_error_arcsec", sm.get("theta_error_arcsec"))
                    model = b.get("size_model", sm["line"].get("size_model"))
                    shape = (f"Gaussian, FWHM {theta:.2f} ± {err:.2f} arcsec" if theta
                             else "point source")
                    own = sm.get("size_own")
                    if sm.get("size_from") == "continuum" and own:
                        why = (f"its own Gaussian improves the fit by delta chi2 = "
                               f"{own['delta_chi2_point_minus_gauss']:.1f} only (at least "
                               f"{own['threshold']:.0f} needed to call it extended), and the "
                               f"continuum is extended: the size of the continuum is used")
                        shape = f"Gaussian, FWHM {theta:.2f} arcsec"
                    elif sm.get("size_from") == "continuum":
                        why = "the source measured on the continuum was asked"
                    elif st:
                        dchi2, need = st["delta_chi2_point_minus_gauss"], st["threshold"]
                        why = (f"it fits better than a point source: delta chi2 = {dchi2:.1f} "
                               f"(at least {need:.0f} needed to call it extended)"
                               if st["adopted"] == "gauss" else
                               f"a Gaussian source improves the fit by delta chi2 = {dchi2:.1f} "
                               f"only (at least {need:.0f} needed to call it extended)")
                    else:
                        why = {"fixed": "the size was fixed",
                               "gauss (prior)": "a size prior was given"}.get(
                                   model, "asked")
                    rows.append(f"  source   {shape}")
                    rows.append(f"           because {why}")
        sm = (self.line or self.continuum).summary if (self.line or self.continuum) else None
        cf = sm.get("continuum_fit") if self.line is not None else None
        rc = (sm.get("continuum_residual") or {}) if self.line is not None else {}
        if cf:
            rows.append(f"Continuum  fitted with the line: {cf['flux_density'] * fd:.3g} ± "
                        f"{cf['flux_density_error'] * fd:.2g} {ud}")
        elif self.continuum is not None:
            c = self.continuum.summary
            value = f"{c['flux_total'] * fd:.3g} ± {c['flux_total_error'] * fd:.2g} {ud}"
            found = c["flux_total"] > 2 * c["flux_total_error"]
            theta = c.get("theta_fwhm_arcsec") or 0.0
            err = c.get("theta_error_arcsec") or 0.0
            ct = c.get("size_test") or {}
            how = ct.get("adopted")
            shape = ("point source" if not theta else
                     f"Gaussian, FWHM {theta:.2f} arcsec" if how in ("line", "fixed") else
                     f"Gaussian, FWHM {theta:.2f} ± {err:.2f} arcsec")
            head = (f"{c['n_sources']} sources, every channel: "
                    if c.get("channels") == "every channel"
                    else "subtracted while stacking and measured on its own: " if self.line
                    else f"{c['n_sources']} sources, the channels outside the line windows: ")
            rows.append("Continuum  " + head + (f"{value} (S/N "
                        f"{c['flux_total'] / c['flux_total_error']:.1f}), {shape}"
                        if found else f"none detected ({value})"))
            if ct and found:
                d, need = ct["delta_chi2_point_minus_gauss"], ct["threshold"]
                own = (f"its own Gaussian improves the fit by delta chi2 = {d:.1f} "
                       f"(at least {need:.0f} needed to call it extended)")
                rows.append("           because " + {
                    "gauss": own if d > need else "a Gaussian was asked",
                    "line": ("the source measured on the line is used: " + own
                             if d <= need else "the source measured on the line was asked"),
                    "point": ("a point source was asked" if ct.get("asked") == "point" else
                              own + (": point" if self.line is None else
                                     ", and the line is a point too")),
                    "fixed": "the size was given"}[how])
        elif rc.get("continuum_subtracted"):
            rows.append("Continuum  subtracted while stacking")
        elif self.line is not None:
            rows.append("Continuum  not subtracted, not fitted")
        if rc:
            rows.append(f"           outside the line window: {rc['offset'] * fd:.2g} ± "
                        f"{rc['offset_error'] * fd:.2g} {ud} "
                        f"({rc['offset_nsigma']:+.1f} sigma): "
                        + ("STILL THERE, the line flux includes it" if rc["detected"]
                           else "nothing left"))
        sizes = self._sizes() if sm else {}
        if "line" in sizes and "continuum" in sizes:
            fmt = lambda v: f"{v[0]:.2f}" + (f" ± {v[1]:.2f}" if v[1] else "") + " arcsec"
            agree = sizes.get("difference_sigma")
            rows.append(f"Sizes      each on its own: line {fmt(sizes['line'])}, continuum "
                        f"{fmt(sizes['continuum'])}"
                        + ("" if agree is None else
                           f": {'compatible' if agree < 2 else 'marginally different' if agree < 3 else 'DIFFERENT'}"
                           f" ({agree:.1f} sigma apart)"))
        if sm:
            ref = [f"{sm['weighting_scheme']} weighting"]
            if sm.get("z_ref") is not None:
                ref.append(f"fluxes at z_ref = {sm['z_ref']:.4f} "
                           f"({'given' if sm.get('z_ref_given') else 'median of the sample'})")
            else:
                ref.append("no redshift rescaling")
            if sm.get("flux_ref") is not None:
                ref.append(f"normalised to flux_ref = {sm['flux_ref']:.4g} "
                           f"({'given' if sm.get('flux_ref_given') else 'median'})")
            rows.append("Reference  " + ", ".join(ref))
        return "\n".join(rows or ["nothing run"])

    def __repr__(self):
        return self.report()


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
        self._reported = None                 # z_ref and flux_ref, as last said
        self.stack_options = None
        self.extract_options = None

    # ---- the stacking -------------------------------------------------------
    def stack(self, *, line_freq=None, line_window=600.0, velocity_range=None, nchan=None,
              channel_width=None, frame="LSRK", data_column="auto", continuum="subtract",
              continuum_order=0, exclude_between=False):
        """What makes the stack.  Every option is given by name:
        ``v.stack(line_freq=345.796, line_window=500)``.

        Parameters
        ----------
        line_freq : float, str, a list of them, or None
            Rest frequency of the line: a number in GHz, or a string with its
            unit (``"345.796GHz"``, ``"345796 MHz"``; Hz, kHz, MHz, GHz,
            THz).  Several lines: a list; the first is the one the stack is
            centred on, the velocities refer to it.  None (default): no
            line, the stack of the continuum: every channel of every
            dataset is kept, whatever its rest frequency, and
            :meth:`extract` measures the continuum (the options of the lines
            and of the continuum subtraction are not used).
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

        no_line = line_freq is None or (np.ndim(line_freq) > 0 and len(line_freq) == 0)
        lines = None if no_line else Lines(line_freq, line_window, exclude_between=_check_bool(
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
            for k in range(0 if no_line else lines.n):   # every window inside the band
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
            frame=fr, data_column=data_column,
            continuum="only" if no_line else continuum, continuum_order=int(order))
        if self.extract_options is not None:
            self._check_extract(self.extract_options)
        return self

    # ---- the measurement ----------------------------------------------------
    def extract(self, *, annuli=(3.0, 3000.0, 18), weighting_scheme="natural", z_norm=True,
                z_ref=None, pb_corr=True, flux_norm=False, flux_ref=None, v0_range=None,
                fwhm_range=None, model="auto", continuum_model="auto", model_prior=None,
                theta_range=(0.0, 15.0), flux_measure="auto", shape_nsigma=2.0, window=None,
                fit_span=1500.0, measure=None, min_sources_per_annulus=2, n_bootstrap=500,
                n_null=1000, seed=42, figures_draw=True, figures_unit="Jy", **advanced):
        """What is measured on the stack.  Without this call, :meth:`run`
        only writes the stack.  Every option is given by name.

        Every line is fitted on every channel and annulus of baseline length,
        its centre and width searched within their ranges.  Lines whose
        windows (``line_window`` of :meth:`stack`) overlap, directly or
        through another line, are fitted together, each with its own flux,
        centre and width and sharing the size, so the fit separates them;
        every other line is measured on its own (its own size and model, on
        the channels within ``fit_span`` of it and outside the windows of
        the other lines).

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
        model : 'auto', 'point', 'gauss', 'continuum' or a number, default 'auto'
            The source of the lines: a Gaussian only if the data prefer it
            (Δχ² ≥ 9 over a point source; the Δχ² is reported), always a
            point, always a Gaussian, a Gaussian of this FWHM (arcsec), or
            (``'continuum'``) the model and the size measured on the
            continuum: for a weak line on a strong continuum.
        continuum_model : 'auto', 'point', 'gauss', 'line' or a number, default 'auto'
            The source of the continuum measured on its own, as ``model``.
            ``'line'``: the model and the size measured on the line, for a
            weak continuum under a strong line.  ``'auto'``: its own
            Gaussian if the data prefer it (the same Δχ² ≥ 9); otherwise the
            size of the line, if the line is extended (the same galaxies:
            the continuum cannot be a point); otherwise a point.  The two
            sizes and how well they agree are always reported.
        model_prior : (theta_arcsec, sigma_dex) or None
            Soft tie of the size of the Gaussian (e.g. to the continuum size).
        theta_range : (min, max), default (0, 15)
            Allowed size (FWHM) of the Gaussian source, arcsec, for the lines
            and the continuum.  A minimum above 0 leaves the point source
            out (``model='auto'`` fits the Gaussian only).
        flux_measure : {'auto', 'gauss', 'area'}, default 'auto'
            Which flux is quoted for every line (both are always given and
            drawn): the integral of the Gaussian, the sum of the spectrum
            over the window, or the Gaussian unless the line is not Gaussian
            (the two differ by more than ``shape_nsigma``).
        shape_nsigma : float, default 2
            Threshold of that test, in sigma of the difference of the two.
        window : number, (lo, hi), a list of them, or None
            Window of the area flux of every line; None: the line windows.
            It can be changed and the measurement repeated without stacking.
        measure : None, 'line' or 'continuum'
            Measure only the lines, or only the continuum (e.g. to repeat
            one of the two with another model).  None: everything the stack
            has.
        fit_span : float or None, default 1500
            Channels farther than this (km/s) from every line are not
            fitted, so that a wide band does not slow the fit down or bring
            in a continuum that is no longer flat; None: every channel.
        min_sources_per_annulus : int, default 2
            Annuli with fewer sources are left out, so that no point is set
            by a single object (one source alone: 1).
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
            ``noise`` ('scatter': the noise of every cell from the scatter of
            its visibilities; 'weights': from the weights alone),
            ``channel_correlation`` (measure and use the correlation between
            adjacent channels), ``n_null_phase`` (stacks with a random phase
            per source, for a significance that does not assume Gaussian
            noise), ``residual_nsigma`` (threshold of the residual-continuum
            check); for the redshift rescaling ``H0``, ``Om0``.
        """
        from .extract.config import (BinningConfig, BootstrapConfig, LineFitConfig,
                                     WeightingConfig)

        line_keys = {"noise", "channel_correlation", "n_null_phase", "residual_nsigma"}
        weight_keys = {"H0", "Om0"}
        bad = sorted(set(advanced) - (line_keys | weight_keys))
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
        elif model == "continuum":
            size = "auto"                      # set from the continuum, when it is measured
        elif model not in ("auto", "point", "gauss"):
            raise ValueError(f"model must be 'auto', 'point', 'gauss', 'continuum' or a size "
                             f"in arcsec, not {model!r}")
        if isinstance(continuum_model, str):
            if continuum_model not in ("auto", "point", "gauss", "line"):
                raise ValueError(f"continuum_model must be 'auto', 'point', 'gauss', 'line' "
                                 f"or a size in arcsec, not {continuum_model!r}")
        elif float(continuum_model) < 0:
            raise ValueError(f"continuum_model: a fixed size must be >= 0 arcsec, "
                             f"not {continuum_model}")
        if measure not in (None, "line", "continuum"):
            raise ValueError(f"measure must be None, 'line' or 'continuum', not {measure!r}")
        if model == "continuum" and continuum_model == "line":
            raise ValueError("model='continuum' and continuum_model='line': one of the two "
                             "must measure its own size")
        if ((model == "continuum" and measure == "line")
                or (continuum_model == "line" and measure == "continuum")):
            raise ValueError("to take the size of one from the other both are measured: "
                             "leave measure=None")
        if model_prior is not None:
            model_prior = tuple(float(x) for x in model_prior)
            if len(model_prior) != 2 or model_prior[0] <= 0 or model_prior[1] <= 0:
                raise ValueError("model_prior: (theta_arcsec > 0, sigma_dex > 0)")
        if flux_measure not in ("auto", "gauss", "area"):
            raise ValueError(f"flux_measure must be 'auto', 'gauss' or 'area', "
                             f"not {flux_measure!r}")
        try:
            th_min, th_max = (float(x) for x in theta_range)
        except (TypeError, ValueError):
            raise ValueError(f"theta_range: (min, max) in arcsec, not {theta_range!r}") from None
        if theta_fixed and not th_min <= theta_fixed <= th_max:
            raise ValueError(f"model={theta_fixed:g}: outside theta_range ({th_min:g}, "
                             f"{th_max:g})")
        if fit_span is not None:
            try:
                fit_span = float(fit_span)
            except (TypeError, ValueError):
                raise ValueError(f"fit_span: km/s or None, not {fit_span!r}") from None
        n_boot = int(n_bootstrap)
        if n_boot < 0 or int(n_null) < 0:
            raise ValueError("n_bootstrap and n_null must be >= 0")
        line_cfg = LineFitConfig(
            v0_range_kms=v0_range, fwhm_range_kms=fwhm_range, size=size,
            theta_fixed_arcsec=theta_fixed if theta_fixed else None,
            theta_min_arcsec=th_min, theta_max_arcsec=th_max,
            theta_prior_arcsec=model_prior, flux=flux_measure,
            shape_nsigma=float(shape_nsigma), window_kms=window,
            fit_span_kms=fit_span,
            min_sources_per_bin=_check_min_sources(min_sources_per_annulus),
            n_null=int(n_null),
            **{k: v for k, v in advanced.items() if k in line_keys})
        opts = dict(binning=binning, weighting=weighting, line=line_cfg,
                    bootstrap=BootstrapConfig(n_realisations=n_boot, seed=int(seed)),
                    draw=figures_draw, unit=str(figures_unit), measure=measure,
                    line_model=model, continuum_model=continuum_model,
                    values=dict(annuli=(float(b_min), float(b_max), int(n_bins)),
                                weighting_scheme=weighting_scheme, z_norm=z_norm, z_ref=z_ref,
                                pb_corr=pb_corr, flux_norm=flux_norm, flux_ref=flux_ref,
                                v0_range=v0_range, fwhm_range=fwhm_range, model=model,
                                continuum_model=continuum_model, measure=measure,
                                model_prior=model_prior, theta_range=(th_min, th_max),
                                flux_measure=flux_measure, shape_nsigma=float(shape_nsigma),
                                window=window, fit_span=fit_span,
                                min_sources_per_annulus=int(min_sources_per_annulus),
                                n_bootstrap=n_boot,
                                n_null=int(n_null), seed=int(seed), figures_draw=figures_draw,
                                figures_unit=str(figures_unit)))
        if self.stack_options is not None:
            self._check_extract(opts)
        self.extract_options = opts
        return self

    def _check_extract(self, opts):
        """The options of the measurement that depend on the number of lines."""
        from .extract.config import per_line, ranges_per_line
        if self.lines is None:                 # the continuum only: no line option to check
            if opts["measure"] == "line" or opts["line_model"] == "continuum" \
                    or opts["continuum_model"] == "line":
                raise ValueError("this stack has no line (v.stack() without line_freq): "
                                 "only the continuum can be measured")
            return
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
            if ln is None:
                return {"line_freq": None, "velocity_range": o["velocity_range_kms"],
                        "nchan": o["nchan_out"], "channel_width": o["channel_width_hz"],
                        "frame": o["frame"], "data_column": o["data_column"],
                        "continuum": "(the continuum only)"}
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
            raise RuntimeError("call v.stack(...) first: with line_freq for the lines, "
                               "without it for the continuum only (also to measure an "
                               "existing stack)")
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
        out = RunResult(files=dict(self._files(tag)),
                        unit=ex["unit"] if ex is not None else None)
        d = os.path.dirname(os.path.abspath(self.output_name))
        os.makedirs(d, exist_ok=True)
        if "stack" in steps:
            out.stack = self._stack()
        if "continuum" in steps:
            self._continuum_again()
        if "measure" in steps:
            out.line, out.continuum = self._measure(tag)
        subtract = self.stack_options["continuum"] in ("subtract", "only")
        if out.continuum is None:
            out.files.pop("continuum_results", None)
        if out.line is None and "measure" in steps or self.lines is None:
            for k in ("line_statistics", "results"):
                out.files.pop(k, None)
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
        if self.lines is None:
            # the continuum only: every channel of every dataset, around the
            # rest frequency at the middle of the sample (each dataset keeps
            # its own band); its statistics are those of DATA
            centre = self._continuum_frequency()
            _say(f"No line: the stack of the continuum, around {centre / 1e9:.2f} GHz "
                 f"(rest frame)", bool(self.verbose))
            out = Stacker(self.input_list, verbose=self.verbose).run(
                output=f["stack"], central_freq=centre, output_format=self.output_format,
                **o, **self.run_options)
            if ex is not None:
                self._compress()
            return out
        return Stacker(self.input_list, verbose=self.verbose).run(
            output=f["stack"], central_freq=self.lines.rest_freq_hz,
            output_format=self.output_format, line=self.lines,
            continuum=ContinuumConfig(order=order) if subtract else None,
            stats_line=f["line_statistics"] if ex is not None else None,
            stats_continuum=f["continuum_statistics"] if ex is not None and subtract else None,
            binning=ex["binning"] if ex is not None else None, **o, **self.run_options)

    def _continuum_frequency(self):
        """Rest frequency [Hz] at the middle of the bands of the sample: the
        centre of a stack without lines."""
        from .ms_input import read_ms_metadata

        centres = []
        for e in self._entries:
            meta = read_ms_metadata(e.ms, e.field, e.spws)
            freq = np.concatenate(meta.freq)
            centres.append(0.5 * (freq.min() + freq.max()) * (1.0 + e.z))
        return float(np.median(centres))

    def _compress(self):
        """The statistics again from the stack (annuli or continuum changed)."""
        from .extract.statistics import compress_visibilities

        f, ex = self.files, self.extract_options
        if self.lines is None:
            compress_visibilities(f["stack"], self.input_list, None, binning=ex["binning"],
                                  out_line=f["continuum_statistics"], line_column="DATA",
                                  resume=False, verbose=bool(self.verbose), one_channel=True)
            return
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

    def _stack_has(self, column):
        from .stackio import open_stack
        try:
            stack = open_stack(self.files["stack"])
            return column in stack.columns(stack.dd_ids()[0])
        except Exception:
            return False

    def _statistics_ready(self):
        """Compress the stack again when its statistics are missing, or were
        made with other annuli or from another column."""
        from .extract.statistics import SufficientStatistics

        f, ex = self.files, self.extract_options
        subtract = self.stack_options["continuum"] == "subtract"
        column = "CORRECTED_DATA" if subtract else "DATA"
        why = None
        main = f["continuum_statistics" if self.lines is None else "line_statistics"]
        if not os.path.isfile(main):
            why = "there are no statistics"
        else:
            st = SufficientStatistics.load(main)
            edges = np.asarray(ex["binning"].edges_lambda(), float)
            if (st.bin_edges_lambda.shape != edges.shape
                    or not np.allclose(st.bin_edges_lambda, edges, rtol=1e-9)):
                why = "the annuli changed"
            elif (st.meta or {}).get("line_column", column) != column:
                if subtract and not self._stack_has("CORRECTED_DATA"):
                    raise RuntimeError(
                        "the stack was made without the continuum subtraction: subtract "
                        "it first, run(only=['continuum', 'measure'])")
                why = ("the continuum subtraction is used again" if subtract
                       else "the continuum is no longer subtracted: the line from DATA")
        if why is None:
            return
        if not os.path.isdir(f["stack"]):
            raise FileNotFoundError(f"{f['stack']} not found: run the stacking first (run())")
        _say(f"Compressing the stack again ({why})", bool(self.verbose))
        self._compress()

    def _measure(self, tag=None):
        from dataclasses import replace

        from .extract.measure import measure_line
        from .extract.plots_line import plot_line_result

        self._statistics_ready()
        f, ex = self._files(tag), self.extract_options
        prefix = f["results"][:-len("_results.json")]
        verbose = bool(self.verbose)
        mode = self.stack_options["continuum"]
        want_line = self.lines is not None and ex["measure"] != "continuum"
        has_cont = (mode in ("subtract", "only") and ex["measure"] != "line"
                    and os.path.isfile(f["continuum_statistics"]))
        if has_cont and (self.n_entries < 2 or ex["bootstrap"].n_realisations < 2):
            _say("Continuum not measured: its errors come from the bootstrap over the "
                 "sources (at least 2 sources and n_bootstrap >= 2)", verbose)
            has_cont = False
        if not want_line and not has_cont:
            raise RuntimeError("nothing to measure: no line asked and no continuum measured "
                               "on its own (v.stack(..., continuum='subtract'))")

        def the_line(cfg):
            return measure_line(f["line_statistics"], self.input_list, self.lines,
                                ex["weighting"], replace(cfg, continuum=mode == "fit"),
                                ex["bootstrap"], output_prefix=prefix, verbose=verbose)

        def the_continuum(line_size):
            _say("Continuum, " + ("every channel:" if self.lines is None else
                                  "from the channels outside the line windows (MODEL_DATA):"),
                 verbose)
            return self._measure_continuum(f, ex, f"{prefix}_continuum", verbose, line_size)

        line = cont = None
        if want_line and ex["line_model"] == "continuum":
            # the continuum first: the line takes its model and its size
            if not has_cont:
                raise RuntimeError("model='continuum': no continuum measured on its own "
                                   "(v.stack(..., continuum='subtract'))")
            cont = the_continuum(None)
            theta = cont.summary.get("theta_fwhm_arcsec") or 0.0
            _say("Line with the source of the continuum: "
                 + (f"Gaussian, FWHM {theta:.2f} arcsec" if theta else "point"), verbose)
            line = the_line(replace(ex["line"], size="gauss" if theta else "point",
                                    theta_fixed_arcsec=theta or None))
            line.summary["size_from"] = "continuum"
        else:
            if want_line:
                line = the_line(ex["line"])
            if has_cont:
                cont = the_continuum(None if line is None
                                     else line.summary.get("theta_fwhm_arcsec") or 0.0)
            ct = cont.summary["size_test"] if cont is not None else {}
            if (line is not None and ex["line_model"] == "auto"
                    and not line.summary.get("theta_fwhm_arcsec")
                    and ct.get("adopted") == "gauss" and cont.summary.get("theta_fwhm_arcsec")):
                # the line cannot tell its size, the continuum can: the same
                # galaxies, so the line is measured again with that size
                theta = cont.summary["theta_fwhm_arcsec"]
                _say(f"The line does not show its size, the continuum does: the line again "
                     f"with the size of the continuum, FWHM {theta:.2f} arcsec", verbose)
                own = line.summary.get("size_test")
                line = the_line(replace(ex["line"], size="gauss", theta_fixed_arcsec=theta))
                line.summary["size_from"] = "continuum"
                line.summary["size_own"] = own
        if line is not None and line.summary.get("size_from"):
            import json
            from .extract.measure import _jsonable
            with open(f"{prefix}_results.json", "w") as fh:        # with where the size is from
                json.dump(line.summary, fh, indent=2, default=_jsonable)
        self._reported_refs((line if line is not None else cont).summary)
        if ex["draw"]:
            plot_line_result(line, self.lines, prefix, unit=ex["unit"], verbose=verbose,
                             continuum=cont)
        return line, cont

    def _reported_refs(self, sm):
        """Say once which references the fluxes have (again when they change)."""
        refs = []
        if sm.get("z_ref") is not None:
            refs.append(f"z_ref = {sm['z_ref']:.4f} "
                        f"({'given' if sm.get('z_ref_given') else 'median of the sample'})")
        if sm.get("flux_ref") is not None:
            refs.append(f"flux_ref = {sm['flux_ref']:.4g} "
                        f"({'given' if sm.get('flux_ref_given') else 'median of the sample'})")
        text = "; ".join(refs) if refs else "no redshift rescaling, no flux normalisation"
        if text != self._reported:
            _notice(text)
            self._reported = text
        return True

    def _measure_continuum(self, f, ex, prefix, verbose, line_size=None):
        """The continuum on its own.  Its source model (``continuum_model``):
        ``'auto'``: its own Gaussian when it fits better than a point source
        by the same delta chi2 as for the line; otherwise the size of the
        line (``line_size``, arcsec; None: no line measured) when the line
        is extended; otherwise a point.  The Gaussian fitted freely is always
        kept in ``summary["size_test"]``, to compare the sizes."""
        import json

        from .extract.config import RESOLVED_DELTA_CHI2, Lines, UVFitConfig
        from .extract.linefit import uv_gaussian
        from .extract.stack import measure_continuum
        from .extract.statistics import SufficientStatistics

        cfg, asked = ex["line"], ex["continuum_model"]
        stats, lines = f["continuum_statistics"], self.lines
        if lines is None:
            # no line: the statistics have one channel, every dataset summed
            # over its own (the bands need not overlap in the rest frame)
            freq = float(SufficientStatistics.load(stats).freq_grid[0])
            lines = Lines(freq / 1e9, 1.0)

        def fit(model, theta=None, say=False):
            return measure_continuum(
                stats, self.input_list, lines, ex["weighting"],
                uvfit=UVFitConfig(model=model, theta_fixed_arcsec=theta,
                                  theta_min_arcsec=cfg.theta_min_arcsec,
                                  theta_max_arcsec=cfg.theta_max_arcsec,
                                  min_sources_per_bin=cfg.min_sources_per_bin),
                bootstrap=ex["bootstrap"], output_prefix=prefix, verbose=say)

        # the Gaussian fitted freely: its size, and how much better than a point it fits
        free = fit("gauss", say=verbose)
        c = free.summary
        ok = free.bin_ok & np.isfinite(free.flux) & (free.flux_error > 0)
        y, w = free.flux[ok], 1.0 / free.flux_error[ok] ** 2
        flat = float((w * y).sum() / w.sum())
        curve = c["flux_total"] * uv_gaussian(free.baseline_klambda[ok] * 1e3,
                                              c["theta_fwhm_arcsec"] or 0.0)
        dchi2 = float((w * (y - flat) ** 2).sum() - (w * (y - curve) ** 2).sum())
        resolved = dchi2 > RESOLVED_DELTA_CHI2
        test = {"delta_chi2_point_minus_gauss": dchi2, "threshold": RESOLVED_DELTA_CHI2,
                "flux_gauss": c["flux_total"], "flux_gauss_error": c["flux_total_error"],
                "theta_gauss_arcsec": c["theta_fwhm_arcsec"],
                "theta_gauss_error_arcsec": c["theta_error_arcsec"],
                "flux_point": c.get("flux_point_source")}
        if not isinstance(asked, str):                       # a size given
            cont, how = (fit("gauss", float(asked)) if float(asked) > 0 else fit("point")), "fixed"
        elif asked == "gauss" or (asked == "auto" and resolved):
            cont, how = free, "gauss"
        elif asked == "line" or (asked == "auto" and line_size):
            if line_size is None:
                raise RuntimeError("continuum_model='line': no line measured")
            cont = fit("gauss", line_size) if line_size > 0 else fit("point")
            how = "line"
        else:
            cont, how = fit("point"), "point"
        if how in ("line", "point", "fixed"):
            theta = cont.summary.get("theta_fwhm_arcsec") or 0.0
            _say("  " + ("" if asked != "auto" else
                         f"its own Gaussian improves the fit by delta chi2 = {dchi2:.1f} only "
                         f"(at least {RESOLVED_DELTA_CHI2:.0f} needed): ")
                 + {"line": "the source of the line, ", "point": "", "fixed": "as asked, "}[how]
                 + (f"Gaussian, FWHM {theta:.2f} arcsec" if theta else "point source")
                 + f": S = {cont.summary['flux_total']:.4g} +- "
                   f"{cont.summary['flux_total_error']:.2g}", verbose)
        test["adopted"] = how
        test["asked"] = asked if isinstance(asked, str) else float(asked)
        cont.summary["size_test"] = test
        cont.summary["channels"] = ("every channel" if self.lines is None
                                    else "outside the line windows")
        with open(f"{prefix}_results.json", "w") as fh:
            json.dump(cont.summary, fh, indent=2)
        return cont

    def __repr__(self):
        lines = "no lines yet (v.stack)" if self.lines is None else str(self.lines)
        return (f"ViSta({self.input_list!r}: {self.n_entries} datasets, {lines}, "
                f"→ {self.files['stack']}"
                + (", measured" if self.extract_options is not None else ", stack only") + ")")


__all__ = ["ViSta", "RunResult"]
