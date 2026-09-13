"""
vista.extract.plots
===================
Diagnostic figures for one extraction.  Which ones you get depends on how the
continuum was handled, because that is what decides whether line and continuum
are two measurements or one.

**Continuum subtracted before the extraction.**  Line and continuum come from
two independent fits on two different columns, so they get two independent
sets of figures, one per run, and nothing is ever drawn on a shared axis:
``plot_uv_profile`` with ``component='line'`` or ``'continuum'``, plus
``plot_spectrum`` for the line.

**Continuum kept and fitted jointly.**  The two came out of the same fit, on
the same data, so they belong in one figure: ``plot_joint`` draws the uv
profiles of both and the spectrum with the line sitting on the fitted
continuum level.

**No continuum at all.**  Only the line, over the velocity range of interest:
``plot_uv_profile`` and ``plot_spectrum``.

``plot_all`` picks the right set automatically.  The legends carry the fitted
numbers and nothing else; everything else worth knowing is in the JSON.

``matplotlib`` is imported lazily, so the rest of the package works without it.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .config import LineConfig
from .stack import ExtractionResult
from .uvfit import gaussian_visibility, two_gaussian_visibility

_LINE_COLOUR = "C3"
_SECOND_COLOUR = "C2"
_CONTINUUM_COLOUR = "C0"
_MODEL_COLOUR = "k"


def _pyplot(save: bool):
    import matplotlib
    if save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _unit(unit: Optional[str]) -> str:
    return unit or "data units"


def _line_flux_label(result: ExtractionResult, unit: Optional[str]) -> str:
    if result.summary.get("flux_method") == "continuum":
        return f"flux density [{_unit(unit)}]"
    return f"line flux [{_unit(unit)} km/s]"


def _fit_label(flux, flux_error, theta, theta_error, model="gauss") -> str:
    """The one string that goes in the legend: the fitted numbers."""
    text = f"F = {flux:.3g} $\\pm$ {flux_error:.2g}"
    if model == "point":
        return text + ", unresolved"
    return (text + f"\n$\\theta_{{FWHM}}$ = {theta:.3f}\" "
                   f"$\\pm$ {theta_error:.3f}\"")


def _model_curve(result: ExtractionResult, b_klambda):
    """Fitted source model of the line, sampled finely from b = 0."""
    if result.fit.params is None or not np.isfinite(result.fit.flux):
        return None, None
    grid = np.linspace(0.0, np.nanmax(b_klambda) * 1e3, 400)
    if result.fit.model == "gauss2":
        values = two_gaussian_visibility(grid, *result.fit.params)
    elif result.fit.model == "point":
        values = np.full_like(grid, result.fit.flux)
    else:
        values = gaussian_visibility(grid, result.fit.params[0],
                                     result.fit.params[1])
    return grid / 1e3, values


def fit_summary_lines(result: ExtractionResult) -> List[str]:
    """The fitted numbers as plain text, for logs or captions."""
    s = result.summary
    out = [f"F = {s['flux_total']:.4g} +- {s['flux_total_error']:.3g}"]
    if s["model"] == "point":
        out.append("unresolved (point-source model)")
    else:
        out.append(f"theta_FWHM = {s['theta_fwhm_arcsec']:.3f}\" "
                   f"+- {s['theta_error_arcsec']:.3f}\"")
    if result.shape is not None:
        out.append(f"v0 = {result.shape.centroid_kms:+.0f} km/s, "
                   f"FWHM = {result.shape.fwhm_kms:.0f} km/s")
    out.append(f"chi2_r = {s['chi2_reduced']:.2f}, "
               f"{s['n_bins_in_fit']} bins, {s['n_sources']} sources")
    for tag, label in (("second_line", "second line"),
                       ("continuum", "continuum")):
        block = s.get(tag)
        if block:
            out.append(f"{label}: F = {block['flux_total']:.4g} "
                       f"+- {block['flux_total_error']:.3g}, "
                       f"theta = {block['theta_fwhm_arcsec']:.2f}\"")
    return out


# ---------------------------------------------------------------------------
# uv plane
# ---------------------------------------------------------------------------
def plot_uv_profile(result: ExtractionResult, path: Optional[str] = None,
                    component: str = "line", unit: Optional[str] = None,
                    dpi: int = 150, show_sources: bool = True):
    """Flux per annulus against baseline length, with the fitted model.

    Parameters
    ----------
    component
        ``'line'`` (default) draws the target line, and the second line of a
        blend alongside it since the two share the same units.
        ``'continuum'`` draws the continuum profile instead; it requires a
        result that carries one, i.e. a joint fit or a continuum-mode run.
    show_sources
        Add a lower panel with the number of sources contributing to each
        annulus, and the minimum below which an annulus was dropped.
    """
    if component not in ("line", "continuum"):
        raise ValueError("component must be 'line' or 'continuum'")
    if component == "continuum" and result.continuum is None:
        raise ValueError("this result carries no continuum profile: run the "
                         "extraction with joint_continuum=True, or use "
                         "method='continuum' on the continuum statistics")

    plt = _pyplot(path is not None)
    if show_sources:
        figure, (ax, ax_n) = plt.subplots(
            2, 1, figsize=(7.0, 5.4), sharex=True, layout="constrained",
            gridspec_kw=dict(height_ratios=[3.4, 1.0]))
    else:
        figure, ax = plt.subplots(figsize=(7.0, 4.4), layout="constrained")
        ax_n = None

    b = result.baseline_klambda
    used = result.bin_ok & np.isfinite(result.flux)

    if component == "line":
        flux, error = result.flux, result.flux_error
        imaginary = result.flux_imaginary
        colour = _LINE_COLOUR
        grid, model = _model_curve(result, b)
        label = _fit_label(result.summary["flux_total"],
                           result.summary["flux_total_error"],
                           result.summary["theta_fwhm_arcsec"],
                           result.summary["theta_error_arcsec"],
                           result.summary["model"])
        ylabel = _line_flux_label(result, unit)
    else:
        block = result.continuum
        flux, error = block["flux"], block["flux_error"]
        imaginary = block["flux_imaginary"]
        colour = _CONTINUUM_COLOUR
        grid = model = None
        if np.isfinite(block["flux_total"]):
            fine = np.linspace(0.0, np.nanmax(b) * 1e3, 400)
            grid = fine / 1e3
            model = gaussian_visibility(fine, block["flux_total"],
                                        block["theta_fwhm_arcsec"])
        label = _fit_label(block["flux_total"], block["flux_total_error"],
                           block["theta_fwhm_arcsec"],
                           block["theta_error_arcsec"])
        ylabel = f"continuum flux density [{_unit(unit)}]"

    ax.errorbar(b[used], flux[used], error[used], fmt="o", color=colour, ms=5)
    if (~used).any():
        ax.errorbar(b[~used], flux[~used], error[~used], fmt="o", mfc="white",
                    color=colour, ms=5, alpha=0.5)
    ax.plot(b, imaginary, "x", color="0.65", ms=5)
    if model is not None:
        ax.plot(grid, model, "-", color=_MODEL_COLOUR, lw=1.5, label=label)

    if component == "line" and result.second is not None:
        block = result.second
        ax.errorbar(b[used], block["flux"][used], block["flux_error"][used],
                    fmt="s", color=_SECOND_COLOUR, ms=4, alpha=0.8)
        if np.isfinite(block["flux_total"]):
            fine = np.linspace(0.0, np.nanmax(b) * 1e3, 400)
            ax.plot(fine / 1e3,
                    gaussian_visibility(fine, block["flux_total"],
                                        block["theta_fwhm_arcsec"]),
                    "--", color=_SECOND_COLOUR, lw=1.3,
                    label=_fit_label(block["flux_total"],
                                     block["flux_total_error"],
                                     block["theta_fwhm_arcsec"],
                                     block["theta_error_arcsec"]))

    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9, loc="upper right", frameon=False)
    if ax_n is None:
        ax.set_xlabel(r"uv distance $b$ [k$\lambda$]")
    else:
        ax_n.fill_between(b, 0, result.sources_per_annulus, step="mid",
                          color="0.85")
        ax_n.step(b, result.sources_per_annulus, where="mid", color="0.4",
                  lw=1.0)
        ax_n.axhline(result.summary["min_sources_per_bin"], color=colour,
                     lw=0.8, ls="--")
        ax_n.set_ylabel("sources", fontsize=9)
        ax_n.set_xlabel(r"uv distance $b$ [k$\lambda$]")
        ax_n.set_ylim(bottom=0)

    if path:
        figure.savefig(path, dpi=dpi)
        print(f"[plot] {path}")
        plt.close(figure)
    return figure


# ---------------------------------------------------------------------------
# spectrum
# ---------------------------------------------------------------------------
def _velocity_limits(result: ExtractionResult, line: LineConfig,
                     v_limits=None, pad_frac: float = 0.02):
    """Velocity range actually covered by the stacked spectrum."""
    if v_limits is not None:
        return tuple(v_limits)
    covered = result.velocity_kms[np.isfinite(result.spectrum)]
    if not covered.size:
        return None
    lo, hi = float(covered.min()), float(covered.max())
    pad = pad_frac * (hi - lo)
    return lo - pad, hi + pad


def plot_spectrum(result: ExtractionResult, line: LineConfig,
                  path: Optional[str] = None, unit: Optional[str] = None,
                  dpi: int = 150, v_limits=None, show_sources: bool = True):
    """Spatially integrated spectrum with the fitted line profile.

    The plotted velocity range is the one the data cover, unless ``v_limits``
    says otherwise.  The shaded band is the bootstrap uncertainty and the
    shaded strip the velocity window.
    """
    plt = _pyplot(path is not None)
    figure, ax = plt.subplots(figsize=(7.4, 4.6), layout="constrained")

    order = np.argsort(result.velocity_kms)
    v = result.velocity_kms[order]
    spectrum = result.spectrum[order]
    error = result.spectrum_error[order]

    ax.step(v, spectrum, where="mid", color="0.3", lw=1.0)
    ax.fill_between(v, spectrum - error, spectrum + error, step="mid",
                    color="0.6", alpha=0.25)

    shape = result.shape
    if shape is not None:
        offset = shape.baseline or 0.0
        model = offset + shape.amplitude * np.exp(
            -0.5 * ((result.velocity_kms - shape.centroid_kms)
                    / shape.sigma_kms) ** 2)
        label = (f"v$_0$ = {shape.centroid_kms:+.0f} km/s, "
                 f"FWHM = {shape.fwhm_kms:.0f} km/s")
        if shape.blended and line.second_rest_freq_hz is not None:
            v2 = line.velocity(result.freq_hz, second=True)
            second = shape.amplitude2 * np.exp(
                -0.5 * ((v2 - shape.centroid2_kms) / shape.sigma2_kms) ** 2)
            ax.plot(v, (offset + second)[order], "--", color=_SECOND_COLOUR,
                    lw=1.3,
                    label=f"v$_0$ = {shape.centroid2_kms:+.0f} km/s, "
                          f"FWHM = {2.3548 * shape.sigma2_kms:.0f} km/s")
            ax.plot(v, (model + second)[order], "-", color="C1", lw=1.0,
                    alpha=0.8)
        ax.plot(v, model[order], "-", color=_MODEL_COLOUR, lw=1.6, label=label)
        if shape.baseline is not None:
            ax.axhline(offset, color=_CONTINUUM_COLOUR, lw=1.0, ls=":",
                       label=f"continuum = {offset:.3g} {_unit(unit)}")
        else:
            ax.axhline(0.0, color=_MODEL_COLOUR, lw=0.8, ls=":",
                       label="baseline fixed at 0")

    ax.axvspan(*line.v_window_kms, color="C1", alpha=0.08)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("velocity [km/s]")
    ax.set_ylabel(f"Re(V), stack [{_unit(unit)}]")
    limits = _velocity_limits(result, line, v_limits)
    if limits:
        ax.set_xlim(*limits)
    if shape is not None:
        ax.legend(fontsize=9, loc="upper right", frameon=False)

    if show_sources:
        twin = ax.twinx()
        twin.step(v, result.sources_per_channel[order], where="mid",
                  color=_LINE_COLOUR, lw=0.8, alpha=0.4)
        twin.set_ylabel("sources per channel", color=_LINE_COLOUR, fontsize=8)
        twin.tick_params(axis="y", labelsize=8, colors=_LINE_COLOUR)
        twin.set_ylim(bottom=0)

    if path:
        figure.savefig(path, dpi=dpi)
        print(f"[plot] {path}")
        plt.close(figure)
    return figure


# ---------------------------------------------------------------------------
# line and continuum fitted together
# ---------------------------------------------------------------------------
def plot_joint(result: ExtractionResult, line: LineConfig,
               path: Optional[str] = None, unit: Optional[str] = None,
               dpi: int = 150, v_limits=None):
    """The single figure of a joint line-plus-continuum fit.

    Three panels: the line profile in the uv plane, the continuum profile in
    the uv plane, and the integrated spectrum with the line sitting on the
    fitted continuum level.  The two uv panels keep their own axes, since a
    velocity-integrated flux and a flux density do not share a scale, but they
    belong in the same figure because they came out of the same fit.

    Raises
    ------
    ValueError
        If the result carries no continuum: then the continuum was either
        subtracted beforehand, and the two fits have their own figures, or not
        fitted at all.
    """
    if result.continuum is None:
        raise ValueError("this result has no jointly fitted continuum: use "
                         "plot_uv_profile and plot_spectrum instead")
    plt = _pyplot(path is not None)
    figure = plt.figure(figsize=(12.6, 4.2), layout="constrained")
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25])
    ax_line = figure.add_subplot(grid[0, 0])
    ax_cont = figure.add_subplot(grid[0, 1], sharex=ax_line)
    ax_spec = figure.add_subplot(grid[0, 2])

    b = result.baseline_klambda
    used = result.bin_ok & np.isfinite(result.flux)

    # --- line in the uv plane ---------------------------------------------
    ax_line.errorbar(b[used], result.flux[used], result.flux_error[used],
                     fmt="o", color=_LINE_COLOUR, ms=4)
    if (~used).any():
        ax_line.errorbar(b[~used], result.flux[~used],
                         result.flux_error[~used], fmt="o", mfc="white",
                         color=_LINE_COLOUR, ms=4, alpha=0.5)
    ax_line.plot(b, result.flux_imaginary, "x", color="0.65", ms=4)
    curve, model = _model_curve(result, b)
    if model is not None:
        ax_line.plot(curve, model, "-", color=_MODEL_COLOUR, lw=1.4,
                     label=_fit_label(result.summary["flux_total"],
                                      result.summary["flux_total_error"],
                                      result.summary["theta_fwhm_arcsec"],
                                      result.summary["theta_error_arcsec"],
                                      result.summary["model"]))
    if result.second is not None:
        block = result.second
        ax_line.errorbar(b[used], block["flux"][used],
                         block["flux_error"][used], fmt="s",
                         color=_SECOND_COLOUR, ms=3.5, alpha=0.8)
        if np.isfinite(block["flux_total"]):
            fine = np.linspace(0.0, np.nanmax(b) * 1e3, 400)
            ax_line.plot(fine / 1e3,
                         gaussian_visibility(fine, block["flux_total"],
                                             block["theta_fwhm_arcsec"]),
                         "--", color=_SECOND_COLOUR, lw=1.2,
                         label=_fit_label(block["flux_total"],
                                          block["flux_total_error"],
                                          block["theta_fwhm_arcsec"],
                                          block["theta_error_arcsec"]))
    ax_line.axhline(0.0, color="k", lw=0.5)
    ax_line.set_xlabel(r"uv distance $b$ [k$\lambda$]")
    ax_line.set_ylabel(_line_flux_label(result, unit))
    ax_line.set_title("line", fontsize=10)
    ax_line.legend(fontsize=8, loc="upper right", frameon=False)

    # --- continuum in the uv plane ----------------------------------------
    block = result.continuum
    ax_cont.errorbar(b[used], block["flux"][used], block["flux_error"][used],
                     fmt="o", color=_CONTINUUM_COLOUR, ms=4)
    if (~used).any():
        ax_cont.errorbar(b[~used], block["flux"][~used],
                         block["flux_error"][~used], fmt="o", mfc="white",
                         color=_CONTINUUM_COLOUR, ms=4, alpha=0.5)
    ax_cont.plot(b, block["flux_imaginary"], "x", color="0.65", ms=4)
    if np.isfinite(block["flux_total"]):
        fine = np.linspace(0.0, np.nanmax(b) * 1e3, 400)
        ax_cont.plot(fine / 1e3,
                     gaussian_visibility(fine, block["flux_total"],
                                         block["theta_fwhm_arcsec"]),
                     "-", color=_MODEL_COLOUR, lw=1.4,
                     label=_fit_label(block["flux_total"],
                                      block["flux_total_error"],
                                      block["theta_fwhm_arcsec"],
                                      block["theta_error_arcsec"]))
    ax_cont.axhline(0.0, color="k", lw=0.5)
    ax_cont.set_xlabel(r"uv distance $b$ [k$\lambda$]")
    ax_cont.set_ylabel(f"continuum flux density [{_unit(unit)}]")
    ax_cont.set_title("continuum", fontsize=10)
    ax_cont.legend(fontsize=8, loc="upper right", frameon=False)

    # --- spectrum ----------------------------------------------------------
    order = np.argsort(result.velocity_kms)
    v = result.velocity_kms[order]
    ax_spec.step(v, result.spectrum[order], where="mid", color="0.3", lw=1.0)
    ax_spec.fill_between(v, (result.spectrum - result.spectrum_error)[order],
                         (result.spectrum + result.spectrum_error)[order],
                         step="mid", color="0.6", alpha=0.25)
    shape = result.shape
    if shape is not None:
        offset = shape.baseline or 0.0
        curve = offset + shape.amplitude * np.exp(
            -0.5 * ((result.velocity_kms - shape.centroid_kms)
                    / shape.sigma_kms) ** 2)
        ax_spec.plot(v, curve[order], "-", color=_MODEL_COLOUR, lw=1.5,
                     label=f"v$_0$ = {shape.centroid_kms:+.0f} km/s, "
                           f"FWHM = {shape.fwhm_kms:.0f} km/s")
        if shape.blended and line.second_rest_freq_hz is not None:
            v2 = line.velocity(result.freq_hz, second=True)
            second = shape.amplitude2 * np.exp(
                -0.5 * ((v2 - shape.centroid2_kms) / shape.sigma2_kms) ** 2)
            ax_spec.plot(v, (offset + second)[order], "--",
                         color=_SECOND_COLOUR, lw=1.2)
            ax_spec.plot(v, (curve + second)[order], "-", color="C1", lw=0.9,
                         alpha=0.8)
        ax_spec.axhline(offset, color=_CONTINUUM_COLOUR, lw=1.0, ls=":",
                        label=f"continuum = {offset:.3g} {_unit(unit)}")
        ax_spec.legend(fontsize=8, loc="upper right", frameon=False)
    ax_spec.axvspan(*line.v_window_kms, color="C1", alpha=0.08)
    ax_spec.axhline(0.0, color="k", lw=0.5)
    ax_spec.set_xlabel("velocity [km/s]")
    ax_spec.set_ylabel(f"Re(V), stack [{_unit(unit)}]")
    ax_spec.set_title("spectrum", fontsize=10)
    limits = _velocity_limits(result, line, v_limits)
    if limits:
        ax_spec.set_xlim(*limits)

    if path:
        figure.savefig(path, dpi=dpi)
        print(f"[plot] {path}")
        plt.close(figure)
    return figure


# ---------------------------------------------------------------------------
# the line profile in flux units
# ---------------------------------------------------------------------------
def plot_line_profile(result: ExtractionResult, line: LineConfig,
                      path: Optional[str] = None, unit: Optional[str] = None,
                      dpi: int = 150, v_limits=None):
    """The line profile in flux units, as an image-plane fit would give it.

    For every channel the flux is measured on the whole radial profile with
    the fitted size held fixed, so the resolution of each annulus is divided
    out instead of being averaged in.  The y axis is a flux density and the
    area under the fitted Gaussian is the total line flux, unlike the stacked
    ``Re(V)`` spectrum whose amplitude is diluted by the long baselines.
    """
    if result.spectrum_zero_baseline is None:
        raise ValueError("this result carries no zero-baseline spectrum: it "
                         "needs a converged uv fit on a line")
    plt = _pyplot(path is not None)
    figure, ax = plt.subplots(figsize=(7.4, 4.6), layout="constrained")

    order = np.argsort(result.velocity_kms)
    v = result.velocity_kms[order]
    flux = result.spectrum_zero_baseline[order]
    error = result.spectrum_zero_baseline_error
    error = None if error is None else error[order]

    ax.step(v, flux, where="mid", color="0.3", lw=1.0)
    if error is not None:
        ax.fill_between(v, flux - error, flux + error, step="mid",
                        color="0.6", alpha=0.25)

    shape = result.zero_baseline_shape
    if shape is not None:
        model = shape.amplitude * np.exp(
            -0.5 * ((result.velocity_kms - shape.centroid_kms)
                    / shape.sigma_kms) ** 2)
        if shape.baseline is not None:
            model = model + shape.baseline
        area = shape.amplitude * shape.sigma_kms * np.sqrt(2.0 * np.pi)
        ax.plot(v, model[order], "-", color=_MODEL_COLOUR, lw=1.6,
                label=f"v$_0$ = {shape.centroid_kms:+.0f} km/s, "
                      f"FWHM = {shape.fwhm_kms:.0f} km/s\n"
                      f"area = {area:.3g} {_unit(unit)} km/s")
        ax.legend(fontsize=9, loc="upper right", frameon=False)

    ax.axvspan(*line.v_window_kms, color="C1", alpha=0.08)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("velocity [km/s]")
    ax.set_ylabel(f"flux density at b = 0 [{_unit(unit)}]")
    theta = result.summary.get("theta_fwhm_arcsec", float("nan"))
    ax.set_title(f"size held at theta = {theta:.2f} arcsec", fontsize=10)
    limits = _velocity_limits(result, line, v_limits)
    if limits:
        ax.set_xlim(*limits)

    if path:
        figure.savefig(path, dpi=dpi)
        print(f"[plot] {path}")
        plt.close(figure)
    return figure


def plot_all(result: ExtractionResult, line: LineConfig, prefix: str,
             unit: Optional[str] = None, dpi: int = 150,
             v_limits=None) -> List[str]:
    """Write the figures that fit the way the continuum was handled.

    - jointly fitted continuum  -> ``<prefix>.png``, one figure for the one
      fit that produced both;
    - continuum-mode run        -> ``<prefix>_uvamp.png``, the continuum
      profile alone;
    - anything else, i.e. a line on a subtracted stack or with no continuum at
      all -> ``<prefix>_uvamp.png`` and ``<prefix>_spectrum.png``.
    """
    if result.continuum is not None:
        path = f"{prefix}.png"
        plot_joint(result, line, path=path, unit=unit, dpi=dpi,
                   v_limits=v_limits)
        return [path]

    if result.summary.get("flux_method") == "continuum":
        path = f"{prefix}_uvamp.png"
        plot_uv_profile(result, path=path, unit=unit, dpi=dpi)
        return [path]

    written = [f"{prefix}_uvamp.png", f"{prefix}_spectrum.png"]
    plot_uv_profile(result, path=written[0], unit=unit, dpi=dpi)
    plot_spectrum(result, line, path=written[1], unit=unit, dpi=dpi,
                  v_limits=v_limits)
    if result.spectrum_zero_baseline is not None:
        path = f"{prefix}_lineprofile.png"
        plot_line_profile(result, line, path=path, unit=unit, dpi=dpi,
                          v_limits=v_limits)
        written.append(path)
    return written


__all__ = ["plot_uv_profile", "plot_spectrum", "plot_line_profile",
           "plot_joint", "plot_all", "fit_summary_lines"]
