"""
vista.extract.plots_line
========================
Figures of :func:`vista.extract.measure_line`:

``<prefix>_profile.png``
    The line profile: flux density at zero spacing against velocity, the
    fitted lines, the two fluxes of every line (area of the Gaussian, and the
    window over which the spectrum is summed), the noise (+-1 sigma band), the
    imaginary part, the residual continuum outside the windows, and the
    residuals in units of sigma.
``<prefix>_uv.png``
    Line flux of every annulus against baseline length, with the point and
    Gaussian models.
``<prefix>_noise.png``
    The noise: per channel from the scatter and from the weights, the
    distribution of the standardised imaginary and line-free cells against
    N(0, 1), and the null distribution of the search S/N with the measured one.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .. import _say
from .config import Lines
from .linefit import uv_gaussian


def _pyplot():
    """pyplot, on the Agg backend (files only) unless inside IPython/Jupyter,
    whose inline display a backend switch would turn off."""
    import sys
    import matplotlib
    if "IPython" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _rebin(v, y, e, corr, n):
    """Average groups of n channels; the error accounts for the correlation
    (``corr``: channel x channel matrix in the order of ``v``, or None)."""
    if n <= 1:
        return v, y, e
    m = len(v) // n
    vb = v[:m * n].reshape(m, n).mean(axis=1)
    yb = np.nanmean(y[:m * n].reshape(m, n), axis=1)
    eb = np.empty(m)
    for k in range(m):
        sl = slice(k * n, (k + 1) * n)
        ek = e[sl]
        r = np.eye(n) if corr is None else corr[sl, sl]
        eb[k] = np.sqrt(np.nansum(r * np.outer(ek, ek))) / n
    return vb, yb, eb


def _pm(value, error):
    """'value +- error' with the error to two significant digits."""
    if not (np.isfinite(value) and np.isfinite(error)) or error <= 0:
        return f"{value:.3g}"
    digits = max(0, 1 - int(np.floor(np.log10(abs(error)))))
    return f"{value:.{digits}f} $\\pm$ {error:.{digits}f}"


def _scale(unit, typical=None):
    """Factor and label for the flux-density axis: mJy for data in Jy whose
    typical value (e.g. the noise of a channel) is below 1 Jy."""
    if unit and unit.lower() == "jy" and (typical is None or not np.isfinite(typical)
                                          or abs(typical) < 1.0):
        return 1e3, "mJy"
    return 1.0, unit or "data units"


def _typical(result):
    e = np.asarray(result.profile_error, float)
    return float(np.nanmedian(e)) if np.isfinite(e).any() else None


def _finish(fig, path):
    """Save the figure to ``path`` and close it (returns the path), or return
    it open (``path`` None)."""
    if path is None:
        return fig
    fig.savefig(path, dpi=140, bbox_inches="tight")
    _pyplot().close(fig)
    return path


def _window_label(lo, hi):
    return (f"line window, ±{hi:.0f} km/s" if abs(lo + hi) < 1e-6 * max(abs(hi), 1)
            else f"line window, {lo:+.0f} to {hi:+.0f} km/s")


def _continuum_label(s, f, ulabel, continuum=None):
    """What happened to the continuum, for the legend: its value when it was
    measured, and what is left outside the line windows."""
    rc = s.get("continuum_residual") or {}
    cf = s.get("continuum_fit")
    left = (f"{_pm(rc['offset'] * f, rc['offset_error'] * f)} {ulabel} "
            f"({rc['offset_nsigma']:+.1f}$\\sigma$)") if rc else ""
    if cf:
        return (f"continuum, fitted with the line: "
                f"{_pm(cf['flux_density'] * f, cf['flux_density_error'] * f)} {ulabel}")
    if rc.get("continuum_subtracted"):
        value = ""
        if continuum is not None:
            c = continuum.summary
            value = (f" ({_pm(c['flux_total'] * f, c['flux_total_error'] * f)} {ulabel})"
                     if c["flux_total"] > 2 * c["flux_total_error"] else " (none detected)")
        return (f"continuum: subtracted{value}; "
                + (f"still there: {left}" if rc.get("detected") else "nothing left outside the window"))
    if rc.get("detected"):
        return f"continuum left in the data: {left}"
    return f"continuum: none found outside the window: {left}" if rc else "continuum: not measured"


def plot_profile(result, line: Optional[Lines] = None, path: Optional[str] = None,
                 unit: Optional[str] = None, v_limits=None, rebin: int = 1, continuum=None):
    """The stacked line: the flux density at zero spacing against velocity
    (black), its noise (grey band), the fitted Gaussian (red, its area
    hatched), the window over which the spectrum is summed (shaded) and the
    level of the continuum outside it (dashed).  The box gives, for every
    line, the flux quoted and where it comes from (the Gaussian fit, or the
    area under the spectrum in the window), the other of the two, and the
    centre, width and peak of the Gaussian.  Below: the residuals of the fit
    and the imaginary part (crosses), both in units of the noise: they should
    scatter around zero.

    ``continuum``: the continuum measured on its own (the result of
    ``extract_flux``), to quote its value.  Without ``path`` the figure is
    returned."""
    plt = _pyplot()
    s = result.summary
    f, ulabel = _scale(unit, _typical(result))
    v = result.velocity_kms
    order = np.argsort(v)
    corr = result.channel_correlation
    rho = None if corr is None else corr[np.ix_(order, order)]
    v, y, e = v[order], result.profile[order] * f, result.profile_error[order] * f
    yi = result.profile_imaginary[order] * f
    used = result.fitted[order]
    vb, yb, eb = _rebin(v, np.where(used, y, np.nan), e, rho, rebin)
    _, yib, _ = _rebin(v, np.where(used, yi, np.nan), e, rho, rebin)
    dvb = abs(vb[1] - vb[0]) if len(vb) > 1 else 1.0
    blocks = s.get("lines") or [s["line"]]
    several = len(blocks) > 1

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.05})
    # the windows of the area fluxes, one per line
    for k, b in enumerate(blocks):
        lo, hi = b.get("window_kms_line1_velocity") or s["window"]["user_kms"]
        own = b.get("window_kms") or (lo, hi)
        ax.axvspan(lo, hi, color="C1", alpha=0.10, lw=0,
                   label=_window_label(*own) + (f" (line {k + 1})" if several else ""))
    ax.fill_between(vb, -eb, eb, step="mid", color="0.6", alpha=0.35, lw=0,
                    label=r"noise, $\pm1\sigma$")
    ax.step(vb, yb, where="mid", color="k", lw=1.2, label="stacked spectrum")
    # the model averaged over the channels, as it was fitted
    _, mb, _ = _rebin(v, np.where(used, result.profile_model[order] * f, np.nan), e, rho, rebin)
    ax.step(vb, mb, where="mid", color="C3", lw=1.5, label="Gaussian fit")
    # the area of the Gaussian: above the continuum when it was fitted with the line
    cf = s.get("continuum_fit")
    base = cf["flux_density"] * f if cf else 0.0
    ax.fill_between(vb, base, np.where(np.isfinite(mb), mb, base), step="mid",
                    facecolor="none", edgecolor="C3", hatch="////", lw=0, alpha=0.5)
    # the continuum: the fitted level, or what is left outside the windows
    rc = s.get("continuum_residual")
    label = _continuum_label(s, f, ulabel, continuum)
    if cf:
        ax.axhline(base, color="C0", lw=1.2, ls="--", label=label)
    elif rc:
        vv = np.linspace(*rc["velocity_range_kms"], 50)
        ax.plot(vv, (rc["offset"] + rc["slope_per_1000kms"] * vv / 1000) * f,
                color="C1" if rc["detected"] else "C0", lw=1.2, ls="--", label=label)
    ax.axhline(0, color="0.3", lw=0.6)

    rows = []
    separate = len(s.get("line_groups") or [[1]]) > 1
    for k, b in enumerate(blocks):
        d = b["detection"]
        gauss = f"{_pm(b['flux_gauss'] * f, b['flux_gauss_error'] * f)} {ulabel} km/s"
        area = f"{_pm(b['flux_area'] * f, b['flux_area_error'] * f)} {ulabel} km/s"
        if b.get("flux_method") == "area":
            fluxes = (f"Flux = {area}  (area under the spectrum, in the window)\n"
                      f"   from the Gaussian fit: {gauss}")
        else:
            fluxes = (f"Flux = {gauss}  (area of the Gaussian fit)\n"
                      f"   under the spectrum, in the window: {area}")
        sig = (f"S/N {d['snr']:.1f} (centre and width fixed)" if d.get("fixed_shape")
               else f"detected at {d['significance_sigma']:.1f}$\\sigma$")
        rows.append(
            (f"Line {k + 1} ({b['rest_freq_ghz']:.6g} GHz), " if several else "") + sig + "\n"
            + fluxes
            + f"\n$\\mu$ = {b['v0_kms']:+.0f} $\\pm$ {b.get('v0_error_kms', np.nan):.0f} km/s,  "
              f"FWHM = {b['fwhm_kms']:.0f} $\\pm$ {b.get('fwhm_error_kms', np.nan):.0f} km/s,  "
              f"peak = {_pm(b['peak'] * f, b.get('peak_error', np.nan) * f)} {ulabel}"
            + (f"\nsource: {_size_text(b['size_model'], b.get('theta_fwhm_arcsec'))}"
               if separate and "size_model" in b else ""))
    ax.text(0.02, 0.97, "\n".join(rows), transform=ax.transAxes, va="top", fontsize=7.8,
            linespacing=1.35, bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.92))
    ax.set_ylabel(f"flux density [{ulabel}]")
    lo_y, hi_y = ax.get_ylim()
    n_rows = sum(r.count("\n") + 1 for r in rows)
    ax.set_ylim(lo_y - 0.22 * (hi_y - lo_y), hi_y + (0.10 + 0.085 * n_rows) * (hi_y - lo_y))
    ax.legend(loc="lower right", fontsize=7.5, frameon=False, ncol=2)
    ax.set_title(f"stacked line{'s' if several else ''} at zero spacing: {s['n_sources']} "
                 f"sources, {s['weighting_scheme']} weighting"
                 + (f", {rebin} channels per bin" if rebin > 1 else ""), fontsize=9)
    # residuals and the imaginary part, in units of the noise
    with np.errstate(invalid="ignore", divide="ignore"):
        axr.step(vb, (yb - mb) / eb, where="mid", color="k", lw=1, label="spectrum − fit")
        axr.plot(vb, yib / eb, "x", color="C0", ms=3.5, mew=0.8, label="imaginary part")
    axr.axhspan(-1, 1, color="0.6", alpha=0.3, lw=0)
    axr.axhline(0, color="0.3", lw=0.6)
    axr.set_ylabel(r"in units of $\sigma$")
    axr.set_xlabel("velocity [km/s]")
    axr.legend(fontsize=7, frameon=False, ncol=2, loc="upper right", borderaxespad=0.1)
    if v_limits:
        axr.set_xlim(*v_limits)
    else:
        axr.set_xlim(v[used].min() - dvb, v[used].max() + dvb)
    return _finish(fig, path)


def _size_text(model, theta):
    """'Gaussian, FWHM 0.46"' or 'point source'."""
    if not theta:
        return "point source"
    return f"Gaussian, FWHM {theta:.2f}\"" + (" (fixed)" if model == "fixed" else "")


def plot_uv(result, path: Optional[str] = None, unit: Optional[str] = None):
    """The line flux measured in every annulus of baseline length (points),
    and the source model fitted to all of them: flat for a point source,
    falling with the baseline for an extended one.  The flux quoted for the
    Gaussian fit is where the model meets zero baseline.  With
    ``model="auto"`` both models are drawn, the preferred one thicker.
    Without ``path`` the figure is returned."""
    plt = _pyplot()
    s = result.summary
    f, ulabel = _scale(unit, _typical(result))
    b = result.baseline_klambda
    ok = result.annulus_ok & np.isfinite(result.annulus_flux)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.errorbar(b[ok], result.annulus_flux[ok] * f, result.annulus_flux_error[ok] * f,
                fmt="o", color="k", ms=4, capsize=2, label="line flux in every annulus")
    bb = np.geomspace(max(b[ok].min() * 0.5, 0.1), b[ok].max() * 1.5, 300)
    p = s["line"]
    ax.axhline(0, color="0.3", lw=0.6)
    st = s.get("size_test")
    unit_f = f"{ulabel} km/s"
    if st and st.get("weight_gauss") is not None:
        gauss_wins = st["adopted"] == "gauss"
        ax.plot(bb, np.full_like(bb, st["flux_point"] * f), color="C0",
                lw=1.0 if gauss_wins else 2.2, ls="--" if gauss_wins else "-",
                label=f"point source: flux {_pm(st['flux_point'] * f, st['flux_point_error'] * f)}")
        if st.get("theta_gauss_arcsec"):
            ax.plot(bb, st["flux_gauss"] * f * uv_gaussian(bb * 1e3, st["theta_gauss_arcsec"]),
                    color="C3", lw=2.2 if gauss_wins else 1.0, ls="-" if gauss_wins else "--",
                    label=f"Gaussian source, FWHM {st['theta_gauss_arcsec']:.2f}\": flux "
                          f"{_pm(st['flux_gauss'] * f, st['flux_gauss_error'] * f)}")
        chosen = (f"extended: the Gaussian source fits better ($\\Delta\\chi^2$ = "
                  f"{st['delta_chi2_point_minus_gauss']:.1f}, needed {st['threshold']:.0f})"
                  if gauss_wins else
                  f"compatible with a point source ($\\Delta\\chi^2$ = "
                  f"{st['delta_chi2_point_minus_gauss']:.1f}, needed {st['threshold']:.0f})")
    elif p["theta_fwhm_arcsec"]:
        ax.plot(bb, p["flux_gauss"] * f * uv_gaussian(bb * 1e3, p["theta_fwhm_arcsec"]),
                color="C3", lw=2.2,
                label=f"Gaussian source, FWHM {p['theta_fwhm_arcsec']:.2f}\": flux "
                      f"{_pm(p['flux_gauss'] * f, p['flux_gauss_error'] * f)}")
        chosen = "Gaussian source, as asked"
    else:
        ax.plot(bb, np.full_like(bb, p["flux_gauss"] * f), color="C0", lw=2.2,
                label=f"point source: flux {_pm(p['flux_gauss'] * f, p['flux_gauss_error'] * f)}")
        chosen = "point source, as asked"
    ax.set_xscale("log")
    ax.set_title(("line 1: " if len(s.get("lines") or []) > 1 else "") + chosen, fontsize=9)
    ax.set_xlabel(r"baseline length [k$\lambda$]   (longer = finer details)")
    ax.set_ylabel(f"line flux [{unit_f}]")
    ax.legend(fontsize=8, frameon=False, title=f"flux of the Gaussian fit in velocity, {unit_f}",
              title_fontsize=7.5)
    return _finish(fig, path)


def plot_noise(result, path: Optional[str] = None, unit: Optional[str] = None):
    """The noise checks.  Without ``path`` the figure is returned."""
    plt = _pyplot()
    s = result.summary
    n = s["noise"]
    f, ulabel = _scale(unit, _typical(result))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    v = result.velocity_kms
    o = np.argsort(v)
    ax.step(v[o], result.noise_channel[o] * f, where="mid", color="k",
            label="from the scatter of the visibilities")
    ax.step(v[o], result.noise_channel_weights[o] * f, where="mid", color="C0", ls="--",
            label="from the weights")
    ax.set_xlabel("velocity [km/s]")
    ax.set_ylabel(f"$\\sigma$ per channel [{ulabel}]")
    ax.set_title(f"$\\kappa$ = scatter / weights = {n['kappa_scatter_over_weights']:.3f}",
                 fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[1]
    grid = np.linspace(-5, 5, 200)
    ax.plot(grid, np.exp(-0.5 * grid ** 2) / np.sqrt(2 * np.pi), color="0.3", lw=1,
            label="N(0, 1)")
    bins = np.linspace(-5, 5, 41)
    if result.z_imaginary.size:
        ax.hist(np.clip(result.z_imaginary, -5, 5), bins=bins, density=True, histtype="step",
                color="C0", label=f"imaginary, rms {n.get('imaginary_rms_over_noise', np.nan):.2f}")
    if result.z_line_free.size:
        ax.hist(np.clip(result.z_line_free, -5, 5), bins=bins, density=True, histtype="step",
                color="C3", label=f"real, line-free, rms {n.get('line_free_rms_over_noise', np.nan):.2f}")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("cell / $\\sigma$")
    rho = n.get("channel_correlation", [])
    ax.set_title("stacked cells off the line; channel correlation "
                 + ", ".join(f"{r:+.2f}" for r in rho[:2]), fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[2]
    d = s["detection"]
    if result.null_snr.size:
        ax.hist(result.null_snr, bins=40, density=True, histtype="stepfilled", color="0.75",
                label="search on noise")
    if result.null_snr_phase.size:
        ax.hist(result.null_snr_phase, bins=30, density=True, histtype="step", color="C0",
                label="search on random-phase stacks")
    if not result.null_snr.size and not result.null_snr_phase.size:
        x = np.linspace(-4, max(5, min(d["snr"], 60) + 1), 200)
        ax.plot(x, np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi), color="0.3",
                label="N(0, 1) (fixed shape)")
    ax.axvline(d["snr"], color="C3", lw=1.5, label=f"measured S/N {d['snr']:.1f}")
    ax.set_xlabel("S/N")
    ax.set_title(f"significance {d['significance_sigma']:.1f}$\\sigma$", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    return _finish(fig, path)


def plot_line_result(result, line: Lines, prefix: str, unit: Optional[str] = None,
                     v_limits=None, rebin: int = 1, verbose=True, continuum=None) -> List[str]:
    """All three figures; returns their paths."""
    out = [plot_profile(result, line, f"{prefix}_profile.png", unit, v_limits, rebin,
                        continuum),
           plot_uv(result, f"{prefix}_uv.png", unit),
           plot_noise(result, f"{prefix}_noise.png", unit)]
    _say("  figures: " + ", ".join(out), verbose)
    return out


def display(fig, save: Optional[str] = None, show: bool = True):
    """Save the figure (``save``: a path) and show it: in a notebook it is
    drawn now and closed (returns None, so it is not drawn twice); with
    ``show=False``, or outside a notebook, it is returned."""
    import sys
    if save:
        fig.savefig(save, dpi=140, bbox_inches="tight")
    if show and "IPython" in sys.modules:
        from IPython.display import display as _display
        _display(fig)
        _pyplot().close(fig)
        return None
    return fig


def _line_of(result):
    """The LineResult of a RunResult (or the object itself)."""
    return getattr(result, "line", None) or result


_COLOURS = ["#000000", "#c0392b", "#2471a3", "#d68910", "#7d3c98", "#17a589", "#7f8c8d"]


def compare(results, expected: Optional[dict] = None, unit: Optional[str] = "Jy",
            rebin: int = 1, v_limits=None, save: Optional[str] = None, show: bool = True,
            table: bool = True):
    """Several measurements side by side: the profiles at zero spacing with
    their fits, the line flux per annulus with the source model, and the
    fluxes with their errors.

    Parameters
    ----------
    results : dict
        ``{label: result}``, results of :meth:`vista.ViSta.run` (or their
        ``.line``), in the order to draw them.
    expected : dict, optional
        What a perfect measurement would give, when it is known (a
        simulation): ``flux`` (line flux, or a list, one per line),
        ``profile`` (``(velocity_kms, flux_density)``), ``uv``
        (``(baseline_klambda, line_flux)``), ``continuum``.  Drawn in green.
    unit : str
        Unit of the data (``"Jy"``: drawn in mJy).
    rebin : int
        Channels averaged in the profile.
    save, show
        As for :func:`display`.
    table : bool
        Also print the numbers of every measurement.
    """
    plt = _pyplot()
    items = [(str(k), _line_of(r)) for k, r in dict(results).items()]
    if not items:
        raise ValueError("compare: no result")
    truth = expected or {}
    f, ulabel = _scale(unit, _typical(items[0][1]))
    fig = plt.figure(figsize=(12, 7.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])
    ax_p, ax_u, ax_f = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ---- the profiles --------------------------------------------------
    # the windows of the measurement with the most lines
    n_lines = [len(r.summary.get("lines") or [1]) for _, r in items]
    s0 = items[int(np.argmax(n_lines))][1].summary
    windows = {tuple(np.round(r.summary["line"]["window_kms"], 3)) for _, r in items}
    for k, b in enumerate(s0.get("lines") or [s0["line"]]):
        lo, hi = b.get("window_kms_line1_velocity") or s0["window"]["user_kms"]
        if len(windows) == 1:
            ax_p.axvspan(lo, hi, color="C1", alpha=0.07, lw=0,
                         label=("line windows" if len(s0.get("lines") or []) > 1
                                else _window_label(lo, hi)) if k == 0 else None)
    vmin, vmax = np.inf, -np.inf
    spectra = []               # the spectrum of every measurement, unless the same as drawn
    dashes = [(6, 2), (2, 2), (8, 2, 2, 2), (4, 4), (1, 1.5), (10, 3)]
    for i, (label, r) in enumerate(items):
        c = _COLOURS[i % len(_COLOURS)]
        v = r.velocity_kms
        o = np.argsort(v)
        corr = r.channel_correlation
        rho = None if corr is None else corr[np.ix_(o, o)]
        used = r.fitted[o]
        vb, yb, eb = _rebin(v[o], np.where(used, r.profile[o] * f, np.nan),
                            r.profile_error[o] * f, rho, rebin)
        _, mb, _ = _rebin(v[o], np.where(used, r.profile_model[o] * f, np.nan),
                          r.profile_error[o] * f, rho, rebin)
        if i == 0:
            ax_p.fill_between(vb, -eb, eb, step="mid", color="0.6", alpha=0.3, lw=0,
                              label=r"$\pm1\sigma$ noise")
        if not any(len(yb) == len(y0) and np.allclose(np.nan_to_num(yb), np.nan_to_num(y0),
                                                      atol=0.02 * np.nanmedian(eb))
                   for y0 in spectra):
            spectra.append(yb)
            ax_p.step(vb, yb, where="mid", color=c if len(spectra) > 1 else "0.25", lw=1.0,
                      alpha=0.75)
        # different dashes: fits that coincide stay visible, one over the other
        ax_p.step(vb, mb, where="mid", color=c, lw=1.8, dashes=dashes[i % len(dashes)],
                  label=label)
        if len(windows) > 1:                 # the window of the area flux of each
            for edge in r.summary["line"]["window_kms"]:
                ax_p.axvline(edge, color=c, lw=1.0, ls=":")
        if used.any():
            vmin, vmax = min(vmin, v[o][used].min()), max(vmax, v[o][used].max())
    if "profile" in truth:
        tv, ty = truth["profile"]
        ax_p.plot(tv, np.asarray(ty) * f, color="#1e8449", lw=2, alpha=0.8, label="expected")
    ax_p.axhline(0, color="0.3", lw=0.6)
    ax_p.set_xlim(*(v_limits or (vmin, vmax)))
    ax_p.set_xlabel("velocity [km/s]")
    ax_p.set_ylabel(f"flux density [{ulabel}]")
    ax_p.set_title("stacked spectrum at zero spacing (thin) and the fits (dashed)"
                   + ("; dotted: the window of each area flux" if len(windows) > 1 else ""),
                   fontsize=9)
    ax_p.legend(fontsize=8, frameon=False, loc="upper right")

    # ---- the line flux per annulus ----------------------------------------
    bmin, bmax = np.inf, 0.0
    drawn = []                 # the points of every measurement, unless the same as drawn
    for i, (label, r) in enumerate(items):
        b = r.baseline_klambda
        ok = r.annulus_ok & np.isfinite(r.annulus_flux)
        if not ok.any():
            continue
        bmin, bmax = min(bmin, b[ok].min()), max(bmax, b[ok].max())
        y, e = r.annulus_flux, r.annulus_flux_error
        if any(len(y) == len(y0) and np.allclose(np.where(ok, y, 0), np.where(ok, y0, 0),
                                                 atol=0.25 * np.nanmedian(e[ok]))
               for y0 in drawn):
            continue
        drawn.append(y)
        shift = 1 + 0.025 * (len(drawn) - 1)                  # side by side
        ax_u.errorbar(b[ok] * shift, y[ok] * f, e[ok] * f, fmt="o",
                      color=_COLOURS[i % len(_COLOURS)], ms=3.5, capsize=1.5, lw=0.9)
    if bmax > 0:
        bb = np.geomspace(max(bmin * 0.5, 0.1), bmax * 1.5, 300)
        for i, (label, r) in enumerate(items):
            p = r.summary["line"]
            th = p.get("theta_fwhm_arcsec") or 0.0
            curve = p["flux_gauss"] * (uv_gaussian(bb * 1e3, th) if th else np.ones_like(bb))
            ax_u.plot(bb, curve * f, color=_COLOURS[i % len(_COLOURS)], lw=1.5,
                      dashes=dashes[i % len(dashes)],
                      label=f"{label}: " + (f"Gaussian {th:.2f}\"" if th else "point source"))
        if "uv" in truth:
            tb, ty = truth["uv"]
            ax_u.plot(tb, np.asarray(ty) * f, color="#1e8449", lw=2, alpha=0.8, label="expected")
    ax_u.axhline(0, color="0.3", lw=0.6)
    ax_u.set_xscale("log")
    ax_u.set_xlabel(r"baseline [k$\lambda$]")
    ax_u.set_ylabel(f"line flux [{ulabel} km/s]")
    ax_u.set_title("line flux in every annulus, and the source models", fontsize=9)
    ax_u.legend(fontsize=7.5, frameon=False)

    # ---- the fluxes ---------------------------------------------------------
    rows = []
    for i, (label, r) in enumerate(items):
        blocks = r.summary.get("lines") or [r.summary["line"]]
        for k, b in enumerate(blocks):
            rows.append((label + (f", line {k + 1}" if len(blocks) > 1 else ""), i, k, b))
    y = np.arange(len(rows))[::-1]
    for yi, (label, i, k, b) in zip(y, rows):
        c = _COLOURS[i % len(_COLOURS)]
        ax_f.errorbar(b["flux"] * f, yi, xerr=b["flux_error"] * f, fmt="o", color=c, ms=6,
                      capsize=3, lw=1.4)
        ax_f.plot(b["flux_gauss"] * f, yi - 0.22, marker="s", mfc="none", mec=c, ms=5, ls="none")
        ax_f.plot(b["flux_area"] * f, yi - 0.22, marker="^", mfc="none", mec=c, ms=5, ls="none")
    tf = truth.get("flux")
    if tf is not None:
        for t in np.atleast_1d(tf):
            ax_f.axvline(t * f, color="#1e8449", ls="--", lw=1.8)
    ax_f.plot([], [], "o", color="0.3", label="flux quoted")
    ax_f.plot([], [], "s", mfc="none", mec="0.3", ls="none", label="Gaussian fit")
    ax_f.plot([], [], "^", mfc="none", mec="0.3", ls="none", label="area in the window")
    if tf is not None:
        ax_f.plot([], [], color="#1e8449", ls="--", label="expected")
    ax_f.set_yticks(y)
    ax_f.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax_f.set_ylim(-0.8, len(rows) - 0.4)
    ax_f.set_xlabel(f"line flux [{ulabel} km/s]")
    ax_f.legend(fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                ncol=4, handletextpad=0.3, columnspacing=1.2)

    if table:
        print(_table(items, truth))
    return display(fig, save, show)


def _table(items, truth):
    """The numbers of every measurement, one row per line."""
    width = max([len(label) + 4 for label, _ in items] + [8])
    head = (f"{'':{width}s} {'flux':>8s} {'± err':>7s} {'quoted':>7s} {'Gauss':>8s} "
            f"{'area':>8s} {'v0':>6s} {'FWHM':>5s} {'size':>6s} {'S/N':>5s} {'signif.':>7s}"
            f"  continuum")
    out = [head]
    tf = truth.get("flux")
    for label, r in items:
        s = r.summary
        cf = s.get("continuum_fit")
        rc = s.get("continuum_residual") or {}
        cont = (f"fitted {cf['flux_density']:.3g} ± {cf['flux_density_error']:.2g}" if cf
                else ("residual FLAGGED" if rc.get("detected") else ""))
        for k, b in enumerate(s.get("lines") or [s["line"]]):
            th = b.get("theta_fwhm_arcsec", s.get("theta_fwhm_arcsec")) or 0.0
            d = b["detection"]
            sig = d.get("significance_sigma", np.nan)
            name = label + (f" ({k + 1})" if len(s.get("lines") or []) > 1 else "")
            out.append(f"{name:{width}s} {b['flux']:8.4g} {b['flux_error']:7.2g} "
                       f"{b['flux_method']:>7s} {b['flux_gauss']:8.4g} {b['flux_area']:8.4g} "
                       f"{b['v0_kms']:+6.0f} {b['fwhm_kms']:5.0f} "
                       f"{(f'{th:.2f}' if th else 'point'):>6s} {d['snr']:5.1f} "
                       + (f"{sig:6.1f}σ" if np.isfinite(sig) else f"{'':7s}")
                       + (f"  {cont}" if k == 0 else ""))
    if tf is not None:
        out.append(f"{'expected':{width}s} " + ", ".join(f"{t:.4g}" for t in np.atleast_1d(tf))
                   + (f"   continuum {truth['continuum']:.3g}" if truth.get("continuum") else ""))
    return "\n".join(out)


__all__ = ["plot_line_result", "plot_profile", "plot_uv", "plot_noise", "compare", "display"]
