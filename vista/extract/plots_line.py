"""
vista.extract.plots_line
========================
Figures of :func:`vista.extract.measure_line`:

``<prefix>_profile.png``
    The line profile: real part of the flux density at zero spacing against
    velocity, the fitted lines, the two fluxes of every line (area of the
    Gaussian, and of the spectrum in the window), the noise (+-1 sigma band),
    the continuum, and the residuals in units of sigma.
``<prefix>_uv.png``
    Line flux of every annulus against baseline length, real and imaginary
    part, with the point and Gaussian models.
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


# Sizes of the figures: large enough to read on a slide or in a notebook.
_FONTS = {"font.size": 15, "axes.labelsize": 16, "axes.titlesize": 14, "legend.fontsize": 13,
          "xtick.labelsize": 14, "ytick.labelsize": 14}
_ORANGE = "#c8650a"            # the text of what belongs to the window


def _window_label(lo, hi):
    return (f"Window ±{hi:.0f} km/s" if abs(lo + hi) < 1e-6 * max(abs(hi), 1)
            else f"Window {lo:+.0f}, {hi:+.0f} km/s")


def _continuum_label(s, f, ulabel, continuum=None):
    """The continuum in the legend: its value and what was done with it."""
    rc = s.get("continuum_residual") or {}
    cf = s.get("continuum_fit")
    if cf:
        return f"Cont. {cf['flux_density'] * f:.2g} {ulabel} (fitted)"
    if rc.get("detected"):
        return f"Cont. {rc['offset'] * f:.2g} {ulabel} (left in the data)"
    if rc.get("continuum_subtracted"):
        if continuum is not None:
            c = continuum.summary
            if c["flux_total"] > 2 * c["flux_total_error"]:
                return f"Cont. {c['flux_total'] * f:.2g} {ulabel} (subtracted)"
        return "Cont. subtracted"
    return "Cont. none found" if rc else "Cont. not measured"


def plot_profile(result, line: Optional[Lines] = None, path: Optional[str] = None,
                 unit: Optional[str] = None, v_limits=None, rebin: int = 1, continuum=None,
                 show: str = "line"):
    """The stacked spectrum.

    ``show="line"``: the line as it was measured, on its own after the
    continuum subtraction or above the continuum fitted with it.
    ``show="continuum"``: the channels outside the line windows, with the
    continuum measured on its own put back, and its level.

    The line: the real part of the flux density at zero spacing
    against velocity (black), its noise (grey band), the fitted Gaussian
    (red), the window over which the spectrum is summed (orange) and the
    level of the continuum (dashed).  On the left the two fluxes of every
    line, the one quoted in bold: ``F_fit``, the area of the Gaussian, with
    its centre, width and peak, and ``F_window``, the area under the spectrum
    in the window.  Below: the residuals of the fit in units of the noise.

    ``continuum``: the continuum measured on its own (the result of
    ``measure_continuum``).  Without ``path`` the figure is returned."""
    plt = _pyplot()
    s = result.summary
    f, ulabel = _scale(unit, _typical(result))
    v = result.velocity_kms
    order = np.argsort(v)
    corr = result.channel_correlation
    rho = None if corr is None else corr[np.ix_(order, order)]
    v, y, e = v[order], result.profile[order] * f, result.profile_error[order] * f
    used = result.fitted[order]
    vb, yb, eb = _rebin(v, np.where(used, y, np.nan), e, rho, rebin)
    dvb = abs(vb[1] - vb[0]) if len(vb) > 1 else 1.0
    blocks = s.get("lines") or [s["line"]]
    several = len(blocks) > 1
    cf = s.get("continuum_fit")
    rc = s.get("continuum_residual") or {}
    # the level of the continuum under the line: fitted with it, or (asked
    # with with_continuum) measured on its own and put back for the drawing
    if show not in ("line", "continuum"):
        raise ValueError(f"show: 'line' or 'continuum', not {show!r}")
    only_cont = show == "continuum"
    what, note = "lines" if several else "line", ""
    level = 0.0
    if cf:
        level, what = cf["flux_density"] * f, what + " + cont"
    elif only_cont:
        if continuum is None:
            raise ValueError("no continuum measured on its own to draw")
        level = continuum.summary["flux_total"] * f
        yb = yb + level
    elif rc.get("continuum_subtracted"):
        note = "contsub"
    if only_cont:
        # the channels outside the line windows only
        what = "cont"
        for b in blocks:
            lo, hi = b.get("window_kms_line1_velocity") or s["window"]["user_kms"]
            yb = np.where((vb >= lo) & (vb <= hi), np.nan, yb)

    with plt.rc_context(_FONTS):
        fig, (ax, axr, axn) = plt.subplots(
            3, 1, figsize=(9, 8.6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 0.8], "hspace": 0.06})
        # the windows of the area fluxes, one per line
        for k, b in enumerate(blocks):
            lo, hi = b.get("window_kms_line1_velocity") or s["window"]["user_kms"]
            own = b.get("window_kms") or (lo, hi)
            ax.axvspan(lo, hi, color="C1", alpha=0.15, lw=0,
                       label=("Line window (left out)" if k == 0 else None) if only_cont
                       else _window_label(*own) if not several
                       else ("Windows" if k == 0 else None))
        ax.fill_between(vb, level - eb, level + eb, step="mid", color="0.6", alpha=0.4, lw=0,
                        label=r"Noise $\pm1\sigma$")
        ax.step(vb, yb, where="mid", color="k", lw=1.4,
                label="Line-free channels" if only_cont else "Stacked spectrum")
        # the model averaged over the channels, as it was fitted
        _, mb, _ = _rebin(v, np.where(used, result.profile_model[order] * f, np.nan), e, rho,
                          rebin)
        if not cf:
            mb = mb + level
        if only_cont:
            mb = np.full_like(mb, level)
        # the fitted Gaussians, drawn on a fine grid
        vf = np.linspace(v[used].min(), v[used].max(), 1500)
        mf = np.full_like(vf, level)
        for b in blocks:
            shift = ((b["window_kms_line1_velocity"][0] - b["window_kms"][0])
                     if b.get("window_kms_line1_velocity") and b.get("window_kms") else 0.0)
            mf += b["peak"] * f * np.exp(-4 * np.log(2) * ((vf - shift - b["v0_kms"])
                                                          / b["fwhm_kms"]) ** 2)
        if not only_cont:
            ax.plot(vf, mf, color="C3", lw=2.4, label="Fit")
        # the continuum: its level under the line, or what is left of it
        if level:
            ax.axhline(level, color="C0", lw=2.4 if only_cont else 1.5,
                       ls="-" if only_cont else "--",
                       label="Cont. model" if only_cont else f"Cont. {level:.2g} {ulabel}")
        elif rc.get("detected"):
            vv = np.linspace(*rc["velocity_range_kms"], 50)
            ax.plot(vv, (rc["offset"] + rc["slope_per_1000kms"] * vv / 1000) * f,
                    color="C0", lw=1.5, ls="--",
                    label=f"Cont. left: {rc['offset'] * f:.2g} {ulabel}")
        ax.axhline(0, color="0.3", lw=0.6)

        # the numbers: per line, F_fit (red), the Gaussian, F_window (orange)
        rows = []                                  # (text, colour, bold, size)
        if only_cont:
            c = cf or {}
            value, error = ((c["flux_density"], c["flux_density_error"]) if cf else
                            (continuum.summary["flux_total"],
                             continuum.summary["flux_total_error"]))
            rows.append((rf"$F_{{\rm cont}}$ = {value * f:.3g} ± {error * f:.2g} {ulabel}",
                         "C0", True, 15))
            rows.append((f"S/N = {value / error:.1f}", "k", True, 15))
        for k, b in enumerate([] if only_cont else blocks):
            area = b.get("flux_method") == "area"
            head = f"Line {k + 1}:  " if several else ""
            rows.append((head + rf"$F_{{\rm window}}$ = {b['flux_area'] * f:.3g} ± "
                         rf"{b['flux_area_error'] * f:.2g} {ulabel} km/s", _ORANGE, area, 15))
            rows.append((rf"$F_{{\rm fit}}$ = {b['flux_gauss'] * f:.3g} ± "
                         rf"{b['flux_gauss_error'] * f:.2g} {ulabel} km/s", "C3", not area, 15))
            shape = (rf"$\mu$ = {b['v0_kms']:+.0f} km/s", f"FWHM = {b['fwhm_kms']:.0f} km/s",
                     f"peak = {b['peak'] * f:.3g} {ulabel}")
            for t in ([",  ".join(shape)] if several else shape):
                rows.append((t, "k", False, 13))
            rows.append((f"S/N = {b['detection']['snr']:.1f}", "k", True, 15))
        step = 0.078
        for i, (t, colour, bold, size) in enumerate(rows):
            ax.text(0.03, 0.95 - i * step, t, transform=ax.transAxes, va="top", fontsize=size,
                    color=colour, weight="bold" if bold else "normal")
        ax.set_ylabel(f"Re flux density [{ulabel}]")
        # room above the spectrum for the numbers and the legend
        lo_y, hi_y = min(np.nanmin(yb - eb), 0.0), max(np.nanmax(yb), np.nanmax(mb))
        if only_cont:
            hi_y = max(hi_y, np.nanmax(yb + eb))
        text = 0.07 + len(rows) * step
        span = (hi_y - lo_y) / max(1.0 - text - 0.08, 0.2)
        ax.set_ylim(lo_y - 0.08 * span, lo_y + 0.92 * span)
        ax.legend(loc="upper right", frameon=False)
        ax.set_title(_title(what, note) + (f", {rebin} channels per bin" if rebin > 1 else ""))
        # residuals of the fit, in units of the noise
        with np.errstate(invalid="ignore", divide="ignore"):
            axr.step(vb, (yb - mb) / eb, where="mid", color="k", lw=1.2)
        axr.axhspan(-1, 1, color="0.6", alpha=0.3, lw=0)
        axr.axhline(0, color="0.3", lw=0.6)
        axr.set_ylabel(r"residual / $\sigma$", fontsize=13)
        # how many sources have data at every velocity
        ns = np.asarray(result.sources_per_channel, float)[order]
        axn.step(v, np.where(used, ns, np.nan), where="mid", color="k", lw=1.4)
        axn.set_ylim(0, 1.15 * max(np.nanmax(ns), 1))
        axn.set_ylabel("sources", fontsize=13)
        axn.set_xlabel("velocity [km/s]")
        if v_limits:
            axn.set_xlim(*v_limits)
        else:
            axn.set_xlim(v[used].min() - dvb, v[used].max() + dvb)
    return _finish(fig, path)


def _size_text(model, theta):
    """'Gaussian, FWHM 0.46"' or 'point source'."""
    if not theta:
        return "point source"
    return f"Gaussian, FWHM {theta:.2f}\"" + (" (fixed)" if model == "fixed" else "")


_GREEN = "#1e8449"             # the point-source model (blue is the continuum)


def _title(what, note=""):
    """'Stacked <what>' with what is stacked in bold."""
    bold = " + ".join(rf"$\bf{{{w.strip()}}}$" for w in what.split("+"))
    return f"Stacked {bold}" + (f" ({note})" if note else "")


def plot_uv(result, path: Optional[str] = None, unit: Optional[str] = None, continuum=None,
            show: str = "line"):
    """What is measured in every annulus of baseline length, real part
    (points) and imaginary part (grey crosses, noise only for a source at the
    centre), with the two source models: Gaussian (red) and point (green,
    dashed); the one used for the flux in bold.  The flux of a model is where
    it meets zero baseline.

    ``show``: ``"line"``, the line flux; ``"continuum"``, the flux density of
    the continuum measured on its own (``continuum``: the result of
    ``measure_continuum``).  Without ``path`` the figure is returned."""
    if show not in ("line", "continuum"):
        raise ValueError(f"show: 'line' or 'continuum', not {show!r}")
    if show == "continuum" and continuum is None:
        raise ValueError("no continuum measured on its own to draw")
    if result is None:                    # the continuum alone
        show = "continuum"
        f, ulabel = _scale(unit, continuum.summary["flux_total_error"])
    else:
        f, ulabel = _scale(unit, _typical(result))
    if show == "line":
        return _uv_figure(*_uv_line(result, f, ulabel), path)
    return _uv_figure(*_uv_continuum(continuum, f, ulabel), path)


def _uv_line(result, f, ulabel):
    """What plot_uv draws for the line: data, models, labels."""
    s = result.summary
    p = s["line"]
    st = s.get("size_test") or {}
    several = len(s.get("lines") or []) > 1
    note = ""
    b = result.baseline_klambda
    ok = result.annulus_ok & np.isfinite(result.annulus_flux)
    y, e = result.annulus_flux, result.annulus_flux_error
    imag = getattr(result, "annulus_flux_imaginary", None)
    if st.get("weight_gauss") is not None:
        gauss = (st["flux_gauss"], st.get("theta_gauss_arcsec") or 0.0)
        point = st["flux_point"]
        winner = "Gauss" if st["adopted"] == "gauss" else "Point"
    elif p["theta_fwhm_arcsec"]:
        gauss, point, winner = (p["flux_gauss"], p["theta_fwhm_arcsec"]), None, "Gauss"
    else:
        gauss, point, winner = None, p["flux_gauss"], "Point"
    if s.get("size_from") == "continuum":
        note = " (size from cont)"
        if gauss is not None:                 # the point source of its own test, to compare
            point = (s.get("size_own") or {}).get("flux_point")
    title = _title("line" + (" 1" if several else ""),
                   "contsub" if (s.get("continuum_residual") or {}).get("continuum_subtracted")
                   else "")
    return (b, ok, y, e, imag, gauss, point, winner, note, f, f"{ulabel} km/s",
            "Re line flux", title)


def _uv_continuum(continuum, f, ulabel):
    """What plot_uv draws for the continuum measured on its own."""
    c = continuum.summary
    ct = c.get("size_test") or {}
    b, y, e = continuum.baseline_klambda, continuum.flux, continuum.flux_error
    ok, imag = continuum.bin_ok & np.isfinite(y), continuum.flux_imaginary
    theta = c.get("theta_fwhm_arcsec") or 0.0
    how, note = ct.get("adopted"), ""
    if how in ("line", "fixed") and theta:          # the Gaussian used is not its own
        gauss, point, winner = (c["flux_total"], theta), ct.get("flux_point"), "Gauss"
        note = " (size from line)" if how == "line" else " (size given)"
    elif ct:
        gauss = (ct["flux_gauss"], ct["theta_gauss_arcsec"] or 0.0)
        point = c["flux_total"] if not theta else ct.get("flux_point")
        winner = "Gauss" if theta else "Point"
    elif theta:
        gauss, point, winner = (c["flux_total"], theta), None, "Gauss"
    else:
        gauss, point, winner = None, c["flux_total"], "Point"
    return (b, ok, y, e, imag, gauss, point, winner, note, f, ulabel,
            "Re continuum flux density", _title("cont"))


def _uv_figure(b, ok, y, e, imag, gauss, point, winner, note, f, unit_f, ylabel, title, path):
    plt = _pyplot()
    with plt.rc_context(_FONTS):
        fig, ax = plt.subplots(figsize=(8, 5.6))
        ax.errorbar(b[ok], y[ok] * f, e[ok] * f, fmt="o", color="k", ms=7, capsize=3, zorder=3,
                    label="Stacked data")
        bb = np.geomspace(max(b[ok].min() * 0.6, 0.1), b[ok].max() * 1.4, 300)
        if gauss is not None:
            ax.plot(bb, gauss[0] * f * uv_gaussian(bb * 1e3, gauss[1]), color="C3", lw=2.6,
                    label=f"Gauss: {gauss[1]:.2f}\", {gauss[0] * f:.3g} {unit_f}{note}")
        if point is not None:
            ax.plot(bb, np.full_like(bb, point * f), color=_GREEN, lw=2.0, ls="--",
                    label=f"Point: {point * f:.3g} {unit_f}"
                          + (note if gauss is None else ""))
        if imag is not None:
            ax.plot(b[ok], np.asarray(imag)[ok] * f, "x", color="0.5", ms=8, mew=1.0, label="Im")
        ax.axhline(0, color="0.3", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel(r"baseline length [k$\lambda$]")
        ax.set_ylabel(f"{ylabel} [{unit_f}]")
        handles, labels = ax.get_legend_handles_labels()
        rank = {"Stacked": 0, "Gauss:": 1, "Point:": 2, "Im": 3}
        keep = sorted(range(len(labels)), key=lambda i: rank[labels[i].split()[0]])
        legend = ax.legend([handles[i] for i in keep], [labels[i] for i in keep], frameon=False)
        for t in legend.get_texts():                 # the model used in bold
            if t.get_text().startswith(winner):
                t.set_weight("bold")
        ax.set_title(title)
    return _finish(fig, path)


def plot_noise(result, path: Optional[str] = None, unit: Optional[str] = None):
    """The noise checks.  Without ``path`` the figure is returned."""
    plt = _pyplot()
    s = result.summary
    n = s["noise"]
    f, ulabel = _scale(unit, _typical(result))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax in axes:
        ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
    ax = axes[0]
    v = result.velocity_kms
    o = np.argsort(v)
    ax.step(v[o], result.noise_channel[o] * f, where="mid", color="k",
            label="from the scatter")
    ax.step(v[o], result.noise_channel_weights[o] * f, where="mid", color="C0", ls="--",
            label="from the weights")
    ax.set_xlabel("velocity [km/s]")
    ax.set_ylabel(f"$\\sigma$ per channel [{ulabel}]")
    ax.set_title(f"$\\kappa$ = scatter / weights = {n['kappa_scatter_over_weights']:.3f}",
                 fontsize=13)
    ax.legend(fontsize=11, frameon=False)

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
                 + ", ".join(f"{r:+.2f}" for r in rho[:2]), fontsize=13)
    ax.legend(fontsize=11, frameon=False)

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
    ax.set_title(f"significance {d['significance_sigma']:.1f}$\\sigma$", fontsize=13)
    ax.legend(fontsize=11, frameon=False)
    fig.tight_layout()
    return _finish(fig, path)


def plot_line_result(result, line: Lines, prefix: str, unit: Optional[str] = None,
                     v_limits=None, rebin: int = 1, verbose=True, continuum=None) -> List[str]:
    """The figures of a measurement, written as ``<prefix>_*.png``: the line
    (``profile``, ``uv``), the noise, and the continuum when it was measured
    on its own (``continuum``, ``continuum_uv``).  Returns their paths."""
    out = []
    if result is not None:
        out += [plot_profile(result, line, f"{prefix}_profile.png", unit, v_limits, rebin,
                             continuum),
                plot_uv(result, f"{prefix}_uv.png", unit),
                plot_noise(result, f"{prefix}_noise.png", unit)]
    if continuum is not None and result is not None:
        out.append(plot_profile(result, line, f"{prefix}_continuum.png", unit, v_limits, rebin,
                                continuum, show="continuum"))
    if continuum is not None:
        out.append(plot_uv(result, f"{prefix}_continuum_uv.png", unit, continuum, "continuum"))
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


def _compare(results, expected: Optional[dict] = None, unit: Optional[str] = "Jy",
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
    fig = plt.figure(figsize=(16, 11), layout="constrained")
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
    ax_p.set_ylabel(f"Re flux density [{ulabel}]")
    ax_p.set_title("Stacked spectra (thin) and their fits (dashed)")
    ax_p.legend(frameon=False, loc="upper right")

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
    ax_u.set_ylabel(f"Re line flux [{ulabel} km/s]")
    ax_u.set_title("Against the baseline length")
    ax_u.legend(frameon=False)

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
    ax_f.set_yticklabels([r[0] for r in rows])
    ax_f.set_ylim(-0.8, len(rows) - 0.4)
    ax_f.set_xlabel(f"line flux [{ulabel} km/s]")
    ax_f.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                ncol=4, handletextpad=0.3, columnspacing=1.2)

    if table:
        print(_table(items, truth))
    return display(fig, save, show)


def compare(results, expected: Optional[dict] = None, unit: Optional[str] = "Jy",
            rebin: int = 1, v_limits=None, save: Optional[str] = None, show: bool = True,
            table: bool = True):
    with _pyplot().rc_context(_FONTS):
        return _compare(results, expected, unit, rebin, v_limits, save, show, table)


compare.__doc__ = _compare.__doc__


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
