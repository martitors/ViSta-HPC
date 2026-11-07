#!/usr/bin/env python
"""
plot_line.py -- the figures of a ViSta line measurement, from its output files
==============================================================================

``measure_line`` (``python -m vista.extract flux ... --method joint``, or
``run_vista.sh``) writes, for an output prefix ``P``:

=====================  ==========================================================
``P_results.json``     every number: line, detection, window, size test, noise
``P_profile.txt``      one row per channel (see PROFILE_COLUMNS below)
``P_uvamp.txt``        one row per uv annulus (see UVAMP_COLUMNS below)
``P_noise.txt``        noise of the profile per channel, from the scatter of the
                       visibilities and from the weights
``P_diagnostics.npz``  channel correlation matrix, standardised noise samples,
                       null distributions of the detection S/N
=====================  ==========================================================

This script makes the three figures from those files alone: it needs only
``numpy`` and ``matplotlib`` (not ViSta), so it can be copied anywhere and
changed freely.  Everything about the look is in ``STYLE`` (or in a
matplotlib style file, ``--style``); every figure is one function, with
comments on what each element is and how it was computed.

Usage::

    python plot_line.py P                         # P_profile.png, P_uv.png, P_noise.png
    python plot_line.py P --unit Jy --rebin 2 --vlim -800 800 --format pdf
    python plot_line.py P --style thesis.mplstyle --only profile

In Python::

    import plot_line as pl
    pl.STYLE["model"]["color"] = "C2"     # any change to STYLE, before plotting
    pl.apply_style()                      # or pl.apply_style("thesis.mplstyle")
    d = pl.load("P")
    fig = pl.plot_profile(d, unit="Jy", rebin=2)
    fig.savefig("my_profile.pdf")
"""

import argparse
import json
import math

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# STYLE: change here (or pass --style file.mplstyle, applied on top)
# =============================================================================
STYLE = {
    # figure sizes, inches
    "figsize_profile": (7.0, 5.6),
    "figsize_uv": (6.2, 4.2),
    "figsize_noise": (13.0, 3.8),
    "height_ratios": (3.2, 1.0),           # profile : residuals
    "dpi": 200,
    # matplotlib defaults
    "rc": {
        "font.size": 10,
        "font.family": "serif",            # "sans-serif", ...
        "mathtext.fontset": "dejavuserif",  # "cm" for Computer Modern
        "text.usetex": False,              # True needs a LaTeX installation
        "axes.linewidth": 0.8,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "legend.frameon": False, "legend.fontsize": 8,
    },
    # colours and lines
    "data": dict(color="k", lw=1.2),                       # the stacked profile
    "model": dict(color="#c0392b", lw=1.6),                # the fit
    "imaginary": dict(color="0.45", lw=0.8, ls=":"),       # null test
    "noise_band": dict(color="0.65", alpha=0.35, lw=0),    # +-1 sigma around 0
    "line_window": dict(color="#e67e22", alpha=0.10, lw=0),
    "zero": dict(color="0.3", lw=0.6),
    "points": dict(fmt="o", color="k", ms=4, capsize=2, elinewidth=1),
    "point_model": dict(color="#2471a3"),
    "gauss_model": dict(color="#c0392b"),
    "total_flux": dict(color="k", ls=":", lw=1),
    "reference": dict(color="#1e8449", ls="--", lw=1.6),   # e.g. the true signal of a simulation
    "hist_null": dict(color="0.75"),
    "hist_imag": dict(color="#2471a3"),
    "hist_free": dict(color="#c0392b"),
    # text
    "milli": "auto",           # data in Jy: show mJy? "auto", True or False
    "show_textbox": True,
    "headroom": 0.45,          # empty space above the profile, for legend and text (fraction)
    "uv_clip_errors": 3.0,     # y range of the uv plot ignores points with errors above
                               # this many times the median error (None: show all)
    "textbox": dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9),
    "textbox_fontsize": 8.5,
    "velocity_label": "velocity [km s$^{-1}$]",
    "baseline_label": r"baseline [k$\lambda$]",
}

# columns of the text files (the header of each file says the same)
PROFILE_COLUMNS = [
    "v_kms",          # channel velocity w.r.t. the rest frequency (radio convention)
    "freq_GHz",       # rest-frame frequency of the channel
    "flux_density",   # flux density at zero spacing (see below)
    "err_thermal",    # its error, from the noise of the visibilities (+ the size model)
    "err_bootstrap",  # its error, from resampling the sources
    "imaginary",      # the same estimate on the imaginary part: the null test
    "model",          # the fitted line, averaged over the channel, at zero spacing
    "fitted",         # 1: channel used in the fit
    "line_free",      # 1: channel outside the line window(s)
    "n_sources",      # sources with data in the channel
]
UVAMP_COLUMNS = [
    "b_klambda",      # mean baseline of the annulus, rest frame
    "line_flux",      # line flux of the annulus, at the fitted line shape
    "err_thermal",    # its error
    "used",           # 1: annulus used in the fit
    "n_sources",      # sources with data in the annulus
]
# How flux_density is computed: in every channel the visibilities of each
# annulus are averaged (real part, phase centre on the sources), then the
# annuli are combined as  S = sum_b g_b y_b / var_b / sum_b g_b^2 / var_b,
# where g_b is the visibility of the fitted source shape (1 for a point
# source): the flux density the source would show at baseline zero.  Its
# integral over velocity is the line flux.


def apply_style(style_file=None):
    """STYLE["rc"] into matplotlib, then a style file on top, if given."""
    plt.rcParams.update(STYLE["rc"])
    if style_file:
        plt.style.use(style_file)


# =============================================================================
# reading
# =============================================================================
def _table(path, columns):
    a = np.atleast_2d(np.loadtxt(path))
    return {c: a[:, i] for i, c in enumerate(columns[:a.shape[1]])}


def load(prefix):
    """Everything the figures need, as a dict."""
    d = {"prefix": prefix}
    with open(f"{prefix}_results.json") as fh:
        d["res"] = json.load(fh)
    d["prof"] = _table(f"{prefix}_profile.txt", PROFILE_COLUMNS)
    d["uv"] = _table(f"{prefix}_uvamp.txt", UVAMP_COLUMNS)
    d["noise"] = _table(f"{prefix}_noise.txt", ["v_kms", "sigma_scatter", "sigma_weights"])
    try:
        z = np.load(f"{prefix}_diagnostics.npz")
        d["diag"] = {k: z[k] for k in z.files}
    except FileNotFoundError:             # outputs of older versions
        d["diag"] = {}
    return d


# =============================================================================
# helpers
# =============================================================================
def flux_scale(unit, d=None):
    """Factor and label of the flux axes.  With data in Jy, STYLE["milli"]
    = "auto" shows mJy when the line peak and the noise of a channel are
    both below 0.1 Jy; True / False force mJy / Jy.  Other units are shown
    as they are."""
    if not (unit and unit.lower() == "jy"):
        return 1.0, unit or "data units"
    milli = STYLE["milli"]
    if milli == "auto":
        typical = 0.0
        if d is not None:
            typical = max(abs(_num(d["res"]["line"].get("peak"))),
                          float(np.nanmedian(d["prof"]["err_thermal"])))
        milli = not typical >= 0.1
    return (1e3, "mJy") if milli else (1.0, "Jy")


def rebin_channels(y, err, corr, n):
    """Average groups of ``n`` channels.

    The error of a mean of correlated channels is not err / sqrt(n): with the
    correlation matrix ``R`` of the noise (from ``_diagnostics.npz``; Hanning
    smoothing, rebinning and the continuum subtraction correlate channels)
    var(mean) = (1 / n^2) sum_ij R_ij err_i err_j.
    """
    if n <= 1:
        return y, err
    m = len(y) // n
    yb, eb = np.full(m, np.nan), np.full(m, np.nan)
    for k in range(m):
        idx = np.arange(k * n, (k + 1) * n)
        idx = idx[np.isfinite(y[idx])]          # channels not fitted are NaN
        if idx.size == 0:
            continue
        r = np.eye(idx.size) if corr is None or corr.size == 0 else corr[np.ix_(idx, idx)]
        yb[k] = y[idx].mean()
        eb[k] = math.sqrt(np.sum(r * np.outer(err[idx], err[idx]))) / idx.size
    return yb, eb


def pm(value, error):
    """'value +- error', error to two significant digits."""
    if not (np.isfinite(value) and np.isfinite(error)) or error <= 0:
        return f"{value:.3g}"
    digits = max(0, 1 - int(math.floor(math.log10(abs(error)))))
    return f"{value:.{digits}f} $\\pm$ {error:.{digits}f}"


def _has(dg, key):
    return dg.get(key) is not None and np.size(dg[key]) > 0


def _num(x):
    """None (a quantity not fitted, e.g. v0 in detect mode) -> nan."""
    return float("nan") if x is None else float(x)


def significance_text(det):
    """How significant the line is.  detect: S/N of the matched filter at the
    given v0 and FWHM (one trial per source size scanned).  search: the
    probability that noise alone gives a peak as high anywhere in the scanned
    range of v0, FWHM and size, from null realisations of the noise."""
    s = _num(det.get("significance_sigma"))
    if np.isinf(s):
        n = det.get("n_null") or 0
        return (f"S/N {det['snr']:.1f}" + (f" (p < 1/{n} from {n} null runs)" if n else ""))
    if det["mode"] == "detect" and not det.get("n_null"):
        return f"S/N {det['snr']:.1f} ({s:.1f}$\\sigma$)"
    return f"S/N {det['snr']:.1f}: {s:.1f}$\\sigma$ (p = {det['p_value']:.1g})"


def summary_text(res, f, ulab):
    """The text box of the profile: line flux, peak, centre, width, size,
    detection.  Errors are those of results.json (the larger of thermal and
    bootstrap; v0 and FWHM only when they were fitted)."""
    ln = res["line"]
    v0 = f"{ln['v0_kms']:+.0f}"
    if ln.get("v0_error_kms"):
        v0 += f" $\\pm$ {ln['v0_error_kms']:.0f}"
    fw = f"{ln['fwhm_kms']:.0f}"
    if ln.get("fwhm_error_kms"):
        fw += f" $\\pm$ {ln['fwhm_error_kms']:.0f}"
    rows = [f"F = {pm(ln['flux'] * f, ln['flux_error'] * f)} {ulab} km s$^{{-1}}$",
            f"peak = {pm(ln['peak'] * f, _num(ln.get('peak_error')) * f)} {ulab}",
            f"v$_0$ = {v0} km s$^{{-1}}$",
            f"FWHM = {fw} km s$^{{-1}}$"]
    if ln.get("theta_fwhm_arcsec"):
        # the larger of the thermal and the bootstrap error is at the top level
        te = res.get("theta_error_arcsec", ln.get("theta_error_arcsec"))
        rows.append(f"size = {pm(ln['theta_fwhm_arcsec'], _num(te))}\"")
    rows.append(significance_text(res["detection"]))
    return "\n".join(rows)


def uv_gaussian(b_klambda, theta_arcsec):
    """Visibility of a circular Gaussian of unit flux and FWHM theta."""
    t = math.radians(theta_arcsec / 3600.0)
    return np.exp(-(math.pi * t * np.asarray(b_klambda) * 1e3) ** 2 / (4 * math.log(2)))


# =============================================================================
# figure 1: the line profile
# =============================================================================
def plot_profile(d, unit=None, rebin=1, vlim=None, rest_axis=True, reference=None,
                 reference_label="true signal"):
    """Flux density at zero spacing against velocity, with the fit, the
    noise band, the imaginary part, the line window, and the residuals.

    ``reference``: an optional curve to compare with, (velocity [km/s],
    flux density [data units]) -- the true signal of a simulation, the
    profile of another line, a model."""
    res, p = d["res"], d["prof"]
    f, ulab = flux_scale(unit, d)

    # channels in increasing velocity; only those used in the fit
    o = np.argsort(p["v_kms"])
    used = p["fitted"][o] == 1
    v = p["v_kms"][o]
    y = np.where(used, p["flux_density"][o], np.nan) * f
    e = p["err_thermal"][o] * f
    yi = np.where(used, p["imaginary"][o], np.nan) * f
    ym = np.where(used, p["model"][o], np.nan) * f
    corr = d["diag"].get("channel_correlation")
    if corr is not None and corr.size:
        corr = corr[np.ix_(o, o)]
    # rebinning (display only: the fit used every channel)
    vb = v if rebin <= 1 else v[:len(v) // rebin * rebin].reshape(-1, rebin).mean(1)
    yb, eb = rebin_channels(y, e, corr, rebin)
    yib, _ = rebin_channels(yi, e, corr, rebin)
    ymb, _ = rebin_channels(ym, e, corr, rebin)

    fig, (ax, axr) = plt.subplots(2, 1, figsize=STYLE["figsize_profile"], sharex=True,
                                  gridspec_kw={"height_ratios": STYLE["height_ratios"],
                                               "hspace": 0.05})
    # the line window: the channels left out of the continuum fit
    w0, w1 = res["window"]["user_kms"]
    ax.axvspan(w0, w1, label="line window", **STYLE["line_window"])
    # +-1 sigma of the noise of every (rebinned) channel, around zero
    ax.fill_between(vb, -eb, eb, step="mid", label=r"$\pm1\sigma$ noise", **STYLE["noise_band"])
    # the imaginary part through the same estimator: must be noise
    ax.step(vb, yib, where="mid", label="imaginary (null test)", **STYLE["imaginary"])
    # the stacked profile and the fit (both averaged over the channels)
    ax.step(vb, yb, where="mid", label="stacked flux density", **STYLE["data"])
    ax.step(vb, ymb, where="mid", label="fit", **STYLE["model"])
    if reference is not None:
        ax.plot(reference[0], np.asarray(reference[1]) * f, label=reference_label,
                **STYLE["reference"])
    ax.axhline(0, **STYLE["zero"])

    if STYLE["show_textbox"]:
        ax.text(0.02, 0.97, summary_text(res, f, ulab), transform=ax.transAxes, va="top",
                fontsize=STYLE["textbox_fontsize"], bbox=STYLE["textbox"])
    ax.set_ylabel(f"flux density [{ulab}]")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + STYLE["headroom"] * (hi - lo))
    ax.legend(loc="upper right")

    # residuals in units of the noise
    with np.errstate(invalid="ignore", divide="ignore"):
        axr.step(vb, (yb - ymb) / eb, where="mid", **STYLE["data"])
    axr.axhspan(-1, 1, **STYLE["noise_band"])
    axr.axhline(0, **STYLE["zero"])
    axr.set_ylabel(r"resid./$\sigma$")
    axr.set_xlabel(STYLE["velocity_label"])
    if vlim:
        axr.set_xlim(*vlim)
    else:
        dvb = abs(vb[1] - vb[0]) if len(vb) > 1 else 1.0
        axr.set_xlim(np.nanmin(np.where(used, v, np.nan)) - dvb,
                     np.nanmax(np.where(used, v, np.nan)) + dvb)

    if rest_axis:
        # rest frequency on top (radio convention: v = c (nu0 - nu) / nu0)
        nu0 = res["rest_freq_ghz"]
        c = 299792.458
        top = ax.secondary_xaxis("top", functions=(lambda vv: nu0 * (1 - vv / c),
                                                   lambda nu: c * (1 - nu / nu0)))
        top.set_xlabel("rest frequency [GHz]")
    return fig


# =============================================================================
# figure 2: amplitude against baseline
# =============================================================================
def plot_uv(d, unit=None, reference=None, reference_label="true signal"):
    """Line flux of every uv annulus against baseline length, with the point
    and the Gaussian source models.  A point source is flat; an extended one
    falls on the baselines that resolve it; the value at zero baseline is the
    total flux.

    ``reference``: an optional curve to compare with, (baseline [klambda],
    line flux [data units x km/s]), e.g. the true signal of a simulation."""
    res, u = d["res"], d["uv"]
    ln, st = res["line"], res.get("size_test") or {}
    f, ulab = flux_scale(unit, d)
    ok = (u["used"] == 1) & np.isfinite(u["line_flux"])
    b = u["b_klambda"]

    fig, ax = plt.subplots(figsize=STYLE["figsize_uv"])
    # each point: the line flux measured with the visibilities of one
    # annulus only; points with big errors (few visibilities, usually the
    # shortest and longest baselines) weigh little in the fit (1 / err^2)
    ax.errorbar(b[ok], u["line_flux"][ok] * f, u["err_thermal"][ok] * f,
                label="line flux per annulus", **STYLE["points"])
    bb = np.geomspace(max(b[ok].min() * 0.5, 0.1), b[ok].max() * 1.5, 300)
    if st.get("weight_gauss") is not None:
        # both size models, drawn thicker the more the data prefer them
        wg = st["weight_gauss"]
        ax.plot(bb, np.full_like(bb, st["flux_point"] * f), lw=0.8 + 1.6 * (1 - wg),
                label=f"point source (weight {1 - wg:.2f})", **STYLE["point_model"])
        if st.get("theta_gauss_arcsec"):
            ax.plot(bb, st["flux_gauss"] * f * uv_gaussian(bb, st["theta_gauss_arcsec"]),
                    lw=0.8 + 1.6 * wg,
                    label=f"Gaussian, FWHM {st['theta_gauss_arcsec']:.2f}\" (weight {wg:.2f})",
                    **STYLE["gauss_model"])
        ax.axhline(ln["flux"] * f, label="total flux (model average)", **STYLE["total_flux"])
    elif ln["theta_fwhm_arcsec"]:
        ax.plot(bb, ln["flux"] * f * uv_gaussian(bb, ln["theta_fwhm_arcsec"]),
                label=f"Gaussian, FWHM {ln['theta_fwhm_arcsec']:.2f}\"", **STYLE["gauss_model"])
    else:
        ax.plot(bb, np.full_like(bb, ln["flux"] * f), label="point source",
                **STYLE["point_model"])
    if reference is not None:
        ax.plot(reference[0], np.asarray(reference[1]) * f, label=reference_label,
                **STYLE["reference"])
    ax.axhline(0, **STYLE["zero"])
    if STYLE["uv_clip_errors"]:
        # the outermost annuli have few visibilities and huge errors: keep
        # them out of the y range (their error bars run off the axes)
        e = u["err_thermal"][ok]
        keep = e <= STYLE["uv_clip_errors"] * np.median(e)
        yy = np.concatenate([(u["line_flux"][ok] - e)[keep], (u["line_flux"][ok] + e)[keep],
                             [0.0, ln["flux"]]]) * f
        pad = 0.08 * (yy.max() - yy.min())
        ax.set_ylim(yy.min() - pad, yy.max() + pad)
    ax.set_xscale("log")
    ax.set_xlabel(STYLE["baseline_label"])
    ax.set_ylabel(f"line flux [{ulab} km s$^{{-1}}$]")
    ax.legend()
    return fig


# =============================================================================
# figure 3: the noise
# =============================================================================
def plot_noise(d, unit=None):
    """Three panels: noise per channel from the scatter of the visibilities
    and from the weights (their ratio is kappa); the stacked cells off the
    line, divided by their noise, against N(0, 1) (imaginary part, and real
    part in the line-free channels); the null distribution of the detection
    S/N with the measured value."""
    res, nz, dg = d["res"], d["noise"], d["diag"]
    n, det = res["noise"], res["detection"]
    f, ulab = flux_scale(unit, d)
    fig, axes = plt.subplots(1, 3, figsize=STYLE["figsize_noise"])

    ax = axes[0]
    o = np.argsort(nz["v_kms"])
    ax.step(nz["v_kms"][o], nz["sigma_scatter"][o] * f, where="mid",
            label="from the scatter of the visibilities", **STYLE["data"])
    ax.step(nz["v_kms"][o], nz["sigma_weights"][o] * f, where="mid", ls="--",
            label="from the weights", **STYLE["point_model"])
    ax.set_xlabel(STYLE["velocity_label"])
    ax.set_ylabel(f"$\\sigma$ per channel [{ulab}]")
    ax.set_title(f"$\\kappa$ = {n['kappa_scatter_over_weights']:.3f}", fontsize=9)
    ax.legend()

    ax = axes[1]
    x = np.linspace(-5, 5, 200)
    ax.plot(x, np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi), color="0.3", lw=1,
            label="N(0, 1)")
    bins = np.linspace(-5, 5, 41)
    if _has(dg, "z_imaginary"):
        # imaginary part of every (channel, annulus) cell over its noise: a
        # symmetric source has no imaginary signal, so this is pure noise
        ax.hist(np.clip(dg["z_imaginary"], -5, 5), bins=bins, density=True, histtype="step",
                label=f"imaginary, rms {_num(n.get('imaginary_rms_over_noise')):.2f}",
                **STYLE["hist_imag"])
    if _has(dg, "z_line_free"):
        # real part in the channels outside the line window
        ax.hist(np.clip(dg["z_line_free"], -5, 5), bins=bins, density=True, histtype="step",
                label=f"real, line-free, rms {_num(n.get('line_free_rms_over_noise')):.2f}",
                **STYLE["hist_free"])
    if not (_has(dg, "z_imaginary") or _has(dg, "z_line_free")):
        ax.text(0.5, 0.5, f"no {d['prefix'].split('/')[-1]}_diagnostics.npz\n"
                f"imaginary rms / noise {_num(n.get('imaginary_rms_over_noise')):.2f}\n"
                f"line-free rms / noise {_num(n.get('line_free_rms_over_noise')):.2f}",
                transform=ax.transAxes, ha="center", va="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel(r"cell / $\sigma$")
    ax.legend()

    ax = axes[2]
    if _has(dg, "null_snr"):
        # the highest S/N that the same search finds on noise realisations
        # with the measured noise and channel correlation
        ax.hist(dg["null_snr"], bins=40, density=True, histtype="stepfilled",
                label="same search on noise", **STYLE["hist_null"])
    if _has(dg, "null_snr_phase"):
        # the same search on the real data, with the phase of every source
        # rotated at random: the signal averages out, the systematics stay
        ax.hist(dg["null_snr_phase"], bins=30, density=True, histtype="step",
                label="random-phase stacks", **STYLE["hist_imag"])
    if not _has(dg, "null_snr"):
        # one template and no null realisations: S/N is N(0, 1) under noise
        x = np.linspace(-4, max(5, det["snr"] + 1), 400)
        ax.plot(x, np.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi), color="0.3",
                label="N(0, 1)" + ("" if det.get("n_null") else " (one template)"))
    ax.axvline(det["snr"], label=f"measured S/N {det['snr']:.1f}", **STYLE["model"])
    ax.set_xlabel("S/N")
    ax.set_title(significance_text(det), fontsize=9)
    ax.legend()
    fig.tight_layout()
    return fig


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix", help="output prefix of measure_line (P in P_results.json)")
    ap.add_argument("--unit", default=None, help="unit of the data, e.g. Jy (then mJy is shown)")
    ap.add_argument("--rebin", type=int, default=1, help="channels per bin in the profile")
    ap.add_argument("--vlim", nargs=2, type=float, default=None, metavar=("LO", "HI"))
    ap.add_argument("--format", default="png", help="png, pdf, svg, ...")
    ap.add_argument("--style", default=None, help="a matplotlib .mplstyle file")
    ap.add_argument("--only", choices=["profile", "uv", "noise"], default=None)
    ap.add_argument("--out", default=None, help="output prefix (default: the input prefix)")
    args = ap.parse_args()

    matplotlib.use("Agg")                  # files only, no window
    apply_style(args.style)
    d = load(args.prefix)
    out = args.out or args.prefix
    figures = {
        "profile": lambda: plot_profile(d, args.unit, args.rebin, args.vlim),
        "uv": lambda: plot_uv(d, args.unit),
        "noise": lambda: plot_noise(d, args.unit),
    }
    for name, make in figures.items():
        if args.only and name != args.only:
            continue
        fig = make()
        path = f"{out}_{name}.{args.format}"
        fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight")
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
