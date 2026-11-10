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


def plot_profile(result, line: Lines, path: str, unit: Optional[str] = None,
                 v_limits=None, rebin: int = 1):
    """The profile at zero spacing with the fitted lines, the two fluxes of
    every line (the area of the Gaussian, hatched, and the window over which
    the spectrum is summed) and the check of the residual continuum."""
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

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(8.6, 6.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.05})
    # the windows of the area fluxes, one per line
    for k, b in enumerate(blocks):
        lo, hi = b.get("window_kms_line1_velocity") or s["window"]["user_kms"]
        ax.axvspan(lo, hi, color="C1", alpha=0.08, lw=0,
                   label="window of the area flux" if k == 0 else None)
    ax.fill_between(vb, -eb, eb, step="mid", color="0.6", alpha=0.35, lw=0,
                    label=r"$\pm1\sigma$ noise")
    ax.step(vb, yib, where="mid", color="0.45", lw=0.8, ls=":", label="imaginary (null)")
    ax.step(vb, yb, where="mid", color="k", lw=1.2, label="stacked flux density")
    # the model averaged over the channels, as it was fitted
    _, mb, _ = _rebin(v, np.where(used, result.profile_model[order] * f, np.nan), e, rho, rebin)
    ax.step(vb, mb, where="mid", color="C3", lw=1.5, label="Gaussian fit")
    # the area of the Gaussian: above the continuum when it was fitted with the line
    cf = s.get("continuum_fit")
    base = cf["flux_density"] * f if cf else 0.0
    ax.fill_between(vb, base, np.where(np.isfinite(mb), mb, base), step="mid",
                    facecolor="none", edgecolor="C3", hatch="////", lw=0, alpha=0.5,
                    label="area of the Gaussian")
    if cf:
        ax.axhline(base, color="C0", lw=1.0, ls="-.",
                   label=f"continuum fitted with the line: {_pm(base, cf['flux_density_error'] * f)} {ulabel}")
    # residual continuum: offset and slope outside the windows
    rc = s.get("continuum_residual")
    if rc:
        vv = np.linspace(*rc["velocity_range_kms"], 50)
        c0 = (s.get("continuum_fit") or {}).get("flux_density", 0.0)   # on the fitted one
        ax.plot(vv, (c0 + rc["offset"] + rc["slope_per_1000kms"] * vv / 1000) * f,
                color="C2" if not rc["detected"] else "C1", lw=1.4, ls="--",
                label=("continuum outside the windows: "
                       + ("FLAGGED" if rc["detected"] else "none") + f" ({rc['offset_nsigma']:+.1f}"
                       r"$\sigma$)"))
    ax.axhline(0, color="0.3", lw=0.6)
    rows = []
    separate = len(s.get("line_groups") or [[1]]) > 1
    for k, b in enumerate(blocks):
        d = b["detection"]
        sc = b.get("shape_check", {})
        name = "" if len(blocks) == 1 else f"line {k + 1}: "
        quoted = "area" if b.get("flux_method") == "area" else "Gaussian"
        sig = (f"S/N {d['snr']:.1f} (fixed shape)" if d.get("fixed_shape")
               else f"{d['significance_sigma']:.1f}$\\sigma$")
        rows.append(
            f"{name}F = {_pm(b['flux'] * f, b['flux_error'] * f)} {ulabel} km/s ({quoted}), {sig}\n"
            f"   Gaussian {_pm(b['flux_gauss'] * f, b['flux_gauss_error'] * f)}, "
            f"area {_pm(b['flux_area'] * f, b['flux_area_error'] * f)}"
            + (f" ($\\Delta$ {sc['nsigma']:+.1f}$\\sigma$)" if np.isfinite(sc.get("nsigma", np.nan)) else "")
            + f"\n   v$_0$ = {b['v0_kms']:+.0f}, FWHM = {b['fwhm_kms']:.0f} km/s, "
            f"peak {_pm(b['peak'] * f, b.get('peak_error', np.nan) * f)} {ulabel}"
            # lines fitted apart: each has its own size
            + (f", size {b['size_model']}" + (f" {b['theta_fwhm_arcsec']:.2f}\""
                                              if b.get("theta_fwhm_arcsec") else "")
               if separate and "size_model" in b else ""))
    ax.text(0.02, 0.97, "\n".join(rows), transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.9))
    ax.set_ylabel(f"flux density [{ulabel}]")
    lo_y, hi_y = ax.get_ylim()
    ax.set_ylim(lo_y, hi_y + (0.35 + 0.12 * len(blocks)) * (hi_y - lo_y))
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    p = s["line"]
    ax.set_title(f"stacked line{'s' if len(blocks) > 1 else ''}, zero spacing "
                 f"({s['n_sources']} sources, {s['weighting_scheme']}, size {p['size_model']}"
                 + (f" {p['theta_fwhm_arcsec']:.2f}\"" if p['theta_fwhm_arcsec'] else "")
                 + (f", {rebin} channels per bin" if rebin > 1 else "") + ")", fontsize=9)
    # residuals
    with np.errstate(invalid="ignore", divide="ignore"):
        axr.step(vb, (yb - mb) / eb, where="mid", color="k", lw=1)
    axr.axhspan(-1, 1, color="0.6", alpha=0.3, lw=0)
    axr.axhline(0, color="0.3", lw=0.6)
    axr.set_ylabel(r"resid./$\sigma$")
    axr.set_xlabel("velocity [km/s]")
    if v_limits:
        axr.set_xlim(*v_limits)
    else:
        axr.set_xlim(v[used].min() - dvb, v[used].max() + dvb)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_uv(result, path: str, unit: Optional[str] = None):
    plt = _pyplot()
    s = result.summary
    f, ulabel = _scale(unit, _typical(result))
    b = result.baseline_klambda
    ok = result.annulus_ok & np.isfinite(result.annulus_flux)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.errorbar(b[ok], result.annulus_flux[ok] * f, result.annulus_flux_error[ok] * f,
                fmt="o", color="k", ms=4, capsize=2, label="line flux per annulus")
    bb = np.geomspace(max(b[ok].min() * 0.5, 0.1), b[ok].max() * 1.5, 300)
    p = s["line"]
    ax.axhline(0, color="0.3", lw=0.6)
    st = s.get("size_test")
    if st and st.get("weight_gauss") is not None:
        wg = st["weight_gauss"]
        ax.plot(bb, np.full_like(bb, st["flux_point"] * f), color="C0",
                lw=0.8 + 1.6 * (1 - wg), label=f"point source (weight {1 - wg:.2f})")
        if st.get("theta_gauss_arcsec"):
            ax.plot(bb, st["flux_gauss"] * f * uv_gaussian(bb * 1e3, st["theta_gauss_arcsec"]),
                    color="C3", lw=0.8 + 1.6 * wg,
                    label=f"Gaussian, FWHM {st['theta_gauss_arcsec']:.2f}\" (weight {wg:.2f}, "
                          f"$\\Delta\\chi^2$ = {st['delta_chi2_point_minus_gauss']:.1f})")
        ax.axhline(p["flux"] * f, color="k", ls=":", lw=1,
                   label=f"total flux, model average: {p['flux'] * f:.3g}")
    elif p["theta_fwhm_arcsec"]:
        ax.plot(bb, p["flux"] * f * uv_gaussian(bb * 1e3, p["theta_fwhm_arcsec"]), color="C3",
                label=f"Gaussian, FWHM {p['theta_fwhm_arcsec']:.2f}\"")
    else:
        ax.plot(bb, np.full_like(bb, p["flux"] * f), color="C3", label="point source")
    ax.set_xscale("log")
    if len(s.get("lines") or []) > 1:
        ax.set_title(f"line 1 ({s['rest_freq_ghz']:.6g} GHz)", fontsize=9)
    ax.set_xlabel(r"baseline [k$\lambda$]")
    ax.set_ylabel(f"line flux [{ulabel} km/s]")
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_noise(result, path: str, unit: Optional[str] = None):
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
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_line_result(result, line: Lines, prefix: str, unit: Optional[str] = None,
                     v_limits=None, rebin: int = 1, verbose=True) -> List[str]:
    """All three figures; returns their paths."""
    out = [plot_profile(result, line, f"{prefix}_profile.png", unit, v_limits, rebin),
           plot_uv(result, f"{prefix}_uv.png", unit),
           plot_noise(result, f"{prefix}_noise.png", unit)]
    _say("  figures: " + ", ".join(out), verbose)
    return out


__all__ = ["plot_line_result", "plot_profile", "plot_uv", "plot_noise"]
