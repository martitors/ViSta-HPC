"""
vista.extract.profiles
======================
Line shape and per-annulus flux (Sec. 5.3.2).

The extraction is a two-step procedure.  First the line shape is measured
once on the spatially integrated spectrum, obtained by averaging the real
part of the visibilities across all annuli, by weighted nonlinear least
squares over a window wide enough to capture the full profile without
truncating its wings.  Only the shape parameters, the centroid and the width
of each component, are retained.  Then, with the shape held fixed, the
amplitude is fitted independently in each annulus by weighted *linear* least
squares, and the flux follows analytically as

    F(b) = A(b) * sigma * sqrt(2 pi)                                 (Eq. 32)

Fixing the shape isolates one component of a blend analytically, avoiding
the truncation and the cross-contamination that a fixed velocity window
would introduce.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from .config import C_KMS, LineConfig, ProfileConfig

SQRT_2PI = float(np.sqrt(2.0 * np.pi))
FWHM_PER_SIGMA = 2.3548200450309493


# ---------------------------------------------------------------------------
# channel masks
# ---------------------------------------------------------------------------
def velocity_window_mask(freq_hz, rest_freq_hz, window_kms) -> np.ndarray:
    """Channels whose velocity falls inside ``window_kms``."""
    v = C_KMS * (rest_freq_hz - np.asarray(freq_hz, float)) / rest_freq_hz
    return (v >= window_kms[0]) & (v <= window_kms[1])


def line_free_mask(freq_hz, line: LineConfig,
                   half_width_kms: Optional[float] = None,
                   window_kms: Optional[Tuple[float, float]] = None,
                   redshift: Optional[float] = None,
                   source_redshift: Optional[float] = None) -> np.ndarray:
    """Channels free of line emission, for a continuum fit.

    The target window is excluded, widened by the window of the second line
    and by every interval in ``line.exclude_v_kms``.  ``half_width_kms`` or
    ``window_kms`` override the width taken from ``line.v_window_kms``.
    """
    freq_hz = np.asarray(freq_hz, float)
    if window_kms is not None:
        target = window_kms
    elif half_width_kms is not None:
        target = (-abs(half_width_kms), abs(half_width_kms))
    else:
        target = line.v_window_kms

    mask = ~velocity_window_mask(freq_hz, line.rest_freq_hz, target)
    if line.second_rest_freq_hz is not None:
        mask &= ~velocity_window_mask(freq_hz, line.second_rest_freq_hz,
                                      line.second_v_window_kms)
    for lo, hi in line.exclude_v_kms:
        mask &= ~velocity_window_mask(freq_hz, line.rest_freq_hz, (lo, hi))
    return mask


def shifted_line_mask(freq_hz, rest_freq_hz, z_grid, z_source, window_kms):
    """Window of a line emitted at ``z_source`` on the grid of a source at
    ``z_grid``.  Used for companions and for interlopers in the band."""
    nu = rest_freq_hz * (1.0 + z_grid) / (1.0 + z_source)
    return velocity_window_mask(freq_hz, nu, window_kms)


# ---------------------------------------------------------------------------
# line shape
# ---------------------------------------------------------------------------
@dataclass
class LineShape:
    """Shape of the stacked line profile, in velocity."""

    centroid_kms: float
    sigma_kms: float
    amplitude: float = float("nan")
    centroid2_kms: Optional[float] = None
    sigma2_kms: Optional[float] = None
    amplitude2: Optional[float] = None
    baseline: Optional[float] = None
    chi2_reduced: float = float("nan")
    raw_params: Optional[list] = field(default=None, repr=False)

    @property
    def fwhm_kms(self) -> float:
        return FWHM_PER_SIGMA * self.sigma_kms

    @property
    def blended(self) -> bool:
        return self.sigma2_kms is not None

    def as_dict(self) -> dict:
        out = {"centroid_kms": self.centroid_kms,
               "sigma_kms": self.sigma_kms,
               "fwhm_kms": self.fwhm_kms,
               "chi2_reduced": self.chi2_reduced}
        if self.blended:
            out.update({"second_centroid_kms": self.centroid2_kms,
                        "second_sigma_kms": self.sigma2_kms,
                        "second_fwhm_kms": FWHM_PER_SIGMA * self.sigma2_kms})
        if self.baseline is not None:
            out["baseline"] = self.baseline
        return out


def _gaussian(v, centroid, sigma):
    return np.exp(-0.5 * ((v - centroid) / sigma) ** 2)


def fit_line_shape(spectrum, weights, freq_hz, line: LineConfig,
                   config: ProfileConfig = ProfileConfig(),
                   start: Optional[list] = None) -> Optional[LineShape]:
    """Fit one or two Gaussians to the spatially integrated spectrum.

    Parameters
    ----------
    spectrum
        Real part of the stacked visibilities averaged over all annuli,
        shape ``(nchan,)``.
    weights
        Either the ``(nchan, nbins)`` weight array or the ``(nchan,)`` weight
        per channel; it sets the least-squares weights.
    start
        Initial guess in the parameterisation of the chosen tie, typically
        the nominal solution when refitting a bootstrap realisation.

    Returns
    -------
    LineShape or None
        ``None`` when the fit does not converge; the caller then falls back
        to the boxcar integration.
    """
    from scipy.optimize import curve_fit

    spectrum = np.asarray(spectrum, float)
    weights = np.asarray(weights, float)
    w_chan = np.nansum(weights, axis=1) if weights.ndim == 2 else weights

    v1 = line.velocity(freq_hz)
    blend = line.second_rest_freq_hz is not None
    v2 = line.velocity(freq_hz, second=True) if blend else None

    span = config.fit_span_kms
    use = np.abs(v1) < span
    if blend:
        use |= np.abs(v2) < span
    use &= np.isfinite(spectrum)
    if use.sum() < (8 if blend else 6):
        return None

    baseline = bool(config.joint_continuum)

    if config.fixed_shape_kms is not None:
        c, sig = (float(x) for x in config.fixed_shape_kms)
        return LineShape(centroid_kms=c, sigma_kms=sig,
                         centroid2_kms=c if blend else None,
                         sigma2_kms=sig if blend else None)

    sigma_err = 1.0 / np.sqrt(np.maximum(w_chan[use], 1e-30))
    y = spectrum[use]
    x1 = v1[use]
    x2 = v2[use] if blend else None
    a0 = float(np.nanmax(y)) if np.isfinite(np.nanmax(y)) else 1.0
    dummy = np.arange(int(use.sum()))

    # Bounds from the data, not from a fixed idea of how wide a line is.  A
    # channel sets the floor, since no fit can resolve a width below it, and
    # the fitted span sets the ceiling.  Hardcoded limits would silently pin
    # the solution to a bound for a narrow line or a narrow band.
    dv_chan = float(np.median(np.abs(np.diff(np.sort(v1[np.isfinite(v1)])))))
    if not np.isfinite(dv_chan) or dv_chan <= 0:
        dv_chan = 1.0
    sigma_lo = 0.5 * dv_chan
    sigma_hi = max(4.0 * dv_chan, 0.5 * span)
    centroid_bound = max(2.0 * dv_chan, 0.5 * span)

    # Start from the moments of the data.  A fixed starting width is a poor
    # guess when the line is only a few channels across: the optimiser can
    # settle on a broad, shallow solution that fits the noise as well as the
    # line, and never come back.
    positive = np.clip(y - np.median(y), 0.0, None)
    total = float(np.sum(positive))
    if total > 0:
        centroid_0 = float(np.sum(positive * x1) / total)
        variance = float(np.sum(positive * (x1 - centroid_0) ** 2) / total)
        sigma_0 = math.sqrt(variance) if variance > 0 else 2.0 * dv_chan
    else:
        centroid_0, sigma_0 = 0.0, 2.0 * dv_chan
    sigma_0 = float(np.clip(sigma_0, 2.0 * sigma_lo, 0.8 * sigma_hi))
    centroid_0 = float(np.clip(centroid_0, -centroid_bound, centroid_bound))
    a0 = float(np.nanmax(y) - np.median(y))
    if not np.isfinite(a0) or a0 <= 0:
        a0 = 1.0

    # parameterisation: (initial guess, lower bound, upper bound, unpack)
    # unpack -> (c1, s1, A1, c2, s2, A2); the optional baseline is always the
    # last parameter, so the indices below never move.
    if not blend:
        p0 = [a0, centroid_0, sigma_0]
        lo = [-np.inf, -centroid_bound, sigma_lo]
        hi = [np.inf, centroid_bound, sigma_hi]
        unpack = lambda p: (p[1], abs(p[2]), p[0], None, None, None)

        def components(p):
            return p[0] * _gaussian(x1, p[1], p[2])
    elif config.shape_tie == "free":
        p0 = [a0, centroid_0, sigma_0, 0.3 * a0, centroid_0, sigma_0]
        lo = [-np.inf, -centroid_bound, sigma_lo,
              -np.inf, -centroid_bound, sigma_lo]
        hi = [np.inf, centroid_bound, sigma_hi,
              np.inf, centroid_bound, sigma_hi]
        unpack = lambda p: (p[1], abs(p[2]), p[0], p[4], abs(p[5]), p[3])

        def components(p):
            return (p[0] * _gaussian(x1, p[1], p[2])
                    + p[3] * _gaussian(x2, p[4], p[5]))
    elif config.shape_tie == "centroid":
        p0 = [a0, centroid_0, sigma_0, 0.3 * a0, sigma_0]
        lo = [-np.inf, -centroid_bound, sigma_lo, -np.inf, sigma_lo]
        hi = [np.inf, centroid_bound, sigma_hi, np.inf, sigma_hi]
        unpack = lambda p: (p[1], abs(p[2]), p[0], p[1], abs(p[4]), p[3])

        def components(p):
            return (p[0] * _gaussian(x1, p[1], p[2])
                    + p[3] * _gaussian(x2, p[1], p[4]))
    else:                                            # 'centroid+width'
        p0 = [a0, centroid_0, sigma_0, 0.3 * a0]
        lo = [-np.inf, -centroid_bound, sigma_lo, -np.inf]
        hi = [np.inf, centroid_bound, sigma_hi, np.inf]
        unpack = lambda p: (p[1], abs(p[2]), p[0], p[1], abs(p[2]), p[3])

        def components(p):
            return (p[0] * _gaussian(x1, p[1], p[2])
                    + p[3] * _gaussian(x2, p[1], p[2]))

    if baseline:
        p0 = p0 + [float(np.nanmedian(y))]
        lo = lo + [-np.inf]
        hi = hi + [np.inf]

    def model(_x, *p):
        value = components(p)
        return value + p[-1] if baseline else value

    if start is not None and len(start) == len(p0):
        p0 = list(start)
    try:
        p, _ = curve_fit(model, dummy, y, p0=p0, sigma=sigma_err,
                         absolute_sigma=False, bounds=(lo, hi), maxfev=20000)
    except Exception:
        return None

    c1, s1, a1, c2, s2, a2 = unpack(p)
    residual = (y - model(dummy, *p)) / sigma_err
    dof = max(int(use.sum()) - len(p), 1)
    return LineShape(centroid_kms=float(c1), sigma_kms=float(s1),
                     amplitude=float(a1),
                     centroid2_kms=None if c2 is None else float(c2),
                     sigma2_kms=None if s2 is None else float(s2),
                     amplitude2=None if a2 is None else float(a2),
                     baseline=float(p[-1]) if baseline else None,
                     chi2_reduced=float(np.sum(residual ** 2) / dof),
                     raw_params=[float(x) for x in p])


# ---------------------------------------------------------------------------
# flux per annulus
# ---------------------------------------------------------------------------
def flux_per_annulus_template(real, weight, freq_hz, line: LineConfig,
                              shape: LineShape,
                              config: ProfileConfig = ProfileConfig()):
    """Amplitude per annulus at fixed shape, by weighted linear least squares.

    Parameters
    ----------
    real, weight
        Stacked real part and summed weights, both ``(nchan, nbins)``.

    Returns
    -------
    (flux, flux_second, continuum, used_channels)
        Line fluxes in ``[data units] * km/s``; ``flux_second`` is ``None``
        for an isolated line, ``continuum`` the flux density of the constant
        term and is ``None`` unless ``config.joint_continuum`` is set.
    """
    v1 = line.velocity(freq_hz)
    g1 = _gaussian(v1, shape.centroid_kms, shape.sigma_kms)
    used = np.abs(v1) < config.fit_span_kms

    blend = shape.blended and line.second_rest_freq_hz is not None
    if blend:
        v2 = line.velocity(freq_hz, second=True)
        g2 = _gaussian(v2, shape.centroid2_kms, shape.sigma2_kms)
        used = used | (np.abs(v2) < config.fit_span_kms)

    with_continuum = bool(config.joint_continuum)
    n_bins = real.shape[1]
    amp1 = np.full(n_bins, np.nan)
    amp2 = np.full(n_bins, np.nan) if blend else None
    continuum = np.full(n_bins, np.nan) if with_continuum else None
    n_min = 4 if (blend or with_continuum) else 2

    for b in range(n_bins):
        w = weight[:, b]
        good = used & (w > 0) & np.isfinite(real[:, b])
        if good.sum() < n_min:
            continue
        columns = [g1[good]]
        if blend:
            columns.append(g2[good])
        if with_continuum:
            columns.append(np.ones(int(good.sum())))
        if len(columns) == 1:                        # closed form
            root = w[good]
            num = float(np.nansum(root * g1[good] * real[good, b]))
            den = float(np.nansum(root * g1[good] ** 2))
            if den > 0:
                amp1[b] = num / den
            continue
        design = np.column_stack(columns)
        root = np.sqrt(w[good])
        try:
            solution, *_ = np.linalg.lstsq(design * root[:, None],
                                           real[good, b] * root, rcond=None)
        except np.linalg.LinAlgError:
            continue
        amp1[b] = solution[0]
        if blend:
            amp2[b] = solution[1]
        if with_continuum:
            continuum[b] = solution[-1]

    flux1 = amp1 * shape.sigma_kms * SQRT_2PI
    flux2 = (amp2 * shape.sigma2_kms * SQRT_2PI) if blend else None
    return flux1, flux2, continuum, used


def flux_per_annulus_window(real, weight, freq_hz, rest_freq_hz, window_kms):
    """Boxcar integration of the line over a velocity window.

    Returns ``(flux, weight_sum, velocity, channel_width, in_window)``.
    """
    v = C_KMS * (rest_freq_hz - np.asarray(freq_hz, float)) / rest_freq_hz
    dv = np.abs(np.gradient(v))
    inside = (v >= window_kms[0]) & (v <= window_kms[1])
    flux = np.nansum(real[inside] * dv[inside, None]
                     * (weight[inside] > 0), axis=0)
    w = weight[inside].sum(axis=0)
    flux[w <= 0] = np.nan
    return flux, w, v, dv, inside


def flux_per_annulus_continuum(real, weight, channel_mask=None):
    """Weighted mean flux density per annulus over the selected channels."""
    if channel_mask is None:
        channel_mask = np.ones(real.shape[0], dtype=bool)
    w = np.nansum(weight[channel_mask], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        flux = np.where(w > 0,
                        np.nansum((real * weight)[channel_mask], axis=0) / w,
                        np.nan)
    return flux, w


def integrated_spectrum(real, weight):
    """Spatially integrated spectrum: weighted average over all annuli."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(real * weight, axis=1) / np.nansum(weight, axis=1)


__all__ = ["LineShape", "fit_line_shape", "flux_per_annulus_template",
           "flux_per_annulus_window", "flux_per_annulus_continuum",
           "integrated_spectrum", "line_free_mask", "velocity_window_mask",
           "shifted_line_mask", "FWHM_PER_SIGMA", "SQRT_2PI"]
