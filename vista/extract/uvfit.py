"""
vista.extract.uvfit
===================
The source model fitted to the amplitude-versus-baseline profile
(Sec. 5.3.2, Eq. 33):

    V(b) = F_tot * exp[ -(pi * theta_FWHM * b)^2 / (4 ln 2) ]

which gives the total flux as the zero-baseline value ``F_tot = V(b=0)`` and
the effective source size as ``theta_FWHM``.  When the stack is unresolved
the Gaussian is replaced by a fixed point-source model, which avoids
introducing a size the data cannot constrain and that a free Gaussian would
tend to overfit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import ARCSEC_PER_RAD, UVFitConfig

_FOUR_LN2 = 4.0 * np.log(2.0)


def gaussian_visibility(b_lambda, flux, theta_arcsec):
    """Circular Gaussian visibility profile, Eq. (33)."""
    theta = np.abs(theta_arcsec) / ARCSEC_PER_RAD
    return flux * np.exp(-(np.pi * theta * np.asarray(b_lambda, float)) ** 2
                         / _FOUR_LN2)


@dataclass
class UVFitResult:
    """Outcome of one uv-plane fit."""

    flux: float = float("nan")
    theta_arcsec: float = float("nan")
    params: Optional[list] = None
    chi2_reduced: float = float("nan")
    n_bins: int = 0
    model: str = "gauss"
    converged: bool = False
    theta_at_bound: bool = False
    flux_error: float = float("nan")          # from the errors given to the fit
    theta_error: float = float("nan")

    def as_dict(self, prefix: str = "") -> dict:
        return {f"{prefix}flux": self.flux,
                f"{prefix}theta_fwhm_arcsec": self.theta_arcsec,
                f"{prefix}chi2_reduced": self.chi2_reduced,
                f"{prefix}n_bins_in_fit": self.n_bins,
                f"{prefix}model": self.model,
                f"{prefix}converged": self.converged,
                f"{prefix}theta_at_bound": self.theta_at_bound}


def fit_uv_profile(b_lambda, flux, sigma=None,
                   config: UVFitConfig = UVFitConfig(),
                   bin_ok=None,
                   theta_fixed: Optional[float] = None) -> UVFitResult:
    """Fit the flux-versus-baseline profile with the configured source model.

    Parameters
    ----------
    b_lambda
        Weighted mean baseline length of each annulus, in units of the
        wavelength.
    flux
        Flux per annulus (line flux, or flux density in continuum mode).
    sigma
        Per-annulus uncertainties, normally the bootstrap ones.  When given,
        they are used as absolute errors, so the reduced chi square is
        meaningful.
    bin_ok
        Extra boolean mask of usable annuli, typically the minimum number of
        contributing sources.
    theta_fixed
        FWHM to freeze, overriding the config.  ``0`` means point source.

    Returns
    -------
    UVFitResult
    """
    from scipy.optimize import curve_fit

    b_lambda = np.asarray(b_lambda, float)
    flux = np.asarray(flux, float)
    good = np.isfinite(b_lambda) & np.isfinite(flux)
    if sigma is not None:
        sigma = np.asarray(sigma, float)
        good &= np.isfinite(sigma) & (sigma > 0)
    if config.b_max_klambda is not None:
        good &= b_lambda <= config.b_max_klambda * 1e3
    if bin_ok is not None:
        good &= np.asarray(bin_ok, bool)
    n_used = int(good.sum())
    err = None if sigma is None else sigma[good]

    if theta_fixed is None:
        theta_fixed = config.resolved_theta_fixed()

    def reduced_chi2(model_values, n_par):
        if sigma is None or n_used <= n_par:
            return float("nan")
        return float(np.sum(((flux[good] - model_values) / err) ** 2)
                     / (n_used - n_par))

    # -- point source or frozen size: one free parameter, the flux ----------
    if theta_fixed is not None:
        if n_used < 2:
            return UVFitResult(n_bins=n_used, model="point"
                               if theta_fixed == 0 else "fixed_theta")
        theta = float(theta_fixed)
        name = "point" if theta == 0.0 else "fixed_theta"

        def model(b, f):
            return np.full_like(np.asarray(b, float), f) if theta == 0.0 \
                else gaussian_visibility(b, f, theta)

        try:
            p, cov = curve_fit(model, b_lambda[good], flux[good],
                               p0=[np.nanmax(flux[good])], sigma=err,
                               absolute_sigma=sigma is not None, maxfev=10000)
        except Exception:
            return UVFitResult(n_bins=n_used, model=name)
        return UVFitResult(flux=float(p[0]), theta_arcsec=theta,
                           flux_error=float(np.sqrt(cov[0, 0])), theta_error=0.0,
                           params=[float(p[0]), theta],
                           chi2_reduced=reduced_chi2(model(b_lambda[good], *p), 1),
                           n_bins=n_used, model=name, converged=True)

    n_par = 2
    if n_used < n_par + 1:
        return UVFitResult(n_bins=n_used, model=config.model)
    theta_lo = max(1e-3, float(config.theta_min_arcsec))
    theta_start = float(np.clip(1.0, theta_lo * 1.001, config.theta_max_arcsec * 0.999))

    # -- circular Gaussian: flux and size -----------------------------------
    try:
        p, cov = curve_fit(gaussian_visibility, b_lambda[good], flux[good],
                           p0=[float(np.nanmax(flux[good])), theta_start], sigma=err,
                           absolute_sigma=sigma is not None,
                           bounds=([-np.inf, theta_lo], [np.inf, config.theta_max_arcsec]),
                           maxfev=10000)
    except Exception:
        return UVFitResult(n_bins=n_used, model=config.model)
    f_tot, theta = float(p[0]), abs(float(p[1]))
    values = gaussian_visibility(b_lambda[good], *p)
    at_bound = theta > 0.95 * config.theta_max_arcsec
    errors = (np.sqrt(np.diag(cov)) if cov is not None and np.all(np.isfinite(cov))
              else [float("nan")] * 2)
    return UVFitResult(flux=f_tot, theta_arcsec=theta,
                       flux_error=float(errors[0]), theta_error=float(errors[1]),
                       params=[float(x) for x in np.atleast_1d(p)],
                       chi2_reduced=reduced_chi2(values, n_par),
                       n_bins=n_used, model=config.model, converged=True,
                       theta_at_bound=bool(at_bound))


__all__ = ["gaussian_visibility", "fit_uv_profile", "UVFitResult"]
