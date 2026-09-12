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
from typing import Optional, Sequence

import numpy as np

from .config import ARCSEC_PER_RAD, UVFitConfig

_FOUR_LN2 = 4.0 * np.log(2.0)


def gaussian_visibility(b_lambda, flux, theta_arcsec):
    """Circular Gaussian visibility profile, Eq. (33)."""
    theta = np.abs(theta_arcsec) / ARCSEC_PER_RAD
    return flux * np.exp(-(np.pi * theta * np.asarray(b_lambda, float)) ** 2
                         / _FOUR_LN2)


def two_gaussian_visibility(b_lambda, flux1, theta1, flux2, theta2):
    """Sum of two circular Gaussians: compact plus extended."""
    return (gaussian_visibility(b_lambda, flux1, theta1)
            + gaussian_visibility(b_lambda, flux2, theta2))


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
                   theta_prior: Optional[Sequence[float]] = None,
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
    theta_prior
        ``(theta_ref, sigma_dex)`` soft tie, overriding ``config.theta_prior``.
    theta_fixed
        FWHM to freeze, overriding the config.  ``0`` means point source.

    Returns
    -------
    UVFitResult
    """
    from scipy.optimize import curve_fit, least_squares

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
    if theta_prior is None:
        theta_prior = config.theta_prior

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
            p, _ = curve_fit(model, b_lambda[good], flux[good],
                             p0=[np.nanmax(flux[good])], sigma=err,
                             absolute_sigma=sigma is not None, maxfev=10000)
        except Exception:
            return UVFitResult(n_bins=n_used, model=name)
        return UVFitResult(flux=float(p[0]), theta_arcsec=theta,
                           params=[float(p[0]), theta],
                           chi2_reduced=reduced_chi2(model(b_lambda[good], *p), 1),
                           n_bins=n_used, model=name, converged=True)

    n_par = 2 if config.model == "gauss" else 4
    if n_used < n_par + 1:
        return UVFitResult(n_bins=n_used, model=config.model)
    flux0 = float(np.nanmax(flux[good]))

    # -- circular Gaussian with a soft tie on the size ----------------------
    if theta_prior is not None and config.model == "gauss":
        theta_ref, sigma_dex = float(theta_prior[0]), float(theta_prior[1])
        if np.isfinite(theta_ref) and theta_ref > 0 and sigma_dex > 0:
            scale = np.ones(n_used) if sigma is None else err

            def residuals(p):
                data = (flux[good] - gaussian_visibility(b_lambda[good],
                                                         p[0], p[1])) / scale
                prior = np.log10(max(abs(p[1]), 1e-6) / theta_ref) / sigma_dex
                return np.concatenate([data, [prior]])

            try:
                solution = least_squares(
                    residuals, x0=[flux0, theta_ref],
                    bounds=([-np.inf, 1e-3], [np.inf, config.theta_max_arcsec]),
                    max_nfev=20000)
            except Exception:
                return UVFitResult(n_bins=n_used, model="gauss_prior")
            f_tot, theta = float(solution.x[0]), abs(float(solution.x[1]))
            values = gaussian_visibility(b_lambda[good], *solution.x)
            return UVFitResult(flux=f_tot, theta_arcsec=theta,
                               params=[f_tot, theta],
                               chi2_reduced=reduced_chi2(values, n_par),
                               n_bins=n_used, model="gauss_prior",
                               converged=True,
                               theta_at_bound=theta > 0.95 * config.theta_max_arcsec)

    # -- free fit -----------------------------------------------------------
    try:
        if config.model == "gauss":
            p, _ = curve_fit(gaussian_visibility, b_lambda[good], flux[good],
                             p0=[flux0, 1.0], sigma=err,
                             absolute_sigma=sigma is not None,
                             bounds=([-np.inf, 1e-3],
                                     [np.inf, config.theta_max_arcsec]),
                             maxfev=10000)
            f_tot, theta = float(p[0]), abs(float(p[1]))
            values = gaussian_visibility(b_lambda[good], *p)
        else:
            p, _ = curve_fit(two_gaussian_visibility, b_lambda[good],
                             flux[good],
                             p0=[0.6 * flux0, 2.5, 0.4 * flux0, 0.4],
                             sigma=err, absolute_sigma=sigma is not None,
                             bounds=([0, 1e-3, 0, 1e-3],
                                     [np.inf, config.theta_max_arcsec,
                                      np.inf, config.theta_max_arcsec]),
                             maxfev=20000)
            if p[1] < p[3]:                     # extended component first
                p = np.array([p[2], p[3], p[0], p[1]])
            f_tot = float(p[0] + p[2])
            theta = abs(float(p[1]))            # size of the extended one
            values = two_gaussian_visibility(b_lambda[good], *p)
    except Exception:
        return UVFitResult(n_bins=n_used, model=config.model)

    at_bound = theta > 0.95 * config.theta_max_arcsec
    if config.model == "gauss2":
        at_bound = max(p[1], p[3]) > 0.95 * config.theta_max_arcsec
    return UVFitResult(flux=f_tot, theta_arcsec=theta,
                       params=[float(x) for x in np.atleast_1d(p)],
                       chi2_reduced=reduced_chi2(values, n_par),
                       n_bins=n_used, model=config.model, converged=True,
                       theta_at_bound=bool(at_bound))


__all__ = ["gaussian_visibility", "two_gaussian_visibility", "fit_uv_profile",
           "UVFitResult"]
