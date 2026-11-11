"""
vista.extract.linefit
=====================
The line measured on the whole stacked cube, channels x uv annuli, at once.

Model of the real part of every cell (channel c, annulus b)::

    Re V(c, b) = [F phi_c(v0, sigma) + F2 phi2_c + ...] g(b; theta) [+ C g(b; theta_c)]

``phi_c`` is a Gaussian of unit integral *averaged over the channel* (not
sampled at its centre: a line a few channels wide is not distorted),
``F`` the velocity-integrated flux (``[data units] km/s``), ``g`` the
visibility of a circular Gaussian of FWHM ``theta`` (1 for a point source)
at the mean baseline of the cell, ``Fk phik`` the other lines of the band
(each in its own velocity, sharing the size) and ``C`` an optional
continuum.  The cells are fitted by weighted least
squares with the noise of :mod:`vista.extract.noise`: every cell weighted by
the inverse of its variance (from the scatter of its visibilities), and the
correlation between adjacent channels taken into account in every error, S/N
and null realisation (a sandwich covariance).  The correlation is not
inverted: Hanning smoothing makes it singular (it removes the highest
spectral frequency), and a generalised least squares on an estimated, nearly
singular matrix would amplify its errors; the weighted fit loses almost
nothing on a line several channels wide.  Compared with fitting the shape on
a spectrum averaged over the annuli and the amplitude annulus by annulus,
every parameter uses all the data at once, with the right weights, and the
errors come out of the same fit.

Detection and significance
--------------------------
The centre and the width are searched within their ranges: the best S/N
over a grid of centres, widths and sizes is the statistic, and its
significance comes from the same search run on noise (:func:`null_max_snr`):
this accounts for the freedom of the fit (the look-elsewhere effect), which
would otherwise make noise peaks look like detections.  With the centre and
the width fixed (ranges with lo = hi) and the size fixed, the flux is linear
in the data -- a matched filter: unbiased, Gaussian, and its S/N is the
significance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.special import erf

from .config import ARCSEC_PER_RAD

FWHM_PER_SIGMA = 2.3548200450309493
_FOUR_LN2 = 4.0 * np.log(2.0)


# ---------------------------------------------------------------------------
# model pieces
# ---------------------------------------------------------------------------
def channel_gaussian(v, dv, v0, sigma):
    """Gaussian of unit integral averaged over channels of width ``dv``
    centred at ``v`` (km/s): the flux density per unit line flux."""
    s = max(float(sigma), 1e-6) * math.sqrt(2.0)
    hi = (np.asarray(v) + 0.5 * dv - v0) / s
    lo = (np.asarray(v) - 0.5 * dv - v0) / s
    return 0.5 * (erf(hi) - erf(lo)) / dv


def uv_gaussian(b_lambda, theta_arcsec):
    """Visibility of a circular Gaussian of unit flux and FWHM theta."""
    t = abs(float(theta_arcsec)) / ARCSEC_PER_RAD
    return np.exp(-(np.pi * t * np.asarray(b_lambda, float)) ** 2 / _FOUR_LN2)


# ---------------------------------------------------------------------------
# the data
# ---------------------------------------------------------------------------
@dataclass
class Cube:
    """The stacked cells and their noise."""

    real: np.ndarray                    # (nchan, nbins)
    variance: np.ndarray                # (nchan, nbins)
    baseline: np.ndarray                # (nchan, nbins) mean baseline [lambda]
    velocity: np.ndarray                # (nchan,) km/s, target line
    dv: float                           # channel width, km/s (> 0)
    ok: np.ndarray                      # (nchan, nbins) cells to fit
    corr: Optional[np.ndarray] = None   # (nchan, nchan) noise correlation, None: independent
    velocities: Optional[Sequence[np.ndarray]] = None   # [(nchan,)] km/s, lines 2, 3, ...
    edges: Optional[np.ndarray] = None  # (nbins+1,) annulus edges [lambda]: average the
                                        # source model over each annulus, not at its mean

    @property
    def shape(self):
        return self.real.shape

    def uv(self, theta_arcsec):
        """Visibility of the source model in every cell (``(nchan, nbins)``).

        With ``edges`` it is the mean over the baselines of the annulus, not
        the value at their mean: a resolved source changes across an
        annulus (18 annuli over 3 decades are +-20% wide), and on the
        steep, convex part of the profile the value at the mean is lower
        than the mean, which made the source look smaller and its total flux
        lower.  The baselines inside an annulus are taken to follow a power
        law in b whose mean is the measured mean baseline of the cell.
        """
        if self.edges is None or float(theta_arcsec) == 0.0:
            return uv_gaussian(self.baseline, theta_arcsec)
        nodes, weights = self._nodes()
        return np.sum(weights * uv_gaussian(nodes, theta_arcsec), axis=-1)

    def _nodes(self):
        cache = getattr(self, "_node_cache", None)
        if cache is None:
            cache = annulus_nodes(self.baseline, self.edges)
            object.__setattr__(self, "_node_cache", cache)
        return cache


def annulus_nodes(baseline, edges, k=8):
    """Quadrature nodes and weights of every cell over its annulus.

    The density of baselines inside annulus [lo, hi] is modelled as
    ``b^gamma``, with gamma such that its mean is the measured mean baseline
    of the cell; returns ``(nodes, weights)``, both ``(nchan, nbins, k)``.
    """
    nchan, nbins = baseline.shape
    lo = np.broadcast_to(edges[:-1][None, :], (nchan, nbins))
    hi = np.broadcast_to(edges[1:][None, :], (nchan, nbins))
    target = np.clip(np.where(np.isfinite(baseline) & (baseline > 0), baseline,
                              np.sqrt(lo * hi)), lo * 1.0001, hi * 0.9999)
    x, wx = np.polynomial.legendre.leggauss(k)
    # nodes uniform in log b (Gauss-Legendre), density b^gamma -> weight b^(gamma+1) in log b
    logs = (np.log(lo)[..., None] + (x[None, None, :] + 1) / 2
            * np.log(hi / lo)[..., None])
    nodes = np.exp(logs)

    def mean_for(gamma):
        w = wx[None, None, :] * np.exp((gamma[..., None] + 1) * (logs - logs[..., :1]))
        return np.sum(w * nodes, axis=-1) / np.sum(w, axis=-1), w

    g_lo, g_hi = np.full(target.shape, -30.0), np.full(target.shape, 30.0)
    for _ in range(60):                                  # bisection, vectorised
        mid = 0.5 * (g_lo + g_hi)
        m, _ = mean_for(mid)
        up = m < target
        g_lo = np.where(up, mid, g_lo)
        g_hi = np.where(up, g_hi, mid)
    _, w = mean_for(0.5 * (g_lo + g_hi))
    return nodes, w / np.sum(w, axis=-1, keepdims=True)


_CORR_CACHE: dict = {}


def _correlation(corr, idx):
    """Channel correlation restricted to the channels ``idx`` and its
    positive semi-definite square root (cached: most annuli use the same
    channels)."""
    key = (id(corr), corr.shape, idx.tobytes())
    hit = _CORR_CACHE.get(key)
    if hit is None:
        r = corr[np.ix_(idx, idx)]
        vals, vecs = np.linalg.eigh(r)
        root = (vecs * np.sqrt(np.clip(vals, 0, None))) @ vecs.T
        if len(_CORR_CACHE) > 256:
            _CORR_CACHE.clear()
        hit = _CORR_CACHE[key] = (r, root)
    return hit


class Whitener:
    """The cells divided by their noise, annulus by annulus, and the
    correlation that remains between adjacent channels.

    Different annuli hold different visibilities, so their noise is
    independent; inside an annulus adjacent channels are correlated as
    ``rho``.  ``self(cube)`` divides every cell by its noise and returns the
    vector of the usable cells, whose noise has unit variance and, per
    annulus, the correlation ``R_b``; :meth:`quad` and :meth:`simulate` use it.
    """

    def __init__(self, variance, ok, corr=None):
        nchan, nbins = variance.shape
        self.correlated = corr is not None
        self.blocks = []
        offset = 0
        for b in range(nbins):
            idx = np.flatnonzero(ok[:, b])
            if idx.size == 0:
                continue
            inv_sd = 1.0 / np.sqrt(variance[idx, b])
            r, root = _correlation(corr, idx) if self.correlated else (None, None)
            self.blocks.append((b, idx, inv_sd, r, root, slice(offset, offset + idx.size)))
            offset += idx.size
        self.size = offset

    def __call__(self, cube):
        """``(nchan, nbins[, k])`` -> ``(n[, k])``: usable cells / noise."""
        return np.concatenate([(inv_sd * cube[idx, b].T).T
                               for b, idx, inv_sd, _, _, _ in self.blocks])

    def per_bin(self, cube):
        """Like __call__, but one array per annulus: ``[(b, values)]``."""
        return [(b, (inv_sd * cube[idx, b].T).T) for b, idx, inv_sd, _, _, _ in self.blocks]

    def quad(self, u, v=None):
        """``u^T R v`` for whitened vectors or matrices: the covariance of
        the linear statistics ``u^T z`` and ``v^T z``."""
        v = u if v is None else v
        if not self.correlated:
            return u.T @ v
        out = 0.0
        for _, _, _, r, _, sl in self.blocks:
            out = out + u[sl].T @ (r @ v[sl])
        return out

    def quad_diag(self, u):
        """Diagonal of ``u^T R u`` for a matrix of many columns (the scan)."""
        if not self.correlated:
            return np.sum(u * u, axis=0)
        out = np.zeros(u.shape[1])
        for _, _, _, r, _, sl in self.blocks:
            out += np.sum(u[sl] * (r @ u[sl]), axis=0)
        return out

    def simulate(self, m, rng):
        """``m`` realisations of the whitened noise, ``(n, m)``."""
        e = rng.standard_normal((self.size, m))
        if not self.correlated:
            return e
        for _, _, _, _, root, sl in self.blocks:
            e[sl] = root @ e[sl]
        return e


# ---------------------------------------------------------------------------
# configuration of one fit
# ---------------------------------------------------------------------------
@dataclass
class FitSpec:
    """Which parameters are free, their bounds and starting values."""

    free: Tuple[str, ...]                    # among F, v0, fwhm, theta, Fk, v0k, fwhmk, C, theta_c
    values: Dict[str, float]                 # start (free) or value (fixed)
    bounds: Dict[str, Tuple[float, float]]
    n_lines: int = 1                         # lines in the model (2, 3, ...: Fk, v0k, fwhmk)
    tie: str = "free"                        # 'free' | 'centroid' | 'centroid+width'
    continuum: bool = False
    theta_prior: Optional[Tuple[float, float]] = None   # (theta_ref, sigma_dex): soft tie

    def params(self, x):
        p = dict(self.values)
        for name, value in zip(self.free, x):
            p[name] = float(value)
        for k in range(2, self.n_lines + 1):
            if self.tie in ("centroid", "centroid+width"):
                p[f"v0{k}"] = p["v0"]
            if self.tie == "centroid+width":
                p[f"fwhm{k}"] = p["fwhm"]
        return p


def model_cube(cube: Cube, p: dict):
    """The model of every cell for the parameters ``p``."""
    s1 = p["fwhm"] / FWHM_PER_SIGMA
    spec = p["F"] * channel_gaussian(cube.velocity, cube.dv, p["v0"], s1)
    g = cube.uv(p.get("theta", 0.0))
    m = spec[:, None] * g
    for k, vk in enumerate(cube.velocities or (), start=2):
        if f"F{k}" in p:
            sk = p[f"fwhm{k}"] / FWHM_PER_SIGMA
            speck = p[f"F{k}"] * channel_gaussian(vk, cube.dv, p[f"v0{k}"], sk)
            m = m + speck[:, None] * g
    if "C" in p:
        m = m + p["C"] * cube.uv(p.get("theta_c", p.get("theta", 0.0)))
    return m


@dataclass
class FitResult:
    params: Dict[str, float]
    errors: Dict[str, float]
    covariance: Optional[np.ndarray]
    free: Tuple[str, ...]
    chi2: float
    dof: int
    converged: bool
    size_model: str = "point"
    jac: Optional[np.ndarray] = None      # whitened Jacobian of the data rows, (n_cells, n_free)
    scale: float = 1.0                    # factor applied to the covariance (chi2_r if > 1)

    @property
    def chi2_reduced(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else float("nan")

    def as_dict(self) -> dict:
        p = self.params
        out = {"flux": p.get("F"), "flux_error": self.errors.get("F"),
               "v0_kms": p.get("v0"), "v0_error_kms": self.errors.get("v0"),
               "fwhm_kms": p.get("fwhm"), "fwhm_error_kms": self.errors.get("fwhm"),
               "peak": peak_flux_density(p.get("F"), p.get("fwhm")),
               "theta_fwhm_arcsec": p.get("theta", 0.0),
               "theta_error_arcsec": self.errors.get("theta", 0.0),
               "size_model": self.size_model,
               "chi2": self.chi2, "dof": self.dof, "chi2_reduced": self.chi2_reduced,
               "converged": self.converged, "free": list(self.free)}
        if "F" in self.errors and self.errors["F"] > 0:
            out["peak_error"] = peak_flux_density(self.errors["F"], p.get("fwhm"))
        extra = [f"{x}{k}" for k in range(2, 100) if f"F{k}" in p for x in ("F", "v0", "fwhm")]
        for k in extra + ["C", "theta_c"]:
            if k in p:
                out[k] = p[k]
                if k in self.errors:
                    out[k + "_error"] = self.errors[k]
        return out


def peak_flux_density(flux, fwhm):
    """Peak of a Gaussian line of integral ``flux`` and FWHM ``fwhm``."""
    if flux is None or fwhm is None or not np.isfinite(fwhm) or fwhm <= 0:
        return float("nan")
    return float(flux) / (float(fwhm) / FWHM_PER_SIGMA * math.sqrt(2 * math.pi))


def fit(cube: Cube, spec: FitSpec, whitener: Optional[Whitener] = None) -> FitResult:
    """Weighted least squares of the cube for the free parameters of
    ``spec``.  Errors from the sandwich covariance (weights and channel
    correlation of the noise model, absolute); scaled by sqrt(chi2_r) when
    that exceeds 1 (the correlation leaves the expected chi2 at the number of
    cells)."""
    from scipy.optimize import least_squares

    wh = whitener or Whitener(cube.variance, cube.ok, cube.corr)
    yw = wh(np.where(cube.ok, cube.real, 0.0))

    use_prior = spec.theta_prior is not None and "theta" in spec.free

    def residual(x):
        p = spec.params(x)
        r = wh(model_cube(cube, p)) - yw
        if use_prior:
            ref, sdex = spec.theta_prior
            r = np.append(r, np.log10(max(abs(p["theta"]), 1e-6) / ref) / sdex)
        return r

    names = spec.free
    x0 = np.array([spec.values[n] for n in names], dtype=float)
    lo = np.array([spec.bounds.get(n, (-np.inf, np.inf))[0] for n in names], dtype=float)
    hi = np.array([spec.bounds.get(n, (-np.inf, np.inf))[1] for n in names], dtype=float)
    pad = 1e-6 * (hi - lo)
    pad = np.where(np.isfinite(pad), pad, 0.0)
    x0 = np.minimum(np.maximum(x0, np.where(np.isfinite(lo), lo + pad, x0)),
                    np.where(np.isfinite(hi), hi - pad, x0))
    if len(names) == 0:
        r = residual(x0)
        return FitResult(spec.params(x0), {}, None, names, float(r @ r),
                         int(r.size), True)
    try:
        sol = least_squares(residual, x0, bounds=(lo, hi), x_scale="jac", method="trf",
                            max_nfev=200 * (len(names) + 1))
        x, ok = sol.x, bool(sol.success)
        jac, r = sol.jac, sol.fun
    except Exception:
        x, ok = x0, False
        r = residual(x0)
        jac = None
    chi2 = float(r @ r)
    dof = int(r.size - len(names))
    errors, cov = {}, None
    scale = 1.0
    if jac is not None:
        try:
            inv = np.linalg.pinv(jac.T @ jac)
            jd = jac[:wh.size]                              # the data rows (not the prior)
            middle = np.atleast_2d(wh.quad(jd))
            if use_prior:
                middle = middle + np.outer(jac[wh.size:], jac[wh.size:])
            cov = inv @ middle @ inv                        # sandwich: channel correlation
            scale = max(1.0, chi2 / dof) if dof > 0 else 1.0
            cov = cov * scale
            for i, n in enumerate(names):
                errors[n] = float(np.sqrt(max(cov[i, i], 0.0)))
        except np.linalg.LinAlgError:
            pass
    return FitResult(spec.params(x), errors, cov, names, chi2, dof, ok,
                     jac=None if jac is None else np.asarray(jac[:wh.size]), scale=scale)


# ---------------------------------------------------------------------------
# linear pieces: matched filter, scan, spectra
# ---------------------------------------------------------------------------
def _project_out(a, c):
    """Remove from the columns of ``a`` their projection on the columns of
    ``c`` (all whitened): the continuum term is linear, so it is fitted
    exactly by this projection in every linear step."""
    if c is None:
        return a
    q, _ = np.linalg.qr(c)
    return a - q @ (q.T @ a)


def scan_templates(cube: Cube, wh: Whitener, v0_grid, fwhm_grid, theta=0.0,
                   continuum=False, theta_c=None):
    """Weighted templates of a unit line on a grid of centres and widths.

    Returns ``(templates (n, G), norms (G,), grid [(v0, fwhm)], continuum
    column or None)``.  For data ``z = wh(y)`` the flux of grid point g is
    ``t_g^T z / (t_g^T t_g)`` and ``templates[:, g] = t_g / sd_g`` with
    ``sd_g^2 = t_g^T R t_g`` (the correlation), so ``templates.T @ z`` is the
    S/N of every grid point and ``norms = (t^T t) / sd`` converts it into a
    flux.  The continuum, if fitted, is projected out of the templates."""
    g = cube.uv(theta)
    grid = [(v0, w) for w in fwhm_grid for v0 in v0_grid]
    spec = np.stack([channel_gaussian(cube.velocity, cube.dv, v0, w / FWHM_PER_SIGMA)
                     for v0, w in grid], axis=1)                  # (nchan, G)
    cols = spec[:, None, :] * g[:, :, None]                         # (nchan, nbins, G)
    a = wh(np.where(cube.ok[:, :, None], cols, 0.0))
    ccol = None
    if continuum:
        gc = cube.uv(theta if theta_c is None else theta_c)
        ccol = wh(np.where(cube.ok, gc, 0.0))[:, None]
        a = _project_out(a, ccol)
    tt = np.sum(a * a, axis=0)
    sd = np.sqrt(np.maximum(wh.quad_diag(a), 0.0))
    ok = (tt > 0) & (sd > 0)
    sd = np.where(ok, sd, np.inf)
    norms = np.where(ok, tt / sd, np.inf)
    return (a / sd).astype(np.float32), norms, grid, ccol


class Scan:
    """The matched filter over a grid of centres, widths and source sizes.

    One block of templates per size (a point source and a few Gaussians):
    for a resolved source a point-source template collects only the
    compact part of the signal, and its S/N can be a fraction of the real
    one.  The statistic is the best S/N over the whole grid; its null
    distribution is computed with the same grid, so the extra freedom is
    paid for.
    """

    def __init__(self, cube, wh, v0_grid, fwhm_grid, thetas, continuum=False):
        self.cube, self.wh = cube, wh
        self.thetas = [float(t) for t in thetas]
        self.blocks = [scan_templates(cube, wh, v0_grid, fwhm_grid, theta=t,
                                      continuum=continuum) for t in self.thetas]

    def best(self, y=None):
        out = None
        for t, (tpl, norms, grid, ccol) in zip(self.thetas, self.blocks):
            r = scan(self.cube, self.wh, tpl, norms, grid, ccol, y=y)
            r["theta"] = t
            r["k"] = int(np.argmax(r["snr_grid"]))
            r["templates"] = tpl
            if out is None or r["snr"] > out["snr"]:
                out = r
        return out

    def null(self, n=1000, seed=1, chunk=200):
        rng = np.random.default_rng(seed)
        out, left = [], n
        while left > 0:
            m = min(chunk, left)
            z = self.wh.simulate(m, rng).astype(np.float32)
            out.append(np.max(np.stack([np.max(tpl.T @ z, axis=0)
                                        for tpl, _, _, _ in self.blocks]), axis=0))
            left -= m
        return np.concatenate(out).astype(float)


def scan(cube: Cube, wh: Whitener, templates, norms, grid, ccol=None, y=None):
    """Matched filter over the grid: best grid point and its S/N, flux and
    flux error.  ``y`` overrides the real part (e.g. the imaginary one)."""
    yw = wh(np.where(cube.ok, cube.real if y is None else y, 0.0))
    snr = (templates.T @ yw.astype(np.float32)).astype(float)   # continuum projected in the templates
    k = int(np.argmax(snr))
    return {"v0": grid[k][0], "fwhm": grid[k][1], "snr": float(snr[k]),
            "flux": float(snr[k] / norms[k]), "flux_error": float(1.0 / norms[k]),
            "snr_grid": snr}


def null_max_snr(templates, wh: Whitener, n=1000, seed=1, chunk=200):
    """Maximum S/N of the scan on ``n`` realisations of pure noise, drawn
    with the noise model of the fit (unit variance per cell after the
    weighting, correlated between adjacent channels)."""
    rng = np.random.default_rng(seed)
    out = []
    left = n
    while left > 0:
        m = min(chunk, left)
        out.append(np.max(templates.T @ wh.simulate(m, rng), axis=0))
        left -= m
    return np.concatenate(out)


def global_significance(snr, null):
    """p-value of a scan S/N from its null distribution, and the number of
    effective independent trials, which extrapolates beyond the simulated
    range: P(max < x) = Phi(x)^N_eff."""
    from scipy.stats import norm
    null = np.sort(np.asarray(null))
    n = null.size
    p_mc = (1 + int(np.sum(null >= snr))) / (n + 1)
    # N_eff from the upper half of the null distribution
    q = np.arange(1, n + 1) / (n + 1)
    upper = q > 0.5
    logphi = norm.logcdf(null[upper])
    n_eff = float(np.median(np.log(q[upper]) / np.where(logphi < 0, logphi, -1e-300)))
    n_eff = max(1.0, n_eff)
    p_ext = float(-np.expm1(n_eff * norm.logcdf(snr)))
    # its logarithm, which stays finite where p itself underflows (S/N > 37)
    log_p_ext = (math.log(n_eff) + float(norm.logsf(snr)) if p_ext < 1e-12
                 else math.log(max(p_ext, 1e-300)))
    p = p_mc if p_mc > 5.0 / (n + 1) else min(p_mc, p_ext)
    log_p = math.log(p) if p > 1e-300 else log_p_ext
    # one-sided Gaussian equivalent, 0 for p >= 0.5 (no evidence at all)
    sigma = max(0.0, _sigma_from_log_p(min(log_p, math.log(0.5))))
    return {"p_value": float(p), "log10_p_value": log_p / math.log(10),
            "p_value_mc": float(p_mc), "p_value_extrapolated": p_ext,
            "n_effective_trials": n_eff, "significance_sigma": sigma, "n_null": int(n)}


def _sigma_from_log_p(log_p):
    """One-sided Gaussian equivalent of a p-value given by its logarithm."""
    try:
        from scipy.special import ndtri_exp
        return float(-ndtri_exp(log_p))
    except ImportError:                            # SciPy < 1.10: asymptotic inverse
        from scipy.stats import norm
        if log_p > -700:
            return float(norm.isf(math.exp(log_p)))
        y = -2.0 * log_p
        return float(math.sqrt(y - math.log(y) - math.log(2 * math.pi)))


def zero_spacing_spectrum(cube: Cube, theta=0.0, y=None, variance=None):
    """Flux density of every channel at zero spacing, with its error.

    With the size fixed the model of a channel is linear in its flux,
    ``V(b) = S g(b)``, and the annuli are independent, so
    ``S = sum_b g y / var / sum_b g^2 / var`` and
    ``var(S) = 1 / sum_b g^2 / var``.  For a point source this is the
    noise-weighted mean over the annuli; for a resolved one each annulus is
    first corrected for the resolution.  Its integral over velocity is the
    line flux."""
    var = cube.variance if variance is None else variance
    y = cube.real if y is None else y
    g = cube.uv(theta)
    ok = cube.ok & np.isfinite(y) & (var > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        wgt = np.where(ok, g * g / var, 0.0)
        num = np.where(ok, g * y / var, 0.0).sum(axis=1)
        den = wgt.sum(axis=1)
        s = np.where(den > 0, num / den, np.nan)
        e = np.where(den > 0, 1.0 / np.sqrt(den), np.nan)
    return s, e


def window_flux(spectrum, error, dv, mask, corr=None):
    """Sum of a spectrum over the channels of ``mask`` times the channel
    width, with its error including the correlation between channels."""
    idx = np.flatnonzero(mask & np.isfinite(spectrum) & np.isfinite(error))
    if idx.size == 0:
        return float("nan"), float("nan"), 0
    flux = float(np.sum(spectrum[idx]) * dv)
    r = np.eye(idx.size) if corr is None else corr[np.ix_(idx, idx)]
    cov = r * np.outer(error[idx], error[idx])
    return flux, float(np.sqrt(max(cov.sum(), 0.0)) * dv), int(idx.size)


def influence(result: FitResult, name: str):
    """How a fitted parameter moves with the data: the whitened vector ``a``
    such that a change ``dz`` of the whitened cells changes the parameter by
    ``a^T dz`` (the least squares linearised at the solution).  None when the
    parameter is fixed or the Jacobian is not available."""
    if result.jac is None or name not in result.free:
        return None
    jac = result.jac
    inv = np.linalg.pinv(jac.T @ jac)
    return jac @ inv[:, result.free.index(name)]


def window_influence(cube: Cube, wh: Whitener, theta, mask, dv):
    """The same for the window flux ``dv * sum_W S_c`` of the zero-spacing
    spectrum at size ``theta`` (:func:`zero_spacing_spectrum`)."""
    var = cube.variance
    g = cube.uv(theta)
    ok = cube.ok & np.isfinite(cube.real) & (var > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.where(ok, g * g / var, 0.0).sum(axis=1)
        use = ok & np.asarray(mask, bool)[:, None] & (den > 0)[:, None]
        # coefficient of every cell y_cb, times var: wh() divides by sigma
        coef_var = np.where(use, dv * g / np.where(den > 0, den, 1.0)[:, None], 0.0)
    return wh(coef_var)


def gaussian_fraction(lo, hi, v0, fwhm):
    """Fraction of a Gaussian line (centre ``v0``, FWHM ``fwhm``) inside the
    velocity interval ``[lo, hi]``."""
    if fwhm is None or not np.isfinite(fwhm) or fwhm <= 0:
        return float("nan")
    s = float(fwhm) / FWHM_PER_SIGMA * math.sqrt(2.0)
    return float(0.5 * (erf((hi - v0) / s) - erf((lo - v0) / s)))


def residual_continuum(spectrum, error, velocity, mask, corr=None):
    """Offset and slope of a spectrum over the channels of ``mask`` (those
    outside the windows of the lines): ``S(v) = a + b v / 1000``, ``v`` in
    km/s from the first line, by least squares weighted by the errors, with
    the correlation between channels in the covariance (sandwich), scaled by
    the reduced chi square when it exceeds 1.  None with fewer than 4
    channels."""
    spectrum, error = np.asarray(spectrum, float), np.asarray(error, float)
    idx = np.flatnonzero(np.asarray(mask, bool) & np.isfinite(spectrum)
                         & np.isfinite(error) & (error > 0))
    if idx.size < 4:
        return None
    y, e = spectrum[idx], error[idx]
    x = np.asarray(velocity, float)[idx] / 1000.0
    X = np.column_stack([np.ones_like(x), x])
    w = 1.0 / e ** 2
    a_inv = np.linalg.pinv(X.T @ (w[:, None] * X))
    beta = a_inv @ (X.T @ (w * y))
    r = np.eye(idx.size) if corr is None else corr[np.ix_(idx, idx)]
    wx = w[:, None] * X
    cov = a_inv @ (wx.T @ (r * np.outer(e, e)) @ wx) @ a_inv
    chi2 = float(np.sum(w * (y - X @ beta) ** 2))
    dof = idx.size - 2
    cov = cov * max(1.0, chi2 / dof)
    ea, eb = (float(np.sqrt(max(cov[i, i], 0.0))) for i in (0, 1))
    a, b = float(beta[0]), float(beta[1])
    return {"offset": a, "offset_error": ea,
            "offset_nsigma": a / ea if ea > 0 else float("nan"),
            "slope_per_1000kms": b, "slope_error": eb,
            "slope_nsigma": b / eb if eb > 0 else float("nan"),
            "n_channels": int(idx.size),
            "velocity_range_kms": [float(np.min(velocity[idx])), float(np.max(velocity[idx]))],
            "chi2_reduced": chi2 / dof if dof > 0 else float("nan")}


def flux_per_annulus(cube: Cube, wh: Whitener, p: dict):
    """Line flux of every annulus at the fitted line shape (amplitude of the
    shape fitted annulus by annulus, continuum and second line held at their
    fitted values): the points of the amplitude-versus-baseline plot."""
    nb = cube.shape[1]
    flux, err = np.full(nb, np.nan), np.full(nb, np.nan)
    unit = dict(p, F=1.0, theta=0.0)
    for k in [k for k in unit if k[0] == "F" and k[1:].isdigit()]:
        unit.pop(k)
    unit.pop("C", None)
    shape = model_cube(cube, unit)                     # phi_c, constant over b
    rest = model_cube(cube, dict(p, F=0.0))            # second line + continuum
    data = dict(wh.per_bin(np.where(cube.ok, cube.real - rest, 0.0)))
    corr = {blk[0]: blk[3] for blk in wh.blocks}
    for b, a in wh.per_bin(np.where(cube.ok, shape, 0.0)):
        yw = data[b]
        den = float(a @ a)
        if den > 0:
            var = float(a @ (corr[b] @ a)) if corr.get(b) is not None else den
            flux[b] = float(a @ yw) / den
            err[b] = math.sqrt(max(var, 0.0)) / den
    return flux, err


__all__ = ["Cube", "Whitener", "FitSpec", "FitResult", "fit", "model_cube", "Scan",
           "annulus_nodes",
           "channel_gaussian", "uv_gaussian", "scan_templates", "scan", "null_max_snr",
           "global_significance", "zero_spacing_spectrum", "window_flux",
           "influence", "window_influence", "gaussian_fraction", "residual_continuum",
           "flux_per_annulus", "peak_flux_density", "FWHM_PER_SIGMA"]
