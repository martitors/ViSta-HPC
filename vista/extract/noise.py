"""
vista.extract.noise
===================
The noise of the stacked visibilities, measured four ways.

1. **From the weights**: a cell (channel x annulus) that averages
   visibilities with weights ``w`` has variance ``1 / sum w`` if the weights
   are ``1 / sigma^2`` in absolute units.  Often they are not: the weights may
   be relative, statwt may not have been run, Hanning smoothing correlates
   adjacent channels, and the ``WEIGHT`` column is the same for every channel,
   so it ignores the extra noise near atmospheric lines.
2. **From the scatter inside each cell** (statistics of format 4): the
   compression also accumulates ``sum w Re^2`` and ``sum w Im^2``, so for every
   cell

       sum w (x - xbar)^2 = sum w x^2 - (sum w x)^2 / sum w,
       E[...] = (N - 1) kappa^2

   where ``kappa^2 / w`` is the true variance of a visibility.  ``kappa``
   calibrates the weights, for every entry and every channel, without any
   assumption on their absolute scale.  A single visibility has S/N far below
   1, so the signal does not inflate the scatter.  This is the reference.
3. **From the imaginary part of the stack**: for sources centred at the phase
   centre and symmetric on average, Im V is pure noise.  ``mean (Im / sigma)^2``
   over the cells must be 1.
4. **From the real part in the line-free channels** after the continuum
   subtraction: ``mean (Re / sigma)^2`` there must be 1 too; residual continuum
   or baseline ripples would raise it.

Two more ingredients follow from the same data: the **correlation of the
noise between channels**, which a flux summed over channels needs, and the
**bootstrap over the sources** (in the extraction), whose excess over the
thermal noise is the scatter of the population.

The correlation between channels has two sources: the spectral response of
the correlator and the rebinning (adjacent channels, e.g. +0.67 at lag 1 for
Hanning smoothing), and the continuum subtraction, which removes from every
row a level fitted on the line-free channels: that level is noisy, so every
channel of the row shares its error -- a correlation at *all* lags, of order
1 / (effective number of fit channels), larger inside the line window than
outside.  On a narrow band it can double the error of a flux summed over the
line.  So the whole channel x channel correlation matrix is measured, from
the imaginary part of every source in every annulus (one noise vector each,
with the same spectral response and the same subtraction as the real part,
and no line), with the mean over the sources removed; it needs a few tens of
such vectors (otherwise: a model of the first lags, from the line-free cells).
For any fixed sum of channels its noise is then known to sqrt(2 / N_vectors).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import warnings

import numpy as np

# below this many degrees of freedom a channel of an entry takes the
# kappa of the whole entry instead of its own
MIN_DOF_CHANNEL = 50


# ---------------------------------------------------------------------------
# per entry
# ---------------------------------------------------------------------------
def scatter_kappa2(block: np.ndarray):
    """kappa^2 of one entry from the scatter inside its cells.

    Parameters
    ----------
    block
        ``(nchan, nbins, 7)`` accumulators of one entry.

    Returns
    -------
    (kappa2_channel, kappa2_entry, dof_entry, scatter_channel, dof_channel)
        kappa^2 per channel (the entry value where a channel has fewer than
        ``MIN_DOF_CHANNEL`` degrees of freedom; NaN where the entry has no
        data), the pooled value and its degrees of freedom, and the sums
        behind the channel values (to pool several entries).
    """
    w = block[..., 2]
    n = block[..., 4]
    ok = (w > 0) & (n >= 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        d_re = np.where(ok, block[..., 5] - block[..., 0] ** 2 / np.where(ok, w, 1), 0.0)
        d_im = np.where(ok, block[..., 6] - block[..., 1] ** 2 / np.where(ok, w, 1), 0.0)
    d = np.clip(d_re, 0, None) + np.clip(d_im, 0, None)
    dof = np.where(ok, 2.0 * (n - 1.0), 0.0)
    d_c, dof_c = d.sum(axis=1), dof.sum(axis=1)
    dof_e = float(dof_c.sum())
    k2_e = float(d_c.sum() / dof_e) if dof_e > 0 else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        k2_c = np.where(dof_c >= MIN_DOF_CHANNEL, d_c / np.maximum(dof_c, 1), k2_e)
    k2_c = np.where(w.sum(axis=1) > 0, k2_c, np.nan)
    return k2_c, k2_e, dof_e, d_c, dof_c


def entry_variance(block: np.ndarray, pb_gain: float = 1.0, use_scatter: bool = True):
    """Variance accumulator of one entry: ``kappa_c^2 * sum w`` per cell.

    It combines like the weights: the variance of ``sum w Re`` of a cell is
    ``kappa^2 sum w``.  The primary-beam correction scales ``sum w Re`` by A
    and ``sum w`` by A^2, so ``kappa^2`` is unchanged and this is computed on
    the corrected weights.

    Returns ``(variance (nchan, nbins), kappa2_entry, dof, scatter_channel,
    dof_channel)``.
    """
    w = block[..., 2] * (pb_gain ** 2 if pb_gain > 0 else 1.0)
    if use_scatter and block.shape[-1] >= 7:
        k2_c, k2_e, dof, d_c, dof_c = scatter_kappa2(block)
        k2 = np.where(np.isfinite(k2_c), k2_c, 0.0)
    else:
        k2, k2_e, dof = np.ones(block.shape[0]), 1.0, 0.0
        d_c = dof_c = np.zeros(block.shape[0])
    return k2[:, None] * w, k2_e, dof, d_c, dof_c


# ---------------------------------------------------------------------------
# stacked cells
# ---------------------------------------------------------------------------
def combine_variance(variance, sums, factors, multiplicity=None, renormalise=True):
    """Variance of the stacked real (and imaginary) part of every cell.

    Same combination as :func:`vista.extract.stack.combine`: with per-source
    gains ``g_s`` (multiplicity, divided by the total weight of the source
    when renormalised) and amplitude factors ``f_s``

        Re = sum_s g_s f_s R_s / sum_s g_s W_s
        Var(Re) = sum_s (g_s f_s)^2 V_s / (sum_s g_s W_s)^2

    A source drawn twice by the bootstrap enters with its noise twice, fully
    correlated, which the square of the gain accounts for.
    """
    n_src = sums.shape[0]
    if multiplicity is None:
        multiplicity = np.ones(n_src)
    multiplicity = np.asarray(multiplicity, dtype=float)
    if renormalise:
        total = sums[:, :, :, 2].sum(axis=(1, 2))
        gain = np.zeros_like(total)
        ok = total > 0
        gain[ok] = multiplicity[ok] / total[ok]
    else:
        gain = multiplicity
    num = np.einsum("s,scb->cb", (gain * factors) ** 2, variance)
    den = np.einsum("s,scb->cb", gain, sums[:, :, :, 2])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den ** 2, np.nan)


def _detrend(z):
    """Remove a straight line along the last axis (over the finite values of
    each row): a smooth signal left in the cells (residual continuum, the
    imaginary part of an offset source) would otherwise look like noise
    correlated over many channels."""
    ok = np.isfinite(z)
    x = np.broadcast_to(np.arange(z.shape[-1], dtype=float), z.shape)
    n = ok.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = np.where(ok, x, 0).sum(axis=-1, keepdims=True) / n
        zm = np.where(ok, z, 0).sum(axis=-1, keepdims=True) / n
        dx = np.where(ok, x - xm, 0.0)
        slope = (dx * np.where(ok, z - zm, 0)).sum(axis=-1, keepdims=True) / (dx ** 2).sum(
            axis=-1, keepdims=True)
    slope = np.where(np.isfinite(slope), slope, 0.0)
    out = z - zm - slope * (x - xm)
    return np.where(ok & (n >= 3), out, np.nan)


def channel_correlation(values: List[np.ndarray], max_lag: int = 3):
    """Correlation of the noise between channels ``k`` apart.

    Parameters
    ----------
    values
        Standardised noise samples ``x / sigma``, each ``(..., nchan)`` with
        NaN where unusable (e.g. the line channels): per-source cells of the
        imaginary part and of the line-free real part.

    Returns
    -------
    (rho, rho_error)
        ``rho[0] = 1`` and ``rho[k]`` for k = 1..max_lag, with their
        statistical errors.  A lag whose correlation is within 2 sigma of 0
        is set to 0: the noise of the estimate would otherwise leak into
        every error bar.
    """
    rho, err = [1.0], [0.0]
    for k in range(1, max_lag + 1):
        num, n, s0, s1 = 0.0, 0, 0.0, 0.0
        for z in values:
            if z.shape[-1] <= k:
                continue
            a, b = z[..., :-k], z[..., k:]
            ok = np.isfinite(a) & np.isfinite(b)
            num += float(np.sum(a[ok] * b[ok]))
            s0 += float(np.sum(a[ok] ** 2))
            s1 += float(np.sum(b[ok] ** 2))
            n += int(ok.sum())
        if n < 10 or s0 <= 0 or s1 <= 0:
            rho.append(0.0)
            err.append(float("nan"))
            continue
        r = float(np.clip(num / np.sqrt(s0 * s1), -0.45, 0.9))
        e = 1.0 / np.sqrt(n)
        rho.append(r if abs(r) > 2 * e else 0.0)
        err.append(float(e))
    return np.asarray(rho), np.asarray(err)


def correlation_matrix(z: np.ndarray, min_pairs: int = 10):
    """Channel x channel correlation of the noise from noise vectors.

    Parameters
    ----------
    z
        ``(n_vectors, nchan)`` standardised noise (NaN where absent): the
        imaginary part of every source in every annulus.

    Returns
    -------
    (R, n_vectors, rms)
        The correlation matrix (unit diagonal, positive semi-definite), the
        number of vectors with at least two channels, and the rms of the
        samples (about 1 if they are noise).  The mean over the vectors is
        removed first, so a signal common to all sources does not enter.
    """
    ok = np.isfinite(z)
    keep = ok.sum(axis=1) >= 2
    z, ok = z[keep], ok[keep]
    nchan = z.shape[1]
    if z.shape[0] == 0:
        return np.eye(nchan), 0, float("nan")
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)      # channels with no sample
        mean = np.nanmean(np.where(ok, z, np.nan), axis=0)
    z0 = np.where(ok, z - np.where(np.isfinite(mean), mean, 0.0), 0.0)
    rms = float(np.sqrt(np.sum(z0 ** 2) / max(ok.sum(), 1)))
    okf = ok.astype(float)
    pairs = okf.T @ okf
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(pairs >= min_pairs, (z0.T @ z0) / np.maximum(pairs, 1), 0.0)
        d = np.sqrt(np.diag(c))
        r = np.where(np.outer(d, d) > 0, c / np.outer(d, d), 0.0)
    np.fill_diagonal(r, 1.0)
    vals, vecs = np.linalg.eigh(r)
    if vals.min() < 0:
        r = (vecs * np.clip(vals, 0, None)) @ vecs.T
        d = np.sqrt(np.clip(np.diag(r), 1e-12, None))
        r = r / np.outer(d, d)
    return r, int(z.shape[0]), rms


def lag_summary(r: np.ndarray, mask: Optional[np.ndarray] = None, max_lag: int = 3):
    """Mean correlation at lags 1..max_lag (over the channels of ``mask``)."""
    n = r.shape[0]
    m = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    out = []
    for k in range(1, max_lag + 1):
        i = np.arange(n - k)
        sel = m[i] & m[i + k]
        out.append(float(np.mean(r[i[sel], i[sel] + k])) if sel.any() else float("nan"))
    return out


def toeplitz_correlation(rho: np.ndarray, n: int) -> np.ndarray:
    """``n x n`` correlation matrix of the channels from ``rho[k]``; made
    positive definite if the measured lags are not (clipping the
    eigenvalues)."""
    r = np.zeros(n)
    r[:min(n, len(rho))] = rho[:n]
    idx = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    m = r[idx]
    if len(rho) > 1 and np.any(rho[1:] != 0):
        vals, vecs = np.linalg.eigh(m)
        if vals.min() <= 1e-6:
            vals = np.clip(vals, 1e-6, None)
            m = (vecs * vals) @ vecs.T
            d = np.sqrt(np.diag(m))
            m = m / np.outer(d, d)
    return m


# ---------------------------------------------------------------------------
# the summary of the four estimates
# ---------------------------------------------------------------------------
@dataclass
class NoiseModel:
    """The noise of the stacked cells and how its estimates compare."""

    variance: np.ndarray                 # (nchan, nbins) adopted variance of Re (= of Im)
    variance_weights: np.ndarray         # (nchan, nbins) from the weights alone
    rho: np.ndarray                      # channel correlation at lags 0..K (summary)
    rho_error: np.ndarray
    kappa: float                         # sqrt(scatter / weights), all entries
    corr: Optional[np.ndarray] = None    # (nchan, nchan) adopted correlation, None: independent
    kappa_entries: Dict[int, float] = field(default_factory=dict)
    kappa_channel: Optional[np.ndarray] = None   # (nchan,) all entries
    source: str = "scatter"              # 'scatter' | 'weights'
    checks: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = {"adopted": self.source,
               "kappa_scatter_over_weights": self.kappa,
               "channel_correlation_method": self.checks.get("channel_correlation_method"),
               "channel_correlation": [float(x) for x in self.rho[1:]],
               "channel_correlation_error": [float(x) for x in self.rho_error[1:]],
               "kappa_per_entry": {str(k): float(v) for k, v in self.kappa_entries.items()}}
        out.update(self.checks)
        return out


def stack_checks(real, imaginary, variance, variance_weights, line_free, contsub,
                 rho=None):
    """The stacked-cell versions of estimates 1, 3 and 4, relative to 2.

    Returns a dict: ``imaginary_rms_over_noise`` (estimate 3),
    ``line_free_rms_over_noise`` (4, only after a continuum subtraction),
    ``weights_noise_over_noise`` (1), each expected to be 1, and the number
    of cells behind them.
    """
    out = {}
    ok = np.isfinite(variance) & (variance > 0)
    free = ok & line_free[:, None]
    zi = imaginary[free & np.isfinite(imaginary)] / np.sqrt(variance[free & np.isfinite(imaginary)])
    if zi.size:
        out["imaginary_rms_over_noise"] = float(np.sqrt(np.mean(zi ** 2)))
        out["imaginary_n_cells"] = int(zi.size)
    if contsub:
        sel = free & np.isfinite(real)
        zr = real[sel] / np.sqrt(variance[sel])
        if zr.size:
            out["line_free_rms_over_noise"] = float(np.sqrt(np.mean(zr ** 2)))
            out["line_free_n_cells"] = int(zr.size)
    both = ok & np.isfinite(variance_weights) & (variance_weights > 0)
    if both.any():
        out["weights_noise_over_noise"] = float(np.sqrt(np.median(variance_weights[both]
                                                                  / variance[both])))
    return out


def build_noise_model(sums, variance, variance_weights_src, factors, renormalise,
                      real, imaginary, line_free, contsub, kappa2_entries, dofs,
                      kappa_channel=None, source="scatter", max_lag=3,
                      correlate=True) -> NoiseModel:
    """Everything above, for the nominal stack.

    ``variance`` and ``variance_weights_src`` are the per-source variance
    accumulators (scatter-based and weights-based, ``(n_src, nchan, nbins)``).
    """
    var = combine_variance(variance, sums, factors, renormalise=renormalise)
    var_w = combine_variance(variance_weights_src, sums, factors, renormalise=renormalise)

    # the correlation between channels
    rho, rho_err = np.array([1.0]), np.array([0.0])
    corr, method = None, "not used"
    correlation_note = None
    extra = {}
    if correlate:
        nchan = sums.shape[1]
        w = sums[..., 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.where((w > 0) & (variance > 0), np.sqrt(variance) / w, np.nan)
            zi = sums[..., 1] / w / sd
            zr = sums[..., 0] / w / sd
        # 1. the whole matrix, from the imaginary part of every source and
        #    annulus (all channels: the imaginary part has no line)
        vectors = np.moveaxis(zi, 1, -1).reshape(-1, nchan)
        r_emp, n_vec, rms_im = correlation_matrix(vectors)
        if n_vec >= 30 and rms_im < 1.3:
            corr = r_emp
            method = f"measured, whole matrix (imaginary part, {n_vec} noise vectors)"
            rho = np.r_[1.0, lag_summary(corr, line_free, max_lag)]
            rho_err = np.r_[0.0, np.full(max_lag, 1.0 / np.sqrt(n_vec))]
            inside = ~line_free
            if inside.sum() > 2:
                ii = np.flatnonzero(inside)
                block = corr[np.ix_(ii, ii)]
                far = np.abs(np.subtract.outer(ii, ii)) > max_lag
                if far.any():
                    extra["channel_correlation_far_in_line_window"] = float(np.mean(block[far]))
        else:
            # 2. a model of the first lags, from the line-free cells
            mask = ~line_free
            zi_f = np.where(mask[None, :, None], np.nan, zi)
            samples = [np.moveaxis(zi_f, 1, -1)]
            if contsub:
                zr_f = np.where(mask[None, :, None], np.nan, zr)
                samples.append(np.moveaxis(zr_f, 1, -1))

            def rms_of(values):
                r = [float(np.sqrt(np.nanmean(z ** 2))) for z in values if np.isfinite(z).any()]
                return max(r) if r else float("nan")

            why = (f"only {n_vec} noise vectors" if n_vec < 30 else
                   f"the imaginary part scatters {rms_im:.1f} x the noise")
            rms = rms_of(samples)
            if rms < 1.2:
                rho, rho_err = channel_correlation(samples, max_lag)
            elif rms_of([_detrend(z) for z in samples]) < 1.5:
                rho, rho_err = channel_correlation([_detrend(z) for z in samples], max_lag)
                correlation_note = ("measured after removing a straight line per source and "
                                    "annulus: the cells off the line hold a smooth signal")
            else:
                rms = rms_of([_detrend(z) for z in samples])
                rho, rho_err = np.zeros(max_lag + 1), np.full(max_lag + 1, np.nan)
                rho[0], rho_err[0] = 1.0, 0.0
                correlation_note = (f"not measured: the cells off the line scatter "
                                    f"{rms:.1f} x the noise (signal or systematics, not noise)")
            if np.any(rho[1:] != 0):
                corr = toeplitz_correlation(rho, nchan)
                method = (f"model of lags 1-{max_lag} from the line-free cells ({why}); the "
                          f"correlation left by the continuum subtraction is not included")
            else:
                method = f"independent channels assumed ({why})"

    k2 = np.array([v for v in kappa2_entries.values() if np.isfinite(v)])
    d = np.array([dofs[k] for k, v in kappa2_entries.items() if np.isfinite(v)])
    kappa = float(np.sqrt(np.sum(k2 * d) / np.sum(d))) if d.size and d.sum() > 0 else 1.0
    checks = stack_checks(real, imaginary, var, var_w, line_free, contsub)
    checks["channel_correlation_method"] = method
    checks.update(extra)
    if correlation_note:
        checks["channel_correlation_note"] = correlation_note
    checks["entries_kappa_spread"] = (float(np.std(np.sqrt(k2)) / np.mean(np.sqrt(k2)))
                                      if k2.size > 1 else 0.0)
    return NoiseModel(variance=var, variance_weights=var_w, rho=rho, rho_error=rho_err,
                      corr=corr, kappa=kappa,
                      kappa_entries={k: float(np.sqrt(v)) for k, v in kappa2_entries.items()
                                     if np.isfinite(v)},
                      kappa_channel=kappa_channel, source=source, checks=checks)


__all__ = ["scatter_kappa2", "entry_variance", "combine_variance", "channel_correlation",
           "correlation_matrix", "lag_summary",
           "toeplitz_correlation", "NoiseModel", "stack_checks", "build_noise_model",
           "MIN_DOF_CHANNEL"]
