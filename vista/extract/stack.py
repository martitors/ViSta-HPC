"""
vista.extract.stack
===================
The stack of the sources from their compressed statistics, and the
measurement of the continuum (Sec. 5.3.2, 5.3.3).

The statistics of the individual sources are combined in a weighted average
(:func:`build_stack_input`, :func:`combine`), keeping only the annuli of
baseline length to which enough sources contribute, so that no point is set
by a single object.  The weighting, the redshift rescaling, the primary-beam
correction and the flux normalisation enter here and never touch the
visibilities on disk: the same stack can be measured in different ways.

:func:`measure_continuum` measures the flux density and the size of the
stacked continuum, with errors from the bootstrap over the sources.  The
lines are measured by :func:`vista.extract.measure_line`.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .. import _reset_clock, _say, _warn
from .config import BootstrapConfig, Lines, UVFitConfig, WeightingConfig
from .sources import (amplitude_factors, group_by_position,
                      read_input_list)
from .statistics import SufficientStatistics
from .uvfit import UVFitResult, fit_uv_profile


# ---------------------------------------------------------------------------
# per-source combination
# ---------------------------------------------------------------------------
@dataclass
class StackInput:
    """Statistics of the selected sources, ready to be stacked."""

    sums: np.ndarray                  # (n_src, nchan, nbins, 7) (5: format 3)
    factors: np.ndarray               # (n_src,) amplitude normalisation
    coordinates: List[tuple]          # (n_src,) (RA, Dec) in degrees
    redshifts: np.ndarray
    freq_grid: np.ndarray
    bin_edges_lambda: np.ndarray
    n_entries: int = 0
    z_ref: Optional[float] = None
    norm_ref: Optional[float] = None
    # noise (vista.extract.noise): per-source variance accumulators of the
    # real part, from the scatter inside the cells and from the weights alone
    variance: Optional[np.ndarray] = None            # (n_src, nchan, nbins)
    variance_weights: Optional[np.ndarray] = None    # (n_src, nchan, nbins)
    kappa2_entries: Dict[int, float] = field(default_factory=dict)
    dof_entries: Dict[int, float] = field(default_factory=dict)
    kappa2_channel: Optional[np.ndarray] = None      # (nchan,) all entries
    has_scatter: bool = False
    meta: dict = field(default_factory=dict)         # of the statistics file

    @property
    def n_sources(self) -> int:
        return self.sums.shape[0]


def build_stack_input(statistics: Union[str, SufficientStatistics],
                      input_list: Union[str, Sequence],
                      weighting: WeightingConfig = WeightingConfig(),
                      match_tol_arcsec: float = 2.0,
                      verbose: bool = True) -> StackInput:
    """Group the entries into physical sources and co-add their statistics.

    Several lines of the input list may describe the same object, observed in
    different epochs, configurations or spectral windows; they are recognised
    by their coordinates, within ``match_tol_arcsec``.  Their accumulators are
    additive, so they are simply summed, which combines the repeated
    observations with their native weights, i.e. naturally, preserving the
    intra-source sensitivity.  The democratic renormalisation is then applied
    to the resulting single measurement, as required by Eq. (26).

    Parameters
    ----------
    statistics
        File written by
        :func:`~vista.extract.statistics.compress_visibilities`, or an already
        loaded :class:`SufficientStatistics`.
    input_list
        The ViSta input list, or the already parsed entries.
    weighting
        Weighting and flux normalisation options.
    match_tol_arcsec
        Angular tolerance used to decide that two entries are the same source.
    """
    stats = (SufficientStatistics.load(statistics)
             if isinstance(statistics, str) else statistics)
    entries = (read_input_list(input_list) if isinstance(input_list, str)
               else list(input_list))

    available = [e for e in entries if e is not None and e.dd in stats.sums]
    if not available:
        raise RuntimeError("no entry of the input list is present in the "
                           "statistics file: are they from the same stack?")

    groups, centres = group_by_position(available, match_tol_arcsec)
    factors, z_ref, norm_ref, missing = amplitude_factors(groups, weighting,
                                                          verbose=verbose)
    if missing:
        dropped = ", ".join(f"dd {groups[k][0].dd}" for k in missing)
        _warn(f"extraction: no normalisation value, sources dropped: {dropped}")
    keep = [k for k in range(len(groups)) if k not in set(missing)]
    if not keep:
        raise RuntimeError("every source was dropped by the normalisation")

    # primary beam: amplitudes / A, weights * A^2, entry by entry, so that
    # different pointings of the same source are corrected separately
    pb_gain = (stats.meta or {}).get("primary_beam", {})
    if weighting.primary_beam:
        if pb_gain:
            values = [pb_gain.get(e.dd, 1.0) for e in available]
            _say(f"  primary beam correction: A from {min(values):.3f} "
                 f"to {max(values):.3f}", verbose)
        else:
            _warn("extraction: no primary beam values in the statistics "
                  "file (compressed with an older version): not corrected")

    from .noise import entry_variance

    has_scatter = stats.has_scatter
    stacked, shapes = [], set()
    variances, variances_w = [], []
    kappa2, dofs = {}, {}
    d_all = dof_all = None
    for k in keep:
        total = var_total = varw_total = None
        for entry in groups[k]:
            raw = np.asarray(stats.sums[entry.dd], dtype=np.float64)
            gain = float(pb_gain.get(entry.dd, 1.0))
            applied = weighting.primary_beam and gain > 0 and gain != 1.0
            block = raw
            if applied:
                block = raw.copy()
                block[..., 0:2] *= gain          # sum w Re, sum w Im
                block[..., 2:4] *= gain * gain   # sum w, sum w b
                # sum w Re^2, sum w Im^2: unchanged (w A^2 (Re / A)^2)
            var, k2_e, dof, d_c, dof_c = entry_variance(raw, gain if applied else 1.0,
                                                        use_scatter=has_scatter)
            kappa2[entry.dd], dofs[entry.dd] = k2_e, dof
            d_all = d_c if d_all is None else d_all + d_c
            dof_all = dof_c if dof_all is None else dof_all + dof_c
            shapes.add(block.shape)
            total = block.copy() if total is None else total + block
            var_total = var if var_total is None else var_total + var
            w_corr = block[..., 2]
            varw_total = w_corr.copy() if varw_total is None else varw_total + w_corr
        stacked.append(total)
        variances.append(var_total)
        variances_w.append(varw_total)
    if len(shapes) > 1:
        raise ValueError(f"the statistics were compressed on different grids: "
                         f"shapes {sorted(shapes)}")

    n_multi = sum(1 for k in keep if len(groups[k]) > 1)
    _say(f"  {len(keep)} sources from {len(available)} entries "
         f"({n_multi} observed more than once), {weighting.scheme} weighting, "
         f"flux normalisation {'on' if norm_ref else 'off'}", verbose)

    with np.errstate(invalid="ignore", divide="ignore"):
        k2_channel = (np.where(dof_all > 0, d_all / np.maximum(dof_all, 1), np.nan)
                      if has_scatter else None)
    return StackInput(
        sums=np.stack(stacked),
        factors=factors[keep],
        coordinates=[centres[k] for k in keep],
        redshifts=np.array([groups[k][0].z for k in keep], dtype=float),
        freq_grid=np.asarray(stats.freq_grid, dtype=float),
        bin_edges_lambda=np.asarray(stats.bin_edges_lambda, dtype=float),
        n_entries=len(available), z_ref=z_ref, norm_ref=norm_ref,
        variance=np.stack(variances), variance_weights=np.stack(variances_w),
        kappa2_entries=kappa2, dof_entries=dofs, kappa2_channel=k2_channel,
        has_scatter=has_scatter, meta=dict(stats.meta or {}))


def combine(sums, factors, multiplicity=None, renormalise=True):
    """Weighted average of the sources, channel by channel and annulus by
    annulus.

    Parameters
    ----------
    sums
        ``(n_src, nchan, nbins, 5)`` accumulators.
    factors
        Per-source amplitude factor (Sec. 3.4.2).
    multiplicity
        Per-source multiplicity of a bootstrap realisation.  ``None`` gives
        the nominal stack, i.e. all multiplicities equal to one.
    renormalise
        ``True`` (democratic) rescales the weights of each source so that
        they sum to one; ``False`` (natural) uses them as they are.

    Returns
    -------
    (real, imaginary, weight, mean_baseline)
        All ``(nchan, nbins)``.
    """
    if multiplicity is None:
        multiplicity = np.ones(sums.shape[0])
    multiplicity = np.asarray(multiplicity, dtype=float)

    if renormalise:
        total = sums[:, :, :, 2].sum(axis=(1, 2))
        ok = total > 0
        gain = np.zeros_like(total)
        gain[ok] = multiplicity[ok] / total[ok]
    else:
        gain = multiplicity

    numerator_re = np.einsum("s,s,scb->cb", gain, factors, sums[:, :, :, 0])
    numerator_im = np.einsum("s,s,scb->cb", gain, factors, sums[:, :, :, 1])
    denominator = np.einsum("s,scb->cb", gain, sums[:, :, :, 2])
    baseline = np.einsum("s,scb->cb", gain, sums[:, :, :, 3])
    with np.errstate(invalid="ignore", divide="ignore"):
        real = np.where(denominator > 0, numerator_re / denominator, np.nan)
        imaginary = np.where(denominator > 0, numerator_im / denominator,
                             np.nan)
        mean_b = np.where(denominator > 0, baseline / denominator, np.nan)
    return real, imaginary, denominator, mean_b


def sources_per_bin(sums, channel_mask) -> np.ndarray:
    """Number of sources contributing to each annulus, over the given channels."""
    weight = sums[..., 2][:, channel_mask, :].sum(axis=1)
    return (weight > 0).sum(axis=0)


def sources_per_channel(sums) -> np.ndarray:
    """Number of sources contributing to each channel of the common grid."""
    return (sums[:, :, :, 2].sum(axis=2) > 0).sum(axis=0)


# ---------------------------------------------------------------------------
# the continuum
# ---------------------------------------------------------------------------
def mean_per_annulus(real, weight, channels=None):
    """Weighted mean over the channels (all, or those of the mask
    ``channels``) in every annulus."""
    if channels is None:
        channels = np.ones(real.shape[0], dtype=bool)
    w = np.nansum(weight[channels], axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(w > 0, np.nansum((real * weight)[channels], axis=0) / w, np.nan)


def mean_per_channel(real, weight):
    """Weighted mean over the annuli in every channel."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nansum(real * weight, axis=1) / np.nansum(weight, axis=1)


@dataclass
class ContinuumResult:
    """The continuum of a stack: ``summary`` (every number), its mean per
    channel (``spectrum``) and per annulus of baseline length (``flux``,
    with the imaginary part as a null test), and the fit of the source."""

    summary: dict
    freq_hz: np.ndarray
    velocity_kms: np.ndarray
    spectrum: np.ndarray
    spectrum_error: np.ndarray
    sources_per_channel: np.ndarray
    baseline_klambda: np.ndarray
    flux: np.ndarray
    flux_error: np.ndarray
    flux_imaginary: np.ndarray
    sources_per_annulus: np.ndarray
    bin_ok: np.ndarray
    fit: UVFitResult

    def save(self, prefix: str) -> List[str]:
        """Write ``<prefix>_results.json``, ``_spectrum.txt``, ``_uvamp.txt``."""
        with open(f"{prefix}_results.json", "w") as fh:
            json.dump(self.summary, fh, indent=2)
        np.savetxt(f"{prefix}_spectrum.txt",
                   np.column_stack([self.velocity_kms, self.spectrum, self.spectrum_error,
                                    self.sources_per_channel]),
                   header="v_kms  Re_stack  err_bootstrap  n_sources")
        np.savetxt(f"{prefix}_uvamp.txt",
                   np.column_stack([self.baseline_klambda, self.flux, self.flux_error,
                                    self.flux_imaginary, self.sources_per_annulus]),
                   header="b_klambda  flux  err_bootstrap  flux_imaginary(null_test)  n_sources")
        return [f"{prefix}_results.json", f"{prefix}_spectrum.txt", f"{prefix}_uvamp.txt"]


def measure_continuum(statistics: Union[str, SufficientStatistics],
                      input_list: Union[str, Sequence],
                      line: Lines,
                      weighting: WeightingConfig = WeightingConfig(),
                      uvfit: UVFitConfig = UVFitConfig(),
                      bootstrap: BootstrapConfig = BootstrapConfig(),
                      overlap_frac: float = 1.0,
                      match_tol_arcsec: float = 2.0,
                      output_prefix: Optional[str] = None,
                      verbose: bool = True) -> ContinuumResult:
    """Flux density and size of the stacked continuum (Sec. 5.3).

    1. The statistics of the sources are combined (weighting, redshift
       rescaling, primary beam: :func:`build_stack_input`, :func:`combine`).
    2. In every annulus of baseline length the continuum is the weighted
       mean over the channels that enough sources cover.
    3. The source model (``uvfit``: a Gaussian or a point) is fitted to
       those means against the baseline: its value at zero baseline is the
       flux density, its width gives the size.
    4. The errors come from the bootstrap over the sources: every
       realisation repeats steps 1 to 3.

    The same on the imaginary part is a null test: it must be zero.

    Parameters
    ----------
    statistics, input_list, weighting, match_tol_arcsec
        Passed to :func:`build_stack_input`: the statistics of the continuum
        (of ``MODEL_DATA`` after a subtraction, or of ``DATA`` for a stack
        without lines).
    line
        The lines of the stack: the velocities of the spectrum refer to the
        first.
    uvfit, bootstrap
        See :mod:`vista.extract.config`.
    overlap_frac
        Only the channels covered by at least this fraction of the sources
        are used (1: by all of them).
    output_prefix
        Also write ``<prefix>_results.json``, ``_spectrum.txt`` and
        ``_uvamp.txt``.
    """
    _reset_clock()
    source = statistics if isinstance(statistics, str) else "statistics in memory"
    _say(f"Continuum: {source}, {uvfit.model} source model, "
         f"{bootstrap.n_realisations} bootstrap realisations", verbose)
    data = build_stack_input(statistics, input_list, weighting,
                             match_tol_arcsec=match_tol_arcsec, verbose=verbose)
    sums, factors, freq = data.sums, data.factors, data.freq_grid
    renormalise = weighting.renormalise
    n_sources = data.n_sources

    # ---- the stack, and the channels to use -------------------------------
    real, imaginary, weight, mean_b = combine(sums, factors, renormalise=renormalise)
    per_channel = sources_per_channel(sums)
    with np.errstate(invalid="ignore", divide="ignore"):    # empty annuli: NaN
        baseline = np.nansum(mean_b * weight, axis=0) / np.nansum(weight, axis=0)
    needed = int(np.ceil(overlap_frac * n_sources))
    channels = per_channel >= needed
    if not channels.any():
        best = int(per_channel.max())
        _warn(f"continuum: no channel is covered by {needed} sources, the best coverage "
              f"({best} sources) is used")
        channels = per_channel >= best
    effective_freq = float(np.nansum(freq[channels]) / channels.sum())
    _say(f"  {int(channels.sum())} channels, effective frequency "
         f"{effective_freq / 1e9:.2f} GHz", verbose)

    # ---- the continuum per annulus, and the annuli to fit -----------------
    flux = mean_per_annulus(real, weight, channels)
    flux_imaginary = mean_per_annulus(imaginary, weight, channels)
    per_annulus = sources_per_bin(sums, channels)
    minimum = max(uvfit.min_sources_per_bin, 2)
    bin_ok = per_annulus >= minimum

    # ---- bootstrap over the sources: the same on every resampling ---------
    rng = np.random.default_rng(bootstrap.seed)
    boot_flux, boot_spectrum = [], []
    for _ in range(bootstrap.n_realisations):
        multiplicity = np.bincount(rng.integers(0, n_sources, n_sources),
                                   minlength=n_sources).astype(float)
        r, _, w, _ = combine(sums, factors, multiplicity=multiplicity, renormalise=renormalise)
        boot_spectrum.append(mean_per_channel(r, w))
        boot_flux.append(mean_per_annulus(r, w, channels))
    boot_flux = np.asarray(boot_flux)
    with warnings.catch_warnings():              # empty annuli / channels: NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        flux_error = np.nanstd(boot_flux, axis=0)
        spectrum_error = np.nanstd(np.asarray(boot_spectrum), axis=0)
    # a floor on the errors: with few sources the bootstrap can make one too small
    usable = np.isfinite(flux_error) & (flux_error > 0) & bin_ok
    if usable.any() and uvfit.error_floor_frac > 0:
        flux_error = np.maximum(flux_error,
                                uvfit.error_floor_frac * np.nanmedian(flux_error[usable]))

    # ---- the source model: on the stack, and on every resampling ----------
    fit = fit_uv_profile(baseline, flux, sigma=flux_error, config=uvfit, bin_ok=bin_ok)
    boot_fits = [fit_uv_profile(baseline, values, sigma=flux_error, config=uvfit, bin_ok=bin_ok)
                 for values in boot_flux]
    boot_fit_flux = np.asarray([b.flux for b in boot_fits], dtype=float)
    boot_fit_theta = np.asarray([b.theta_arcsec for b in boot_fits], dtype=float)
    ok = np.isfinite(boot_fit_flux) & np.isfinite(boot_fit_theta)

    def spread(values):
        return float(np.nanstd(values[ok])) if ok.any() else float("nan")

    def percentiles(values):
        return ([float(x) for x in np.nanpercentile(values[ok], [16, 50, 84])]
                if ok.any() else [float("nan")] * 3)

    finite = np.isfinite(flux)
    weight_per_bin = np.nansum(weight[channels], axis=0)
    flux_point = float(np.nansum(flux[finite] * weight_per_bin[finite])
                       / np.nansum(weight_per_bin[finite]))
    null_test = np.isfinite(flux_imaginary) & bin_ok
    summary = {
        "n_sources": n_sources,
        "n_entries": data.n_entries,
        "source_coordinates_deg": [[round(ra, 6), round(dec, 6)]
                                   for ra, dec in data.coordinates],
        "weighting_scheme": weighting.scheme,
        "weights_renormalised": renormalise,
        "redshift_rescaling": weighting.redshift_rescaling,
        "z_ref": data.z_ref,
        "z_ref_given": weighting.z_ref is not None,
        "flux_normalised": data.norm_ref is not None,
        "flux_ref": data.norm_ref,
        "flux_ref_given": weighting.norm_ref is not None,
        "rest_freq_ghz": line.rest_freq_ghz,
        "flux_total": fit.flux,
        "flux_total_error": spread(boot_fit_flux),
        "flux_percentiles_16_50_84": percentiles(boot_fit_flux),
        "theta_fwhm_arcsec": fit.theta_arcsec,
        "theta_error_arcsec": spread(boot_fit_theta),
        "theta_percentiles_16_50_84": percentiles(boot_fit_theta),
        "theta_at_bound": fit.theta_at_bound,
        "theta_max_arcsec": uvfit.theta_max_arcsec,
        "bootstrap_fraction_at_bound": (
            float(np.mean(boot_fit_theta[np.isfinite(boot_fit_theta)]
                          > 0.95 * uvfit.theta_max_arcsec))
            if np.isfinite(boot_fit_theta).any() else float("nan")),
        "n_bootstrap": bootstrap.n_realisations,
        "n_bootstrap_failed": int((~ok).sum()),
        "flux_point_source": flux_point,
        "min_sources_per_bin": minimum,
        "n_bins_in_fit": fit.n_bins,
        "chi2_reduced": fit.chi2_reduced,
        "model": fit.model,
        "fit_params": fit.params,
        "n_sources_min_in_window": int(per_channel[channels].min()),
        "null_test_max_imaginary_over_sigma": (
            float(np.nanmax(np.abs(flux_imaginary[null_test])
                            / np.maximum(flux_error[null_test], 1e-30)))
            if null_test.any() else float("nan")),
        "units": "flux density in [data units]; the flux normalisation rescales it by "
                 "flux_ref / NORM",
        "continuum_effective_freq_ghz": effective_freq / 1e9,
        "continuum_n_channels": int(channels.sum()),
    }
    result = ContinuumResult(
        summary=summary, freq_hz=freq, velocity_kms=line.velocity(freq),
        spectrum=mean_per_channel(real, weight), spectrum_error=spectrum_error,
        sources_per_channel=per_channel, baseline_klambda=baseline / 1e3,
        flux=flux, flux_error=flux_error, flux_imaginary=flux_imaginary,
        sources_per_annulus=per_annulus, bin_ok=bin_ok, fit=fit)
    if output_prefix:
        _say("  written: " + ", ".join(result.save(output_prefix)), verbose)
    _say(f"Continuum done: S = {fit.flux:.4g} +- {summary['flux_total_error']:.3g}   "
         f"theta = {fit.theta_arcsec:.3f}\" +- {summary['theta_error_arcsec']:.3f}\"   "
         f"chi2r = {fit.chi2_reduced:.2f}", verbose)
    return result


__all__ = ["StackInput", "build_stack_input", "combine", "sources_per_bin",
           "sources_per_channel", "mean_per_annulus", "mean_per_channel",
           "measure_continuum", "ContinuumResult"]
