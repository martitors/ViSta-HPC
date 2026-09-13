"""
vista.extract.stack
===================
Combination of the per-source statistics into the stack, and the full
extraction chain (Sec. 5.3.2, 5.3.3).

The compressed statistics of the individual sources are combined through a
weighted average, retaining only the radial bins to which at least a minimum
number of sources contribute, so that no point of the profile is dominated
by a single, potentially unrepresentative source.  The weighting scheme and
the flux normalisation enter here, at the extraction stage, and never touch
the visibilities stored on disk: the stacked MS always keeps its native
amplitudes and the same dataset can be reused for different tests.

All the uncertainties come from bootstrap resampling of the source sample.
Each realisation repeats the whole chain, from the combination of the
statistics through the line shape and the uv fits, so that the scatter
across realisations captures both the measurement noise and the variance
among the sources of the population.  As a null test, the same analysis
applied to the imaginary part is verified to be consistent with zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .config import (BootstrapConfig, LineConfig, ProfileConfig, UVFitConfig,
                     WeightingConfig)
from .profiles import (LineShape, fit_line_shape, flux_per_annulus_continuum,
                       flux_per_annulus_template, flux_per_annulus_window,
                       integrated_spectrum)
from .sources import (Entry, amplitude_factors, group_by_position,
                      read_input_list)
from .statistics import SufficientStatistics
from .uvfit import UVFitResult, fit_uv_profile, gaussian_visibility


# ---------------------------------------------------------------------------
# per-source combination
# ---------------------------------------------------------------------------
@dataclass
class StackInput:
    """Statistics of the selected sources, ready to be stacked."""

    sums: np.ndarray                  # (n_src, nchan, nbins, 5)
    factors: np.ndarray               # (n_src,) amplitude normalisation
    coordinates: List[tuple]          # (n_src,) (RA, Dec) in degrees
    redshifts: np.ndarray
    freq_grid: np.ndarray
    bin_edges_lambda: np.ndarray
    n_entries: int = 0
    z_ref: Optional[float] = None
    norm_ref: Optional[float] = None

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
    if missing and verbose:
        dropped = ", ".join(f"dd={groups[k][0].dd}" for k in missing)
        print(f"[norm] no normalisation value, dropped: {dropped}")
    keep = [k for k in range(len(groups)) if k not in set(missing)]
    if not keep:
        raise RuntimeError("every source was dropped by the normalisation")

    stacked, shapes = [], set()
    for k in keep:
        total = None
        for entry in groups[k]:
            block = np.asarray(stats.sums[entry.dd], dtype=np.float64)
            shapes.add(block.shape)
            total = block.copy() if total is None else total + block
        stacked.append(total)
    if len(shapes) > 1:
        raise ValueError(f"the statistics were compressed on different grids: "
                         f"shapes {sorted(shapes)}")

    if verbose:
        n_multi = sum(1 for k in keep if len(groups[k]) > 1)
        print(f"[stack] {len(keep)} sources from {len(available)} entries "
              f"({n_multi} observed more than once)  "
              f"scheme={weighting.scheme}  "
              f"flux-normalised={'yes' if norm_ref else 'no'}")

    return StackInput(
        sums=np.stack(stacked),
        factors=factors[keep],
        coordinates=[centres[k] for k in keep],
        redshifts=np.array([groups[k][0].z for k in keep], dtype=float),
        freq_grid=np.asarray(stats.freq_grid, dtype=float),
        bin_edges_lambda=np.asarray(stats.bin_edges_lambda, dtype=float),
        n_entries=len(available), z_ref=z_ref, norm_ref=norm_ref)


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


def flux_per_channel(real, weight, baseline_lambda, theta_arcsec,
                     bin_ok=None):
    """Flux density of the source in each channel, extrapolated to b = 0.

    With the size held fixed the source model is linear in the flux,
    ``V(b) = F * g(b)`` with ``g(b) = exp[-(pi theta b)^2 / (4 ln 2)]``, so
    the flux of a channel is the weighted least squares solution

    ``F = sum_b w g Re(V) / sum_b w g^2``

    over the annuli.  This is the spectrum an image-plane fit would give:
    the amplitude is corrected for the resolution of every annulus instead of
    being averaged with it, so its units are those of the data, not of a
    visibility average, and its integral over velocity is the total line flux.

    Parameters
    ----------
    real, weight
        Stacked real part and summed weights, both ``(nchan, nbins)``.
    baseline_lambda
        Mean baseline length of each annulus, in units of the wavelength.
    theta_arcsec
        Source size to assume, normally the one fitted on the line profile.
        ``0`` treats the source as unresolved and the result is the plain
        weighted average.
    bin_ok
        Annuli to use.
    """
    g = gaussian_visibility(np.asarray(baseline_lambda, float), 1.0,
                            theta_arcsec)
    usable = np.isfinite(g) & (g > 1e-3)
    if bin_ok is not None:
        usable &= np.asarray(bin_ok, bool)
    if not usable.any():
        return np.full(real.shape[0], np.nan)
    w = np.where(np.isfinite(weight[:, usable]), weight[:, usable], 0.0)
    r = np.where(np.isfinite(real[:, usable]), real[:, usable], 0.0)
    numerator = np.nansum(w * g[usable] * r, axis=1)
    denominator = np.nansum(w * g[usable] ** 2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, numerator / denominator, np.nan)


def sources_per_bin(sums, channel_mask) -> np.ndarray:
    """Number of sources contributing to each annulus, over the given channels."""
    weight = sums[..., 2][:, channel_mask, :].sum(axis=1)
    return (weight > 0).sum(axis=0)


def sources_per_channel(sums) -> np.ndarray:
    """Number of sources contributing to each channel of the common grid."""
    return (sums[:, :, :, 2].sum(axis=2) > 0).sum(axis=0)


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------
@dataclass
class ExtractionResult:
    """Everything the extraction produces, ready to be saved or plotted."""

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
    shape: Optional[LineShape] = None
    second: Optional[dict] = None
    continuum: Optional[dict] = None
    spectrum_zero_baseline: Optional[np.ndarray] = None
    spectrum_zero_baseline_error: Optional[np.ndarray] = None
    zero_baseline_shape: Optional[LineShape] = None

    def _save_profile(self, prefix: str, tag: str, block: dict) -> str:
        path = f"{prefix}_uvamp_{tag}.txt"
        np.savetxt(path, np.column_stack([self.baseline_klambda,
                                          block["flux"], block["flux_error"],
                                          block["flux_imaginary"],
                                          self.sources_per_annulus]),
                   header="b_klambda  flux  err_bootstrap  "
                          "flux_imaginary(null_test)  n_sources")
        return path

    def save(self, prefix: str) -> List[str]:
        """Write ``<prefix>_results.json``, ``_spectrum.txt``, ``_uvamp.txt``."""
        written = []
        with open(f"{prefix}_results.json", "w") as fh:
            json.dump(self.summary, fh, indent=2)
        written.append(f"{prefix}_results.json")
        np.savetxt(f"{prefix}_spectrum.txt",
                   np.column_stack([self.velocity_kms, self.spectrum,
                                    self.spectrum_error,
                                    self.sources_per_channel]),
                   header="v_kms  Re_stack  err_bootstrap  n_sources")
        written.append(f"{prefix}_spectrum.txt")
        if self.spectrum_zero_baseline is not None:
            path = f"{prefix}_spectrum_zerob.txt"
            np.savetxt(path, np.column_stack([
                self.velocity_kms, self.spectrum_zero_baseline,
                self.spectrum_zero_baseline_error,
                self.sources_per_channel]),
                header="v_kms  flux_density  err_bootstrap  n_sources")
            written.append(path)
        np.savetxt(f"{prefix}_uvamp.txt",
                   np.column_stack([self.baseline_klambda, self.flux,
                                    self.flux_error, self.flux_imaginary,
                                    self.sources_per_annulus]),
                   header="b_klambda  flux  err_bootstrap  "
                          "flux_imaginary(null_test)  n_sources")
        written.append(f"{prefix}_uvamp.txt")
        if self.second is not None:
            written.append(self._save_profile(prefix, "second_line",
                                              self.second))
        if self.continuum is not None:
            written.append(self._save_profile(prefix, "continuum",
                                              self.continuum))
        return written


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
def extract_flux(statistics: Union[str, SufficientStatistics],
                 input_list: Union[str, Sequence],
                 line: LineConfig,
                 weighting: WeightingConfig = WeightingConfig(),
                 profile: ProfileConfig = ProfileConfig(),
                 uvfit: UVFitConfig = UVFitConfig(),
                 bootstrap: BootstrapConfig = BootstrapConfig(),
                 match_tol_arcsec: float = 2.0,
                 output_prefix: Optional[str] = None,
                 verbose: bool = True) -> ExtractionResult:
    """Recover the stacked flux from the compressed statistics.

    The chain is the one of Sec. 5.3: the statistics of the sources are
    combined, the line shape is fitted on the spatially integrated spectrum,
    the amplitude is fitted annulus by annulus at fixed shape, the resulting
    profile is fitted with the configured source model, and every uncertainty
    comes from bootstrap resampling of the sample.

    With ``profile.joint_continuum`` the line is fitted on top of a constant
    continuum, whose amplitude is itself measured annulus by annulus and
    fitted in the uv plane: use it when the stack was not continuum
    subtracted, so that line and continuum come out of the same fit.

    Parameters
    ----------
    statistics, input_list, weighting, match_tol_arcsec
        Passed to :func:`build_stack_input`.
    line, profile, uvfit, bootstrap
        See the corresponding classes in :mod:`vista.extract.config`.
    output_prefix
        When given, the results are also written to
        ``<prefix>_results.json``, ``<prefix>_spectrum.txt`` and
        ``<prefix>_uvamp.txt``.

    Returns
    -------
    ExtractionResult
    """
    data = build_stack_input(statistics, input_list, weighting,
                             match_tol_arcsec=match_tol_arcsec,
                             verbose=verbose)
    sums, factors = data.sums, data.factors
    freq = data.freq_grid
    renormalise = weighting.renormalise
    n_sources = data.n_sources

    second_freq = line.second_rest_freq_hz
    with_continuum = (profile.joint_continuum
                      and profile.method == "template")

    # ---- nominal stack ---------------------------------------------------
    real, imaginary, weight, mean_b = combine(sums, factors,
                                              renormalise=renormalise)
    per_channel = sources_per_channel(sums)
    baseline = (np.nansum(mean_b * weight, axis=0)
                / np.nansum(weight, axis=0))
    spectrum = integrated_spectrum(real, weight)
    velocity = line.velocity(freq)

    continuum_mask = None
    effective_freq = None
    if profile.method == "continuum":
        needed = int(np.ceil(profile.continuum_overlap_frac * n_sources))
        continuum_mask = per_channel >= needed
        if not continuum_mask.any():
            best = int(per_channel.max())
            if verbose:
                print(f"[continuum] no channel covered by {needed} sources, "
                      f"falling back to the best coverage ({best})")
            continuum_mask = per_channel >= best
        effective_freq = float(np.nansum(freq[continuum_mask])
                               / continuum_mask.sum())
        if verbose:
            print(f"[continuum] {int(continuum_mask.sum())} channels, "
                  f"nu_eff = {effective_freq / 1e9:.2f} GHz")

    def measure(real_part, weight_part, spectrum_part, shape_start=None):
        """One realisation: (line flux, second line, continuum, mask, shape)."""
        if profile.method == "continuum":
            values, _ = flux_per_annulus_continuum(real_part, weight_part,
                                                   continuum_mask)
            return values, None, None, continuum_mask, None
        if profile.method == "window":
            values, _, _, _, used = flux_per_annulus_window(
                real_part, weight_part, freq, line.rest_freq_hz,
                line.v_window_kms)
            second_values = None
            if second_freq is not None:
                second_values, _, _, _, _ = flux_per_annulus_window(
                    real_part, weight_part, freq, second_freq,
                    line.second_v_window_kms)
            return values, second_values, None, used, None
        shape = fit_line_shape(spectrum_part, weight_part, freq, line,
                               profile, start=shape_start)
        if shape is None:
            values, _, _, _, used = flux_per_annulus_window(
                real_part, weight_part, freq, line.rest_freq_hz,
                line.v_window_kms)
            return values, None, None, used, None
        values, second_values, cont, used = flux_per_annulus_template(
            real_part, weight_part, freq, line, shape, profile)
        return values, second_values, cont, used, shape

    flux, flux_second, flux_continuum, used_channels, shape = measure(
        real, weight, spectrum)
    if profile.method == "template" and shape is None and verbose:
        print("[shape] the profile fit did not converge, falling back to the "
              "boxcar integration")
    if verbose and shape is not None:
        message = (f"[shape] centroid={shape.centroid_kms:+.0f} km/s  "
                   f"FWHM={shape.fwhm_kms:.0f} km/s  "
                   f"chi2r={shape.chi2_reduced:.2f}")
        if shape.blended:
            message += (f" | second: centroid={shape.centroid2_kms:+.0f}  "
                        f"FWHM={2.3548 * shape.sigma2_kms:.0f} km/s")
        if shape.baseline is not None:
            message += f" | continuum level={shape.baseline:.4g}"
        print(message)

    # the imaginary part: the null test follows exactly the same path
    second_imaginary = continuum_imaginary = None
    if shape is not None and profile.method == "template":
        flux_imaginary, second_imaginary, continuum_imaginary, _ = \
            flux_per_annulus_template(imaginary, weight, freq, line, shape,
                                      profile)
    elif profile.method == "continuum":
        flux_imaginary, _ = flux_per_annulus_continuum(imaginary, weight,
                                                       continuum_mask)
    else:
        flux_imaginary, _, _, _, _ = flux_per_annulus_window(
            imaginary, weight, freq, line.rest_freq_hz, line.v_window_kms)
        if second_freq is not None:
            second_imaginary, _, _, _, _ = flux_per_annulus_window(
                imaginary, weight, freq, second_freq,
                line.second_v_window_kms)

    per_annulus = sources_per_bin(sums, used_channels)
    minimum = max(uvfit.min_sources_per_bin, 2)
    bin_ok = per_annulus >= minimum

    # ---- bootstrap, first pass: profiles and spectra ---------------------
    rng = np.random.default_rng(bootstrap.seed)
    boot_flux, boot_second, boot_continuum, boot_spectrum = [], [], [], []
    for _ in range(bootstrap.n_realisations):
        multiplicity = np.bincount(rng.integers(0, n_sources, n_sources),
                                   minlength=n_sources).astype(float)
        r, _, w, _ = combine(sums, factors, multiplicity=multiplicity,
                             renormalise=renormalise)
        s = integrated_spectrum(r, w)
        boot_spectrum.append(s)
        f1, f2, fc, _, _ = measure(
            r, w, s, shape_start=None if shape is None else shape.raw_params)
        boot_flux.append(f1)
        if f2 is not None:
            boot_second.append(f2)
        if fc is not None:
            boot_continuum.append(fc)

    boot_flux = np.asarray(boot_flux)
    boot_spectrum = np.asarray(boot_spectrum)
    flux_error = np.nanstd(boot_flux, axis=0)
    spectrum_error = np.nanstd(boot_spectrum, axis=0)

    def apply_floor(errors, mask):
        usable = np.isfinite(errors) & (errors > 0) & mask
        if usable.any() and uvfit.error_floor_frac > 0:
            return np.maximum(errors, uvfit.error_floor_frac
                              * np.nanmedian(errors[usable]))
        return errors

    flux_error = apply_floor(flux_error, bin_ok)

    # ---- bootstrap, second pass: the fits, weighted like the nominal one --
    boot_fit_flux, boot_fit_theta = [], []
    for values in boot_flux:
        result = fit_uv_profile(baseline, values, sigma=flux_error,
                                config=uvfit, bin_ok=bin_ok)
        boot_fit_flux.append(result.flux)
        boot_fit_theta.append(result.theta_arcsec)
    boot_fit_flux = np.asarray(boot_fit_flux, dtype=float)
    boot_fit_theta = np.asarray(boot_fit_theta, dtype=float)

    fit = fit_uv_profile(baseline, flux, sigma=flux_error, config=uvfit,
                         bin_ok=bin_ok)
    ok = np.isfinite(boot_fit_flux) & np.isfinite(boot_fit_theta)

    def percentiles(values, mask):
        return ([float(x) for x in np.nanpercentile(values[mask],
                                                    [16, 50, 84])]
                if mask.any() else [float("nan")] * 3)

    finite = np.isfinite(flux)
    weight_per_bin = np.nansum(weight[used_channels], axis=0)
    flux_point = float(np.nansum(flux[finite] * weight_per_bin[finite])
                       / np.nansum(weight_per_bin[finite]))

    def fit_companion(values, errors, imaginary_values, theta_reference,
                      tie_dex, boot_values):
        """uv fit of a secondary profile, with its bootstrap errors."""
        errors = apply_floor(errors, bin_ok)
        boot_f, boot_t = [], []
        for i, realisation in enumerate(boot_values):
            reference = (boot_fit_theta[i]
                         if (tie_dex is not None and i < len(boot_fit_theta))
                         else None)
            prior = ((reference, tie_dex)
                     if (reference is not None and np.isfinite(reference)
                         and reference > 0) else None)
            outcome = fit_uv_profile(baseline, realisation, sigma=errors,
                                     config=uvfit, bin_ok=bin_ok,
                                     theta_prior=prior)
            boot_f.append(outcome.flux)
            boot_t.append(outcome.theta_arcsec)
        nominal_prior = ((theta_reference, tie_dex)
                         if (tie_dex is not None
                             and np.isfinite(theta_reference)
                             and theta_reference > 0) else None)
        nominal = fit_uv_profile(baseline, values, sigma=errors, config=uvfit,
                                 bin_ok=bin_ok, theta_prior=nominal_prior)
        boot_f = np.asarray(boot_f, dtype=float)
        boot_t = np.asarray(boot_t, dtype=float)
        ok_f, ok_t = np.isfinite(boot_f), np.isfinite(boot_t)
        return {
            "flux": values, "flux_error": errors,
            "flux_imaginary": (imaginary_values if imaginary_values is not None
                               else np.full_like(values, np.nan)),
            "fit": nominal,
            "flux_total": nominal.flux,
            "flux_total_error": (float(np.nanstd(boot_f[ok_f]))
                                 if ok_f.any() else float("nan")),
            "flux_percentiles_16_50_84": percentiles(boot_f, ok_f),
            "theta_fwhm_arcsec": nominal.theta_arcsec,
            "theta_error_arcsec": (float(np.nanstd(boot_t[ok_t]))
                                   if ok_t.any() else float("nan")),
            "theta_at_bound": nominal.theta_at_bound,
            "chi2_reduced": nominal.chi2_reduced,
            "theta_tied_to_line": tie_dex is not None,
            "theta_tie_sigma_dex": tie_dex,
        }

    # ---- second line -----------------------------------------------------
    second = None
    if second_freq is not None and boot_second:
        if profile.method == "window":
            second_flux, _, _, _, _ = flux_per_annulus_window(
                real, weight, freq, second_freq, line.second_v_window_kms)
        else:
            second_flux = flux_second
        second = fit_companion(second_flux,
                               np.nanstd(np.asarray(boot_second), axis=0),
                               second_imaginary, fit.theta_arcsec,
                               uvfit.second_line_theta_tie_dex,
                               np.asarray(boot_second))
        second["rest_freq_ghz"] = line.second_rest_freq_ghz

    # ---- continuum fitted jointly with the line --------------------------
    continuum = None
    if with_continuum and boot_continuum:
        continuum = fit_companion(
            flux_continuum, np.nanstd(np.asarray(boot_continuum), axis=0),
            continuum_imaginary, fit.theta_arcsec,
            uvfit.continuum_theta_tie_dex, np.asarray(boot_continuum))
        continuum["effective_freq_ghz"] = float(
            np.nansum(freq[used_channels]) / max(used_channels.sum(), 1) / 1e9)

    # ---- the spectrum at zero baseline -----------------------------------
    # Per-channel flux with the fitted size held fixed: the spectrum an
    # image-plane fit would produce, in flux units, whose integral over
    # velocity is the total line flux.
    spectrum_zero = spectrum_zero_error = zero_shape = None
    if profile.method != "continuum" and np.isfinite(fit.theta_arcsec):
        theta_fixed = float(fit.theta_arcsec)
        spectrum_zero = flux_per_channel(real, weight, baseline, theta_fixed,
                                         bin_ok)
        boot_zero = []
        rng_zero = np.random.default_rng(bootstrap.seed)
        for _ in range(bootstrap.n_realisations):
            multiplicity = np.bincount(
                rng_zero.integers(0, n_sources, n_sources),
                minlength=n_sources).astype(float)
            r, _, w, _ = combine(sums, factors, multiplicity=multiplicity,
                                 renormalise=renormalise)
            boot_zero.append(flux_per_channel(r, w, baseline, theta_fixed,
                                              bin_ok))
        spectrum_zero_error = np.nanstd(np.asarray(boot_zero), axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            channel_weight = np.where(spectrum_zero_error > 0,
                                      1.0 / spectrum_zero_error ** 2, 0.0)
        zero_shape = fit_line_shape(spectrum_zero, channel_weight, freq, line,
                                    profile,
                                    start=None if shape is None
                                    else shape.raw_params)
        if verbose and zero_shape is not None:
            integral = (zero_shape.amplitude * zero_shape.sigma_kms
                        * np.sqrt(2.0 * np.pi))
            print(f"[zero-b] peak = {zero_shape.amplitude:.4g}  "
                  f"FWHM = {zero_shape.fwhm_kms:.0f} km/s  "
                  f"integral = {integral:.4g} "
                  f"({100 * integral / fit.flux:.0f}% of F)")

    # ---- summary ---------------------------------------------------------
    null_test = np.isfinite(flux_imaginary) & bin_ok
    summary = {
        "n_sources": n_sources,
        "n_entries": data.n_entries,
        "source_coordinates_deg": [[round(ra, 6), round(dec, 6)]
                                   for ra, dec in data.coordinates],
        "weighting_scheme": weighting.scheme,
        "weights_renormalised": renormalise,
        "redshift_rescaling": (weighting.redshift_rescaling
                               and not weighting.already_applied),
        "z_ref": data.z_ref,
        "flux_normalised": data.norm_ref is not None,
        "norm_ref": data.norm_ref,
        "rest_freq_ghz": line.rest_freq_ghz,
        "v_window_kms": list(line.v_window_kms),
        "flux_method": profile.method,
        "fit_span_kms": profile.fit_span_kms,
        "joint_continuum": with_continuum,
        "flux_total": fit.flux,
        "flux_total_error": (float(np.nanstd(boot_fit_flux[ok]))
                             if ok.any() else float("nan")),
        "flux_percentiles_16_50_84": percentiles(boot_fit_flux, ok),
        "theta_fwhm_arcsec": fit.theta_arcsec,
        "theta_error_arcsec": (float(np.nanstd(boot_fit_theta[ok]))
                               if ok.any() else float("nan")),
        "theta_percentiles_16_50_84": percentiles(boot_fit_theta, ok),
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
        "n_sources_min_in_window": (int(per_channel[used_channels].min())
                                    if np.any(used_channels) else 0),
        "null_test_max_imaginary_over_sigma": (
            float(np.nanmax(np.abs(flux_imaginary[null_test])
                            / np.maximum(flux_error[null_test], 1e-30)))
            if null_test.any() else float("nan")),
        "units": ("flux in [data units] * km/s for a line, [data units] for "
                  "the continuum; the flux normalisation rescales them by "
                  "norm_ref / norm"),
    }
    if shape is not None:
        summary["line_shape"] = shape.as_dict()
    if zero_shape is not None:
        block = zero_shape.as_dict()
        block["peak"] = zero_shape.amplitude
        block["integral"] = float(zero_shape.amplitude * zero_shape.sigma_kms
                                  * np.sqrt(2.0 * np.pi))
        block["theta_assumed_arcsec"] = float(fit.theta_arcsec)
        summary["zero_baseline_spectrum"] = block
    if effective_freq is not None:
        summary["continuum_effective_freq_ghz"] = effective_freq / 1e9
        summary["continuum_n_channels"] = int(continuum_mask.sum())
    for tag, block in (("second_line", second), ("continuum", continuum)):
        if block is not None:
            summary[tag] = {k: v for k, v in block.items()
                            if not isinstance(v, np.ndarray) and k != "fit"}

    result = ExtractionResult(
        summary=summary, freq_hz=freq, velocity_kms=velocity,
        spectrum=spectrum, spectrum_error=spectrum_error,
        sources_per_channel=per_channel, baseline_klambda=baseline / 1e3,
        flux=flux, flux_error=flux_error, flux_imaginary=flux_imaginary,
        sources_per_annulus=per_annulus, bin_ok=bin_ok, fit=fit, shape=shape,
        second=second, continuum=continuum,
        spectrum_zero_baseline=spectrum_zero,
        spectrum_zero_baseline_error=spectrum_zero_error,
        zero_baseline_shape=zero_shape)

    if output_prefix:
        written = result.save(output_prefix)
        if verbose:
            print("[output] " + ", ".join(written))
    if verbose:
        print(f"[result] F = {fit.flux:.4g} "
              f"+- {summary['flux_total_error']:.3g}   "
              f"theta = {fit.theta_arcsec:.3f}\" "
              f"+- {summary['theta_error_arcsec']:.3f}\"   "
              f"chi2r = {fit.chi2_reduced:.2f}")
        if continuum is not None:
            print(f"[result] continuum S = {continuum['flux_total']:.4g} "
                  f"+- {continuum['flux_total_error']:.3g}   "
                  f"theta = {continuum['theta_fwhm_arcsec']:.3f}\"")
    return result


__all__ = ["StackInput", "build_stack_input", "combine", "flux_per_channel",
           "sources_per_bin",
           "sources_per_channel", "extract_flux", "ExtractionResult"]
