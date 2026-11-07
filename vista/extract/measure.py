"""
vista.extract.measure
=====================
The stacked line, measured on the whole channel x annulus cube
(:mod:`vista.extract.linefit`) with the noise measured from the data
(:mod:`vista.extract.noise`).

What it gives, in ``<prefix>_results.json`` and next to it:

* the **profile**: flux density at zero spacing against velocity, channel by
  channel, with its thermal and bootstrap errors, the fitted model and the
  imaginary part (``_profile.txt``); its integral is the line flux;
* the **line**: flux, peak, centre, FWHM, size, from one fit of all the
  cells, with thermal (fit) and bootstrap (population) errors;
* the **significance**: S/N of the matched filter, and for a free centre and
  width the probability that noise alone gives as much (null realisations);
* the **window flux**: the plain sum of the profile in a window, with an
  error that includes the correlation between channels;
* the **noise**: the four estimates of :mod:`vista.extract.noise` side by
  side (``_noise.txt`` per channel), and the per-annulus fluxes
  (``_uvamp.txt``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np

from .. import _reset_clock, _say, _warn
from .config import BootstrapConfig, LineConfig, LineFitConfig, WeightingConfig, C_KMS
from .linefit import (FWHM_PER_SIGMA, Cube, FitResult, FitSpec, Scan, Whitener, channel_gaussian,
                      fit, flux_per_annulus, global_significance, model_cube,
                      peak_flux_density, uv_gaussian, window_flux, zero_spacing_spectrum)
from .noise import build_noise_model, combine_variance
from .profiles import line_free_mask, velocity_window_mask
from .stack import build_stack_input, combine, sources_per_bin, sources_per_channel
from .statistics import SufficientStatistics


@dataclass
class LineResult:
    """Everything :func:`measure_line` produces."""

    summary: dict
    freq_hz: np.ndarray
    velocity_kms: np.ndarray
    profile: np.ndarray                 # zero-spacing flux density per channel
    profile_error: np.ndarray           # thermal
    profile_error_bootstrap: np.ndarray
    profile_imaginary: np.ndarray
    profile_model: np.ndarray
    fitted: np.ndarray                  # channels used in the fit
    line_free: np.ndarray               # channels outside the line window(s)
    sources_per_channel: np.ndarray
    baseline_klambda: np.ndarray
    annulus_flux: np.ndarray
    annulus_flux_error: np.ndarray
    annulus_ok: np.ndarray
    sources_per_annulus: np.ndarray
    noise_channel: np.ndarray           # thermal error of the profile from the scatter
    noise_channel_weights: np.ndarray   # the same from the weights alone
    z_imaginary: np.ndarray = field(default_factory=lambda: np.zeros(0))
    z_line_free: np.ndarray = field(default_factory=lambda: np.zeros(0))
    null_snr: np.ndarray = field(default_factory=lambda: np.zeros(0))
    null_snr_phase: np.ndarray = field(default_factory=lambda: np.zeros(0))
    channel_correlation: Optional[np.ndarray] = None    # (nchan, nchan), None: independent

    def save(self, prefix: str) -> List[str]:
        written = []
        with open(f"{prefix}_results.json", "w") as fh:
            json.dump(self.summary, fh, indent=2, default=_jsonable)
        written.append(f"{prefix}_results.json")
        np.savetxt(f"{prefix}_profile.txt", np.column_stack([
            self.velocity_kms, self.freq_hz / 1e9, self.profile, self.profile_error,
            self.profile_error_bootstrap, self.profile_imaginary, self.profile_model,
            self.fitted.astype(int), self.line_free.astype(int), self.sources_per_channel]),
            header="v_kms  freq_GHz  flux_density  err_thermal  err_bootstrap  "
                   "imaginary  model  fitted  line_free  n_sources")
        written.append(f"{prefix}_profile.txt")
        np.savetxt(f"{prefix}_uvamp.txt", np.column_stack([
            self.baseline_klambda, self.annulus_flux, self.annulus_flux_error,
            self.annulus_ok.astype(int), self.sources_per_annulus]),
            header="b_klambda  line_flux  err_thermal  used  n_sources")
        written.append(f"{prefix}_uvamp.txt")
        np.savetxt(f"{prefix}_noise.txt", np.column_stack([
            self.velocity_kms, self.noise_channel, self.noise_channel_weights]),
            header="v_kms  sigma_profile_from_scatter  sigma_profile_from_weights")
        written.append(f"{prefix}_noise.txt")
        # what the figures need besides the tables (examples/plot_line.py)
        corr = (self.channel_correlation if self.channel_correlation is not None
                else np.zeros((0, 0)))
        np.savez_compressed(
            f"{prefix}_diagnostics.npz", velocity_kms=self.velocity_kms,
            channel_correlation=corr, z_imaginary=self.z_imaginary,
            z_line_free=self.z_line_free, null_snr=self.null_snr,
            null_snr_phase=self.null_snr_phase)
        written.append(f"{prefix}_diagnostics.npz")
        return written


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def akaike_weight(delta_chi2: float) -> float:
    """Weight of the Gaussian source against the point source from the chi
    square improvement, with the Akaike penalty of its extra parameter."""
    a = 0.5 * (float(delta_chi2) - 2.0)
    if not np.isfinite(a):
        return 0.0
    return float(1.0 / (1.0 + np.exp(-a))) if a > -700 else 0.0


def size_test_correlation(cube: Cube, wh, point: FitResult, spec_point: FitSpec,
                          theta_gauss: float, theta_ref: float) -> float:
    """The factor by which the channel correlation inflates the chi-square
    difference between the point and the Gaussian source.

    The fits weight the cells by their noise alone (the correlation is
    singular with Hanning smoothing, so it is not inverted), so their chi
    square counts the correlated noise as independent.  For nested models the
    difference is the square of the data projected on the direction ``d``
    that only the Gaussian can follow (the change of the model with the size,
    with what the point model can absorb by changing its own parameters
    removed): its true variance is ``d^T R d``, not ``d^T d``.  With Hanning
    smoothing (+0.67 at lag 1) and a line many channels wide this is about
    2: without it, noise made point sources look resolved and pushed the
    model-averaged flux up."""
    if not getattr(wh, "correlated", False):
        return 1.0
    p0 = dict(point.params)
    base = model_cube(cube, p0)
    th = theta_gauss if theta_gauss and theta_gauss > 0.05 * theta_ref else theta_ref
    d = wh(model_cube(cube, {**p0, "theta": float(th)}) - base)
    cols = []
    for name in spec_point.free:
        h = {"v0": 1.0, "fwhm": 1.0}.get(name, max(abs(p0.get(name, 0.0)), 1e-30) * 1e-3)
        cols.append(wh(model_cube(cube, {**p0, name: p0[name] + h}) - base) / h)
    if cols:
        jac = np.column_stack(cols)
        d = d - jac @ np.linalg.lstsq(jac, d, rcond=None)[0]
    norm2 = float(d @ d)
    if not norm2 > 0:
        return 1.0
    return float(np.clip(wh.quad(d) / norm2, 0.25, 10.0))


def _fit_spec(cfg: LineFitConfig, line: LineConfig, start: dict, size: str,
              v0_bounds, fwhm_bounds, second: bool):
    """The FitSpec of the adopted model."""
    free = ["F"]
    values = {"F": start["F"], "v0": start["v0"], "fwhm": start["fwhm"], "theta": 0.0}
    bounds = {"v0": v0_bounds, "fwhm": fwhm_bounds,
              "theta": (1e-3, cfg.theta_max_arcsec)}
    if cfg.mode == "search":
        free += ["v0", "fwhm"]
    if cfg.theta_fixed_arcsec is not None:
        values["theta"] = float(cfg.theta_fixed_arcsec)
    elif size == "gauss":
        free.append("theta")
        values["theta"] = float(np.clip(start.get("theta", 0.5), 0.01, cfg.theta_max_arcsec))
    if second:
        values.update({"F2": start.get("F2", 0.0), "v02": start.get("v02", start["v0"]),
                       "fwhm2": start.get("fwhm2", start["fwhm"])})
        bounds.update({"v02": v0_bounds, "fwhm2": fwhm_bounds})
        free.append("F2")
        if cfg.mode == "search":
            if cfg.shape_tie == "free":
                free += ["v02", "fwhm2"]
            elif cfg.shape_tie == "centroid":
                free.append("fwhm2")
    if cfg.continuum:
        values["C"] = start.get("C", 0.0)
        free.append("C")
    prior = cfg.theta_prior_arcsec if "theta" in free else None
    if prior is not None:
        values["theta"] = float(np.clip(prior[0], 0.01, cfg.theta_max_arcsec))
    return FitSpec(tuple(free), values, bounds, second=second, tie=cfg.shape_tie,
                   continuum=cfg.continuum, theta_prior=prior)


def measure_line(statistics: Union[str, SufficientStatistics],
                 input_list: Union[str, Sequence],
                 line: LineConfig,
                 weighting: WeightingConfig = WeightingConfig(),
                 config: LineFitConfig = LineFitConfig(),
                 bootstrap: BootstrapConfig = BootstrapConfig(),
                 match_tol_arcsec: float = 2.0,
                 output_prefix: Optional[str] = None,
                 verbose: bool = True) -> LineResult:
    """Measure the stacked line: profile, flux, significance, noise.

    Parameters
    ----------
    statistics, input_list, weighting, match_tol_arcsec
        As in :func:`vista.extract.extract_flux`.
    line
        Rest frequency, line window (also the default window of the window
        flux and of the allowed centres), second line, exclusions.
    config
        See :class:`vista.extract.config.LineFitConfig`.
    bootstrap
        Resampling of the sources, for the population errors.
    output_prefix
        Write ``<prefix>_results.json``, ``_profile.txt``, ``_uvamp.txt`` and
        ``_noise.txt``.
    """
    from scipy.stats import norm

    _reset_clock()
    cfg = config
    data = build_stack_input(statistics, input_list, weighting,
                             match_tol_arcsec=match_tol_arcsec, verbose=verbose)
    sums, factors = data.sums, data.factors
    n_src = data.n_sources
    renorm = weighting.renormalise
    freq = data.freq_grid
    velocity = line.velocity(freq)
    dv = float(np.median(np.abs(np.diff(velocity)))) if len(velocity) > 1 else 1.0
    second = line.second_rest_freq_hz is not None
    velocity2 = line.velocity(freq, second=True) if second else None
    meta = data.meta
    contsub = (meta.get("line_column") == "CORRECTED_DATA") or ("contsub" in meta)

    use_scatter = cfg.noise == "scatter" and data.has_scatter
    if cfg.noise == "scatter" and not data.has_scatter:
        _warn("measure: the statistics have no scatter accumulators (format 3): "
              "the noise is taken from the weights, whose absolute scale is not "
              "checked.  Compress again to measure it")
    var_src = data.variance if use_scatter else data.variance_weights

    # ---- nominal stack and its noise -------------------------------------
    real, imag, weight, mean_b = combine(sums, factors, renormalise=renorm)
    line_free = line_free_mask(freq, line)
    noise = build_noise_model(
        sums, var_src, data.variance_weights, factors, renorm, real, imag, line_free,
        contsub, data.kappa2_entries if use_scatter else {},
        data.dof_entries, kappa_channel=data.kappa2_channel,
        source="scatter" if use_scatter else "weights",
        correlate=cfg.channel_correlation)
    var = noise.variance
    corr = noise.corr if cfg.channel_correlation else None

    # ---- cells to fit ------------------------------------------------------
    per_channel = sources_per_channel(sums)
    in_span = np.abs(velocity) < cfg.fit_span_kms
    if second:
        in_span |= np.abs(velocity2) < cfg.fit_span_kms
    per_annulus = sources_per_bin(sums, in_span)
    min_src = 1 if n_src == 1 else max(2, cfg.min_sources_per_bin)
    bin_ok = per_annulus >= min_src
    ok = (np.isfinite(real) & np.isfinite(var) & (var > 0) & in_span[:, None]
          & bin_ok[None, :] & (weight > 0))
    if not ok.any():
        raise RuntimeError("measure: no usable cell (check the line window, the fit "
                           "span and the minimum number of sources per annulus)")
    baseline = np.where(np.isfinite(mean_b), mean_b, 0.0)
    cube = Cube(real=real, variance=var, baseline=baseline, velocity=velocity, dv=dv,
                ok=ok, corr=corr, velocity2=velocity2, edges=data.bin_edges_lambda)
    wh = Whitener(var, ok, corr)
    _say(f"  cube: {int(ok.sum()):,} cells ({int(ok.any(axis=1).sum())} channels x "
         f"{int(bin_ok.sum())} annuli), noise from the "
         f"{'scatter of the visibilities' if use_scatter else 'weights'}; channel "
         f"correlation {', '.join(f'{r:+.2f}' for r in noise.rho[1:]) or 'none'} "
         f"({noise.checks.get('channel_correlation_method')})", verbose)

    # ---- detection: matched filter (detect) or scan (search) -------------
    win = cfg.window_kms or tuple(line.v_window_kms)
    fwhm_lo = 2.0 * dv
    fwhm_hi = max(4.0 * dv, float(win[1] - win[0]))
    if cfg.fwhm_range_kms is not None:
        fwhm_lo, fwhm_hi = (float(x) for x in cfg.fwhm_range_kms)
    if cfg.mode == "detect":
        v0_grid, fwhm_grid = [float(cfg.v0_kms)], [float(cfg.fwhm_kms)]
        v0_bounds = (cfg.v0_kms - 1.0, cfg.v0_kms + 1.0)
    else:
        if cfg.v0_range_kms is not None:
            v_lo, v_hi = (float(x) for x in cfg.v0_range_kms)
        else:
            v_lo, v_hi = float(win[0]) + 0.5 * fwhm_lo, float(win[1]) - 0.5 * fwhm_lo
        n_v = max(1, int(round((v_hi - v_lo) / (0.5 * dv))) + 1)
        v0_grid = list(np.linspace(v_lo, v_hi, n_v))
        fwhm_grid = list(np.geomspace(fwhm_lo, fwhm_hi, 14))
        v0_bounds = (v_lo, v_hi)
    fwhm_bounds = (0.5 * fwhm_lo, fwhm_hi * 1.5)
    if cfg.mode == "search":
        n_v = len(v0_grid)
        if n_v > 160:                    # bound the number of templates
            v0_grid = list(np.linspace(v0_grid[0], v0_grid[-1], 160))
    # source sizes of the scan: a point source, and Gaussians of 0.5, 1 and 2
    # times the size that halves the visibility at the median baseline
    wmed = np.where(ok, 1.0 / np.where(ok, var, 1.0), 0.0)
    order_b = np.argsort(baseline[ok])
    cw = np.cumsum(wmed[ok][order_b])
    b_med = float(baseline[ok][order_b][np.searchsorted(cw, 0.5 * cw[-1])])
    theta_half = 2 * np.log(2) / (np.pi * max(b_med, 1.0)) * 206264.806
    if cfg.theta_fixed_arcsec is not None:
        thetas = [float(cfg.theta_fixed_arcsec)]
    elif cfg.theta_prior_arcsec is not None:
        thetas = [float(cfg.theta_prior_arcsec[0])]
    elif cfg.size == "point":
        thetas = [0.0]
    else:
        thetas = [0.0] + [t for t in (0.5 * theta_half, theta_half, 2 * theta_half)
                          if t < cfg.theta_max_arcsec]
    scanner = Scan(cube, wh, v0_grid, fwhm_grid, thetas, continuum=cfg.continuum)
    best = scanner.best()
    if cfg.mode == "detect" and len(thetas) == 1:
        p = float(norm.sf(best["snr"]))
        signif = {"p_value": p, "significance_sigma": float(best["snr"]),
                  "n_effective_trials": 1.0, "n_null": 0}
        null = np.zeros(0)
    elif not cfg.n_null:
        # no null realisations asked: Bonferroni over the templates (conservative)
        n_templates = len(thetas) * (1 if cfg.mode == "detect"
                                     else len(v0_grid) * len(fwhm_grid))
        p = float(min(1.0, n_templates * norm.sf(best["snr"])))
        signif = {"p_value": p, "significance_sigma": float(norm.isf(p)) if p < 1 else 0.0,
                  "n_effective_trials": float(n_templates), "n_null": 0,
                  "p_value_method": f"Bonferroni over {n_templates} templates (n_null = 0): "
                                    "an upper limit on p"}
        null = np.zeros(0)
    else:
        null = scanner.null(n=cfg.n_null, seed=bootstrap.seed)
        signif = global_significance(best["snr"], null)
    detection = {"mode": cfg.mode, "snr": best["snr"], "v0_kms": best["v0"],
                 "fwhm_kms": best["fwhm"], "theta_arcsec": best["theta"],
                 "thetas_scanned_arcsec": thetas,
                 "flux_matched_filter": best["flux"],
                 "flux_matched_filter_error": best["flux_error"], **signif}
    _say(f"  {'matched filter' if cfg.mode == 'detect' else 'search'}: S/N = "
         f"{best['snr']:.2f} at v0 = {best['v0']:+.0f} km/s, FWHM = {best['fwhm']:.0f} km/s, "
         f"size {best['theta']:.2f}\""
         + ("" if signif.get("n_null", 0) == 0 else
            f"  ->  p = {signif['p_value']:.2g} ({signif['significance_sigma']:.1f} sigma, "
            f"{signif['n_effective_trials']:.0f} effective trials)"), verbose)

    # data-driven null: random phases per source
    null_phase = np.zeros(0)
    if cfg.n_null_phase and n_src >= 2:
        rng = np.random.default_rng(bootstrap.seed + 1)
        vals = []
        for _ in range(int(cfg.n_null_phase)):
            phi = rng.uniform(0, 2 * np.pi, n_src)
            rot = sums.copy()
            rot[..., 0] = (np.cos(phi)[:, None, None] * sums[..., 0]
                           - np.sin(phi)[:, None, None] * sums[..., 1])
            r_rot, _, _, _ = combine(rot, factors, renormalise=renorm)
            vals.append(scanner.best(y=r_rot)["snr"])
        null_phase = np.asarray(vals)
        detection["p_value_phase_null"] = float((1 + np.sum(null_phase >= best["snr"]))
                                                / (1 + null_phase.size))
        detection["phase_null_snr_rms"] = float(np.sqrt(np.mean(null_phase ** 2)))
    # the imaginary part through the same filter, at the best grid point
    yi = wh(np.where(ok, np.where(np.isfinite(imag), imag, 0.0), 0.0))
    detection["snr_imaginary"] = float(best["templates"][:, best["k"]] @ yi.astype(np.float32))

    # ---- the fit: point source, and a Gaussian if asked or preferred -----
    start = {"F": best["flux"], "v0": best["v0"], "fwhm": best["fwhm"],
             "theta": best["theta"] if best["theta"] > 0 else 0.5}
    if second:
        start.update({"F2": 0.0, "v02": best["v0"], "fwhm2": best["fwhm"]})
    if cfg.theta_prior_arcsec is not None and cfg.theta_fixed_arcsec is None:
        sizes = ["gauss"]                # the prior decides the size
    else:
        sizes = ([cfg.size] if cfg.size != "auto" or cfg.theta_fixed_arcsec is not None
                 else ["point", "gauss"])
    fits = {}
    for size in sizes:
        spec = _fit_spec(cfg, line, start, size, v0_bounds, fwhm_bounds, second)
        fits[size] = (fit(cube, spec, wh), spec)
    size_test = None
    w_gauss = None
    size_factor = 1.0
    if len(fits) == 2:
        dchi2_raw = fits["point"][0].chi2 - fits["gauss"][0].chi2
        size_factor = size_test_correlation(cube, wh, fits["point"][0], fits["point"][1],
                                            fits["gauss"][0].params.get("theta", 0.0),
                                            theta_half)
        dchi2 = dchi2_raw / size_factor
        adopted = "gauss" if dchi2 >= cfg.resolved_delta_chi2 else "point"
        w_gauss = akaike_weight(dchi2)
        size_test = {"delta_chi2_point_minus_gauss": float(dchi2),
                     "delta_chi2_uncorrelated": float(dchi2_raw),
                     "correlation_factor": size_factor,
                     "threshold": cfg.resolved_delta_chi2, "adopted": adopted,
                     "flux_point": fits["point"][0].params["F"],
                     "flux_point_error": fits["point"][0].errors.get("F"),
                     "flux_gauss": fits["gauss"][0].params["F"],
                     "flux_gauss_error": fits["gauss"][0].errors.get("F"),
                     "theta_gauss_arcsec": fits["gauss"][0].params.get("theta"),
                     "weight_gauss": w_gauss}
    else:
        adopted = sizes[0]
    result, spec = fits[adopted]
    result.size_model = ("fixed" if cfg.theta_fixed_arcsec is not None
                         else "gauss (prior)" if spec.theta_prior is not None else adopted)
    theta = float(result.params.get("theta", 0.0))

    # ---- bootstrap over the sources --------------------------------------
    rng = np.random.default_rng(bootstrap.seed)
    boot_params, boot_profile, boot_window = [], [], []
    spec_boot = FitSpec(spec.free, dict(result.params), spec.bounds, second=spec.second,
                        tie=spec.tie, continuum=spec.continuum, theta_prior=spec.theta_prior)
    other = None
    if w_gauss is not None:
        o_res, o_spec = fits["point" if adopted == "gauss" else "gauss"]
        other = FitSpec(o_spec.free, dict(o_res.params), o_spec.bounds, second=o_spec.second,
                        tie=o_spec.tie, continuum=o_spec.continuum)
    boot_flux_avg = []
    win_mask = velocity_window_mask(freq, line.rest_freq_hz, win)
    if n_src >= 2:
        for _ in range(int(bootstrap.n_realisations)):
            m = np.bincount(rng.integers(0, n_src, n_src), minlength=n_src).astype(float)
            r_b, _, w_b, mb_b = combine(sums, factors, multiplicity=m, renormalise=renorm)
            v_b = combine_variance(var_src, sums, factors, multiplicity=m,
                                   renormalise=renorm)
            ok_b = ok & np.isfinite(r_b) & np.isfinite(v_b) & (v_b > 0)
            if not ok_b.any():
                continue
            c_b = Cube(real=r_b, variance=v_b,
                       baseline=np.where(np.isfinite(mb_b), mb_b, baseline),
                       velocity=velocity, dv=dv, ok=ok_b, corr=corr, velocity2=velocity2,
                       edges=data.bin_edges_lambda)
            fb = fit(c_b, spec_boot)
            boot_params.append({k: fb.params.get(k) for k in spec.free})
            s_b, e_b = zero_spacing_spectrum(c_b, float(fb.params.get("theta", theta)))
            f_w = window_flux(s_b, e_b, dv, win_mask, corr)[0]
            if other is not None:
                fo = fit(c_b, other)
                fp, fg = (fo, fb) if adopted == "gauss" else (fb, fo)
                wg = akaike_weight((fp.chi2 - fg.chi2) / size_factor)
                boot_flux_avg.append((1 - wg) * fp.params["F"] + wg * fg.params["F"])
                s_o, e_o = zero_spacing_spectrum(c_b, float(fo.params.get("theta", 0.0)))
                f_o = window_flux(s_o, e_o, dv, win_mask, corr)[0]
                w_this = wg if adopted == "gauss" else 1 - wg
                f_w = w_this * f_w + (1 - w_this) * f_o
                s_b = w_this * s_b + (1 - w_this) * s_o
            boot_profile.append(s_b)
            boot_window.append(f_w)
    boot_err, boot_pct = {}, {}
    for k in spec.free:
        vals = np.array([b[k] for b in boot_params if b.get(k) is not None], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 1:
            boot_err[k] = float(np.std(vals))
            boot_pct[k] = [float(x) for x in np.percentile(vals, [16, 50, 84])]

    # ---- profile, window fluxes, annuli ------------------------------------
    zero = Cube(real=real, variance=var, baseline=np.zeros_like(baseline),
                velocity=velocity, dv=dv, ok=ok, corr=corr, velocity2=velocity2)
    yim = np.where(np.isfinite(imag), imag, 0.0)
    prof, prof_err = zero_spacing_spectrum(cube, theta)
    noise_th = prof_err
    prof_im, _ = zero_spacing_spectrum(cube, theta, y=yim)
    pm = dict(result.params)
    model_profile = model_cube(zero, pm)[:, 0]
    if w_gauss is not None:
        # the profile averaged over the two size models, like the flux
        parts = {}
        for k in ("point", "gauss"):
            rk = fits[k][0]
            tk = float(rk.params.get("theta", 0.0))
            pk, ek = zero_spacing_spectrum(cube, tk)
            parts[k] = (pk, ek, zero_spacing_spectrum(cube, tk, y=yim)[0],
                        model_cube(zero, dict(rk.params))[:, 0])
        wts = {"point": 1 - w_gauss, "gauss": w_gauss}
        prof = sum(wts[k] * parts[k][0] for k in parts)
        prof_err = np.sqrt(sum(wts[k] * (parts[k][1] ** 2 + (parts[k][0] - prof) ** 2)
                               for k in parts))
        prof_im = sum(wts[k] * parts[k][2] for k in parts)
        model_profile = sum(wts[k] * parts[k][3] for k in parts)
    prof_boot = (np.nanstd(np.asarray(boot_profile), axis=0) if boot_profile
                 else np.full_like(prof, np.nan))
    var_w = noise.variance_weights
    _, prof_err_w = zero_spacing_spectrum(cube, theta, variance=var_w)

    fitted = ok.any(axis=1)
    fw = result.params["fwhm"]
    opt_mask = np.abs(velocity - result.params["v0"]) <= 0.5 * 1.19 * fw

    def windows(res, sp):
        """Window fluxes of one fit; a resolved source: the profile at zero
        spacing depends on the size, so its uncertainty adds to theirs."""
        th = float(res.params.get("theta", 0.0))
        pr, pe = zero_spacing_spectrum(cube, th)
        fwin, ewin, nwin = window_flux(pr, pe, dv, win_mask, corr)
        fopt, eopt, _ = window_flux(pr, pe, dv, opt_mask, corr)
        e_t = res.errors.get("theta", 0.0) if "theta" in sp.free else 0.0
        if th > 0 and e_t > 0:
            step = max(0.25 * e_t, 1e-3)
            lo_t = max(th - step, 0.0)
            f_up = window_flux(zero_spacing_spectrum(cube, th + step)[0], pe, dv, win_mask, corr)[0]
            f_dn = window_flux(zero_spacing_spectrum(cube, lo_t)[0], pe, dv, win_mask, corr)[0]
            dfdt = (f_up - f_dn) / (th + step - lo_t)
            ewin = float(np.hypot(ewin, dfdt * e_t))
            if fwin:
                eopt = float(np.hypot(eopt, dfdt * e_t * fopt / fwin))
        return fwin, ewin, nwin, fopt, eopt

    f_win, e_win, n_win, f_opt, e_opt = windows(result, spec)
    if w_gauss is not None:
        wins = {k: windows(*fits[k]) for k in ("point", "gauss")}
        wp, wg_ = 1 - w_gauss, w_gauss

        def average(i):
            fa = wp * wins["point"][i] + wg_ * wins["gauss"][i]
            ea = np.sqrt(wp * (wins["point"][i + 1] ** 2 + (wins["point"][i] - fa) ** 2)
                         + wg_ * (wins["gauss"][i + 1] ** 2 + (wins["gauss"][i] - fa) ** 2))
            return float(fa), float(ea)

        f_win, e_win = average(0)
        f_opt, e_opt = average(3)
    e_win_boot = float(np.nanstd(boot_window)) if boot_window else float("nan")
    window = {"user_kms": [float(win[0]), float(win[1])], "flux": f_win,
              "flux_error": float(np.nanmax([e_win, e_win_boot])), "flux_error_thermal": e_win,
              "n_channels": n_win, "flux_error_bootstrap": e_win_boot,
              "optimal_kms": [float(result.params["v0"] - 0.595 * fw),
                              float(result.params["v0"] + 0.595 * fw)],
              "flux_optimal": f_opt, "flux_optimal_error": e_opt,
              "note_optimal": "1.19 FWHM around the fitted centre: best S/N for a "
                              "Gaussian, 84% of its flux; defined on the data, so "
                              "slightly biased high for a weak line"}

    ann_flux, ann_err = flux_per_annulus(cube, wh, pm)

    # standardised noise samples, for the diagnostic plot
    sd = np.sqrt(var)
    with np.errstate(invalid="ignore", divide="ignore"):
        zi = (imag / sd)[line_free[:, None] & np.isfinite(var) & (var > 0)
                         & np.isfinite(imag)]
        zf = ((real / sd)[line_free[:, None] & np.isfinite(var) & (var > 0)
                          & np.isfinite(real)] if contsub else np.zeros(0))

    # ---- summary -------------------------------------------------------------
    flux = result.params["F"]
    e_th = result.errors.get("F", float("nan"))
    e_bt = boot_err.get("F", float("nan"))
    if w_gauss is not None:
        # the size of a weak or barely resolved stack is uncertain, and the
        # total flux with it: the flux is averaged over the point and the
        # Gaussian model with their Akaike weights, and the spread between
        # them enters the error (otherwise the error would be that of
        # whichever model happened to be chosen)
        fp, fg = fits["point"][0], fits["gauss"][0]
        ep, eg = fp.errors.get("F", np.nan), fg.errors.get("F", np.nan)
        flux = (1 - w_gauss) * fp.params["F"] + w_gauss * fg.params["F"]
        e_th = float(np.sqrt((1 - w_gauss) * (ep ** 2 + (fp.params["F"] - flux) ** 2)
                             + w_gauss * (eg ** 2 + (fg.params["F"] - flux) ** 2)))
        if len(boot_flux_avg) > 1:
            e_bt = float(np.std(boot_flux_avg))
        size_test["flux_averaged"] = flux
        size_test["flux_averaged_error_thermal"] = e_th
    adopted_err = float(np.nanmax([e_th, e_bt])) if np.isfinite(e_th) else e_bt
    line_block = result.as_dict()
    line_block["peak"] = peak_flux_density(flux, result.params.get("fwhm"))
    line_block["peak_error"] = peak_flux_density(adopted_err, result.params.get("fwhm"))
    line_block.update({"flux": flux, "flux_error_thermal": e_th, "flux_error_bootstrap": e_bt,
                       "flux_error": adopted_err,
                       "flux_snr": flux / adopted_err if adopted_err > 0 else float("nan"),
                       "bootstrap_errors": boot_err, "bootstrap_percentiles_16_50_84": boot_pct,
                       "n_bootstrap": len(boot_params)})
    if np.isfinite(e_th) and np.isfinite(e_bt) and e_th > 0:
        line_block["bootstrap_over_thermal"] = e_bt / e_th
    summary = {
        "method": "joint fit of the channel x annulus cube",
        "n_sources": n_src, "n_entries": data.n_entries,
        "weighting_scheme": weighting.scheme, "rest_freq_ghz": line.rest_freq_ghz,
        "line_column": meta.get("line_column"), "contsub": meta.get("contsub"),
        "line": line_block, "detection": detection, "window": window,
        "size_test": size_test, "noise": noise.as_dict(),
        "units": "flux in [data units] km/s, flux density and peak in [data units]; "
                 "with the flux normalisation, rescaled by norm_ref / norm",
        "channel_width_kms": dv,
        "n_cells_fitted": int(ok.sum()), "n_annuli_used": int(bin_ok.sum()),
        "min_sources_per_bin": min_src,
    }
    summary["flux_total"] = flux
    summary["flux_total_error"] = adopted_err
    summary["theta_fwhm_arcsec"] = theta
    summary["theta_error_arcsec"] = float(np.nanmax([result.errors.get("theta", 0.0),
                                                     boot_err.get("theta", 0.0)]))
    summary["chi2_reduced"] = result.chi2_reduced

    out = LineResult(
        summary=summary, freq_hz=freq, velocity_kms=velocity, profile=prof,
        profile_error=prof_err, profile_error_bootstrap=prof_boot,
        profile_imaginary=prof_im, profile_model=model_profile, fitted=fitted,
        line_free=line_free, sources_per_channel=per_channel,
        baseline_klambda=np.nansum(mean_b * weight, axis=0)
        / np.maximum(np.nansum(weight, axis=0), 1e-300) / 1e3,
        annulus_flux=ann_flux, annulus_flux_error=ann_err, annulus_ok=bin_ok,
        sources_per_annulus=per_annulus, noise_channel=noise_th,
        noise_channel_weights=prof_err_w, z_imaginary=zi, z_line_free=zf,
        null_snr=null, null_snr_phase=null_phase, channel_correlation=corr)

    chk = noise.checks
    _say(f"  noise: scatter/weights kappa = {noise.kappa:.3f}; imaginary "
         f"{chk.get('imaginary_rms_over_noise', float('nan')):.3f}"
         + (f", line-free {chk['line_free_rms_over_noise']:.3f}"
            if "line_free_rms_over_noise" in chk else "")
         + f" x the noise (expected 1)", verbose)
    for name, key in (("imaginary part", "imaginary_rms_over_noise"),
                      ("real part in the line-free channels", "line_free_rms_over_noise")):
        value, n_cells = chk.get(key), chk.get(key.replace("rms_over_noise", "n_cells"), 0)
        # only an excess matters: below 1 the errors are conservative
        if value is not None and n_cells and value - 1 > max(0.1, 4 / math.sqrt(2 * n_cells)):
            _warn(f"measure: the {name} scatters {value:.2f} x the noise of the "
                  f"visibilities: {'centring or calibration' if 'imag' in key else 'residual continuum or baseline'} "
                  f"effects, the thermal errors are optimistic there")
    if cfg.mode == "search" and signif.get("significance_sigma", 0) < 4:
        _warn(f"measure: weak line ({signif.get('significance_sigma', 0):.1f} sigma in the "
              f"search): the flux of a fit with free centre and width is unreliable at this "
              f"S/N (it follows noise peaks, and its error is underestimated).  Fix the "
              f"width from another tracer (mode='detect', --mode detect --fwhm), or quote "
              f"the window flux")
        summary["warnings"] = summary.get("warnings", []) + ["weak line: free-shape flux unreliable"]
    if output_prefix:
        written = out.save(output_prefix)
        _say("  written: " + ", ".join(written), verbose)
    _say(f"Line: F = {flux:.4g} +- {adopted_err:.3g} (thermal {e_th:.3g}, bootstrap "
         f"{e_bt:.3g})  peak = {line_block['peak']:.4g}  v0 = {result.params['v0']:+.1f} "
         f"km/s  FWHM = {fw:.0f} km/s  size {result.size_model}"
         + (f" {theta:.2f}\"" if theta > 0 else "")
         + f"  chi2r = {result.chi2_reduced:.2f}", verbose)
    return out


__all__ = ["measure_line", "LineResult"]
