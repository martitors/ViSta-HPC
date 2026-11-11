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
* the **lines**: flux, peak, centre, FWHM, size, from one fit of all the
  cells: the lines whose windows overlap are fitted together (sharing the
  size), every other line on its own; thermal (fit) and bootstrap
  (population) errors;
* the **significance** of every line: the probability that noise alone
  gives as much in the same search (null realisations); the S/N of the
  matched filter when the shape is fixed;
* **two fluxes** for every line: the integral of the fitted Gaussian and the
  sum of the profile over the window (no assumption on the shape), the error
  of their difference, and which one is quoted (``flux_measure='auto'``: the
  Gaussian unless the line is not Gaussian);
* the **residual continuum**: offset and slope of the profile outside the
  windows, flagged when significant;
* the **noise**: the four estimates of :mod:`vista.extract.noise` side by
  side (``_noise.txt`` per channel), and the per-annulus fluxes
  (``_uvamp.txt``).
"""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import List, Optional, Sequence, Union

import numpy as np

from .. import _reset_clock, _say, _warn
from .config import (RESOLVED_DELTA_CHI2, BootstrapConfig, Lines, LineFitConfig,
                     WeightingConfig, per_line, ranges_per_line)
from .linefit import (Cube, FitResult, FitSpec, Scan, Whitener, fit, flux_per_annulus,
                      gaussian_fraction, global_significance, influence, model_cube,
                      peak_flux_density, residual_continuum, window_flux, window_influence,
                      zero_spacing_spectrum)
from .noise import build_noise_model, combine_variance
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
    annulus_flux_imaginary: Optional[np.ndarray] = None  # the same fit on the imaginary part

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
            self.annulus_ok.astype(int), self.sources_per_annulus,
            (self.annulus_flux_imaginary if self.annulus_flux_imaginary is not None
             else np.full(len(self.annulus_flux), np.nan))]),
            header="b_klambda  line_flux  err_thermal  used  n_sources  line_flux_imaginary")
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
        h = (1.0 if name.startswith(("v0", "fwhm"))
             else max(abs(p0.get(name, 0.0)), 1e-30) * 1e-3)
        cols.append(wh(model_cube(cube, {**p0, name: p0[name] + h}) - base) / h)
    if cols:
        jac = np.column_stack(cols)
        d = d - jac @ np.linalg.lstsq(jac, d, rcond=None)[0]
    norm2 = float(d @ d)
    if not norm2 > 0:
        return 1.0
    return float(np.clip(wh.quad(d) / norm2, 0.25, 10.0))


def _line_ranges(cfg: LineFitConfig, line: Lines, dv: float):
    """Allowed centre and FWHM of every line, ``[(v0_range, fwhm_range)]``:
    the user's ranges, or the defaults from the windows."""
    v0r = ranges_per_line(cfg.v0_range_kms, line.n, "v0_range")
    fwr = ranges_per_line(cfg.fwhm_range_kms, line.n, "fwhm_range")
    out = []
    for k in range(line.n):
        lo_w, hi_w = line.windows_kms[k]
        fw = fwr[k] or (2.0 * dv, max(4.0 * dv, hi_w - lo_w))
        if fw[0] <= 0:
            raise ValueError("fwhm_range: the FWHM must be > 0")
        v0 = v0r[k]
        if v0 is None:
            v0 = (lo_w + 0.5 * fw[0], hi_w - 0.5 * fw[0])
            if v0[0] > v0[1]:                    # window narrower than the smallest width
                v0 = (0.5 * (lo_w + hi_w),) * 2
        out.append((tuple(v0), tuple(fw)))
    return out


def _grids(v0_range, fwhm_range, dv):
    """Centres (steps of half a channel, at most 160) and widths (14, log)
    of the search; a single value for a fixed one."""
    lo, hi = v0_range
    if lo == hi:
        v0_grid = [float(lo)]
    else:
        n_v = max(1, int(round((hi - lo) / (0.5 * dv))) + 1)
        v0_grid = list(np.linspace(lo, hi, min(n_v, 160)))
    lo, hi = fwhm_range
    fwhm_grid = [float(lo)] if lo == hi else list(np.geomspace(lo, hi, 14))
    return v0_grid, fwhm_grid


def _fit_spec(cfg: LineFitConfig, ranges, start: dict, size: str, n_lines: int):
    """The FitSpec of the adopted model: centres and widths free within
    their ranges (fixed where lo = hi)."""
    (v0r, fwr) = ranges[0]
    free = ["F"]
    values = {"F": start["F"], "v0": start["v0"], "fwhm": start["fwhm"], "theta": 0.0}
    th_lo, th_hi = max(1e-3, cfg.theta_min_arcsec), cfg.theta_max_arcsec
    bounds = {"v0": v0r, "fwhm": fwr, "theta": (th_lo, th_hi)}
    if v0r[0] < v0r[1]:
        free.append("v0")
    else:
        values["v0"] = v0r[0]
    if fwr[0] < fwr[1]:
        free.append("fwhm")
    else:
        values["fwhm"] = fwr[0]
    if cfg.theta_fixed_arcsec is not None:
        values["theta"] = float(cfg.theta_fixed_arcsec)
    elif size == "gauss":
        free.append("theta")
        values["theta"] = float(np.clip(start.get("theta", 0.5), max(0.01, th_lo), th_hi))
    for k in range(2, n_lines + 1):
        v0k, fwk = ranges[k - 1]
        values.update({f"F{k}": start.get(f"F{k}", 0.0),
                       f"v0{k}": start.get(f"v0{k}", 0.5 * (v0k[0] + v0k[1])),
                       f"fwhm{k}": start.get(f"fwhm{k}", np.sqrt(fwk[0] * fwk[1]))})
        bounds.update({f"v0{k}": v0k, f"fwhm{k}": fwk})
        free.append(f"F{k}")
        if v0k[0] < v0k[1]:
            free.append(f"v0{k}")
        else:
            values[f"v0{k}"] = v0k[0]
        if fwk[0] < fwk[1]:
            free.append(f"fwhm{k}")
        else:
            values[f"fwhm{k}"] = fwk[0]
    if cfg.continuum:
        values["C"] = start.get("C", 0.0)
        free.append("C")
    prior = cfg.theta_prior_arcsec if "theta" in free else None
    if prior is not None:
        values["theta"] = float(np.clip(prior[0], max(0.01, th_lo), th_hi))
    return FitSpec(tuple(free), values, bounds, n_lines=n_lines, continuum=cfg.continuum,
                   theta_prior=prior)


def _area(spectrum, error, dv, mask, corr, continuum=None, continuum_error=0.0):
    """Sum of the spectrum over ``mask`` times the channel width, minus the
    continuum when it was fitted with the lines (a constant at zero spacing,
    with the size of the lines): ``(flux, error, n_channels)``."""
    f, e, n = window_flux(spectrum, error, dv, mask, corr)
    if continuum is not None and n:
        f = f - dv * n * float(continuum)
        e = float(np.hypot(e, dv * n * float(continuum_error or 0.0)))
    return f, e, n


def _names(k):
    """Parameter names of line ``k`` (0-based)."""
    return ("F", "v0", "fwhm") if k == 0 else (f"F{k + 1}", f"v0{k + 1}", f"fwhm{k + 1}")


def _significance(best, scanner, n_null, seed, single):
    """p-value and significance of a search: from its S/N for a single
    template, from null realisations of the same search otherwise."""
    from scipy.stats import norm
    if single:
        p = float(norm.sf(best["snr"]))
        return ({"p_value": p, "significance_sigma": float(best["snr"]),
                 "n_effective_trials": 1.0, "n_null": 0,
                 "p_value_method": "one template (centre, width and size fixed): "
                                   "the S/N of the matched filter"}, np.zeros(0))
    if not n_null:
        n_templates = sum(t.shape[1] for t, _, _, _ in scanner.blocks)
        p = float(min(1.0, n_templates * norm.sf(best["snr"])))
        return ({"p_value": p, "significance_sigma": float(norm.isf(p)) if p < 1 else 0.0,
                 "n_effective_trials": float(n_templates), "n_null": 0,
                 "p_value_method": f"Bonferroni over {n_templates} templates (n_null = 0): "
                                   "an upper limit on p"}, np.zeros(0))
    null = scanner.null(n=n_null, seed=seed)
    return global_significance(best["snr"], null), null


def _stack_and_cells(job):
    """The stack of the sources, its noise, the cells (channel x annulus) to fit,
    and the groups of lines: those whose windows overlap are fitted together,
    the others each on its own."""
    cfg, input_list, line = job.cfg, job.input_list, job.line
    match_tol_arcsec, statistics, verbose = job.match_tol_arcsec, job.statistics, job.verbose
    weighting = job.weighting
    _reset_clock()
    data = build_stack_input(statistics, input_list, weighting,
                             match_tol_arcsec=match_tol_arcsec, verbose=verbose)
    sums, factors = data.sums, data.factors
    n_src = data.n_sources
    renorm = weighting.renormalise
    freq = data.freq_grid
    velocity = line.velocity(freq)
    dv = float(np.median(np.abs(np.diff(velocity)))) if len(velocity) > 1 else 1.0
    n_lines = line.n
    vel_k = [line.velocity(freq, k=k) for k in range(n_lines)]
    meta = data.meta
    contsub = (meta.get("line_column") == "CORRECTED_DATA") or ("contsub" in meta)
    ranges = _line_ranges(cfg, line, dv)
    win_k = (per_line(cfg.window_kms, n_lines, "window") if cfg.window_kms is not None
             else list(line.windows_kms))

    use_scatter = cfg.noise == "scatter" and data.has_scatter
    if cfg.noise == "scatter" and not data.has_scatter:
        _warn("measure: the statistics have no scatter accumulators (format 3): "
              "the noise is taken from the weights, whose absolute scale is not "
              "checked.  Compress again to measure it")
    var_src = data.variance if use_scatter else data.variance_weights

    # ---- nominal stack and its noise -------------------------------------
    real, imag, weight, mean_b = combine(sums, factors, renormalise=renorm)
    line_free = line.free_mask(freq)
    noise = build_noise_model(
        sums, var_src, data.variance_weights, factors, renorm, real, imag, line_free,
        contsub, data.kappa2_entries if use_scatter else {},
        data.dof_entries, kappa_channel=data.kappa2_channel,
        source="scatter" if use_scatter else "weights",
        correlate=cfg.channel_correlation)
    var = noise.variance
    corr = noise.corr if cfg.channel_correlation else None

    # ---- cells to fit: the channels within the fit span of a line ----------
    per_channel = sources_per_channel(sums)

    def near(lines_k):
        """The channels within the fit span of any of these lines."""
        if cfg.fit_span_kms is None:
            return np.ones(len(freq), bool)
        out = np.zeros(len(freq), bool)
        for k in lines_k:
            out |= np.abs(vel_k[k]) < cfg.fit_span_kms
        return out

    in_span = near(range(n_lines))
    per_annulus = sources_per_bin(sums, in_span)
    min_src = 1 if n_src == 1 else max(2, cfg.min_sources_per_bin)
    bin_ok = per_annulus >= min_src
    usable = np.isfinite(real) & np.isfinite(var) & (var > 0) & bin_ok[None, :] & (weight > 0)
    ok = usable & in_span[:, None]
    if not ok.any():
        raise RuntimeError("measure: no usable cell (check the line windows, the fit span "
                           "and the minimum number of sources per annulus)")
    baseline = np.where(np.isfinite(mean_b), mean_b, 0.0)

    # ---- the groups: lines whose windows overlap are fitted together; the
    # others each on its own, on the channels within the fit span of its
    # lines and outside the windows of the other lines, sharing nothing
    # (size, model, continuum)
    groups = line.groups()
    group_of = {k: g for g, members in enumerate(groups) for k in members}
    line_windows = [line.window_mask(freq, k) for k in range(n_lines)]
    G = []
    for members in groups:
        span = near(members)
        for j in range(n_lines):
            if j not in members:
                span &= ~line_windows[j]
        ok_g = usable & span[:, None]
        if not ok_g.any():
            raise RuntimeError(f"measure: no usable cell for line(s) "
                               f"{', '.join(str(k + 1) for k in members)}")
        cube_g = Cube(real=real, variance=var, baseline=baseline, velocity=vel_k[members[0]],
                      dv=dv, ok=ok_g, corr=corr,
                      velocities=[vel_k[j] for j in members[1:]] or None,
                      edges=data.bin_edges_lambda)
        G.append(SimpleNamespace(members=members, ok=ok_g, cube=cube_g,
                                 wh=Whitener(var, ok_g, corr)))
    _say(f"  cube: {int(ok.sum()):,} cells ({int(ok.any(axis=1).sum())} channels x "
         f"{int(bin_ok.sum())} annuli), noise from the "
         f"{'scatter of the visibilities' if use_scatter else 'weights'}; channel "
         f"correlation {', '.join(f'{r:+.2f}' for r in noise.rho[1:]) or 'none'} "
         f"({noise.checks.get('channel_correlation_method')})", verbose)
    if n_lines > 1:
        _say("  " + "; ".join(
            f"lines {' + '.join(str(k + 1) for k in m)} fitted together (their windows "
            f"overlap)" if len(m) > 1 else f"line {m[0] + 1} measured on its own"
            for m in groups), verbose)
    job.G, job.baseline, job.bin_ok, job.contsub, job.corr = G, baseline, bin_ok, contsub, corr
    job.data, job.dv, job.factors, job.freq, job.group_of = data, dv, factors, freq, group_of
    job.groups, job.imag, job.line_free, job.mean_b = groups, imag, line_free, mean_b
    job.meta, job.min_src, job.n_lines, job.n_src = meta, min_src, n_lines, n_src
    job.noise, job.ok, job.per_annulus, job.per_channel = noise, ok, per_annulus, per_channel
    job.ranges, job.real, job.renorm, job.sums, job.usable = ranges, real, renorm, sums, usable
    job.var, job.var_src, job.vel_k, job.velocity = var, var_src, vel_k, velocity
    job.weight, job.win_k = weight, win_k


def _search(job):
    """The search: every line is looked for over the allowed centres, widths and
    source sizes with a matched filter; the best S/N and its significance,
    from the same search on noise (and on stacks with random phases)."""
    G, baseline, bootstrap, cfg, dv = job.G, job.baseline, job.bootstrap, job.cfg, job.dv
    factors, group_of, imag, line = job.factors, job.group_of, job.imag, job.line
    n_lines, n_src, ok, ranges, renorm = job.n_lines, job.n_src, job.ok, job.ranges, job.renorm
    sums, var, vel_k, verbose = job.sums, job.var, job.vel_k, job.verbose
    # ---- the search, line by line, within the ranges ----------------------
    # source sizes of the scan: a point source, and Gaussians of 0.5, 1 and 2
    # times the size that halves the visibility at the median baseline
    wmed = np.where(ok, 1.0 / np.where(ok, var, 1.0), 0.0)
    order_b = np.argsort(baseline[ok])
    cw = np.cumsum(wmed[ok][order_b])
    b_med = float(baseline[ok][order_b][np.searchsorted(cw, 0.5 * cw[-1])])
    theta_half = 2 * np.log(2) / (np.pi * max(b_med, 1.0)) * 206264.806
    th_min, th_max = cfg.theta_min_arcsec, cfg.theta_max_arcsec
    if cfg.theta_fixed_arcsec is not None:
        thetas = [float(cfg.theta_fixed_arcsec)]
    elif cfg.theta_prior_arcsec is not None:
        thetas = [float(np.clip(cfg.theta_prior_arcsec[0], th_min, th_max))]
    elif cfg.size == "point":
        thetas = [0.0]
    else:
        thetas = ([0.0] if th_min == 0 else []) + [
            t for t in (0.5 * theta_half, theta_half, 2 * theta_half) if th_min <= t <= th_max]
        if not thetas:                       # a narrow range: its middle
            thetas = [0.5 * (th_min + th_max)]
    scanners, bests, detections, nulls = [], [], [], []
    for k in range(n_lines):
        gs = G[group_of[k]]
        cube_k = (gs.cube if k == gs.members[0]
                  else replace(gs.cube, velocity=vel_k[k], velocities=None))
        v0_grid, fwhm_grid = _grids(*ranges[k], dv)
        scanner = Scan(cube_k, gs.wh, v0_grid, fwhm_grid, thetas, continuum=cfg.continuum)
        best = scanner.best()
        single = len(v0_grid) == 1 and len(fwhm_grid) == 1 and len(thetas) == 1
        signif, null = _significance(best, scanner, cfg.n_null, bootstrap.seed + 7 * k,
                                     single)
        det = {"snr": best["snr"], "v0_kms": best["v0"], "fwhm_kms": best["fwhm"],
               "theta_arcsec": best["theta"], "thetas_scanned_arcsec": thetas,
               "v0_range_kms": list(ranges[k][0]), "fwhm_range_kms": list(ranges[k][1]),
               "fixed_shape": bool(single),
               "flux_matched_filter": best["flux"],
               "flux_matched_filter_error": best["flux_error"], **signif}
        yi = gs.wh(np.where(gs.ok, np.where(np.isfinite(imag), imag, 0.0), 0.0))
        det["snr_imaginary"] = float(best["templates"][:, best["k"]] @ yi.astype(np.float32))
        scanners.append(scanner)
        bests.append(best)
        detections.append(det)
        nulls.append(null)
        name = "" if n_lines == 1 else f"line {k + 1} ({line.freqs_ghz[k]:.6g} GHz): "
        _say(f"  {name}{'matched filter' if single else 'search'}: S/N = "
             f"{best['snr']:.2f} at v0 = {best['v0']:+.0f} km/s, FWHM = {best['fwhm']:.0f} "
             f"km/s, size {best['theta']:.2f}\""
             + ("" if signif.get("n_null", 0) == 0 else
                f"  ->  p = {signif['p_value']:.2g} ({signif['significance_sigma']:.1f} sigma, "
                f"{signif['n_effective_trials']:.0f} effective trials)"), verbose)
    detection = detections[0]
    null = nulls[0]

    # data-driven null: random phases per source
    null_phase = np.zeros(0)
    if cfg.n_null_phase and n_src >= 2:
        rng = np.random.default_rng(bootstrap.seed + 1)
        vals = [[] for _ in range(n_lines)]
        for _ in range(int(cfg.n_null_phase)):
            phi = rng.uniform(0, 2 * np.pi, n_src)
            rot = sums.copy()
            rot[..., 0] = (np.cos(phi)[:, None, None] * sums[..., 0]
                           - np.sin(phi)[:, None, None] * sums[..., 1])
            r_rot, _, _, _ = combine(rot, factors, renormalise=renorm)
            for k in range(n_lines):
                vals[k].append(scanners[k].best(y=r_rot)["snr"])
        for k in range(n_lines):
            nk = np.asarray(vals[k])
            detections[k]["p_value_phase_null"] = float(
                (1 + np.sum(nk >= bests[k]["snr"])) / (1 + nk.size))
            detections[k]["phase_null_snr_rms"] = float(np.sqrt(np.mean(nk ** 2)))
        null_phase = np.asarray(vals[0])
    job.bests, job.detection, job.detections, job.null = bests, detection, detections, null
    job.null_phase, job.th_min, job.theta_half = null_phase, th_min, theta_half


def _fit_groups(job):
    """The fit of every group of lines on all its cells at once: with a point
    source and with a Gaussian one, and the choice between the two."""
    G, bests, cfg, ranges, th_min = job.G, job.bests, job.cfg, job.ranges, job.th_min
    theta_half = job.theta_half
    # ---- the fit of every group: point source, and a Gaussian if asked or
    # preferred ---------------------------------------------------------------
    if cfg.theta_prior_arcsec is not None and cfg.theta_fixed_arcsec is None:
        sizes = ["gauss"]                # the prior decides the size
    elif cfg.size == "auto" and cfg.theta_fixed_arcsec is None and th_min > 0:
        sizes = ["gauss"]                # a point source is below the range
    else:
        sizes = ([cfg.size] if cfg.size != "auto" or cfg.theta_fixed_arcsec is not None
                 else ["point", "gauss"])
    for gs in G:
        b0 = bests[gs.members[0]]
        start = {"F": b0["flux"], "v0": b0["v0"], "fwhm": b0["fwhm"],
                 "theta": b0["theta"] if b0["theta"] > 0 else 0.5}
        for i, k in enumerate(gs.members[1:], start=1):
            nf, nv, nw = _names(i)
            start.update({nf: bests[k]["flux"], nv: bests[k]["v0"], nw: bests[k]["fwhm"]})
        ranges_g = [ranges[k] for k in gs.members]
        fits = {}
        for size in sizes:
            spec = _fit_spec(cfg, ranges_g, start, size, len(gs.members))
            fits[size] = (fit(gs.cube, spec, gs.wh), spec)
        size_test, w_gauss, size_factor = None, None, 1.0
        if len(fits) == 2:
            dchi2_raw = fits["point"][0].chi2 - fits["gauss"][0].chi2
            size_factor = size_test_correlation(gs.cube, gs.wh, fits["point"][0],
                                                fits["point"][1],
                                                fits["gauss"][0].params.get("theta", 0.0),
                                                theta_half)
            dchi2 = dchi2_raw / size_factor
            adopted = "gauss" if dchi2 >= RESOLVED_DELTA_CHI2 else "point"
            w_gauss = akaike_weight(dchi2)
            size_test = {"delta_chi2_point_minus_gauss": float(dchi2),
                         "delta_chi2_uncorrelated": float(dchi2_raw),
                         "correlation_factor": size_factor,
                         "threshold": RESOLVED_DELTA_CHI2, "adopted": adopted,
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
        gs.fits, gs.size_test, gs.w_gauss, gs.size_factor = fits, size_test, w_gauss, size_factor
        gs.adopted, gs.result, gs.spec = adopted, result, spec
        gs.theta = float(result.params.get("theta", 0.0))


def _bootstrap(job):
    """The bootstrap over the sources: the fits repeated on resampled stacks,
    for the errors that come from the differences among the galaxies."""
    G, baseline, bootstrap, corr, data = job.G, job.baseline, job.bootstrap, job.corr, job.data
    dv, factors, freq, line, n_lines = job.dv, job.factors, job.freq, job.line, job.n_lines
    n_src, renorm, sums, var_src = job.n_src, job.renorm, job.sums, job.var_src
    win_k = job.win_k
    # ---- bootstrap over the sources --------------------------------------
    rng = np.random.default_rng(bootstrap.seed)
    win_masks = [line.window_mask(freq, k, win_k[k]) for k in range(n_lines)]
    for gs in G:
        spec, result = gs.spec, gs.result
        gs.boot_params, gs.boot_profile = [], []
        gs.boot_window = {k: [] for k in gs.members}
        gs.boot_flux_avg = {k: [] for k in gs.members}
        gs.spec_boot = FitSpec(spec.free, dict(result.params), spec.bounds,
                               n_lines=spec.n_lines, continuum=spec.continuum,
                               theta_prior=spec.theta_prior)
        gs.other = None
        if gs.w_gauss is not None:
            o_res, o_spec = gs.fits["point" if gs.adopted == "gauss" else "gauss"]
            gs.other = FitSpec(o_spec.free, dict(o_res.params), o_spec.bounds,
                               n_lines=o_spec.n_lines, continuum=o_spec.continuum)
    if n_src >= 2:
        for _ in range(int(bootstrap.n_realisations)):
            m = np.bincount(rng.integers(0, n_src, n_src), minlength=n_src).astype(float)
            r_b, _, w_b, mb_b = combine(sums, factors, multiplicity=m, renormalise=renorm)
            v_b = combine_variance(var_src, sums, factors, multiplicity=m,
                                   renormalise=renorm)
            for gs in G:
                ok_b = gs.ok & np.isfinite(r_b) & np.isfinite(v_b) & (v_b > 0)
                if not ok_b.any():
                    continue
                c_b = Cube(real=r_b, variance=v_b,
                           baseline=np.where(np.isfinite(mb_b), mb_b, baseline),
                           velocity=gs.cube.velocity, dv=dv, ok=ok_b, corr=corr,
                           velocities=gs.cube.velocities, edges=data.bin_edges_lambda)
                fb = fit(c_b, gs.spec_boot)
                gs.boot_params.append({k: fb.params.get(k) for k in gs.spec.free})
                s_b, e_b = zero_spacing_spectrum(c_b, float(fb.params.get("theta", gs.theta)))
                f_w = {k: _area(s_b, e_b, dv, win_masks[k], corr, fb.params.get("C"))[0]
                       for k in gs.members}
                if gs.other is not None:
                    fo = fit(c_b, gs.other)
                    fp, fg = (fo, fb) if gs.adopted == "gauss" else (fb, fo)
                    wg = akaike_weight((fp.chi2 - fg.chi2) / gs.size_factor)
                    for i, k in enumerate(gs.members):
                        nf = _names(i)[0]
                        gs.boot_flux_avg[k].append((1 - wg) * fp.params[nf]
                                                   + wg * fg.params[nf])
                    s_o, e_o = zero_spacing_spectrum(c_b, float(fo.params.get("theta", 0.0)))
                    w_this = wg if gs.adopted == "gauss" else 1 - wg
                    f_w = {k: w_this * f_w[k] + (1 - w_this)
                           * _area(s_o, e_o, dv, win_masks[k], corr, fo.params.get("C"))[0]
                           for k in gs.members}
                    s_b = w_this * s_b + (1 - w_this) * s_o
                gs.boot_profile.append(s_b)
                for k in gs.members:
                    gs.boot_window[k].append(f_w[k])
    for gs in G:
        gs.boot_err, gs.boot_pct = {}, {}
        for k in gs.spec.free:
            vals = np.array([b[k] for b in gs.boot_params if b.get(k) is not None],
                            dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size > 1:
                gs.boot_err[k] = float(np.std(vals))
                gs.boot_pct[k] = [float(x) for x in np.percentile(vals, [16, 50, 84])]
    job.win_masks = win_masks


def _profiles(job):
    """The spectrum at zero spacing: every channel extrapolated to zero baseline
    with the size of its group, with its noise, the model and the imaginary part."""
    G, baseline, corr, dv, freq = job.G, job.baseline, job.corr, job.dv, job.freq
    imag, line, noise, real, var = job.imag, job.line, job.noise, job.real, job.var
    vel_k = job.vel_k
    # ---- profile, window fluxes, annuli ------------------------------------
    # every channel takes the profile of the group whose lines are nearest (a
    # resolved source: the zero-spacing profile depends on the size of its group)
    dist = np.full((len(G), len(freq)), np.inf)
    for g, gs in enumerate(G):
        for k in gs.members:
            lo_w, hi_w = line.windows_kms[k]
            dist[g] = np.minimum(dist[g], np.maximum(np.maximum(lo_w - vel_k[k],
                                                                vel_k[k] - hi_w), 0.0))
    owner = np.argmin(dist, axis=0)
    chan = np.arange(len(freq))

    def compose(per_group):
        """One value per channel, from its group (another one where it has none)."""
        arr = np.asarray(per_group, float)
        out = arr[owner, chan]
        for g in range(arr.shape[0]):
            out = np.where(np.isfinite(out), out, arr[g])
        return out

    yim = np.where(np.isfinite(imag), imag, 0.0)
    var_w = noise.variance_weights
    parts = {"prof": [], "err": [], "im": [], "lines": [], "c": [], "noise": [], "boot": [],
             "err_w": []}
    for gs in G:
        cube_g, theta = gs.cube, gs.theta
        zero_g = Cube(real=real, variance=var, baseline=np.zeros_like(baseline),
                      velocity=cube_g.velocity, dv=dv, ok=gs.ok, corr=corr,
                      velocities=cube_g.velocities)

        def lines_and_c(params):
            pl = {k: v for k, v in params.items() if k != "C"}
            return model_cube(zero_g, pl)[:, 0], float(params.get("C", 0.0))

        prof, prof_err = zero_spacing_spectrum(cube_g, theta)
        parts["noise"].append(prof_err)
        prof_im, _ = zero_spacing_spectrum(cube_g, theta, y=yim)
        model_lines, c_model = lines_and_c(gs.result.params)
        if gs.w_gauss is not None:
            # the profile averaged over the two size models, like the flux
            pp = {}
            for k in ("point", "gauss"):
                rk = gs.fits[k][0]
                tk = float(rk.params.get("theta", 0.0))
                pk, ek = zero_spacing_spectrum(cube_g, tk)
                pp[k] = (pk, ek, zero_spacing_spectrum(cube_g, tk, y=yim)[0],
                         *lines_and_c(rk.params))
            wts = {"point": 1 - gs.w_gauss, "gauss": gs.w_gauss}
            prof = sum(wts[k] * pp[k][0] for k in pp)
            prof_err = np.sqrt(sum(wts[k] * (pp[k][1] ** 2 + (pp[k][0] - prof) ** 2)
                                   for k in pp))
            prof_im = sum(wts[k] * pp[k][2] for k in pp)
            model_lines = sum(wts[k] * pp[k][3] for k in pp)
            c_model = sum(wts[k] * pp[k][4] for k in pp)
        parts["prof"].append(prof)
        parts["err"].append(prof_err)
        parts["im"].append(prof_im)
        parts["lines"].append(model_lines)
        parts["c"].append(np.full(len(freq), c_model))
        with warnings.catch_warnings():         # channels outside the fit: all NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            parts["boot"].append(np.nanstd(np.asarray(gs.boot_profile), axis=0)
                                 if gs.boot_profile else np.full_like(prof, np.nan))
        parts["err_w"].append(zero_spacing_spectrum(cube_g, theta, variance=var_w)[1])
    prof, prof_err, prof_im = compose(parts["prof"]), compose(parts["err"]), compose(parts["im"])
    noise_th = compose(parts["noise"])
    model_profile = sum(parts["lines"]) + compose(parts["c"])
    prof_boot = compose(parts["boot"])
    prof_err_w = compose(parts["err_w"])
    job.compose, job.model_profile, job.noise_th = compose, model_profile, noise_th
    job.prof, job.prof_boot, job.prof_err = prof, prof_boot, prof_err
    job.prof_err_w, job.prof_im = prof_err_w, prof_im


def _fluxes(job):
    """The two fluxes of every line, the area of the Gaussian and the area under
    the spectrum in the window, the test that compares them and the one that
    is quoted; the flux per annulus; the continuum fitted with the lines."""
    G, cfg, corr, detections, dv = job.G, job.cfg, job.corr, job.detections, job.dv
    group_of, imag, line, n_lines, ok = job.group_of, job.imag, job.line, job.n_lines, job.ok
    vel_k, win_k, win_masks = job.vel_k, job.win_k, job.win_masks
    fitted = ok.any(axis=1)

    def windows(cube_g, res, sp, mask, opt_mask):
        """Window fluxes of one fit; a resolved source: the profile at zero
        spacing depends on the size, so its uncertainty adds to theirs."""
        th = float(res.params.get("theta", 0.0))
        pr, pe = zero_spacing_spectrum(cube_g, th)
        c_fit, c_err = res.params.get("C"), res.errors.get("C", 0.0)
        fwin, ewin, nwin = _area(pr, pe, dv, mask, corr, c_fit, c_err)
        fopt, eopt, _ = _area(pr, pe, dv, opt_mask, corr, c_fit, c_err)
        e_t = res.errors.get("theta", 0.0) if "theta" in sp.free else 0.0
        if th > 0 and e_t > 0:
            step = max(0.25 * e_t, 1e-3)
            lo_t = max(th - step, 0.0)
            f_up = _area(zero_spacing_spectrum(cube_g, th + step)[0], pe, dv, mask, corr,
                         c_fit)[0]
            f_dn = _area(zero_spacing_spectrum(cube_g, lo_t)[0], pe, dv, mask, corr, c_fit)[0]
            dfdt = (f_up - f_dn) / (th + step - lo_t)
            ewin = float(np.hypot(ewin, dfdt * e_t))
            if fwin:
                eopt = float(np.hypot(eopt, dfdt * e_t * fopt / fwin))
        return fwin, ewin, nwin, fopt, eopt

    def averaged(values, errors, weights):
        fa = sum(w * v for w, v in zip(weights, values))
        ea = np.sqrt(sum(w * (e ** 2 + (v - fa) ** 2)
                         for w, v, e in zip(weights, values, errors)))
        return float(fa), float(ea)

    ann_flux, ann_err = flux_per_annulus(G[0].cube, G[0].wh, dict(G[0].result.params))
    # the same on the imaginary part: no signal for a centred source, a null test
    ann_imag, _ = flux_per_annulus(
        replace(G[0].cube, real=np.where(np.isfinite(imag), imag, 0.0)), G[0].wh,
        dict(G[0].result.params))

    # ---- the continuum fitted with the lines (no subtraction) --------------
    for gs in G:
        gs.continuum_fit = None
        result = gs.result
        if "C" in result.params:
            c_val, c_th = result.params["C"], result.errors.get("C", float("nan"))
            if gs.w_gauss is not None:
                fp, fg = gs.fits["point"][0], gs.fits["gauss"][0]
                c_val, c_th = averaged([fp.params["C"], fg.params["C"]],
                                       [fp.errors.get("C", np.nan), fg.errors.get("C", np.nan)],
                                       [1 - gs.w_gauss, gs.w_gauss])
            c_bt = gs.boot_err.get("C", float("nan"))
            gs.continuum_fit = {
                "flux_density": float(c_val),
                "flux_density_error": float(np.nanmax([c_th, c_bt])),
                "flux_density_error_thermal": float(c_th),
                "flux_density_error_bootstrap": float(c_bt),
                "theta_fwhm_arcsec": gs.theta,
                "note": "constant in frequency, fitted with the lines and sharing their size; "
                        "the area fluxes are the sums over the windows minus it"}
    continuum_fit = G[0].continuum_fit

    # ---- every line: the Gaussian flux, the area flux, and which to quote --
    blocks = []
    flux_choice_notes = []
    for k in range(n_lines):
        gs = G[group_of[k]]
        result, spec, fits, w_gauss = gs.result, gs.spec, gs.fits, gs.w_gauss
        boot_err = gs.boot_err
        nf, nv, nw = _names(gs.members.index(k))
        v0_k, fw_k = result.params[nv], result.params[nw]
        lo_k, hi_k = win_k[k]
        opt_mask = np.abs(vel_k[k] - v0_k) <= 0.5 * 1.19 * fw_k
        # Gaussian: averaged over the two size models with their weights
        f_g = result.params[nf]
        e_g_th = result.errors.get(nf, float("nan"))
        e_g_bt = boot_err.get(nf, float("nan"))
        if w_gauss is not None:
            fp, fg = fits["point"][0], fits["gauss"][0]
            f_g, e_g_th = averaged([fp.params[nf], fg.params[nf]],
                                   [fp.errors.get(nf, np.nan), fg.errors.get(nf, np.nan)],
                                   [1 - w_gauss, w_gauss])
            if len(gs.boot_flux_avg[k]) > 1:
                e_g_bt = float(np.std(gs.boot_flux_avg[k]))
        e_g = float(np.nanmax([e_g_th, e_g_bt])) if np.isfinite(e_g_th) else e_g_bt
        # area: the sum of the spectrum over the window
        f_a, e_a_th, n_a, f_o, e_o = windows(gs.cube, result, spec, win_masks[k], opt_mask)
        if w_gauss is not None:
            wins = {s: windows(gs.cube, *fits[s], win_masks[k], opt_mask)
                    for s in ("point", "gauss")}
            ww = [1 - w_gauss, w_gauss]
            f_a, e_a_th = averaged([wins["point"][0], wins["gauss"][0]],
                                   [wins["point"][1], wins["gauss"][1]], ww)
            f_o, e_o = averaged([wins["point"][3], wins["gauss"][3]],
                                [wins["point"][4], wins["gauss"][4]], ww)
        e_a_bt = (float(np.nanstd(gs.boot_window[k])) if gs.boot_window[k]
                  else float("nan"))
        e_a = float(np.nanmax([e_a_th, e_a_bt]))
        # their difference: its error from the same data, to first order
        a_g = influence(result, nf)
        a_w = window_influence(gs.cube, gs.wh, gs.theta, win_masks[k], dv)
        a_c = influence(result, "C")
        if a_c is not None:                  # the area minus the fitted continuum
            a_w = a_w - dv * n_a * a_c
        d_err = (float(np.sqrt(max(gs.wh.quad(a_w - a_g), 0.0) * max(result.scale, 1.0)))
                 if a_g is not None else float("nan"))
        delta = f_a - f_g
        nsig = delta / d_err if d_err > 0 else float("nan")
        cover = gaussian_fraction(lo_k, hi_k, v0_k, fw_k)
        blended = line.overlaps(k) if cfg.window_kms is None else [
            j for j in range(n_lines) if j != k
            and win_masks[j].any() and (win_masks[j] & win_masks[k]).any()]
        gaussian = not (np.isfinite(nsig) and abs(nsig) > cfg.shape_nsigma)
        if cfg.flux != "auto":
            method, why = cfg.flux, f"flux_measure='{cfg.flux}'"
        elif gaussian:
            method = "gauss"
            why = (f"the area agrees with the Gaussian ({nsig:+.1f} sigma)"
                   if np.isfinite(nsig) else "no area flux to compare with")
        elif blended:
            method = "gauss"
            why = (f"the window overlaps line(s) {', '.join(str(j + 1) for j in blended)}: "
                   f"only the joint Gaussian fit separates them")
        elif not cover >= 0.95:
            method = "gauss"
            why = (f"the window holds {100 * cover:.0f}% of the fitted line: too narrow "
                   f"for a flux without a shape")
        else:
            method = "area"
            why = (f"the area and the Gaussian fit differ by {nsig:+.1f} sigma: the line is "
                   f"not a Gaussian")
        if method == "area":
            flux_k, err_k, eth_k, ebt_k = f_a, e_a, e_a_th, e_a_bt
        else:
            flux_k, err_k, eth_k, ebt_k = f_g, e_g, e_g_th, e_g_bt
        if not gaussian:
            flux_choice_notes.append((k, nsig, method, why, blended, cover))
        win_v1 = line.velocity(np.array(line.window_freq_hz(k, (lo_k, hi_k))))
        blk = {
            "rest_freq_ghz": line.freqs_ghz[k], "window_kms": [float(lo_k), float(hi_k)],
            "window_kms_line1_velocity": sorted(float(v) for v in win_v1),
            "flux": flux_k, "flux_error": err_k, "flux_error_thermal": eth_k,
            "flux_error_bootstrap": ebt_k, "flux_method": method,
            "flux_snr": flux_k / err_k if err_k > 0 else float("nan"),
            "flux_gauss": f_g, "flux_gauss_error": e_g, "flux_gauss_error_thermal": e_g_th,
            "flux_gauss_error_bootstrap": e_g_bt,
            "flux_area": f_a, "flux_area_error": e_a, "flux_area_error_thermal": e_a_th,
            "flux_area_error_bootstrap": e_a_bt, "flux_area_n_channels": n_a,
            "flux_optimal": f_o, "flux_optimal_error": e_o,
            "optimal_kms": [float(v0_k - 0.595 * fw_k), float(v0_k + 0.595 * fw_k)],
            "shape_check": {"area_minus_gauss": delta, "error": d_err, "nsigma": nsig,
                            "threshold": cfg.shape_nsigma, "gaussian": bool(gaussian),
                            "window_fraction_of_gaussian": cover,
                            "blended_with_lines": [j + 1 for j in blended],
                            "adopted": method, "reason": why},
            "v0_kms": v0_k, "v0_error_kms": float(np.nanmax(
                [result.errors.get(nv, 0.0), boot_err.get(nv, 0.0)])),
            "fwhm_kms": fw_k, "fwhm_error_kms": float(np.nanmax(
                [result.errors.get(nw, 0.0), boot_err.get(nw, 0.0)])),
            "peak": peak_flux_density(f_g, fw_k), "peak_error": peak_flux_density(e_g, fw_k),
            "detection": detections[k],
        }
        if n_lines > 1:
            # what this line shares: the fit of its group (the size, the continuum)
            blk.update({
                "fitted_with_lines": [j + 1 for j in gs.members],
                "size_model": result.size_model, "theta_fwhm_arcsec": gs.theta,
                "theta_error_arcsec": float(np.nanmax([result.errors.get("theta", 0.0),
                                                       boot_err.get("theta", 0.0)])),
                "size_test": gs.size_test, "chi2_reduced": result.chi2_reduced,
                "continuum_fit": gs.continuum_fit})
        blocks.append(blk)
    for gs in G:
        if gs.size_test is not None:
            first_g = blocks[gs.members[0]]
            gs.size_test["flux_averaged"] = first_g["flux_gauss"]
            gs.size_test["flux_averaged_error_thermal"] = first_g["flux_gauss_error_thermal"]
    job.ann_err, job.ann_flux, job.ann_imag, job.blocks = ann_err, ann_flux, ann_imag, blocks
    job.continuum_fit, job.fitted = continuum_fit, fitted
    job.flux_choice_notes = flux_choice_notes


def _residual_continuum(job):
    """What is left of the continuum outside the line windows (an offset and a
    slope), and the standardised noise samples of the noise figure."""
    G, baseline, blocks, cfg, compose = job.G, job.baseline, job.blocks, job.cfg, job.compose
    contsub, corr, data, dv, imag = job.contsub, job.corr, job.data, job.dv, job.imag
    line_free, real, usable, var = job.line_free, job.real, job.usable, job.var
    velocity = job.velocity
    # ---- residual continuum: offset and slope outside the windows ---------
    cube_all = Cube(real=real, variance=var, baseline=baseline, velocity=velocity, dv=dv,
                    ok=usable, corr=corr, edges=data.bin_edges_lambda)
    c_fitted = G[0].result.params.get("C")
    if c_fitted is not None:                 # what the fitted continuum leaves
        s_parts, e_parts = [], []
        for gs in G:
            s_g, e_g_all = zero_spacing_spectrum(cube_all, gs.theta)
            s_parts.append(s_g - gs.result.params.get("C", 0.0))
            e_parts.append(e_g_all)
        s_all, e_all = compose(s_parts), compose(e_parts)
    else:                                    # with the size of the first line, as drawn
        s_all, e_all = zero_spacing_spectrum(cube_all, G[0].theta)
    resid = residual_continuum(s_all, e_all, velocity, line_free, corr)
    if resid is not None:
        thr = cfg.residual_nsigma
        resid["threshold_nsigma"] = thr
        resid["continuum_subtracted"] = bool(contsub)
        resid["continuum_fitted"] = c_fitted is not None
        resid["detected"] = bool(abs(resid["offset_nsigma"]) > thr
                                 or abs(resid["slope_nsigma"]) > thr)
        peak0 = blocks[0]["peak"]
        resid["offset_over_line_peak"] = (resid["offset"] / peak0
                                          if peak0 and np.isfinite(peak0) else float("nan"))

    # standardised noise samples, for the diagnostic plot
    sd = np.sqrt(var)
    with np.errstate(invalid="ignore", divide="ignore"):
        zi = (imag / sd)[line_free[:, None] & np.isfinite(var) & (var > 0)
                         & np.isfinite(imag)]
        zf = ((real / sd)[line_free[:, None] & np.isfinite(var) & (var > 0)
                          & np.isfinite(real)] if contsub else np.zeros(0))
    job.c_fitted, job.resid, job.zf, job.zi = c_fitted, resid, zf, zi


def _result(job):
    """The summary (every number) and the result."""
    G, ann_err, ann_flux, ann_imag = job.G, job.ann_err, job.ann_flux, job.ann_imag
    bin_ok, blocks, cfg, continuum_fit = job.bin_ok, job.blocks, job.cfg, job.continuum_fit
    corr, data, detection, dv, fitted = job.corr, job.data, job.detection, job.dv, job.fitted
    freq, groups, line, line_free = job.freq, job.groups, job.line, job.line_free
    mean_b, meta, min_src, model_profile = job.mean_b, job.meta, job.min_src, job.model_profile
    n_src, noise, noise_th, null = job.n_src, job.noise, job.noise_th, job.null
    null_phase, ok, per_annulus = job.null_phase, job.ok, job.per_annulus
    per_channel, prof, prof_boot = job.per_channel, job.prof, job.prof_boot
    prof_err, prof_err_w, prof_im, resid = job.prof_err, job.prof_err_w, job.prof_im, job.resid
    velocity, weight, weighting, zf = job.velocity, job.weight, job.weighting, job.zf
    zi = job.zi
    # ---- summary -------------------------------------------------------------
    result, size_test, theta = G[0].result, G[0].size_test, G[0].theta
    boot_err, boot_pct = G[0].boot_err, G[0].boot_pct
    first = blocks[0]
    flux, adopted_err = first["flux"], first["flux_error"]
    # the parameters of the other lines of the group are in their own blocks
    line_block = {key: value for key, value in result.as_dict().items()
                  if not re.fullmatch(r"(F|v0|fwhm)\d+(_error)?", key)}
    line_block.update(first)
    line_block.update({"bootstrap_errors": boot_err,
                       "bootstrap_percentiles_16_50_84": boot_pct,
                       "n_bootstrap": len(G[0].boot_params)})
    e_th, e_bt = first["flux_error_thermal"], first["flux_error_bootstrap"]
    if np.isfinite(e_th) and np.isfinite(e_bt) and e_th > 0:
        line_block["bootstrap_over_thermal"] = e_bt / e_th
    window = {"user_kms": first["window_kms"], "flux": first["flux_area"],
              "flux_error": first["flux_area_error"],
              "flux_error_thermal": first["flux_area_error_thermal"],
              "flux_error_bootstrap": first["flux_area_error_bootstrap"],
              "n_channels": first["flux_area_n_channels"],
              "optimal_kms": first["optimal_kms"], "flux_optimal": first["flux_optimal"],
              "flux_optimal_error": first["flux_optimal_error"],
              "note_optimal": "1.19 FWHM around the fitted centre: best S/N for a "
                              "Gaussian, 84% of its flux; defined on the data, so "
                              "slightly biased high for a weak line"}
    summary = {
        "method": "joint fit of the channel x annulus cube",
        "n_sources": n_src, "n_entries": data.n_entries,
        "weighting_scheme": weighting.scheme, "rest_freq_ghz": line.rest_freq_ghz,
        "z_ref": data.z_ref, "z_ref_given": weighting.z_ref is not None,
        "flux_normalised": data.norm_ref is not None, "flux_ref": data.norm_ref,
        "flux_ref_given": weighting.norm_ref is not None,
        "lines_setup": line.as_dict(), "flux_choice": cfg.flux,
        "line_groups": [[k + 1 for k in m] for m in groups],
        "line_column": meta.get("line_column"), "contsub": meta.get("contsub"),
        "line": line_block, "lines": [line_block] + blocks[1:],
        "detection": detection, "window": window,
        "size_test": size_test, "continuum_residual": resid,
        "continuum_fit": continuum_fit, "noise": noise.as_dict(),
        "units": "flux in [data units] km/s, flux density and peak in [data units]; "
                 "with the flux normalisation, rescaled by flux_ref / NORM",
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
        annulus_flux_imaginary=ann_imag,
        sources_per_annulus=per_annulus, noise_channel=noise_th,
        noise_channel_weights=prof_err_w, z_imaginary=zi, z_line_free=zf,
        null_snr=null, null_snr_phase=null_phase, channel_correlation=corr)
    job.line_block, job.out, job.summary = line_block, out, summary


def _report(job):
    """The checks said aloud: the noise, a residual continuum, a line that is not
    Gaussian, a weak line; the files; one line of log per line."""
    G, blocks, c_fitted, cfg, contsub = job.G, job.blocks, job.c_fitted, job.cfg, job.contsub
    detection, flux_choice_notes, group_of = job.detection, job.flux_choice_notes, job.group_of
    line, line_block, n_lines, noise = job.line, job.line_block, job.n_lines, job.noise
    out, output_prefix, resid, summary = job.out, job.output_prefix, job.resid, job.summary
    verbose = job.verbose
    chk = noise.checks
    _say(f"  noise: scatter/weights kappa = {noise.kappa:.3f}; imaginary "
         f"{chk.get('imaginary_rms_over_noise', float('nan')):.3f}"
         + (f", line-free {chk['line_free_rms_over_noise']:.3f}"
            if "line_free_rms_over_noise" in chk else "")
         + " x the noise (expected 1)", verbose)
    notes = []
    for name, key in (("imaginary part", "imaginary_rms_over_noise"),
                      ("real part in the line-free channels", "line_free_rms_over_noise")):
        value, n_cells = chk.get(key), chk.get(key.replace("rms_over_noise", "n_cells"), 0)
        # only an excess matters: below 1 the errors are conservative
        if value is not None and n_cells and value - 1 > max(0.1, 4 / math.sqrt(2 * n_cells)):
            _warn(f"measure: the {name} scatters {value:.2f} x the noise of the "
                  f"visibilities: {'centring or calibration' if 'imag' in key else 'residual continuum or baseline'} "
                  f"effects, the thermal errors are optimistic there")
    if resid is not None and resid["detected"]:
        if contsub:
            what = "residual continuum after the subtraction"
            advice = ("Widen the line windows (wings in the continuum fit), use "
                      "continuum_order=1 (a slope), or add a line you did not list")
        elif c_fitted is not None:
            what = "continuum left by the one fitted with the lines"
            advice = ("A slope or a line not listed: subtract the continuum "
                      "(stack(..., continuum='subtract', continuum_order=1)), or add the line")
        else:
            what = "continuum in the data (not subtracted)"
            advice = ("Subtract it (the default) or fit it with the lines "
                      "(stack(..., continuum='fit')): as it is, the line fluxes "
                      "include it")
        _warn(f"measure: {what} outside the line windows: offset "
              f"{resid['offset']:.3g} +- {resid['offset_error']:.2g} "
              f"({resid['offset_nsigma']:+.1f} sigma, {resid['offset_over_line_peak']:+.0%} of "
              f"the line peak), slope {resid['slope_per_1000kms']:.3g} +- "
              f"{resid['slope_error']:.2g} per 1000 km/s ({resid['slope_nsigma']:+.1f} sigma).  "
              + advice)
        notes.append("continuum outside the windows: " + what)
    for k, nsig, method, why, blended, cover in flux_choice_notes:
        name = "the line" if n_lines == 1 else f"line {k + 1}"
        quoted = (f"; quoted: {method} (flux_measure='{cfg.flux}')" if cfg.flux != "auto" else "")
        if blended:
            _warn(f"measure: {name}: its window overlaps line(s) "
                  f"{', '.join(str(j + 1) for j in blended)}, so its area flux holds part of "
                  f"them ({nsig:+.1f} sigma from the Gaussian flux); the Gaussian flux of the "
                  f"joint fit separates them" + (quoted or ", and is quoted"))
        elif not cover >= 0.95:
            _warn(f"measure: {name}: the area flux differs from the Gaussian by {nsig:+.1f} "
                  f"sigma, but the window holds only {100 * cover:.0f}% of the fitted line: "
                  f"widen the window" + (quoted or " (the Gaussian flux is quoted)"))
        else:
            _warn(f"measure: {name} is not Gaussian (double peak, wings?): the area and the "
                  f"Gaussian flux differ by {nsig:+.1f} sigma" + (quoted or "; the area in the window is quoted"))
        notes.append(f"{name}: area and Gaussian flux differ by {nsig:+.1f} sigma")
    sig1 = detection.get("significance_sigma", 0)
    if not detection["fixed_shape"] and sig1 < 4:
        _warn(f"measure: weak line ({sig1:.1f} sigma in the search): the flux of a fit with "
              f"free centre and width is unreliable at this S/N (it follows noise peaks, "
              f"and its error is underestimated).  Fix the width from another tracer "
              f"(fwhm_range=(w, w)), narrow the ranges, or quote the area flux")
        notes.append("weak line: free-shape flux unreliable")
    if notes:
        summary["warnings"] = notes
    if output_prefix:
        written = out.save(output_prefix)
        _say("  written: " + ", ".join(written), verbose)
    for k, blk in enumerate(blocks):
        name = "Line" if n_lines == 1 else f"Line {k + 1} ({line.freqs_ghz[k]:.6g} GHz)"
        lk = line_block if k == 0 else blk
        gs = G[group_of[k]]
        _say(f"{name}: F = {lk['flux']:.4g} +- {lk['flux_error']:.3g} "
             f"({'area' if lk['flux_method'] == 'area' else 'Gaussian'}; Gaussian "
             f"{lk['flux_gauss']:.4g}, area {lk['flux_area']:.4g})  peak = {lk['peak']:.4g}  "
             f"v0 = {lk['v0_kms']:+.1f} km/s  FWHM = {lk['fwhm_kms']:.0f} km/s"
             + (f"  size {gs.result.size_model}"
                + (f" {gs.theta:.2f}\"" if gs.theta > 0 else "")
                + f"  chi2r = {gs.result.chi2_reduced:.2f}" if k == gs.members[0] else ""),
             verbose)


def measure_line(statistics: Union[str, SufficientStatistics],
                 input_list: Union[str, Sequence],
                 line: Lines,
                 weighting: WeightingConfig = WeightingConfig(),
                 config: LineFitConfig = LineFitConfig(),
                 bootstrap: BootstrapConfig = BootstrapConfig(),
                 match_tol_arcsec: float = 2.0,
                 output_prefix: Optional[str] = None,
                 verbose: bool = True) -> LineResult:
    """Measure the stacked lines: profile, fluxes, significance, noise.

    The measurement is a sequence of steps, each a function of this module
    (``_stack_and_cells``, ``_search``, ``_fit_groups``, ``_bootstrap``,
    ``_profiles``, ``_fluxes``, ``_residual_continuum``, ``_result``,
    ``_report``): read them in that order to follow what is done.

    Parameters
    ----------
    statistics, input_list, weighting, match_tol_arcsec
        As in :func:`vista.extract.measure_continuum`.
    line
        The lines (:class:`~vista.extract.config.Lines`): rest frequencies
        and windows.  Lines whose windows overlap (directly or through
        another line) are fitted together, sharing the size; every other
        line is measured on its own, on the channels near it and outside the
        windows of the others.  The first is the line of the stack.
    config
        See :class:`vista.extract.config.LineFitConfig`.
    bootstrap
        Resampling of the sources, for the population errors.
    output_prefix
        Write ``<prefix>_results.json``, ``_profile.txt``, ``_uvamp.txt`` and
        ``_noise.txt``.
    """
    # what the steps share: the inputs here, then what every step works out
    job = SimpleNamespace(statistics=statistics, input_list=input_list, line=line,
                          weighting=weighting, cfg=config, bootstrap=bootstrap,
                          match_tol_arcsec=match_tol_arcsec, output_prefix=output_prefix,
                          verbose=verbose)
    # the steps, in order; each reads what the earlier ones left in ``job``
    _stack_and_cells(job)       # the stack, its noise, the cells, the groups of lines
    _search(job)                # every line searched: best S/N and its significance
    _fit_groups(job)            # the fit: point and Gaussian source, and the choice
    _bootstrap(job)             # the fits again on resampled stacks: population errors
    _profiles(job)              # the spectrum at zero spacing, with its noise and model
    _fluxes(job)                # the two fluxes of every line, and which one is quoted
    _residual_continuum(job)    # what is left of the continuum outside the windows
    _result(job)                # the summary and the result
    _report(job)                # warnings, files, log
    return job.out


__all__ = ["measure_line", "LineResult"]
