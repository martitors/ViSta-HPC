"""
Companions: other sources in the field of a target, removed from the stack.

A stack is centred on its targets.  A second source in the field of one of
them (a companion) is not at the phase centre: its visibilities oscillate
with the baseline, and what is left of them after the average over the
baselines is added to the target.

What is done, for every dataset that has companions (Sec. 6.2.4-6.2.5 of
the thesis):

1. the target and its companions are fitted together, as point sources at
   their known positions, by weighted least squares on the visibilities of
   that dataset.  The model of a source ``k`` at ``(l, m, n)`` from the
   target is ``S_k(nu) exp(2 pi i (u l + v m + w (n - 1)) nu / c)``; the
   target is ``S_0(nu)``.
2. **Continuum**: one flux per source, on the channels outside every line
   window (of the target, and of the companions with a redshift).  It is
   subtracted from all the channels when it is detected (``MIN_SNR``):
   subtracting a flux compatible with zero would only add noise.
3. **Line**: only for the companions with a redshift.  In the window of the
   line of the companion (the windows of the target, at the velocity of the
   companion) one flux per channel is fitted and subtracted.  A companion
   without a redshift has no known line: it enters with its continuum only.
4. **Too close to tell apart**: a companion closer to the target than the
   resolution of the dataset cannot be separated from it.  This is measured
   by the variance inflation factor: by how much the variance of the flux
   of the companion grows because the target is fitted with it (1: not at
   all; 1 / (1 - rho^2) for two sources whose models correlate by rho).
   Above ``MAX_VARIANCE_INFLATION`` the companion is left out of the fit
   and nothing of it is subtracted: it is reported, not removed badly.

Only the companions are subtracted; the target is always fitted and never
touched.  After the subtraction a fit of the target alone gives what the
joint fit gave.

The file of the companions has one line per companion::

    <ms_path>  <RA>  <Dec>  [redshift]

with the path as in the input list.  A dataset that is in the input list
more than once (several targets in one field) takes the companion for each
of them.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .extract.sources import parse_dec, parse_ra, read_input_list, separation_arcsec

C_LIGHT = 299792458.0
C_KMS = 299792.458
MIN_SNR = 3.0                      # continuum of a companion: subtracted above this
MAX_VARIANCE_INFLATION = 2.0       # companion left out of the fit above this
_CELLS = 2_000_000                 # rows x channels handled at once


@dataclass
class Companion:
    """One line of the file of the companions."""

    ms: str
    ra: float                      # radians
    dec: float                     # radians
    z: Optional[float] = None


def read_companions(path: str) -> List[Companion]:
    """Read the file of the companions: ``<ms_path> <RA> <Dec> [redshift]``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file of the companions {path!r} not found (cwd {os.getcwd()})")
    out = []
    with open(path) as fh:
        for n, raw in enumerate(fh, 1):
            text = raw.split("#", 1)[0].strip()
            if not text:
                continue
            t = text.split()
            if len(t) not in (3, 4):
                raise ValueError(f"{path}, line {n}: '<ms_path> <RA> <Dec> [redshift]', "
                                 f"got {text!r}")
            try:
                z = float(t[3]) if len(t) == 4 else None
                out.append(Companion(t[0], parse_ra(t[1]), parse_dec(t[2]), z))
            except ValueError as e:
                raise ValueError(f"{path}, line {n}: {e}") from None
            if z is not None and z < 0:
                raise ValueError(f"{path}, line {n}: redshift {z} < 0")
    return out


def match_companions(companions, entries):
    """``{dd: [Companion, ...]}``: the companions of every entry of the
    input list, by the path of its dataset."""
    by_path = {}
    for e in entries:
        if e is not None:
            by_path.setdefault(os.path.realpath(e.ms), []).append(e)
    out = {}
    for c in companions:
        targets = by_path.get(os.path.realpath(c.ms))
        if not targets:
            raise ValueError(f"companion in {c.ms!r}: this dataset is not in the input list")
        for e in targets:
            if separation_arcsec(e.ra, e.dec, c.ra, c.dec) < 0.05:
                if len(targets) == 1:
                    raise ValueError(f"companion in {c.ms!r}: its position is that of the "
                                     f"target")
                continue                       # one of several targets of this field: itself
            out.setdefault(e.dd, []).append(c)
    return out


def direction(target_ra, target_dec, ra, dec):
    """``(l, m, n)`` of a position seen from a phase centre: the same
    geometry as the phase shift of the stacking."""
    sr, cr, sd, cd = (math.sin(target_ra), math.cos(target_ra),
                      math.sin(target_dec), math.cos(target_dec))
    s = np.array([math.cos(ra) * math.cos(dec), math.sin(ra) * math.cos(dec), math.sin(dec)])
    return (float(np.dot([-sr, cr, 0.0], s)), float(np.dot([-cr * sd, -sr * sd, cd], s)),
            float(np.dot([cr * cd, sr * cd, sd], s)))


def phasors(uvw, freq, lmn):
    """Model of a unit point source at every ``lmn``: complex
    ``[source, row, channel]``; the first is the target (1 everywhere)."""
    out = np.ones((len(lmn) + 1, len(uvw), len(freq)), np.complex128)
    for k, (l, m, n) in enumerate(lmn, 1):
        delay = uvw[:, 0] * l + uvw[:, 1] * m + uvw[:, 2] * (n - 1.0)
        out[k] = np.exp(2j * math.pi / C_LIGHT * delay[:, None] * freq[None, :])
    return out


def line_channels(freq, lines, z_target, z):
    """Channels of the target's rest-frame grid inside the line windows of a
    companion at redshift ``z``: the windows of the target, moved to the
    velocity of the companion."""
    mask = np.zeros(len(freq), bool)
    if lines is None or z is None:
        return mask
    shift = (1.0 + z_target) / (1.0 + z)        # rest frame of the companion -> of the target
    for k in range(lines.n):
        lo, hi = lines.window_freq_hz(k)
        mask |= (freq >= min(lo, hi) * shift) & (freq <= max(lo, hi) * shift)
    return mask


def _blocks(pieces, nchan):
    """The pieces of a dataset cut in blocks of at most _CELLS cells."""
    step = max(1, _CELLS // max(1, nchan))
    for row0, n in pieces:
        for r in range(row0, row0 + n, step):
            yield r, min(step, row0 + n - r)


def normal_equations(stack, dd, freq, lmn):
    """Per channel: ``A[j] S = b[j]`` of the weighted least squares of the
    fluxes ``S`` of the target and its companions."""
    k = len(lmn) + 1
    a = np.zeros((len(freq), k, k))
    b = np.zeros((len(freq), k))
    for row0, n in _blocks(stack.pieces(dd), len(freq)):
        d = stack.read(dd, row0, n, ["DATA", "FLAG", "FLAG_ROW", "UVW", "WEIGHT"])
        good = ~d["FLAG"] & ~d["FLAG_ROW"][:, None, None] & np.isfinite(d["DATA"])
        w = np.where(good, d["WEIGHT"][:, None, :], 0.0).astype(np.float64)
        vw = (np.where(good, d["DATA"], 0).astype(np.complex128) * w).sum(axis=2)
        e = phasors(np.asarray(d["UVW"], float), freq, lmn)
        a += np.einsum("anj,nj,bnj->jab", e.conj(), w.sum(axis=2), e, optimize=True).real
        b += np.einsum("anj,nj->ja", e.conj(), vw, optimize=True).real
    return a, b


def _solve(a, b):
    """Fluxes and their covariance for one set of normal equations."""
    cov = np.linalg.inv(a + 1e-10 * np.mean(np.diag(a)) * np.eye(len(a)))
    return cov @ b, cov


def fit_companions(a, b, free, line_masks, v_step_kms):
    """The fluxes to subtract, from the normal equations of one dataset.

    a, b        : per channel, see :func:`normal_equations`
    free        : channels outside every line window
    line_masks  : per companion, the channels of its line (all False: none)
    v_step_kms  : channel width, for the line fluxes reported

    Returns ``(amplitude, report)``: ``amplitude[k, channel]`` is the flux of
    companion ``k`` to subtract (zero: nothing), ``report`` one dict per
    companion.
    """
    n_comp, nchan = a.shape[1] - 1, a.shape[0]
    valid = a[:, 0, 0] > 0                       # channels with data
    free = free & valid
    amplitude = np.zeros((n_comp, nchan))
    report = [{"variance_inflation": None, "correlation_with_target": None,
               "continuum": None, "line": None} for _ in range(n_comp)]

    # how well every companion is told apart from the rest, on these baselines
    a_all = a[free if free.any() else valid].sum(axis=0)
    cov_all = _solve(a_all, np.zeros(n_comp + 1))[1]
    diag = np.diag(a_all)
    keep = [0]
    for k in range(1, n_comp + 1):
        vif = float(diag[k] * cov_all[k, k])
        report[k - 1]["variance_inflation"] = vif
        report[k - 1]["correlation_with_target"] = float(a_all[0, k]
                                                         / math.sqrt(diag[0] * diag[k]))
        if vif <= MAX_VARIANCE_INFLATION:
            keep.append(k)
    keep = np.array(keep)
    if len(keep) == 1:
        return amplitude, report
    a_k, b_k = a[:, keep][:, :, keep], b[:, keep]

    # one flux per channel, for the lines and for the scatter of the continuum
    per_chan = np.full((nchan, len(keep)), np.nan)
    per_var = np.full((nchan, len(keep)), np.nan)
    for j in np.flatnonzero(valid):
        s, cov = _solve(a_k[j], b_k[j])
        per_chan[j], per_var[j] = s, np.diag(cov)

    # continuum: one flux per source on the line-free channels
    cont = np.zeros(len(keep))
    noise_scale = np.ones(len(keep))             # real variance / variance from the weights
    if free.sum() >= 2:
        cont, cov = _solve(a_k[free].sum(axis=0), b_k[free].sum(axis=0))
        for i, k in enumerate(keep[1:], 1):
            # the error from the scatter of the channels, which knows the
            # real noise (the weights may not) and its correlation between
            # neighbouring channels
            r = (per_chan[free, i] - cont[i]) / np.sqrt(per_var[free, i])
            r1 = float(np.sum(r[1:] * r[:-1]) / np.sum(r * r))
            noise_scale[i] = np.sum(r * r) / (len(r) - 1) * max(1.0, 1.0 + 2.0 * r1)
            error = math.sqrt(cov[i, i] * noise_scale[i])
            snr = float(cont[i] / error)
            subtracted = snr >= MIN_SNR
            report[k - 1]["continuum"] = {"value": float(cont[i]), "error": error,
                                          "snr": snr, "subtracted": bool(subtracted)}
            if subtracted:
                amplitude[k - 1, :] = cont[i]

    # lines: one flux per channel, inside the window of the companion
    for i, k in enumerate(keep[1:], 1):
        window = line_masks[k - 1] & valid
        if not window.any():
            continue
        line = per_chan[window, i] - cont[i]
        amplitude[k - 1, window] = per_chan[window, i]
        report[k - 1]["line"] = {
            "channels": int(window.sum()), "subtracted": True,
            "flux": float(line.sum() * v_step_kms),
            "flux_error": float(math.sqrt(per_var[window, i].sum() * noise_scale[i])
                                * v_step_kms)}
    return amplitude, report


def subtract_model(stack, dd, freq, lmn, amplitude):
    """``DATA -= sum_k amplitude[k] * model_k``, in place."""
    used = np.flatnonzero(np.any(amplitude != 0, axis=1))
    if not len(used):
        return
    for row0, n in _blocks(stack.pieces(dd), len(freq)):
        d = stack.read(dd, row0, n, ["DATA", "UVW"])
        e = phasors(np.asarray(d["UVW"], float), freq, [lmn[k] for k in used])[1:]
        model = np.einsum("kj,knj->nj", amplitude[used], e)
        stack.write(dd, row0, "DATA",
                    (d["DATA"].astype(np.complex128) - model[:, :, None]).astype(d["DATA"].dtype))


def describe(c, unit="Jy"):
    """One companion of the report, in words."""
    f, u = (1e3, "mJy") if unit.lower() == "jy" else (1.0, unit)
    head = (f"{os.path.basename(c['dataset'].rstrip('/'))}, {c['separation_arcsec']:.1f}\" "
            f"from the target" + (f", z = {c['z']:.4f}" if c["z"] is not None else ""))
    if c["status"] == "too close":
        return (f"{head}: not subtracted, too close to the target for these baselines "
                f"(variance inflation {c['variance_inflation']:.1f} > "
                f"{MAX_VARIANCE_INFLATION:g})")
    parts = []
    k = c["continuum"]
    if k is not None:
        parts.append(f"continuum {k['value'] * f:.3g} ± {k['error'] * f:.2g} {u} "
                     + ("subtracted" if k["subtracted"] else
                        f"not subtracted (S/N {k['snr']:.1f} < {MIN_SNR:g})"))
    ln = c["line"]
    if ln is not None:
        parts.append(f"line {ln['flux']:.3g} ± {ln['flux_error']:.2g} {unit} km/s subtracted "
                     f"({ln['channels']} channels)")
    elif c["z"] is not None and c.get("lines", True):
        parts.append("its line is outside the band")
    return f"{head}: " + "; ".join(parts or ["nothing to subtract"])


def subtract_companions(stack_path: str, input_list: str, companions_file: str,
                        lines=None, verbose: bool = True) -> List[dict]:
    """Remove the companions from a stack, in place (column ``DATA``).

    Parameters
    ----------
    stack_path
        The stack: a Zarr store or a Measurement Set, before the continuum
        subtraction.
    input_list
        The input list the stack was made from.
    companions_file
        ``<ms_path> <RA> <Dec> [redshift]``, one companion per line.
    lines
        The lines of the stack (:class:`~vista.extract.config.Lines`); None:
        the stack of the continuum only.

    Returns
    -------
    list of dict
        One per companion and target: where it is, what was fitted, what was
        subtracted.  The same is stored in the stack (record ``companions``).
    """
    from . import _say
    from .stackio import open_stack

    entries = read_input_list(input_list)
    by_dd = match_companions(read_companions(companions_file), entries)
    stack = open_stack(stack_path, writable=True)
    report = []
    try:
        if stack.get_record("companions"):
            raise RuntimeError(f"the companions were already subtracted from {stack_path}: "
                               f"stack again, run() or run(only='stack')")
        dd_freq, present = stack.dd_frequencies(), set(stack.dd_ids())
        _say(f"Companions: {sum(map(len, by_dd.values()))} in {len(by_dd)} datasets", verbose)
        for dd in sorted(by_dd):
            if dd not in present:
                continue
            entry, comps = entries[dd], by_dd[dd]
            freq = np.asarray(dd_freq[dd], float)
            lmn = [direction(entry.ra, entry.dec, c.ra, c.dec) for c in comps]
            masks = [line_channels(freq, lines, entry.z, c.z) for c in comps]
            free = np.ones(len(freq), bool) if lines is None else lines.free_mask(freq)
            for m in masks:
                free &= ~m
            a, b = normal_equations(stack, dd, freq, lmn)
            v_step = (C_KMS * abs(freq[1] - freq[0]) / np.mean(freq)) if len(freq) > 1 else 0.0
            amplitude, rep = fit_companions(a, b, free, masks, v_step)
            subtract_model(stack, dd, freq, lmn, amplitude)
            for c, r, amp in zip(comps, rep, amplitude):
                too_close = r["variance_inflation"] > MAX_VARIANCE_INFLATION
                r.update(dataset=entry.ms, dd=int(dd), ra_deg=math.degrees(c.ra),
                         dec_deg=math.degrees(c.dec), z=c.z, lines=lines is not None,
                         separation_arcsec=float(separation_arcsec(entry.ra, entry.dec,
                                                                   c.ra, c.dec)),
                         status=("too close" if too_close else
                                 "subtracted" if np.any(amp != 0) else "nothing subtracted"))
                r["text"] = describe(r)
                _say("  " + r["text"], verbose)
                report.append(r)
        stack.set_record("companions", {"file": os.path.abspath(companions_file),
                                        "min_snr": MIN_SNR,
                                        "max_variance_inflation": MAX_VARIANCE_INFLATION,
                                        "companions": report})
    finally:
        stack.close()
    return report


__all__ = ["subtract_companions", "read_companions", "Companion", "MIN_SNR",
           "MAX_VARIANCE_INFLATION"]
