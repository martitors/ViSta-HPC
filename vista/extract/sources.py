"""
vista.extract.sources
=====================
The input list that describes the stack, and the per-source factors that make
the amplitudes physically comparable (Sec. 3.4.2).

Input list
----------
The very same file used for the stacking, with one optional extra column::

    # <ms_path>  <redshift>  <RA>  <Dec>  [FIELD_ID]  [SPW_IDS]  [NORM]
    /path/to/obs1.ms  2.310  12:34:56.7  +12:34:56.7  4  27,29  2.91e13
    /path/to/obs2.ms  2.561  00:32:07.6  -30:37:35.2  11 25,27
    /path/to/obs3.ms  2.561  00:32:07.6  -30:37:35.2  12 23,25

The line number is the ``DATA_DESC_ID`` of that entry in the stacked MS,
because the pipeline creates one output spectral window per input line and
tags every row it writes with the index of that line.  Entries skipped at
runtime keep their slot, with zero rows, so the numbering never shifts.

``NORM`` is optional and is the quantity the amplitudes are divided by when
the flux normalisation is switched on: a luminosity, a continuum flux
density, or any proxy that correlates with the stacked emission.  It is
recognised because it is neither an integer nor a comma-separated list of
integers, so it cannot be confused with ``FIELD_ID`` or ``SPW_IDS``; it can
also be written explicitly as ``norm=2.91e13``.

Physical sources
----------------
Several lines may describe the same object, observed in different epochs,
configurations or spectral windows.  They are recognised by their
coordinates, within ``match_tol_arcsec``, and their accumulators are summed
before the population average: this combines the repeated observations of one
source with their native weights, i.e. naturally, which preserves the optimal
intra-source sensitivity.  The democratic renormalisation is then applied to
the resulting single measurement, as required by Eq. (26).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import ARCSEC_PER_RAD, WeightingConfig


# ---------------------------------------------------------------------------
# coordinates
# ---------------------------------------------------------------------------
def parse_ra(text: str) -> float:
    """Right ascension ``hh:mm:ss.s`` (or ``hh mm ss.s``) to radians."""
    h, m, s = (float(x) for x in text.replace(" ", ":").split(":"))
    return math.radians((h + m / 60.0 + s / 3600.0) * 15.0)


def parse_dec(text: str) -> float:
    """Declination ``+dd:mm:ss.s``, ``dd.mm.ss.s`` or ``dd mm ss.s`` to radians."""
    text = text.strip()
    negative = text.startswith("-")
    body = text.lstrip("+-").replace(" ", ":")
    if ":" not in body:                      # dd.mm.ss.ss
        parts = body.split(".")
        body = f"{parts[0]}:{parts[1]}:" + ".".join(parts[2:])
    d, m, s = (float(x) for x in body.split(":"))
    value = d + m / 60.0 + s / 3600.0
    return math.radians(-value if negative else value)


def separation_arcsec(ra1, dec1, ra2, dec2) -> float:
    """Angular separation between two directions, in arcsec."""
    return math.hypot((ra2 - ra1) * math.cos(0.5 * (dec1 + dec2)),
                      dec2 - dec1) * ARCSEC_PER_RAD


# ---------------------------------------------------------------------------
# input list
# ---------------------------------------------------------------------------
def _is_integer(token: str) -> bool:
    try:
        int(token)
        return True
    except ValueError:
        return False


def _is_integer_list(token: str) -> bool:
    parts = [p for p in token.split(",") if p != ""]
    return bool(parts) and all(_is_integer(p) for p in parts)


def split_trailing_columns(tokens: Sequence[str]):
    """Split the tokens after Dec into ``(field, spws, norm)``.

    ``FIELD_ID`` is a bare integer and ``SPW_IDS`` a comma-separated list of
    integers, so any other numeric token is the normalisation factor.  An
    explicit ``norm=`` is honoured first.
    """
    norm = None
    remaining = []
    for token in tokens:
        if token.lower().startswith("norm="):
            norm = float(token.split("=", 1)[1])
        else:
            remaining.append(token)
    if norm is None and remaining and not _is_integer_list(remaining[-1]):
        try:
            norm = float(remaining[-1])
            remaining = remaining[:-1]
        except ValueError:
            pass
    field = int(remaining[0]) if len(remaining) >= 1 else None
    spws = ([int(s) for s in remaining[1].split(",") if s != ""]
            if len(remaining) >= 2 else None)
    return field, spws, norm


@dataclass
class Entry:
    """One line of the input list, i.e. one data descriptor of the stack."""

    dd: int
    ms: str
    z: float
    ra: float                     # radians
    dec: float                    # radians
    field: Optional[int] = None
    spws: Optional[List[int]] = None
    norm: Optional[float] = None

    @property
    def coordinates_deg(self):
        """(RA, Dec) in degrees, for reporting."""
        return (math.degrees(self.ra), math.degrees(self.dec))


def read_input_list(path: str) -> List[Optional[Entry]]:
    """Read the ViSta input list.  Index in the returned list = data descriptor.

    ``None`` marks an ``# EXCLUDED`` placeholder, for lines removed from the
    stack by hand and whose slot must be preserved.
    """
    entries: List[Optional[Entry]] = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.upper().startswith("# EXCLUDED"):
                entries.append(None)
                continue
            if line.startswith("#"):
                continue
            parts = line.split("#", 1)[0].split()
            dd = len(entries)
            if len(parts) < 4:
                raise ValueError(f"{path}: cannot parse line {dd}: {line!r}")
            field, spws, norm = split_trailing_columns(parts[4:])
            entries.append(Entry(dd=dd, ms=parts[0], z=float(parts[1]),
                                 ra=parse_ra(parts[2]),
                                 dec=parse_dec(parts[3]),
                                 field=field, spws=spws, norm=norm))
    return entries


def group_by_position(entries: Sequence[Entry], tol_arcsec: float = 2.0):
    """Cluster the entries into physical sources by their coordinates.

    Sources are identified by position and nothing else: the input list
    carries no names, and none are invented here.

    Returns ``(groups, centres)``, where ``groups[k]`` holds the entries of
    the k-th source and ``centres[k]`` its (RA, Dec) in degrees.
    """
    groups: List[List[Entry]] = []
    centres: List[Tuple[float, float]] = []
    for entry in entries:
        for k, (ra, dec) in enumerate(centres):
            if separation_arcsec(ra, dec, entry.ra, entry.dec) <= tol_arcsec:
                groups[k].append(entry)
                break
        else:
            centres.append((entry.ra, entry.dec))
            groups.append([entry])
    return groups, [(math.degrees(ra), math.degrees(dec))
                    for ra, dec in centres]


# ---------------------------------------------------------------------------
# cosmology and flux normalisation  (Eq. 27)
# ---------------------------------------------------------------------------
def luminosity_distance_mpc(z: float, H0: float = 67.4,
                            Om0: float = 0.315) -> float:
    """Luminosity distance in Mpc for a flat LambdaCDM cosmology.

    Uses astropy when available and falls back to a direct quadrature.
    """
    try:
        from astropy.cosmology import FlatLambdaCDM
        return float(FlatLambdaCDM(H0=H0, Om0=Om0)
                     .luminosity_distance(z).value)
    except ImportError:
        OL = 1.0 - Om0
        DH = 2.99792458e5 / H0
        za = np.linspace(0.0, z, max(5000, int(z * 2000)) + 1)
        E = np.sqrt(Om0 * (1.0 + za) ** 3 + OL)
        return (1.0 + z) * DH * float(np.trapezoid(1.0 / E, za))


def alpha_factor(z: float, z_ref: float, H0: float = 67.4,
                 Om0: float = 0.315) -> float:
    """Amplitude rescaling that transports a source from ``z`` to ``z_ref``.

    Eq. (27): ``alpha = (D_L(z)/D_L(z_ref))^2 * (1+z_ref)/(1+z)``, where the
    last factor accounts for the bandwidth compression between the frames.
    """
    return ((luminosity_distance_mpc(z, H0, Om0)
             / luminosity_distance_mpc(z_ref, H0, Om0)) ** 2
            * (1.0 + z_ref) / (1.0 + z))


def amplitude_factors(groups: Sequence[Sequence[Entry]],
                      weighting: WeightingConfig, verbose: bool = True):
    """Per-source amplitude factor ``f = alpha * norm_ref / norm``.

    Returns ``(factors, z_ref, norm_ref, missing)``, one factor per group;
    ``missing`` lists the sources dropped for lack of a normalisation value.
    """
    redshifts = np.array([g[0].z for g in groups], dtype=float)
    z_ref = weighting.z_ref
    if weighting.redshift_rescaling and z_ref is None:
        z_ref = float(np.median(redshifts))
        if verbose:
            print(f"[norm] z_ref = {z_ref:.4f} (median of the sample)")

    values = []
    for group in groups:
        available = [e.norm for e in group if e.norm is not None]
        values.append(float(available[0]) if available else None)

    norm_ref = None
    if weighting.flux_normalisation:
        present = [v for v in values if v is not None and v > 0]
        if not present:
            raise RuntimeError(
                "flux normalisation requested but the input list carries no "
                "normalisation column: add it as the last value of each line, "
                "or switch the normalisation off")
        norm_ref = (float(weighting.norm_ref) if weighting.norm_ref is not None
                    else float(np.median(present)))
        if verbose:
            origin = ("given" if weighting.norm_ref is not None
                      else "median of the sample")
            print(f"[norm] norm_ref = {norm_ref:.4e} ({origin})")

    factors, missing = [], []
    for k, group in enumerate(groups):
        factor = 1.0
        if weighting.redshift_rescaling and not weighting.already_applied:
            factor *= alpha_factor(group[0].z, z_ref, weighting.H0,
                                   weighting.Om0)
        if weighting.flux_normalisation:
            if values[k] is None or values[k] <= 0:
                missing.append(k)
                factors.append(float("nan"))
                continue
            factor *= norm_ref / values[k]
        factors.append(factor)
    return np.array(factors, dtype=float), z_ref, norm_ref, missing


__all__ = ["Entry", "read_input_list", "group_by_position", "parse_ra",
           "parse_dec", "separation_arcsec", "alpha_factor",
           "luminosity_distance_mpc", "amplitude_factors",
           "split_trailing_columns"]
