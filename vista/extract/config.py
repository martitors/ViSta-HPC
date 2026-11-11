"""
vista.extract.config
====================
Configuration objects for the visibility-domain signal extraction
(Sec. 5.2.3 and 5.3 of Torsello 2026).

Every knob the user may want to touch lives in one of these dataclasses.
All of them have defaults that are safe for a generic run; only
``Lines`` has no default rest frequency, because the rest frequency of
the targeted line is the one thing the code cannot guess.

These objects are internal: the user sets the same options through
:class:`vista.ViSta` (``ViSta(...)``, ``.stack()``, ``.extract()``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

C_KMS = 299792.458
C_MS = 299792458.0
ARCSEC_PER_RAD = 206264.806247


# ---------------------------------------------------------------------------
# the lines
# ---------------------------------------------------------------------------
def _is_number(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def _pair(w, what):
    """A window: a half-width ``w`` (``-w..+w``) or a pair ``(lo, hi)``, km/s."""
    if _is_number(w):
        w = abs(float(w))
        if w == 0:
            raise ValueError(f"{what}: a window of zero width")
        return (-w, w)
    lo, hi = (float(x) for x in w)
    if not lo < hi:
        raise ValueError(f"{what}: the window ({lo:g}, {hi:g}) km/s is empty (lo must be < hi)")
    return (lo, hi)


_FREQ_UNITS = {"hz": 1e-9, "khz": 1e-6, "mhz": 1e-3, "ghz": 1.0, "thz": 1e3}


def parse_freq_ghz(value) -> float:
    """A rest frequency in GHz: a number (GHz) or a string with its unit,
    ``"345.796GHz"``, ``"345796 MHz"``, ``"3.45796e11 Hz"`` (Hz, kHz, MHz,
    GHz, THz; the case does not matter)."""
    if isinstance(value, str):
        import re
        m = re.fullmatch(r"\s*([-+0-9.eE]+)\s*([a-zA-Z]*)\s*", value)
        unit = (m.group(2) if m else "").lower() or "ghz"
        if not m or unit not in _FREQ_UNITS:
            raise ValueError(f"line_freq: a number (GHz) or a string with its unit "
                             f"('345.796GHz', '345796 MHz'), not {value!r}")
        try:
            number = float(m.group(1))
        except ValueError:
            raise ValueError(f"line_freq: not a number: {value!r}") from None
        return number * _FREQ_UNITS[unit]
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"line_freq: a number (GHz) or a string with its unit, "
                         f"not {value!r}") from None


def per_line(value, n, what):
    """One window per line from a number, a pair or a list of them.

    With one line a pair is ``(lo, hi)``.  With ``n`` lines a single number
    or pair applies to all of them, and a list must have one entry (a number
    or a pair) per line."""
    if value is None:
        return [None] * n
    if _is_number(value):
        return [_pair(value, what)] * n
    items = list(value)
    flat = all(_is_number(x) for x in items)
    if n == 1:
        if flat and len(items) == 2:
            return [_pair(items, what)]
        if len(items) == 1:
            return [_pair(items[0], what)]
        raise ValueError(f"{what}: one line, so a half-width or a pair (lo, hi), "
                         f"not {value!r}")
    if flat and len(items) == 2 and n != 2:
        return [_pair(items, what)] * n
    if len(items) != n:
        raise ValueError(f"{what}: {n} lines, so one window per line ({n} entries), "
                         f"not {len(items)}")
    return [_pair(x, f"{what}, line {k + 1}") for k, x in enumerate(items)]


class Lines:
    """The lines of a stack: rest frequencies and velocity windows.

    The first line is the one the stack is centred on: the output grid is
    centred on its rest frequency and velocities refer to it.  Every line
    has its own window (in its own velocity, radio convention): the channels
    inside any window are kept out of the continuum fit, and the window is
    also where the flux of that line is summed.

    Parameters
    ----------
    freq_ghz
        Rest frequency of the line, or a list of them: a number in GHz, or a
        string with its unit (``"345.796GHz"``, ``"345796 MHz"``).
    window_kms
        Half-width (``500``: -500..+500 km/s) or ``(lo, hi)``; with several
        lines one entry per line, or one for all.
    exclude_between
        Also keep out of the continuum fit the channels between the lines
        (everything from the lowest to the highest window).
    """

    def __init__(self, freq_ghz, window_kms=600.0, exclude_between=False):
        items = (list(freq_ghz) if isinstance(freq_ghz, (list, tuple, np.ndarray))
                 else [freq_ghz])
        freqs = [parse_freq_ghz(f) for f in items]
        if not freqs:
            raise ValueError("line_freq: at least one rest frequency")
        for f in freqs:
            if not (np.isfinite(f) and f > 0):
                raise ValueError(f"line_freq: rest frequencies in GHz, > 0, not {f!r}")
        self.freqs_ghz = freqs
        self.windows_kms = per_line(window_kms, len(freqs), "line_window")
        self.exclude_between = bool(exclude_between)

    def __repr__(self):
        parts = [f"{f:.6g} GHz [{w[0]:+g}, {w[1]:+g}] km/s"
                 for f, w in zip(self.freqs_ghz, self.windows_kms)]
        return ("Lines(" + "; ".join(parts)
                + (", excluding the channels between them" if self.exclude_between else "")
                + ")")

    @property
    def n(self) -> int:
        return len(self.freqs_ghz)

    # the first line: the one of the stack
    @property
    def rest_freq_ghz(self) -> float:
        return self.freqs_ghz[0]

    @property
    def rest_freq_hz(self) -> float:
        return self.freqs_ghz[0] * 1e9

    @property
    def v_window_kms(self) -> Tuple[float, float]:
        return self.windows_kms[0]

    def freq_hz(self, k: int = 0) -> float:
        return self.freqs_ghz[k] * 1e9

    def velocity(self, freq_hz: np.ndarray, second: bool = False, k: int = 0) -> np.ndarray:
        """Radio-convention velocity of a frequency grid for line ``k``, km/s."""
        if second:
            k = 1
        if k >= self.n:
            raise ValueError(f"there is no line {k + 1}")
        nu0 = self.freq_hz(k)
        return C_KMS * (nu0 - np.asarray(freq_hz, float)) / nu0

    def window_freq_hz(self, k: int, window=None) -> Tuple[float, float]:
        """Frequency interval of the window of line ``k``."""
        lo, hi = self.windows_kms[k] if window is None else window
        nu0 = self.freq_hz(k)
        return nu0 * (1.0 - hi / C_KMS), nu0 * (1.0 - lo / C_KMS)

    def window_mask(self, freq_hz, k: int = 0, window=None) -> np.ndarray:
        """Channels inside the window of line ``k`` (or inside ``window``)."""
        lo, hi = self.windows_kms[k] if window is None else window
        v = self.velocity(freq_hz, k=k)
        return (v >= lo) & (v <= hi)

    def free_mask(self, freq_hz, first_window=None) -> np.ndarray:
        """Channels outside every window (and, with ``exclude_between``,
        outside the span from the lowest to the highest window): the
        channels of the continuum.  ``first_window`` replaces the window of
        the first line."""
        f = np.asarray(freq_hz, float)
        inside = np.zeros(f.shape, bool)
        for k in range(self.n):
            inside |= self.window_mask(f, k, first_window if k == 0 else None)
        if self.exclude_between and self.n > 1:
            edges = [e for k in range(self.n)
                     for e in self.window_freq_hz(k, first_window if k == 0 else None)]
            inside |= (f >= min(edges)) & (f <= max(edges))
        return ~inside

    def overlaps(self, k: int) -> List[int]:
        """The other lines whose window overlaps the window of line ``k``
        (windows that only touch at an edge do not)."""
        lo, hi = self.window_freq_hz(k)
        out = []
        for j in range(self.n):
            if j != k:
                a, b = self.window_freq_hz(j)
                if a < hi and b > lo:
                    out.append(j)
        return out

    def groups(self) -> List[List[int]]:
        """The lines measured together: those whose windows overlap, directly
        or through another line (A with B and B with C: one group).  Every
        group is fitted on its own, sharing nothing with the others; the
        first group holds the first line.  Indices are 0-based, in order."""
        group = list(range(self.n))

        def root(k):
            while group[k] != k:
                group[k] = group[group[k]]
                k = group[k]
            return k

        for k in range(self.n):
            for j in self.overlaps(k):
                a, b = root(k), root(j)
                if a != b:
                    group[max(a, b)] = min(a, b)
        out = {}
        for k in range(self.n):
            out.setdefault(root(k), []).append(k)
        return [out[r] for r in sorted(out)]

    def as_dict(self) -> dict:
        return {"rest_freq_ghz": list(self.freqs_ghz),
                "window_kms": [list(w) for w in self.windows_kms],
                "exclude_between": self.exclude_between}


# ---------------------------------------------------------------------------
# radial binning of the uv plane
# ---------------------------------------------------------------------------
@dataclass
class BinningConfig:
    """Logarithmic annuli in projected rest-frame baseline length.

    The defaults span the full range of ALMA configurations; the number of
    bins is the usual compromise between resolution in ``b`` and the number
    of visibilities (hence the S/N) per annulus.
    """

    b_min_klambda: float = 3.0
    b_max_klambda: float = 3000.0
    n_bins: int = 18

    def edges_lambda(self) -> np.ndarray:
        """Bin edges in units of the observing wavelength."""
        return np.logspace(np.log10(self.b_min_klambda * 1e3),
                           np.log10(self.b_max_klambda * 1e3),
                           int(self.n_bins) + 1)


# ---------------------------------------------------------------------------
# weighting and flux normalisation  (Sec. 3.4.1, 3.4.2, 5.2.3)
# ---------------------------------------------------------------------------
@dataclass
class WeightingConfig:
    """How the individual sources are combined into the stack.

    Parameters
    ----------
    scheme
        ``'natural'`` (default)  the native visibility weights are used as
        they are, so the deepest observations dominate.  This maximises the
        formal S/N and is preferable for emission too faint to be seen in
        the individual sources.

        ``'democratic'``  each *physical source* contributes with total
        weight 1 (Eq. 26): repeated observations of the same source are
        first co-added with their native weights, i.e. naturally, and the
        resulting single measurement is then renormalised to unity.  This is
        the choice for a population average.
    redshift_rescaling
        Apply the factor alpha of Eq. (27), which transports every source to
        ``z_ref``.  Essential over a broad redshift range, irrelevant over a
        narrow one.
    z_ref
        Common reference redshift.  ``None`` means "use the median redshift
        of the sample", which is the usual choice; fix it explicitly when
        comparing different stacks.
    flux_normalisation
        Rescale the amplitudes by ``norm_ref / norm``, where ``norm`` is the
        last column of the input list: a luminosity, a continuum flux
        density, or any proxy that correlates with the stacked emission.
        It removes the degeneracy by which an intrinsically brighter object
        contributes proportionally more flux and a bright minority skews the
        stack.  Omit it when you want the total emission of the sample
        weighted by the true intrinsic luminosities.
    norm_ref
        Reference value of that normalisation.  ``None`` uses the median of
        the sample, which however changes with the selection: fix it
        explicitly when comparing different stacks.
    primary_beam
        Correct every entry for the primary beam attenuation at the source
        position (default ``True``).  The response A is computed during the
        compression, one value per line of the input list, and applied here:
        the amplitudes are divided by A and the weights multiplied by A**2,
        so an attenuated pointing is corrected and also counts less.
    H0, Om0
        Cosmology used for the luminosity distances in alpha.
    """

    scheme: str = "natural"             # 'natural' | 'democratic'
    redshift_rescaling: bool = True
    z_ref: Optional[float] = None
    flux_normalisation: bool = False
    norm_ref: Optional[float] = None
    primary_beam: bool = True
    H0: float = 67.4
    Om0: float = 0.315

    def __post_init__(self):
        if self.scheme not in ("democratic", "natural"):
            raise ValueError("scheme must be 'democratic' or 'natural', "
                             f"got {self.scheme!r}")

    @property
    def renormalise(self) -> bool:
        """True when the per-source weights must be renormalised to 1."""
        return self.scheme == "democratic"


# ---------------------------------------------------------------------------
# the joint fit of the whole cube  (vista.extract.measure)
# ---------------------------------------------------------------------------
#: Improvement of the chi square over a point source needed to call the stack
#: resolved (a 3 sigma preference); the value found is always reported.
RESOLVED_DELTA_CHI2 = 9.0


@dataclass
class LineFitConfig:
    """The lines fitted on all channels and annuli at once.

    Parameters
    ----------
    v0_range_kms
        Allowed centres, km/s, in the velocity of each line: a pair for all
        the lines, or one pair per line.  ``lo == hi`` fixes the centre.
        None: the window of the line shrunk by half the smallest width.
    fwhm_range_kms
        Allowed FWHM, km/s, as above; ``lo == hi`` fixes the width (with the
        centre fixed too, the flux is a matched filter and its S/N is the
        significance).  None: two channels to the width of the window.
        The fit stays exactly inside both ranges.
    size
        ``'auto'`` (default): a Gaussian source only if it improves the
        chi square by at least :data:`RESOLVED_DELTA_CHI2` over a point
        source, otherwise a point source; ``'point'``; ``'gauss'``.  A free
        size at low S/N is biased high and drags the total flux up with it.
    theta_fixed_arcsec
        Freeze the size (FWHM) to this value instead.
    theta_min_arcsec, theta_max_arcsec
        Range of the size of the Gaussian source.  A minimum above 0 leaves
        the point source out (``'auto'`` fits the Gaussian only).
    theta_prior_arcsec
        ``(theta_ref, sigma_dex)``: soft tie of the size to a reference
        value, e.g. the continuum size of the same sources.
    flux
        Which flux is quoted as *the* flux of each line; both are always
        given.  ``'gauss'``: the integral of the fitted Gaussian.
        ``'area'``: the sum of the spectrum over the window of the line (no
        assumption on the shape).  ``'auto'`` (default): the Gaussian,
        unless the two differ by more than ``shape_nsigma`` times the error
        of their difference (the line is not Gaussian: double peak, wings),
        the window contains the fitted line and no other line overlaps it.
    shape_nsigma
        Threshold of that test.
    window_kms
        Window of the area flux of each line (same forms as the line
        windows).  None: the line windows.
    continuum
        Fit a continuum term (constant in frequency, same size as the line)
        together with the lines.
    fit_span_kms
        Channels farther than this from every line (in its own velocity)
        are not fitted; None: every channel of the stack.
    min_sources_per_bin
        Annuli with fewer contributing sources are left out (min 2, or 1 for
        a single source).
    noise
        ``'scatter'`` (default): the variance of each cell from the scatter of
        its visibilities; ``'weights'``: from the weights alone.
    channel_correlation
        Measure the correlation of the noise between adjacent channels and
        use it; ``False`` assumes independent channels.
    n_null
        Realisations of the search on noise, for its significance.
    n_null_phase
        Data-driven null: the search on stacks whose sources are given a
        random phase.  0 to skip.
    residual_nsigma
        Threshold of the residual-continuum check (offset and slope of the
        spectrum outside the windows).
    """

    v0_range_kms: Optional[Sequence] = None
    fwhm_range_kms: Optional[Sequence] = None
    size: str = "auto"
    theta_fixed_arcsec: Optional[float] = None
    theta_min_arcsec: float = 0.0
    theta_max_arcsec: float = 15.0
    theta_prior_arcsec: Optional[Tuple[float, float]] = None
    flux: str = "auto"
    shape_nsigma: float = 2.0
    window_kms: Optional[Sequence] = None
    continuum: bool = False
    fit_span_kms: Optional[float] = 1500.0
    min_sources_per_bin: int = 2
    noise: str = "scatter"
    channel_correlation: bool = True
    n_null: int = 1000
    n_null_phase: int = 200
    residual_nsigma: float = 3.0

    def __post_init__(self):
        if self.size not in ("auto", "point", "gauss"):
            raise ValueError(f"size must be 'auto', 'point' or 'gauss', got {self.size!r}")
        if self.flux not in ("auto", "gauss", "area"):
            raise ValueError(f"flux must be 'auto', 'gauss' or 'area', got {self.flux!r}")
        if self.noise not in ("scatter", "weights"):
            raise ValueError(f"noise must be 'scatter' or 'weights', got {self.noise!r}")
        if not 0 <= self.theta_min_arcsec < self.theta_max_arcsec:
            raise ValueError(f"theta_range: 0 <= min < max, not ({self.theta_min_arcsec:g}, "
                             f"{self.theta_max_arcsec:g}) (to fix the size: model=theta)")
        if self.theta_fixed_arcsec is not None and self.theta_fixed_arcsec < 0:
            raise ValueError("theta_fixed_arcsec must be >= 0")
        if self.theta_min_arcsec > 0 and (self.size == "point"
                                          or self.theta_fixed_arcsec == 0):
            raise ValueError(f"model='point' but theta_range starts at "
                             f"{self.theta_min_arcsec:g}\": a point source is below it")
        if self.fit_span_kms is not None and not self.fit_span_kms > 0:
            raise ValueError(f"fit_span must be > 0 km/s (or None: every channel), "
                             f"not {self.fit_span_kms!r}")
        if self.min_sources_per_bin < 1:
            raise ValueError("min_sources_per_annulus must be >= 1")
        if self.shape_nsigma <= 0:
            raise ValueError("shape_nsigma must be > 0")


def ranges_per_line(value, n, what):
    """Ranges (lo, hi) per line, ``lo <= hi`` (equal: fixed); None entries
    for the defaults."""
    if value is None:
        return [None] * n
    items = list(value)
    if all(_is_number(x) for x in items):
        if len(items) != 2:
            raise ValueError(f"{what}: a pair (lo, hi), or one pair per line")
        items = [items] * n
    if len(items) != n:
        raise ValueError(f"{what}: {n} line{'s' if n > 1 else ''}, so "
                         f"{'one pair per line' if n > 1 else 'one pair'}, not {len(items)}")
    out = []
    for k, x in enumerate(items):
        if x is None:
            out.append(None)
            continue
        lo, hi = (float(v) for v in x)
        if lo > hi:
            raise ValueError(f"{what}" + (f", line {k + 1}" if n > 1 else "")
                             + f": lo ({lo:g}) > hi ({hi:g})")
        out.append((lo, hi))
    return out


# ---------------------------------------------------------------------------
# uv-plane source model  (Sec. 5.3.2, Eq. 33)
# ---------------------------------------------------------------------------
@dataclass
class UVFitConfig:
    """Source model fitted to the amplitude-versus-baseline profile.

    Parameters
    ----------
    model
        ``'gauss'``  circular Gaussian, Eq. (33): returns the total flux
        ``F_tot = V(b=0)`` and the effective size ``theta_FWHM``.

        ``'point'``  unresolved source, ``V(b) = F_tot``.  Use it when the
        stack is unresolved at the available resolution: a free Gaussian
        would overfit a size the data cannot constrain.

        ``'gauss2'``  two circular Gaussians, for a compact plus extended
        decomposition.  Needs many well-populated annuli.
    theta_fixed_arcsec
        Freeze the FWHM at this value and fit the flux only.  Equivalent to
        ``model='point'`` when set to 0.
    theta_min_arcsec, theta_max_arcsec
        Range of the fitted FWHM.  Lower the maximum to a physically
        sensible value if the fit converges to the bound.
    theta_prior
        ``(theta_ref_arcsec, sigma_dex)``: soft tie of the size to a
        reference value, typically the continuum size in the same band.
        It adds ``log10(theta / theta_ref) / sigma_dex`` to the residuals,
        so the size stays free but pays for wandering off.  A
        non-destructive alternative to freezing it.
    b_max_klambda
        Ignore annuli beyond this baseline length in the fit.
    min_sources_per_bin
        Drop the annuli to which fewer than this many sources contribute, so
        that no point of the profile is set by a single object.  Values below
        2 are raised to 2, because a single source gives a null bootstrap
        error.
    error_floor_frac
        Floor on the per-annulus bootstrap errors, as a fraction of their
        median: with few sources the bootstrap can underestimate them and a
        single annulus would then dominate the chi square.
    """

    model: str = "gauss"                # 'gauss' | 'point' | 'gauss2'
    theta_fixed_arcsec: Optional[float] = None
    theta_min_arcsec: float = 0.0
    theta_max_arcsec: float = 15.0
    theta_prior: Optional[Tuple[float, float]] = None
    b_max_klambda: Optional[float] = None
    min_sources_per_bin: int = 2
    error_floor_frac: float = 0.1

    def __post_init__(self):
        if self.model not in ("gauss", "point", "gauss2"):
            raise ValueError("model must be 'gauss', 'point' or 'gauss2', "
                             f"got {self.model!r}")

    def resolved_theta_fixed(self) -> Optional[float]:
        """FWHM to freeze, taking ``model='point'`` into account."""
        if self.model == "point":
            return 0.0
        return self.theta_fixed_arcsec


# ---------------------------------------------------------------------------
# uncertainties  (Sec. 5.3.3)
# ---------------------------------------------------------------------------
@dataclass
class BootstrapConfig:
    """Bootstrap resampling of the source sample.

    Every realisation repeats the whole chain, from the combination of the
    per-source statistics through the shape fit and the uv fit, so that the
    scatter across realisations contains both the measurement noise and the
    variance of the population.

    Parameters
    ----------
    n_realisations
        Number of resamplings.  500 is enough for 1-sigma errors; push it to
        a few thousand if you quote percentiles far in the tails.
    seed
        Seed of the random generator, for reproducibility.
    """

    n_realisations: int = 500
    seed: int = 42


# ---------------------------------------------------------------------------
# continuum subtraction on the stacked MS  (Sec. 5.2.3)
# ---------------------------------------------------------------------------
@dataclass
class ContinuumConfig:
    """Continuum subtraction, spectral window by spectral window.

    Parameters
    ----------
    order
        ``0`` constant (default): a narrow line-free bandwidth cannot
        constrain a slope, and imposing one introduces a spurious gradient.
        ``1`` straight line.
        ``-1`` choose between 0 and 1 for each spectral window by comparing
        the two through their Bayesian information criterion, so that a
        linear continuum is preferred only when the data support it.
    delta_bic
        BIC margin required to prefer order 1 over order 0.
    require_positive_slope
        With ``order=-1``, accept order 1 only if the fitted slope is
        positive, as expected for dust continuum in this regime.

    The channels left out of the fit are those inside the windows of the
    lines (:class:`Lines`), and between them with ``exclude_between``.
    data_column, model_column, line_column
        Column read, column where the fitted continuum is written, and
        column where the continuum-subtracted visibilities are written.  The
        input column is never modified, so the step can be rerun.
    chunk_rows
        Rows read at a time from a Measurement Set, at most: fewer when the
        spectra are wide, so that a chunk of the data column stays within
        about 256 MB.  A Zarr store is read by its own chunks.
    bic_rows
        With ``order=-1``: the order of a spectral window is chosen on the
        mean spectrum of its first ``bic_rows`` rows (high S/N, one small
        read).  Fixed by rows rather than by chunks, so that the separate step
        and the subtraction during the stacking choose the same order.
    """

    order: int = 0                      # 0 | 1 | -1 (BIC)
    delta_bic: float = 2.0
    require_positive_slope: bool = True
    data_column: str = "DATA"
    model_column: str = "MODEL_DATA"
    line_column: str = "CORRECTED_DATA"
    chunk_rows: int = 50_000
    bic_rows: int = 5000

    def __post_init__(self):
        if self.order not in (-1, 0, 1):
            raise ValueError(f"order must be 0, 1 or -1, got {self.order}")


__all__ = ["Lines", "per_line", "ranges_per_line", "BinningConfig", "WeightingConfig",
           "LineFitConfig",
           "UVFitConfig", "BootstrapConfig", "ContinuumConfig",
           "C_KMS", "C_MS", "ARCSEC_PER_RAD"]
