"""
vista.extract.config
====================
Configuration objects for the visibility-domain signal extraction
(Sec. 5.2.3 and 5.3 of Torsello 2026).

Every knob the user may want to touch lives in one of these dataclasses.
All of them have defaults that are safe for a generic run; only
``LineConfig.rest_freq_ghz`` has no default, because the rest frequency of
the targeted line is the one thing the code cannot guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

C_KMS = 299792.458
C_MS = 299792458.0
ARCSEC_PER_RAD = 206264.806247


# ---------------------------------------------------------------------------
# spectral setup
# ---------------------------------------------------------------------------
@dataclass
class LineConfig:
    """Spectral definition of the targeted line (and of an optional blend).

    Parameters
    ----------
    rest_freq_ghz
        Rest frequency of the line being stacked, in GHz.  Velocities are
        always referred to it.
    v_window_kms
        Velocity window, in km/s, that contains the line.  It is used
        (a) to define the line-free channels for the continuum fit and
        (b) as the integration window when ``ProfileConfig.method='window'``.
        Keep it wide enough to contain the full profile including the wings.
    second_rest_freq_ghz
        Rest frequency of a second line falling in the same band, in GHz.
        When set, the two lines are de-blended analytically instead of being
        integrated over two adjacent windows.
    second_v_window_kms
        Velocity window of the second line, in its own velocity frame.
    exclude_v_kms
        Extra velocity intervals (in the target frame) to exclude from the
        line-free channels, e.g. a known absorption feature.
    """

    rest_freq_ghz: float
    v_window_kms: Tuple[float, float] = (-600.0, 600.0)
    second_rest_freq_ghz: Optional[float] = None
    second_v_window_kms: Tuple[float, float] = (-500.0, 500.0)
    exclude_v_kms: Sequence[Tuple[float, float]] = ()

    @property
    def rest_freq_hz(self) -> float:
        return float(self.rest_freq_ghz) * 1e9

    @property
    def second_rest_freq_hz(self) -> Optional[float]:
        if self.second_rest_freq_ghz is None:
            return None
        return float(self.second_rest_freq_ghz) * 1e9

    def velocity(self, freq_hz: np.ndarray, second: bool = False) -> np.ndarray:
        """Radio-convention velocity of a frequency grid, in km/s."""
        nu0 = self.second_rest_freq_hz if second else self.rest_freq_hz
        if nu0 is None:
            raise ValueError("second_rest_freq_ghz is not set")
        return C_KMS * (nu0 - np.asarray(freq_hz, float)) / nu0


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
        ``'democratic'``  each *physical source* contributes with total
        weight 1 (Eq. 26): repeated observations of the same source are
        first co-added with their native weights, i.e. naturally, and the
        resulting single measurement is then renormalised to unity.  This is
        the choice for a population average.

        ``'natural'``  the native visibility weights are used as they are,
        so the deepest observations dominate.  This maximises the formal S/N
        and is preferable for emission too faint to be seen in the
        individual sources.
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
    already_applied
        Set to ``True`` if the amplitudes on disk already carry alpha, so
        that the factor is not applied twice.
    H0, Om0
        Cosmology used for the luminosity distances in alpha.
    """

    scheme: str = "democratic"          # 'democratic' | 'natural'
    redshift_rescaling: bool = True
    z_ref: Optional[float] = None
    flux_normalisation: bool = False
    norm_ref: Optional[float] = None
    already_applied: bool = False
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
# how the flux per annulus is measured  (Sec. 5.3.2)
# ---------------------------------------------------------------------------
@dataclass
class ProfileConfig:
    """Line profile handling and per-annulus flux measurement.

    Parameters
    ----------
    method
        ``'template'`` (default) fit the line shape once on the spatially
        integrated spectrum, freeze centroid and width, and fit only the
        amplitude in each annulus by weighted linear least squares.  The
        flux then follows analytically as ``F = A * sigma * sqrt(2 pi)``:
        no truncation of the wings, no contamination between blended
        components.

        ``'window'`` plain boxcar integration over ``LineConfig.v_window_kms``.
        Simpler and assumption-free, but it truncates the wings and mixes
        blended lines.

        ``'continuum'`` no line at all: the flux per annulus is the weighted
        average of the line-free channels (use it on the continuum column).
    fit_span_kms
        Half-width, in km/s, of the region around each line used both for
        the shape fit and for the amplitude least squares.
    shape_tie
        Constraints of the two-component shape fit for a blend:
        ``'free'`` (6 parameters), ``'centroid'`` (common centroid),
        ``'centroid+width'`` (common centroid and width).
    fixed_shape_kms
        ``(centroid, sigma)`` in km/s.  When given, the shape fit is skipped
        and this shape is imposed on both components.
    joint_continuum
        Fit the line **on top of** a continuum instead of assuming it was
        already removed: a constant term is added both to the shape fit on
        the integrated spectrum and to the per-annulus least squares.  The
        amplitude of that term is the continuum flux density of the annulus,
        so the continuum gets its own profile and its own uv fit, measured
        jointly with the line and on the same data.

        Switch it on when the stack was **not** continuum-subtracted.  It is
        also worth running on a subtracted stack as a robustness test: if the
        flux or the size move, the fit was absorbing residual continuum.
    continuum_overlap_frac
        In ``'continuum'`` mode, use only the channels covered by at least
        this fraction of the sources (1.0 = all of them).
    """

    method: str = "template"            # 'template' | 'window' | 'continuum'
    fit_span_kms: float = 1500.0
    shape_tie: str = "free"             # 'free' | 'centroid' | 'centroid+width'
    fixed_shape_kms: Optional[Tuple[float, float]] = None
    joint_continuum: bool = False
    continuum_overlap_frac: float = 1.0

    def __post_init__(self):
        if self.method not in ("template", "window", "continuum"):
            raise ValueError("method must be 'template', 'window' or "
                             f"'continuum', got {self.method!r}")
        if self.shape_tie not in ("free", "centroid", "centroid+width"):
            raise ValueError("shape_tie must be 'free', 'centroid' or "
                             f"'centroid+width', got {self.shape_tie!r}")


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
    theta_max_arcsec
        Upper bound on the fitted FWHM.  Lower it to a physically sensible
        value if the fit converges to the bound.
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
    second_line_theta_tie_dex
        Soft tie, in dex, of the size of the second line to the size of the
        target line measured on the same bootstrap realisation.  ``None``
        fits it freely.  The default is permissive: it stabilises poorly
        constrained fits without imposing equality.
    """

    model: str = "gauss"                # 'gauss' | 'point' | 'gauss2'
    theta_fixed_arcsec: Optional[float] = None
    theta_max_arcsec: float = 15.0
    theta_prior: Optional[Tuple[float, float]] = None
    b_max_klambda: Optional[float] = None
    min_sources_per_bin: int = 2
    error_floor_frac: float = 0.1
    second_line_theta_tie_dex: Optional[float] = 0.2
    continuum_theta_tie_dex: Optional[float] = None

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
    exclude_kms
        Half-width, in km/s, of the symmetric velocity window excluded from
        the fit.  Ignored when both ``exclude_kms_lo`` and
        ``exclude_kms_hi`` are given.
    exclude_kms_lo, exclude_kms_hi
        Explicit asymmetric exclusion window, e.g. to widen the mask where a
        neighbouring line falls inside the fitted range.
    data_column, model_column, line_column
        Column read, column where the fitted continuum is written, and
        column where the continuum-subtracted visibilities are written.  The
        input column is never modified, so the step can be rerun.
    chunk_rows
        Rows read per chunk: bounds the memory footprint on a large stack.
    """

    order: int = 0                      # 0 | 1 | -1 (BIC)
    delta_bic: float = 2.0
    require_positive_slope: bool = True
    exclude_kms: float = 600.0
    exclude_kms_lo: Optional[float] = None
    exclude_kms_hi: Optional[float] = None
    data_column: str = "DATA"
    model_column: str = "MODEL_DATA"
    line_column: str = "CORRECTED_DATA"
    chunk_rows: int = 50_000

    def __post_init__(self):
        if self.order not in (-1, 0, 1):
            raise ValueError(f"order must be 0, 1 or -1, got {self.order}")


__all__ = ["LineConfig", "BinningConfig", "WeightingConfig", "ProfileConfig",
           "UVFitConfig", "BootstrapConfig", "ContinuumConfig",
           "C_KMS", "C_MS", "ARCSEC_PER_RAD"]
