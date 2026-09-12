"""
vista.extract
=============
Post-processing and signal extraction for a ViSta stack (Chapter 5).

The stacked Measurement Set produced by :class:`vista.ViSta` is a standard
interferometric product, but imaging it inherits the ill-defined hybrid beam
of a heterogeneous sample, and fitting the visibilities with a general tool
requires holding them in memory.  This subpackage recovers the stacked flux
directly in the uv domain, from a compact set of sufficient statistics, and
never loads the full visibility set.

Typical workflow
----------------
::

    from vista.extract import (LineConfig, WeightingConfig, ContinuumConfig,
                               subtract_continuum, compress_visibilities,
                               extract_flux)

    line = LineConfig(rest_freq_ghz=345.7959899,   # the line you stack, GHz
                      v_window_kms=(-600, 600))

    # 1. continuum subtraction on the stack (writes MODEL_DATA and
    #    CORRECTED_DATA, leaves DATA untouched).  Optional: skip it and set
    #    ProfileConfig(joint_continuum=True) below to fit line and continuum
    #    together instead.
    subtract_continuum("stacked.ms", line, ContinuumConfig(order=0))

    # 2. compression into sufficient statistics, read-only
    compress_visibilities("stacked.ms", "input_list.txt", line,
                          out_line="line_stats.npy",
                          out_continuum="cont_stats.npy")

    # 3. extraction, with the weighting scheme chosen here
    result = extract_flux("line_stats.npy", "input_list.txt", line,
                          weighting=WeightingConfig(scheme="democratic"),
                          output_prefix="stack_democratic")

    print(result.summary["flux_total"], result.summary["theta_fwhm_arcsec"])

Steps 1 and 2 read the MS and need ``casatools``; step 3 works on the
compressed statistics and only needs ``numpy`` and ``scipy``, so it can run
on a laptop and be repeated cheaply for different subsamples, weighting
schemes and source models.
"""

from .config import (BinningConfig, BootstrapConfig, ContinuumConfig,
                     LineConfig, ProfileConfig, UVFitConfig, WeightingConfig)
from .contsub import subtract_continuum
from .profiles import (LineShape, fit_line_shape, flux_per_annulus_continuum,
                       flux_per_annulus_template, flux_per_annulus_window,
                       integrated_spectrum, line_free_mask)
from .sources import (Entry, alpha_factor, amplitude_factors,
                      group_by_position, read_input_list)
from .stack import (ExtractionResult, StackInput, build_stack_input, combine,
                    extract_flux, sources_per_bin, sources_per_channel)
from .statistics import SufficientStatistics, compress_visibilities
from .uvfit import (UVFitResult, fit_uv_profile, gaussian_visibility,
                    two_gaussian_visibility)

__all__ = [
    # configuration
    "LineConfig", "BinningConfig", "WeightingConfig", "ProfileConfig",
    "UVFitConfig", "BootstrapConfig", "ContinuumConfig",
    # pipeline steps
    "subtract_continuum", "compress_visibilities", "extract_flux",
    # building blocks
    "SufficientStatistics", "StackInput", "build_stack_input", "combine",
    "sources_per_bin", "sources_per_channel", "ExtractionResult",
    "LineShape", "fit_line_shape", "flux_per_annulus_template",
    "flux_per_annulus_window", "flux_per_annulus_continuum",
    "integrated_spectrum", "line_free_mask",
    "fit_uv_profile", "UVFitResult", "gaussian_visibility",
    "two_gaussian_visibility",
    "Entry", "read_input_list", "group_by_position",
    "amplitude_factors", "alpha_factor",
]
