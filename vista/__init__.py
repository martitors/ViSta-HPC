"""
ViSta: Visibility Stacking tool for broadband interferometric data.

Main class:
    ViSta  -- HPC-optimised visibility-domain stacking pipeline
              with optional CUDA GPU acceleration

Subpackage:
    vista.extract -- post-processing (continuum subtraction) and
                     visibility-domain signal extraction of the stack

Example usage::

    from vista import ViSta
    p = ViSta("input_list.txt")
    p.run("stacked_output.ms", central_freq=153.253e9, nchan_out=1000)

New in v2.1:
    - Transparent GPU/CUDA acceleration (batch kernel dispatch)
    - Thread-safe Casacore access (global + per-MS locking)
    - Lazy chunk-at-a-time materialisation with adaptive chunk sizing
    - velocity_range_kms parameter for velocity-based output bandwidth
    - Automatic OBSERVE_TARGET state filtering
    - CORRECTED_DATA preference with DATA fallback
    - Full output subtable set (FIELD, DD, ANTENNA, POL, OBS, FEED, SOURCE)

New in v2.2:
    - vista.extract: continuum subtraction, compression into sufficient
      statistics, and uv-domain flux extraction with bootstrap errors
    - Optional normalisation factor as the last column of the input list
"""

__all__ = ["ViSta"]
__version__ = "2.2.0"


def __getattr__(name):
    """Import the stacking pipeline lazily (PEP 562).

    ``vista.pipeline`` needs the compiled kernel ``ms_ops`` and the dask-ms
    stack, which the extraction subpackage does not: importing it eagerly
    here would make ``import vista.extract`` fail on a machine where the
    kernel was never built.  ``from vista import ViSta`` still works.
    """
    if name == "ViSta":
        from .pipeline import ViSta
        return ViSta
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
