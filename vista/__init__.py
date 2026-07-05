"""
ViSta: Visibility Stacking tool for broadband interferometric data.

Main class:
    ViSta  -- HPC-optimised visibility-domain stacking pipeline

Example usage::

    from vista import ViSta
    p = ViSta("input_list.txt")
    p.run("stacked_output.ms", central_freq=153.253e9, nchan_out=1000)"""
ViSta: Visibility Stacking tool for broadband interferometric data.

Main class:
    ViSta  -- HPC-optimised visibility-domain stacking pipeline
              with optional CUDA GPU acceleration

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
"""

from .pipeline import ViSta

__all__ = ["ViSta"]
__version__ = "2.1.0"
"""

from .pipeline import ViSta

__all__ = ["ViSta"]
__version__ = "2.0.0"
