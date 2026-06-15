"""
ViSta: Visibility Stacking tool for broadband interferometric data.

Main class:
    ViSta  -- HPC-optimised visibility-domain stacking pipeline

Example usage::

    from vista import ViSta
    p = ViSta("input_list.txt")
    p.run("stacked_output.ms", central_freq=153.253e9, nchan_out=1000)
"""

from .pipeline import ViSta

__all__ = ["ViSta"]
__version__ = "2.0.0"
