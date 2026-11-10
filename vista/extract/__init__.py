"""
vista.extract
=============
Post-processing and signal extraction of a ViSta stack.

The stacked dataset is a standard interferometric product, but imaging it
inherits the ill-defined hybrid beam of a heterogeneous sample, and fitting
the visibilities with a general tool requires holding them in memory.  This
subpackage recovers the stacked emission directly in the uv domain, from a
compact set of sufficient statistics, and never loads the full visibility
set.

It is driven by :class:`vista.ViSta`::

    from vista import ViSta

    v = ViSta("input_list.txt", output_name="co32")
    v.stack(line_freq=345.796, line_window=500)            # the continuum is subtracted by default
    v.extract(weighting_scheme="natural", fwhm_range=(100, 600))
    result = v.run()

The steps it runs are here:

* :mod:`vista.extract.contsub` -- the continuum subtraction (also done while
  the stack is written, :mod:`vista.extract.inline`);
* :mod:`vista.extract.statistics` -- the compression of the stack into
  sufficient statistics, channel by channel and annulus by annulus;
* :mod:`vista.extract.measure` -- the lines fitted on all channels and
  annuli at once, with the noise measured from the data
  (:mod:`vista.extract.noise`, :mod:`vista.extract.linefit`);
* :mod:`vista.extract.stack` -- the continuum, measured on its own
  statistics (flux per annulus and uv fit, :mod:`vista.extract.uvfit`).

The compression reads the stack, a Zarr store or a Measurement Set, and
runs its per-row work in the compiled kernel ``ms_ops`` (OpenMP) when it is
built, otherwise in NumPy.  The measurement works on the statistics and
only needs ``numpy`` and ``scipy``, so it can run on a laptop and be
repeated cheaply (``ViSta.run(only="measure")``).
"""

from .measure import LineResult, measure_line
from .plots_line import plot_line_result
from .sources import Entry, read_input_list
from .stack import ExtractionResult, extract_flux
from .statistics import compress_visibilities
from .contsub import subtract_continuum

__all__ = ["LineResult", "ExtractionResult", "plot_line_result", "Entry", "read_input_list",
           "measure_line", "extract_flux", "compress_visibilities", "subtract_continuum"]
