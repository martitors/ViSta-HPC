# ViSta — Visibility Stacking

[![Documentation](https://readthedocs.org/projects/vista-hpc/badge/?version=latest)](https://vista-hpc.readthedocs.io/en/latest/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martitors/ViSta-HPC/blob/extract_opt/examples/ViSta_tutorial.ipynb)
[![PASP](https://img.shields.io/badge/PASP-137%2C%20124501-blue)](https://doi.org/10.1088/1538-3873/ae1faf)
[![SPIE](https://img.shields.io/badge/Proc.%20SPIE-14153%2C%20141531X-blue)](https://doi.org/10.1117/12.3103792)

**ViSta** is a visibility-domain stacking method that combines interferometric observations in the Fourier domain from radio to sub-millimeter wavelengths. It combines datasets of sources at different redshifts, observed with different telescopes, array configurations and spectral setups. By stacking visibilities directly and transforming them into the rest frame, ViSta enhances the signal and suppresses noise through extended uv-coverage, without requiring imaging reconstruction.

```{image} _static/stacked_line.png
:alt: A stacked CO(3-2) line measured by ViSta on a simulated sample, with the true signal
:width: 75%
:align: center
```

*A CO(3–2) line stacked from twelve simulated ALMA Band 3 observations at
z = 2–3, none of which detects it alone (from the [Examples](examples/index)).*

```{list-table}
:widths: 20 80

* - **Language**
  - Python ≥ 3.8, with a C++/OpenMP kernel (optional CUDA)
* - **Input**
  - Measurement Sets of any telescope (ALMA, VLA, NOEMA, MeerKAT, ...)
* - **Output**
  - Stacked dataset (Zarr store or Measurement Set), compressed statistics, the measured line and continuum, figures
* - **Repository**
  - [github.com/martitors/ViSta-HPC](https://github.com/martitors/ViSta-HPC)
* - **License**
  - MIT
```

## From ViSta 1.0 to this version

**ViSta 1.0** ([github.com/martitors/ViSta](https://github.com/martitors/ViSta))
is the first implementation of the method, built on CASA and presented in
Torsello et al. (2025), *The ViSta method for stacking in the Fourier domain
and its application to the dusty star-forming galaxies in the ALMA Science
Archive*, PASP 137, 124501
([doi:10.1088/1538-3873/ae1faf](https://doi.org/10.1088/1538-3873/ae1faf)).
Each dataset is centred on its source, brought to the rest frame by scaling
its frequencies, uv coordinates and antenna positions by (1 + z), rebinned
onto a common spectral grid, and concatenated with the others into a single
Measurement Set, as if it were one more epoch or configuration of the same
target. The stack is then measured in the uv plane with UVMultiFit, or
imaged with `tclean`. On simulations ViSta matches image-plane stacking for
point sources and recovers the flux of faint extended sources more
accurately; on the CO(3–2) emission of dusty star-forming galaxies from the
ALMA Science Archive it outperforms image-plane stacking when the individual
sources are hard to detect. Each CASA step, however, writes a full copy of
the data before the next one starts, and the steps run one after the other.

**This version** keeps the method and replaces the CASA functions with a
self-contained Python package and a compiled C++/OpenMP kernel, with
optional GPU acceleration, as described in Torsello et al. (2026), *The ViSta
method for optimized stacking of broadband interferometric data in the
Fourier domain*, Proc. SPIE 14153, 141531X
([doi:10.1117/12.3103792](https://doi.org/10.1117/12.3103792)).

## What it does

- **One pass over the data**: re-centring, rest-framing, regridding, the
  continuum subtraction and the compression into statistics happen while
  the stack is written; nothing is read twice, no intermediate datasets.
- **Fast**: reader processes, a GIL-free C++/OpenMP kernel (optional GPU)
  and parallel writers work at the same time; 15–30 times faster than the
  CASA implementation on the same data.
- **Measured in the uv plane**: the line is fitted on all channels and
  baselines at once, `F · φ(v) · g(b)`, with the noise measured from the
  data, the correlation between channels in every error, the size of the
  source and the significance of the detection.
- **Heterogeneous samples**: every dataset can have its own redshift,
  array, channel width, bandwidth, phase centre and spectral frame; the
  frequencies are always brought to the frame of the redshifts (LSRK by
  default).


## In short

```bash
git clone https://github.com/martitors/ViSta-HPC.git && cd ViSta-HPC
pip install -r requirements.txt pybind11
cmake -S . -B build && cmake --build build -j && cmake --install build
./run_vista.sh --input input_list.txt --line-freq 345.79599 --line-window 500 --extract
```

or from Python:

```python
from vista import ViSta

v = ViSta("input_list.txt")                        # input, output, how it runs
v.stack(line_freq=345.79599, line_window=500)                # the line, the band, the continuum
v.extract()                                        # measure it
result = v.run()                  # stack, subtract the continuum, compress, measure
print(result)                     # the line (both fluxes), the continuum; files: stacked_*
```

New here? Start from the [Quickstart](getting_started/quickstart), then follow
the [Examples](examples/index), a complete worked example on simulated data
that also runs on Google Colab.

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

getting_started/installation
getting_started/quickstart
getting_started/input_list
```

```{toctree}
:maxdepth: 2
:caption: The stacking, step by step
:hidden:

stacking/index
stacking/centring
stacking/restframing
stacking/spectral_grid
stacking/combination
stacking/continuum
stacking/compression
```

```{toctree}
:maxdepth: 2
:caption: Measuring the stack
:hidden:

extraction/index
extraction/measure_line
extraction/weighting
extraction/outputs
extraction/extract_flux
extraction/validation
```

```{toctree}
:maxdepth: 2
:caption: Command line
:hidden:

command_line
```

```{toctree}
:maxdepth: 1
:caption: Examples
:hidden:

examples/index
examples/00_setup
examples/01_the_simulator_and_one_galaxy
examples/02_a_sample_of_twelve_galaxies
examples/03_the_stack
examples/04_the_measurement
examples/05_refining_the_window
examples/06_the_source_model_the_wrong
examples/07_the_normalisations
examples/08_the_weighting
examples/09_the_errors
examples/10_the_continuum
examples/11_a_harder_case_a_line_that_is
examples/12_a_harder_case_sources_that
examples/13_a_harder_case_two_lines_in
examples/14_the_command_line
examples/15_on_your_own_data
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

reference/parameters
reference/api
reference/performance
reference/structure
reference/changelog
reference/citation
```
