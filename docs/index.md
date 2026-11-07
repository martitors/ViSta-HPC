# ViSta — Visibility Stacking

[![Documentation](https://readthedocs.org/projects/vista-hpc/badge/?version=latest)](https://vista-hpc.readthedocs.io/en/latest/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martitors/ViSta-HPC/blob/extract_opt/examples/ViSta_tutorial.ipynb)
[![PASP](https://img.shields.io/badge/PASP-137%2C%20124501-blue)](https://doi.org/10.1088/1538-3873/ae1faf)

**ViSta** stacks interferometric observations of many faint sources **in the
visibility (uv) plane** and measures the stacked emission: line profile,
flux, width, size, continuum and significance. It combines datasets of
sources at different redshifts, observed with different telescopes, array
configurations and spectral setups, by re-centring each one on its source,
bringing it to the rest frame and regridding it onto a common grid, and then
measures the stack directly on the visibilities, without imaging.

```{image} _static/stacked_line.png
:alt: A stacked CO(3-2) line measured by ViSta on a simulated sample, with the true signal
:width: 75%
:align: center
```

*A CO(3–2) line stacked from twelve simulated ALMA Band 3 observations at
z = 2–3, none of which detects it alone (from the [Examples](examples/index)).*

The method is described in Torsello et al. (2025), *The ViSta method for
optimized stacking of broadband interferometric data in the Fourier domain*,
PASP 137, 124501 ([doi:10.1088/1538-3873/ae1faf](https://doi.org/10.1088/1538-3873/ae1faf)).

```{list-table}
:widths: 20 80

* - **Language**
  - Python ≥ 3.8, with a C++/OpenMP kernel (optional CUDA)
* - **Input**
  - Measurement Sets of any telescope (ALMA, VLA, NOEMA, MeerKAT, ...)
* - **Output**
  - a stack (Zarr store or Measurement Set), compressed statistics, the measured line and continuum, figures
* - **Repository**
  - [github.com/martitors/ViSta-HPC](https://github.com/martitors/ViSta-HPC)
* - **License**
  - MIT
```

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
  array, channel width, bandwidth, phase centre and spectral frame.
- **Repeatable**: the measurement runs on a few MB of statistics, in
  seconds, as often as needed (weighting, normalisation, source model).

## In short

```bash
git clone https://github.com/martitors/ViSta-HPC.git && cd ViSta-HPC
pip install -r requirements.txt pybind11
cmake -S . -B build && cmake --build build -j && cmake --install build
./run_vista.sh --input input_list.txt --rest-freq 345.79599 --frame LSRK
```

or from Python:

```python
from vista import ViSta
from vista.extract import LineConfig, ContinuumConfig, measure_line

line = LineConfig(rest_freq_ghz=345.79599, v_window_kms=(-500, 500))
ViSta("input_list.txt").run("stack.zarr", central_freq=line.rest_freq_hz,
                            velocity_range_kms=900, frame="LSRK",
                            line=line, continuum=ContinuumConfig(order=0),
                            stats_line="line_stats.npy", stats_continuum="cont_stats.npy")
result = measure_line("line_stats.npy", "input_list.txt", line, output_prefix="stack")
print(result.summary["line"]["flux"], result.summary["detection"]["significance_sigma"])
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
examples/01_the_simulated_sample
examples/02_one_galaxy_without_and_with
examples/03_stacking
examples/04_measuring_the_stacked_line
examples/05_searching_for_the_line
examples/06_what_each_option_changes
examples/07_the_continuum
examples/08_redoing_the_continuum
examples/09_the_stack_in_the_image
examples/10_from_the_command_line_and_on
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
