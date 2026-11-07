# Repository structure

```text
ViSta-HPC/
├── vista/                   the Python package
│   ├── pipeline.py          the ViSta class: input list, the run, its threads
│   ├── spectral_grid.py     the common rest-frame channel grid
│   ├── ms_input.py          reading the input MSs (metadata, rows, spectral windows)
│   ├── readers.py           reader processes, shared memory
│   ├── compute.py           calls to the kernel, GPU buffers
│   ├── frames.py            spectral frame of the inputs, Doppler factor per row
│   ├── polarization.py      the correlations of the inputs
│   ├── ms_output.py         writing a Measurement Set (main table, subtables)
│   ├── stackio.py           the stack on disk: Zarr writer, Zarr/MS reader, export_ms
│   ├── common.py            log, lock, coordinates, small helpers
│   └── extract/             the measurement
│       ├── config.py        all the options
│       ├── contsub.py       continuum subtraction on the stack
│       ├── statistics.py    compression into statistics
│       ├── inline.py        subtraction and compression during the stacking (one pass)
│       ├── noise.py         the noise: scatter, weights, imaginary part, channel correlation
│       ├── linefit.py       the joint fit of the channel × annulus cube, significance
│       ├── measure.py       measure_line (--method joint)
│       ├── sources.py       input list, source grouping, cosmology, normalisation
│       ├── stack.py         weighting, combination, bootstrap, extract_flux
│       ├── profiles.py      line shape and flux per annulus
│       ├── uvfit.py         circular Gaussian / point-source model
│       ├── plots.py         figures of extract_flux
│       ├── plots_line.py    figures of measure_line
│       └── cli.py           python -m vista.extract
├── src/                     the kernel (Python module ms_ops)
│   ├── ms_ops.cpp           OpenMP kernel, Python bindings, GPU dispatch
│   ├── ms_ops_cuda.cu       CUDA engine
│   └── ms_ops_cuda.h        CPU/GPU interface, shared formulas
├── examples/
│   ├── ViSta_tutorial.ipynb the worked example (Colab)
│   ├── simulate.py          synthetic ALMA-like Measurement Sets
│   ├── plot_line.py         the figures of measure_line, in your own style
│   ├── run_extraction.py    the whole workflow in a script
│   ├── run_vista.py         the stacking in a script
│   └── example_input_list.txt
├── docs/                    this documentation (Sphinx, Read the Docs)
├── run_vista.sh             the command-line driver
├── CMakeLists.txt
├── requirements.txt
└── setup.py
```
