# Repository structure

```text
ViSta-HPC/
├── vista/                   the Python package
│   ├── api.py               the ViSta class: the options of every step, run()
│   ├── cli.py               the command line (python -m vista, run_vista.sh)
│   ├── pipeline.py          the stacking: input list, the run, its threads
│   ├── resources.py         cores and memory: readers, writers, threads, GPU batch
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
│       ├── config.py        the lines and the options of every step (internal)
│       ├── contsub.py       continuum subtraction on the stack
│       ├── statistics.py    compression into statistics
│       ├── inline.py        subtraction and compression during the stacking (one pass)
│       ├── noise.py         the noise: scatter, weights, imaginary part, channel correlation
│       ├── linefit.py       the joint fit of the channel × annulus cube, significance
│       ├── measure.py       the measurement of the lines
│       ├── sources.py       input list, source grouping, cosmology, normalisation
│       ├── stack.py         weighting, combination, bootstrap, the continuum measurement
│       ├── profiles.py      line shape and flux per annulus
│       ├── uvfit.py         circular Gaussian / point-source model
│       ├── plots.py         figures of the continuum
│       └── plots_line.py    figures of the lines
├── src/                     the kernel (Python module ms_ops)
│   ├── ms_ops.cpp           OpenMP kernel, Python bindings, GPU dispatch
│   ├── ms_ops_cuda.cu       CUDA engine
│   └── ms_ops_cuda.h        CPU/GPU interface, shared formulas
├── examples/
│   ├── ViSta_tutorial.ipynb the worked example (Colab)
│   ├── simulate.py          synthetic ALMA-like Measurement Sets
│   ├── plot_line.py         the figures of the lines, in your own style
│   ├── run_vista.py         a run in a script
│   ├── run_extraction.py    the measurement repeated with different options
│   ├── inspect_ms.py        the bands of the datasets: velocity_range, line_window, annuli
│   ├── inspect_line.py      one dataset imaged: where the line is, its width
│   ├── check_rest.py        the rest frequency rest-framed copies were written with
│   └── example_input_list.txt
├── docs/                    this documentation (Sphinx, Read the Docs)
├── benchmarks/              ViSta 1.0 against this version: time, memory, accuracy
├── run_vista.sh             the command line (python -m vista)
├── CMakeLists.txt
├── requirements.txt
└── setup.py
```
