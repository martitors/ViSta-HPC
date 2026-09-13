<p align="left">
  <img src="logo.png" alt="ViSta logo" width="200">
</p>

# ViSta — Visibility Stacking tool

**ViSta** is an HPC-optimised pipeline for stacking interferometric observations directly in the visibility (Fourier) domain. It combines datasets from sources at different redshifts, observed with different telescopes and array configurations, by rescaling, re-centring, and regridding each input Measurement Set onto a common rest-frame *uv*-plane before stacking them into a single unified observation.

The method is described in:

> Torsello et al. (2025), *The ViSta method for optimized stacking of broadband interferometric data in the Fourier domain*, PASP 137, 124501. [doi:10.1088/1538-3873/ae1faf](https://doi.org/10.1088/1538-3873/ae1faf)

This repository contains the HPC reimplementation (ViSta v2), which replaces the original CASA-based pipeline with a self-contained Python package backed by a compiled C++/OpenMP kernel with optional GPU (CUDA) acceleration.

---

## Features

- **>10× faster** than the original CASA-based pipeline on a single CPU node
- **No intermediate data products** written to disk
- **Multi-threaded** read/compute/write pipeline (producer–consumer model)
- **Optional GPU acceleration** via CUDA (transparent fallback to CPU if no GPU is available)
- Supports any telescope using the **Measurement Set** format (ALMA, VLA, MeerKAT, eMERLIN, NOEMA, ...)
- Each input MS can have a different redshift, array configuration, spectral setup, and phase centre

---

## Requirements

### Python dependencies
```
numpy >= 1.21
dask >= 2022.1
dask-ms >= 0.2.18
xarray >= 0.19
```

### Build dependencies
- CMake >= 3.18
- GCC with OpenMP support
- pybind11 (`pip install pybind11`)
- *(optional)* CUDA Toolkit >= 11.0 for GPU support

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/martitors/ViSta-HPC.git
cd ViSta-HPC
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Build the C++/OpenMP kernel

**CPU only:**
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
make install
cd ..
```

**With GPU (CUDA) support:**
```bash
mkdir build && cd build
cmake .. -DWITH_CUDA=ON
make -j$(nproc)
make install
cd ..
```

The compiled module (`ms_ops.so`) will be installed into the `vista/` directory automatically.

---

## Usage

### Input file format

Create a plain text file with one Measurement Set per line:

```
# <ms_path>  <redshift>  <RA>  <Dec>  [FIELD_ID]  [SPW_ID]
/path/to/obs1.ms  2.310  12:34:56.7  +12:34:56.7
/path/to/obs2.ms  1.987  12:34:56.7  +12:34:56.7  0  1
/path/to/obs3.ms  2.105  12:34:56.7  +12:34:56.7
```

- `FIELD_ID` and `SPW_ID` are optional (default: 0)
- Lines starting with `#` are ignored

An optional extra value may follow, the normalisation factor used by
`vista.extract` to put the sources on a common flux scale (a luminosity, a
continuum flux density, any proxy of the stacked emission). It is ignored by
the stacking and is recognised because it is neither a bare integer nor a
comma-separated list of integers, so it cannot be confused with `FIELD_ID` or
`SPW_IDS`:

```
# <ms_path>  <redshift>  <RA>  <Dec>  [FIELD_ID]  [SPW_IDS]  [NORM]
/path/to/obs1.ms  2.369  00:18:02.46  -31:35:05.2  4   27,29  2.91e13
/path/to/obs2.ms  2.561  00:32:07.60  -30:37:35.2  11  25,27  6.05e12
/path/to/obs3.ms  2.561  00:32:07.60  -30:37:35.2  12  23,25  6.05e12
```

Lines sharing the same coordinates, like the last two above, are different
observations of the same physical source; the extraction recognises them and
combines them before the population average.

### Python API

```python
from vista import ViSta

pipeline = ViSta(
    input_file="input_list.txt",
    chunk_rows=5000,   # baseline rows per processing chunk
    verbose=True,
)

pipeline.run(
    ms_out="stacked_output.ms",
    central_freq=153.253e9,   # rest-frame central frequency [Hz]
    nchan_out=1000,           # number of output channels
)
```

See `examples/run_vista.py` for a complete example.

---

## Configuration

| Parameter | Description | Default |
|---|---|---|
| `chunk_rows` | Baseline rows processed per chunk | 5000 |
| `nchan_out` | Number of output channels | auto (widest coverage) |
| `central_freq` | Rest-frame central frequency of output grid [Hz] | required |
| `OMP_NUM_THREADS` | Number of OpenMP threads (set as env variable) | all cores |

For GPU runs, the batch size (number of MSs dispatched per CUDA kernel launch) is set inside the pipeline and defaults to 20. Larger batches improve overlap between GPU compute and I/O; very large batches (> sample size) reduce pipeline overlap.

---

## Repository structure

```
ViSta-HPC/
├── vista/               # Python package
│   ├── __init__.py
│   ├── pipeline.py      # Main ViSta class
│   └── extract/         # post-processing and uv-domain extraction
│       ├── config.py    # all user-facing options
│       ├── contsub.py   # continuum subtraction on the stack
│       ├── statistics.py# compression into sufficient statistics
│       ├── sources.py   # source table, cosmology, flux normalisation
│       ├── stack.py     # weighting, combination, bootstrap, driver
│       ├── profiles.py  # line shape and per-annulus flux
│       ├── uvfit.py     # circular Gaussian / point-source model
│       ├── plots.py     # diagnostic figure
│       └── cli.py       # python -m vista.extract
├── src/                 # C++ / CUDA kernel
│   ├── ms_ops.cpp       # OpenMP kernel (+ GPU dispatch)
│   └── ms_ops_cuda.cu   # CUDA kernel
├── examples/
│   ├── run_vista.py
│   └── example_input_list.txt
├── docs/
│   └── extraction.md
├── run_vista.sh         # end-to-end driver
├── CMakeLists.txt
├── requirements.txt
└── README.md
```

---

## Post-processing and signal extraction

The output MS is a standard Measurement Set and can be processed with any tool
that supports the format (`tclean`, [WSClean](https://wsclean.readthedocs.io/),
[UVMultiFit](https://github.com/marti-vidal-i/UVMultiFit), `uvcontsub`), but
both routes become awkward on a stack of heterogeneous, wide datasets: imaging
inherits the ill-defined hybrid beam of a sample spanning very different
angular resolutions, and a general visibility fitter has to hold everything in
memory.

The `vista.extract` subpackage recovers the stacked flux directly in the
visibility domain, from a compact set of sufficient statistics, without ever
loading the full visibility set:

```python
from vista.extract import (LineConfig, ContinuumConfig, WeightingConfig,
                           subtract_continuum, compress_visibilities,
                           extract_flux)

line = LineConfig(rest_freq_ghz=345.7959899,   # the line you are stacking, GHz
                  v_window_kms=(-600, 600))

# continuum subtraction: writes MODEL_DATA and CORRECTED_DATA, keeps DATA
subtract_continuum("stacked_output.ms", line, ContinuumConfig(order=0))

# compression into sufficient statistics (read-only, checkpointed)
compress_visibilities("stacked_output.ms", "input_list.txt", line,
                      out_line="line_stats.npy",
                      out_continuum="cont_stats.npy")

# extraction: weighting scheme and source model are chosen here
result = extract_flux("line_stats.npy", "input_list.txt", line,
                      weighting=WeightingConfig(scheme="democratic"),
                      output_prefix="stack_democratic")

print(result.summary["flux_total"], result.summary["theta_fwhm_arcsec"])
```

For every channel the visibilities are binned radially into logarithmically
spaced annuli of rest-frame baseline length, storing the weighted sums of the
real and imaginary parts, the sum of the weights, the weighted sum of the
baseline length and the number of samples. Every subsequent fit runs on that
representation, which is what makes the bootstrap inexpensive. The total flux
and the effective source size then come from a circular Gaussian fitted to the
amplitude-versus-baseline profile, with the line shape measured once on the
spatially integrated spectrum and held fixed while the amplitude is fitted
annulus by annulus.

Both weighting schemes are available and are applied at extraction time, so the
stacked MS always keeps its native amplitudes and the same dataset can be reused
for different subsamples and tests:

- **natural** (the default), the native visibility weights, maximum formal S/N;
- **democratic**, each source renormalised to total weight 1, so that the stack
  represents the population average.

The extraction reads the **same input list** used for the stacking: the line
number is the `DATA_DESC_ID` in the stacked MS, so no second file has to be
kept in sync.

The whole workflow, from the stacking to the fits, is driven by
`run_vista.sh`:

```bash
./run_vista.sh --input input_list.txt --rest-freq 345.7959899
./run_vista.sh --input input_list.txt --rest-freq 345.7959899 \
               --contsub after --weighting democratic --norm-flux
./run_vista.sh --input input_list.txt --rest-freq 345.7959899 \
               --only fit --model point        # refit, nothing else
./run_vista.sh --help
```

By default nothing is subtracted and no continuum term is fitted: the data are
treated as line only. `--contsub before` removes the continuum from the stack
and fits the two separately; `--contsub after` keeps it and fits it together
with the line.

or step by step:

```bash
python -m vista.extract contsub  stacked.ms     --rest-freq 345.7959899
python -m vista.extract compress stacked.ms     --input input_list.txt \
        --rest-freq 345.7959899 --out-line line_stats.npy
python -m vista.extract flux     line_stats.npy --input input_list.txt \
        --rest-freq 345.7959899 --weighting democratic --out stack_democratic
```

`casatools` is needed only by the two steps that read the MS; the extraction
itself runs in a plain numpy/scipy environment. See `docs/extraction.md` for the full parameter reference.

---

## Citation

If you use ViSta in your research, please cite:

```bibtex
@article{Torsello2025,
  author  = {Torsello, M. and Massardi, M. and Liuzzo, E. and
             Gururajan, G. and Perrotta, F. and Lapi, A.},
  title   = {The {ViSta} method for optimized stacking of broadband
             interferometric data in the {Fourier} domain},
  journal = {Publ. Astron. Soc. Pac.},
  volume  = {137},
  pages   = {124501},
  year    = {2025},
  doi     = {10.1088/1538-3873/ae1faf}
}
```

---

## License

This project is licensed under the MIT License.

---

## Changelog — v2.1

- **Thread-safe Casacore access**: global + per-MS locking for safe concurrent reads
- **Lazy materialisation**: vis/flag data kept as lazy Dask arrays, materialised one chunk at a time (peak RAM capped)
- **Adaptive chunk sizing**: `chunk_rows` auto-reduced for wide-bandwidth data (target ~512 MB/chunk)
- **Input channel pre-slicing**: only channels overlapping the output window are read from disk
- **SPW frequency sorting**: merged axis always monotonically increasing in the same MS
- **`velocity_range_kms` parameter**: set output bandwidth as a velocity range instead of channel count
- **Automatic `OBSERVE_TARGET` filtering**: calibrator scans excluded via STATE subtable
- **`CORRECTED_DATA` preference**: uses calibrated column when available, falls back to `DATA`
- **Constant-column bulk write**: `FEED1`/`FEED2`/`PROCESSOR_ID`/... written once via `putcol` instead of per-chunk
- **Full subtable set**: output includes FIELD, DATA_DESCRIPTION, ANTENNA, POLARIZATION, OBSERVATION, FEED, SOURCE
- **Dynamic `REST_FREQUENCY`**: SOURCE table uses `central_freq` instead of hardcoded CO(4-3)
- **Parallel reader pool**: multiple reader workers on SLURM, single reader on laptop
- **English docstrings**: all comments and documentation rewritten in English
