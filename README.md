<p align="center">
  <img src="logo.png" alt="ViSta logo" width="400">
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
│   └── pipeline.py      # Main ViSta class
├── src/                 # C++ / CUDA kernel
│   ├── ms_ops.cpp       # OpenMP kernel (+ GPU dispatch)
│   └── ms_ops_cuda.cu   # CUDA kernel
├── examples/
│   ├── run_vista.py
│   └── example_input_list.txt
├── CMakeLists.txt
├── requirements.txt
└── README.md
```

---

## Post-processing

After stacking, the output MS can be processed with any tool that supports the Measurement Set format:

- **Imaging**: `tclean` (CASA) or [WSClean](https://wsclean.readthedocs.io/)
- **Visibility-plane fitting**: [UVMultiFit](https://github.com/marti-vidal-i/UVMultiFit)
- **Continuum subtraction**: `uvcontsub` (CASA)

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
