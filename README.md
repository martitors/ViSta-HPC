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

- **15–30× faster** than the original CASA-based pipeline (ViSta 1.0) on one node, on the same data (see [Performance](#performance))
- **No intermediate data products** written to disk: one pass over the data
- **Parallel pipeline**: reader processes, a GIL-free C++/OpenMP kernel and parallel writers work at the same time
- **Zarr output**: the stack is a compressed Zarr store (layout of dask-ms) written by several threads at once, for any output path not ending in `.ms`; `vista.export_ms()` makes a Measurement Set from it when you need one
- **Fast extraction**: continuum subtraction and compression run in OpenMP kernels with overlapped I/O, on Zarr stores and MSs alike, without `casatools` (see [Performance](#performance))
- **Spectral frame**: `run(frame="LSRK")` removes the motion of the Earth from TOPO inputs (ALMA, VLA) row by row, without resampling the data twice
- **Optional GPU acceleration** via CUDA (`device="gpu"`): many chunks per launch, automatic split to the GPU memory
- **Clear log** for the whole package: elapsed time on every line, what each dataset contributes, warnings for suspicious inputs
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
python-casacore >= 3.5
zarr >= 2.12, < 3     # the stack; dask-ms reads zarr 2 stores
numcodecs >= 0.10
astropy >= 5.0
scipy >= 1.7          # vista.extract
matplotlib >= 3.4     # vista.extract, optional plots
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
cmake .. -DWITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80   # 80 = A100 (default)
make -j$(nproc)
make install
cd ..
```

The compiled module (`ms_ops.so`) will be installed into the `vista/` directory
automatically. Set `CMAKE_CUDA_ARCHITECTURES` to the compute capability of your
GPU (70 V100, 80 A100, 86 RTX 30xx, 89 L40, 90 H100). The CUDA runtime is
linked statically, so the module runs on compute nodes without
`LD_LIBRARY_PATH` settings. A CUDA build also runs on machines without a GPU,
with `device="cpu"`.

---

## Tutorial notebook

`examples/ViSta_tutorial.ipynb` runs everything on a simulated sample of
ALMA Band 3 observations of CO(3–2) at z = 2–3 (`examples/simulate.py`),
written twice, with and without noise, so that every result is compared with
the true signal: one galaxy in the uv plane (mean over the baselines, flux
per annulus) and in a CLEANed CASA cube (peak pixel, aperture, with their
errors), the stacking with the continuum subtraction and the compression in
one pass, the line measurement (`detect` and `search`), what every option
changes, the continuum, a second continuum subtraction, the image of the
stack, the command line; the figures are drawn with `examples/plot_line.py`.
It installs and builds everything itself on
[Google Colab](https://colab.research.google.com/github/martitors/ViSta-HPC/blob/extract_opt/examples/ViSta_tutorial.ipynb)
(*Runtime → Run all*, about 15 minutes on the CPU, CASA's data tables
included; for a private repository put a GitHub token in the Colab secrets
as `GITHUB_TOKEN`), and runs as well in a local Jupyter with `VISTA_REPO` set
to a checkout (CASA: `pip install casatasks casatools`, for the images only).

---

## Usage

### Input file format

Create a plain text file with one Measurement Set per line:

```
# <ms_path>  <redshift>  <RA>  <Dec>  [FIELD_ID]  [SPW_IDS]
/path/to/obs1.ms  2.310  12:34:56.7  +12:34:56.7
/path/to/obs2.ms  1.987  12:34:56.7  +12:34:56.7  0  1
/path/to/obs3.ms  2.105  12:34:56.7  +12:34:56.7  3  0,2
```

- RA and Dec are the position of the source to stack, not the phase centre
- `FIELD_ID` is optional: without it every field of the MS is read, and the
  phase centre is taken from field 0. Give it whenever the MS holds more
  than one field (a mosaic, or calibrators kept in the MS)
- `SPW_IDS` is optional: a comma-separated list of spectral windows; without
  it every SPW of the MS is used. It needs `FIELD_ID` before it
- Text after `#` is a comment

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

### Python API — stacking only

`vista.ViSta` is self-contained: it reads the input list, builds the stack,
and stops there. Nothing in it depends on `vista.extract`,
so if all you want is the stack, this is the whole interface.

```python
from vista import ViSta

pipeline = ViSta(
    input_file="input_list.txt",   # one line per MS / field / spw
    chunk_rows=5000,               # rows read per processing chunk
    verbose=True,                  # or "debug"
)

result = pipeline.run(
    output="stack.zarr",           # Zarr store; a path ending in .ms gives a Measurement Set
    central_freq=345.7959899e9,    # rest frequency of the line [Hz]
    velocity_range_kms=2000.0,     # output bandwidth, ±km/s
    frame="LSRK",                  # frame of the redshifts: removes the Earth's motion
    device="cpu",                  # "gpu" with a CUDA build
)

from vista import export_ms        # only if a Measurement Set is needed
export_ms("stack.zarr", "stack.ms")
```

A script that calls `run()` does not need an `if __name__ == "__main__":`
guard: the reader processes do not re-run it.

The stack holds one spectral window per line of the input list. By default it
is a **Zarr store** with the layout of [dask-ms](https://dask-ms.readthedocs.io)
(`MAIN/MAIN_<k>` with the columns of an MS main table for input line `k`, plus
the subtables): every chunk of every column is a file of its own, so several
threads write it at once, and the columns are compressed (flags and empty
channels take almost no space). `vista.extract` reads it directly; it also
reads back with `daskms.experimental.zarr.xds_from_zarr`. For `tclean` or any
tool that needs a Measurement Set, `export_ms(store, ms)` (or
`python -m vista.extract export-ms stack.zarr stack.ms`) writes one: the same
MS that `output_format="ms"` writes, with the rows of every dataset in input
order, and with `MODEL_DATA`/`CORRECTED_DATA` when the continuum subtraction
added them. `output_format="ms"` writes the MS directly
instead, by one thread.

#### `ViSta(...)` — constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_file` | `str` | required | Path to the input list |
| `chunk_rows` | `int` | `5000` | Rows read and processed per chunk. Bounds the memory footprint; the code lowers it automatically when the input has many channels. Raise it on a machine with plenty of RAM, lower it if the run is killed |
| `verbose` | `bool` or `"debug"` | `True` | `True`: one line per step and per dataset, with the elapsed time, and the warnings. `False`: warnings and errors only. `"debug"`: also metadata, grids per dataset, queues, GPU buffers (see [The log](#the-log)) |

#### `ViSta.run(...)` — the stacking itself

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output` | `str` | required | Output path: a Zarr store (e.g. `stack.zarr`) or a Measurement Set (a path ending in `.ms`, or `output_format="ms"`). Anything already at that path is deleted first. `ms_out` is accepted as an alias |
| `central_freq` | `float` | required | Rest-frame frequency, in **Hz**, on which the output grid is centred. This is the rest frequency of the line being stacked |
| `velocity_range_kms` | `float`, `(lo, hi)`, or `None` | `None` | Output bandwidth as a velocity range. A scalar means ±v. Overrides `nchan_out`. Asking for more than the narrowest dataset covers makes the window slide to that dataset's band edge, so the stack is no longer centred on the line for it |
| `nchan_out` | `int` or `None` | `None` | Number of output channels, set directly. `None` takes the widest coverage any single dataset can provide |
| `channel_width_hz` | `float` or `None` | `None` | Common rest-frame channel width, in Hz. `None` uses the widest rest-framed channel of the sample, which is the finest grid every dataset supports; a finer value is refused with an error naming the dataset that sets the floor |
| `data_column` | `str` | `"auto"` | Column read from each input MS. `"auto"` prefers `CORRECTED_DATA` and falls back to `DATA`. Name the column explicitly when `CORRECTED_DATA` holds something you do not want, as in a simulation whose corrupted copy lives there |
| `scratch_dir` | `str` or `None` | `None` | Build the stack here and move it to `output` at the end. Use a local NVMe or `$TMPDIR` when the destination is on a network filesystem |
| `frame` | `str` or `None` | `None` | The frame of your redshifts: `"LSRK"` or `"BARY"`. The input frequencies are brought to it row by row before the rest-framing, which removes the motion of the Earth from TOPO data (ALMA, VLA). Inputs already in that frame are left as they are. `None` uses the frequencies as they are and warns when the inputs are TOPO. See [Spectral frame of the inputs](#spectral-frame-of-the-inputs) |
| `output_format` | `"zarr"`, `"ms"` or `None` | `None` | `"zarr"`: compressed Zarr store written by `n_writers` threads in parallel. `"ms"`: Measurement Set, written by one thread. `None`: `"ms"` when `output` ends in `.ms`, otherwise `"zarr"` |
| `n_writers` | `int` | `4` | Zarr only: threads writing the stack at the same time |
| `compress` | `bool` | `True` | Zarr only: compress the columns (Blosc lz4) |
| `device` | `"cpu"` or `"gpu"` | `"cpu"` | Where the kernel runs. `"gpu"` needs a CUDA build and a visible GPU; if the GPU fails during the run, the run continues on the CPU with a warning |
| `n_readers` | `int` | `2` | Reader processes, each reading a different MS at the same time. More readers help only on fast filesystems |
| `n_threads` | `int` or `None` | `None` | OpenMP threads of the CPU kernel. `None` keeps `OMP_NUM_THREADS`, else `SLURM_CPUS_PER_TASK`, else all cores |
| `gpu_batch` | `int` | `8` | GPU only: maximum chunks per GPU launch. The compute takes the chunks already read, without waiting for more; larger values use more host memory |
| `timing` | `bool` | `False` | At the end, also log how long setup, reading, computing and writing took, and which stage limited the run |
| `log_file` | `str` or `None` | `None` | Also append the log to this file |
| `reader_mode` | `"process"` or `"thread"` | `"process"` | `"thread"` avoids shared memory (`/dev/shm`), e.g. in containers where it is small, at the cost of reads taking turns |
| `chunk_queue`, `out_queue` | `int` | `n_readers+1`, `max(3, n_writers+1)` | Advanced: chunks waiting for the compute and for the writers. They bound the memory in use |
| `line` | `LineConfig` or `None` | `None` | The line (`vista.extract.LineConfig`), needed by the next two options: its window, with the second line and `exclude_v_kms`, sets the channels left out of the continuum fit |
| `continuum` | `ContinuumConfig` or `None` | `None` | Subtract the continuum **while the stack is written**: the stack gets `MODEL_DATA` and `CORRECTED_DATA` next to `DATA`, bit for bit those of `subtract_continuum` on the finished stack, without reading it back. `DATA` is kept, so the subtraction can be redone later with another window. See [One pass](#one-pass-continuum-subtraction-and-compression-during-the-stacking) |
| `stats_line`, `stats_continuum` | `str` or `None` | `None` | Also **compress** while the stack is written, into these files, as `compress_visibilities` would: the line from `CORRECTED_DATA` (`DATA` without `continuum`), the continuum from `MODEL_DATA` |
| `binning`, `max_amplitude` | | `BinningConfig()`, `None` | Options of that compression |

`run()` returns a dictionary with `output`, `format`, `n_rows`, `n_spw`,
`nchan`, `df_hz`, `flagged_fraction`, `skipped` (indices of the input lines
that gave no data), `bytes_on_disk` and `seconds`; with `continuum`, also
`contsub_orders`, and with `stats_line`, `statistics` (the files written).

**CPU or GPU?** On the Leonardo benchmarks below (MS output, ViSta 2.3) the run
was limited by the single writer of the MS, so the GPU gave the same time as
the CPU. With the parallel Zarr writers the limit moves towards reading and
computing, where the GPU can help: to be measured on Leonardo.

See `examples/run_vista.py` for a complete example.

### How the stacking works

```
setup:  metadata of every MS → common output grid → output subtables

        readers (processes) ──► chunks ──► compute (main thread) ──► results ──► writers (threads)
        one MS each,            bounded     C++ kernel, GIL released  bounded     Zarr: n_writers in
        shared memory           queue       OpenMP or CUDA            queue       parallel; MS: one
```

- **Readers** read each MS chunk by chunk with python-casacore, only the
  rows of the requested field and of the target scans (`OBSERVE_TARGET`), and
  only the channels that fall in the output window. Separate processes are
  needed because python-casacore holds the Python GIL while reading.
- **Compute** rest-frames UVW (×1/(1+z)), rotates them to the stacking
  position, phase-shifts the visibilities in double precision and rebins them
  onto the output grid with channel-overlap weights, as CASA `mstransform`.
  On the GPU, all the chunks already read (up to `gpu_batch`) go in one
  launch, into reused page-locked buffers.
- **Writers** store every processed chunk. Zarr: each chunk goes to its own
  files, at its place in the group of its dataset, so the writers never wait
  for each other and the rows of every dataset stay in input order. MS: one
  thread appends the chunks in the order they are processed (the rows of
  different datasets can interleave; the content does not depend on it).

### Spectral frame of the inputs

ViSta computes the rest frequency of every channel as `f × (1+z)`, which is
right only if the frequencies `f` of the input MS are in the frame in which
`z` is defined. Check the frame of an MS with `listobs` (column "Frame") or
`MEAS_FREQ_REF` in its `SPECTRAL_WINDOW` table (1 LSRK, 3 BARY, 5 TOPO):

| Input | What happens | What to do |
|---|---|---|
| Same frame as `z` | correct | nothing |
| TOPO (ALMA, VLA calibrated data) | the Earth's motion moves the line with the date of each observation, by up to ±30 km/s | `run(frame=...)` with the frame of `z` |
| Another frame (BARY with `z` in LSRK, or the reverse) | constant offset per direction, up to ~20 km/s | `run(frame=...)` |
| NOEMA | already in LSR(K): NOEMA tracks the Doppler shift at the telescope | check that the MS is labelled LSRK; if it says TOPO, fix the label, do not convert |
| REST | already rest frequencies | not accepted |

With `frame` set, for every row the Doppler factor `k = f_frame / f_input` is
computed from the time of the row, the position of the array and the
direction of the source (python-casacore measures), and the kernel processes
the row with the frequencies `f × k` and the redshift `(1+z) × k − 1`. This
is exact: the phase shift keeps using the observed frequency, and the
channels are rebinned once, with no extra resampling (unlike converting the
inputs with `mstransform` first). On a simulated stack of four ALMA
observations three months apart, the line comes back to its true width (60
km/s) and position, while without the correction it is 73 km/s wide and
shifted by 13 km/s.

`k` is computed in the main process, before the readers start, at knots every
60 s around the data, and interpolated linearly for every row: the error is
below 0.2 mm/s, and `k` is not rounded. Computed in a fixed order, it is the
same bit for bit whatever the readers, the chunk size or the device, and the
same in the main, extract and extract_opt branches (casacore measures keeps
caches that change the last bits of `k` with the order of the conversions).

The output is labelled LSRK, so that `tclean` leaves its frequencies as they
are: its default (`specmode='cube'`, `outframe='LSRK'`) and
`specmode='cubedata'` give the same cube. (With a TOPO label, `tclean` would
convert every row with the time of its observation and the (0, 0) direction
of the stack, broadening the line.)

### The log

Every line carries the time since the start of the step. A run on four
simulated ALMA observations (labelled TOPO, three months apart) on a 2-core
machine, with `frame="LSRK"` and `timing=True`:

```
[ViSta       0.0s] ViSta 2.4.0 · 4 MS from input_list.txt → stack.zarr (Zarr) · CPU (2 threads) · 2 readers · 4 writers
[ViSta       2.1s] Frame: inputs 4 TOPO -> LSRK, corrected row by row (without the correction the line moves by -10.3 .. +35.3 km/s from one observation to another)
[ViSta       2.1s] Output grid: 200 channels × 5860.1 kHz (5.08 km/s), ±508 km/s around 345.7960 GHz
[ViSta       2.1s]   (width: the coarsest rest-frame channel of the sample; channels: nchan_out, the MS can fill 119-120 of them)
[ViSta       2.1s]   4 of 4 MS do not cover the whole window around 345.7960 GHz: their window is moved to the edge of their band (shift in the lines below), so their channel frequencies differ from the others
[ViSta       2.2s] Setup done (40 antennas in total); stacking 4 MS...
[ViSta       2.4s] [2/4] obs1_20230530.ms  z=0.5000  2,700 rows  DATA  1 SPW, 120 ch  covers 60% of the grid  offset 1.8"  window +240.3 MHz
[ViSta       2.5s] [1/4] obs0_20230301.ms  z=0.5000  2,700 rows  DATA  1 SPW, 120 ch  covers 60% of the grid  offset 1.8"  window +240.3 MHz
[ViSta       2.8s] [3/4] obs2_20230828.ms  z=0.5000  2,700 rows  DATA  1 SPW, 120 ch  covers 60% of the grid  offset 1.8"  window +240.3 MHz
[ViSta       2.8s] [4/4] obs3_20231126.ms  z=0.5000  2,700 rows  DATA  1 SPW, 120 ch  covers 60% of the grid  offset 1.8"  window +240.3 MHz
[ViSta       3.8s] Timing: setup 2.2 s · stacking 1.1 s · subtables and closing 0.5 s
[ViSta       3.8s]   during the stacking (overlapping): reading 0.5 s per reader · computing 0.5 s · writing 0.1 s per writer  → limited by reading (51% busy)
[ViSta       3.8s] Done in 3.8 s: stack.zarr · 4 SPW × 200 channels · 10,800 rows · 37.3 MB of visibilities, 16.8 MB on disk · 40.0% of the output flagged (no data)
[ViSta       3.8s]   (Zarr store: vista.export_ms() writes it as a Measurement Set)
```

For each dataset: the rows used (and how many the field / target-scan
selection removed), the column read (`DATA` or `CORRECTED_DATA`), the spectral
windows and input channels used, the fraction of the output grid it covers
(the rest is flagged), the distance between its phase centre and the stacking
position, and the shift of its output window when it does not cover the
window centred on the line. Warnings are always shown: stacking position
outside half the primary beam, a missing `data_column`, spectral windows
dropped, datasets skipped and why, a GPU error.

`timing=True` adds, at the end, the time of each stage and the one that
limited the run (with an MS output usually `→ limited by writing`, with the
Zarr store usually reading). `verbose="debug"`
adds the metadata and grid of every dataset, the weight scale, the queues and
the GPU buffers. The extraction steps (`vista.extract`) write to the same log,
in the same format; `vista.set_verbosity(verbose, log_file)` sets the level
and an optional log file for the whole package.

### Performance

Leonardo (CINECA), one node: 32-core Intel Xeon Platinum 8358 and an NVIDIA
A100; input and output on Lustre. 50 datasets, 1000 output channels,
2 readers, 32 OpenMP threads (GPU: 20 chunks per launch); medians of 3 runs.
ViSta 1.0 is the CASA pipeline of the paper (restframing, centering,
rebinning, stacking) on the same data: median of 3 runs (200 channels) and
mean of 2 runs (2048 channels).

| Sample | ViSta 1.0 (CASA) | ViSta-HPC, CPU | ViSta-HPC, GPU |
|---|---|---|---|
| 50 MS × 200 channels, one setup | 308.5 s | 18.5 s (×16.7) | 18.9 s (×16.3) |
| 50 MS × 2048 channels, mixed telescopes, redshifts and channel widths | 833.5 s | 27.4 s (×30) | 28.6 s (×29) |

Adding OpenMP threads, readers or a larger GPU batch does not shorten these
runs: the single writer sets the pace.

The output is the same as the previous version: bit-identical on the CPU
(compared column by column, subtables included, independently of the row
order); on the GPU a few values differ in the last float32 digit, because the
phase is computed per channel instead of by recurrence.

**v2.4, Zarr stack and extraction** — measured on a 2-core test machine (not
Leonardo), with the inputs in the page cache: two simulated ALMA datasets of
541,800 rows × 120 channels, labelled TOPO, stacked with `frame="LSRK"` on 160
channels (30% of the output empty, flagged). v2.3 is the previous version
(MS output, extraction with casatools); "v2.4, MS" is `output_format="ms"`
and the new extraction run on that MS.

| Step | v2.3 (MS, casatools) | v2.4, MS | v2.4, Zarr |
|---|---|---|---|
| stacking, whole run | 13–21 s | 13.5 s | 10–11 s |
| of which writing (per writer) | 9–17 s | 9.6 s | 2.7–4.1 s (4 writers) |
| size of the stack | 2.8 GB | 2.8 GB | 0.86 GB |
| continuum subtraction (order 0) | 224 s | 28 s | 8.6 s |
| compression into statistics | 225 s | 22 s | 4.8 s |
| peak memory of the extraction | 1.9 GB | 1.0 GB | 0.35 GB |
| stack after the subtraction | 8.0 GB | 8.0 GB | 1.5 GB |

`export_ms` of this store takes 14 s. The emptier the grid, the larger the
gain of the store on disk: with 88% of the grid empty (10 datasets × 200
channels on 1000 channels) it was 182 MB against 1.6 GB.

The results are the same: `output_format="ms"` and `export_ms` give the MS
of v2.3 bit for bit (1,083,600 rows, every column and subtable); the
continuum and the subtracted visibilities differ from v2.3 only in the last
float32 digit (closed-form least squares instead of `np.polyfit`), the line
statistics by less than 10⁻⁹ in relative terms, with identical counts. The
continuum statistics change only where they should: the flagged, empty
channels no longer enter the fit of the continuum (see the changelog).

---

## Repository structure

```
ViSta-HPC/
├── vista/               # Python package
│   ├── __init__.py
│   ├── pipeline.py      # Main ViSta class: input list, the run, its threads
│   ├── spectral_grid.py # common rest-frame channel grid
│   ├── ms_input.py      # reading the input MSs (metadata, rows, SPW merging)
│   ├── readers.py       # reader processes, shared memory
│   ├── compute.py       # calls to the kernel, GPU buffers
│   ├── frames.py        # spectral frame of the inputs, Doppler factor per row
│   ├── polarization.py  # the correlations of the inputs (same number in every entry)
│   ├── ms_output.py     # writing an output MS (main table, subtables)
│   ├── stackio.py       # the stack on disk: Zarr writer, Zarr/MS reader, export_ms
│   ├── common.py        # log, lock, coordinates, small helpers
│   └── extract/         # post-processing and uv-domain extraction
│       ├── config.py    # all user-facing options
│       ├── contsub.py   # continuum subtraction on the stack
│       ├── statistics.py# compression into sufficient statistics
│       ├── inline.py    # subtraction and compression during the stacking (one pass)
│       ├── noise.py     # the noise: scatter, weights, imaginary part, channel correlation
│       ├── linefit.py   # the joint fit of the channel x annulus cube, significance
│       ├── measure.py   # measure_line: profile, flux, detection, noise (--method joint)
│       ├── sources.py   # source table, cosmology, flux normalisation
│       ├── stack.py     # weighting, combination, bootstrap, extract_flux (Sec. 5.3)
│       ├── profiles.py  # line shape and per-annulus flux
│       ├── uvfit.py     # circular Gaussian / point-source model
│       ├── plots.py     # figures of extract_flux
│       ├── plots_line.py# figures of measure_line
│       └── cli.py       # python -m vista.extract
├── src/                 # C++ / CUDA kernel (Python module ms_ops)
│   ├── ms_ops.cpp       # OpenMP kernel + Python bindings (+ GPU dispatch)
│   ├── ms_ops_cuda.cu   # CUDA engine
│   └── ms_ops_cuda.h    # CPU/GPU interface and shared formulas
├── examples/
│   ├── ViSta_tutorial.ipynb # worked example on simulations (Colab)
│   ├── simulate.py      # synthetic ALMA-like Measurement Sets
│   ├── plot_line.py     # the figures of measure_line, to restyle
│   ├── run_extraction.py# the whole workflow in a script
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

The stack (exported with `export_ms` when a tool needs a Measurement Set) can
be processed with any tool that supports the format (`tclean`, [WSClean](https://wsclean.readthedocs.io/),
[UVMultiFit](https://github.com/marti-vidal-i/UVMultiFit), `uvcontsub`), but
both routes become awkward on a stack of heterogeneous, wide datasets: imaging
inherits the ill-defined hybrid beam of a sample spanning very different
angular resolutions, and a general visibility fitter has to hold everything in
memory.

The `vista.extract` subpackage recovers the stacked flux directly in the
visibility domain, from a compact set of sufficient statistics, without ever
loading the full visibility set:

```python
from vista import ViSta
from vista.extract import (LineConfig, ContinuumConfig, WeightingConfig,
                           LineFitConfig, measure_line)

line = LineConfig(rest_freq_ghz=345.7959899,   # the line you are stacking, GHz
                  v_window_kms=(-300, 300))    # where you expect it

# stacking, continuum subtraction and compression in one pass
ViSta("input_list.txt").run("stack.zarr", central_freq=345.7959899e9,
                            velocity_range_kms=2000, frame="LSRK",
                            line=line, continuum=ContinuumConfig(order=0),
                            stats_line="line_stats.npy",
                            stats_continuum="cont_stats.npy")

# the line: profile, flux, significance, noise -- on a few MB, repeatable
result = measure_line("line_stats.npy", "input_list.txt", line,
                      weighting=WeightingConfig(scheme="natural"),
                      output_prefix="stack")
print(result.summary["line"]["flux"], result.summary["detection"])
```

The same steps one at a time, on an existing stack:

```python
from vista.extract import subtract_continuum, compress_visibilities, extract_flux

subtract_continuum("stack.zarr", line, ContinuumConfig(order=0))   # MODEL_DATA, CORRECTED_DATA
compress_visibilities("stack.zarr", "input_list.txt", line,
                      out_line="line_stats.npy", out_continuum="cont_stats.npy")
result = extract_flux("line_stats.npy", "input_list.txt", line,      # the chain of Sec. 5.3
                      weighting=WeightingConfig(scheme="democratic"),
                      output_prefix="stack_democratic")
```

For every channel the visibilities are binned radially into logarithmically
spaced annuli of rest-frame baseline length, storing the weighted sums of the
real and imaginary parts, the sum of the weights, the weighted sum of the
baseline length, the number of samples and (since v2.5) the weighted sums of
the squares of the real and imaginary parts, from which the noise is
measured. Every subsequent fit runs on that
representation, which is what makes the bootstrap inexpensive. The total flux
and the effective source size then come from a circular Gaussian fitted to the
amplitude-versus-baseline profile, with the line shape measured once on the
spatially integrated spectrum and held fixed while the amplitude is fitted
annulus by annulus.

Both weighting schemes are available and are applied at extraction time, so the
stack always keeps its native amplitudes and the same dataset can be reused
for different subsamples and tests:

- **natural** (the default), the native visibility weights, maximum formal S/N;
- **democratic**, each source renormalised to total weight 1, so that the stack
  represents the population average.

The extraction reads the **same input list** used for the stacking: the line
number is the `DATA_DESC_ID` in the stack, so no second file has to be kept
in sync.

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
python -m vista.extract contsub  stack.zarr     --rest-freq 345.7959899
python -m vista.extract compress stack.zarr     --input input_list.txt \
        --rest-freq 345.7959899 --out-line line_stats.npy
python -m vista.extract flux     line_stats.npy --input input_list.txt \
        --rest-freq 345.7959899 --weighting democratic --out stack_democratic
python -m vista.extract export-ms stack.zarr    stack.ms     # optional
```

The two steps that read the stack (continuum subtraction and compression)
accept the Zarr store of the stacking and Measurement Sets alike, including
the stacks made with earlier versions (whatever the order of their rows);
`casatools` is no longer needed. The continuum is fitted on the unflagged
samples only, so the empty, flagged channels of a dataset that does not cover
the whole grid do not pull it towards zero. They
read, compute and write at the same time (`n_io` threads on a Zarr store, one
on an MS), and their per-row work runs in the compiled kernel `ms_ops`
(OpenMP) — NumPy is used when the kernel is not built. The extraction itself
runs in a plain numpy/scipy environment. See `docs/extraction.md` for the full parameter reference.

The three steps write to the same log as the stacking (see [The log](#the-log)):
one line per data descriptor, the results, and warnings for what was skipped
or fell back (too few line-free channels, a profile fit that did not converge,
sources without a normalisation value).

### One pass: continuum subtraction and compression during the stacking

The continuum subtraction and the compression work row by row on the output
grid, so `ViSta.run(..., continuum=..., stats_line=...)` does them on every
chunk right after the stacking kernel, before it is written: the stack is
read once instead of three times. The result is the same as with the
separate steps: `DATA`, `MODEL_DATA` and `CORRECTED_DATA` bit for bit (tested
on Zarr and MS, orders 0, 1 and BIC, all the test cases), and the statistics
files bit for bit on a Zarr store (about 1e-15 on an MS, read there in larger
pieces, so the sums are added in another order). With `order=-1` the order of
each entry is chosen on its first `bic_rows` rows (default 5000), in both
paths, so they choose the same.

`run_vista.sh` does this by default whenever it builds the stack and compresses
it in the same run (`--two-pass` for the separate steps).

On the 2-core test machine (1,083,600 rows, 160 channels, inputs in the page
cache): MS stack 51 s → 26 s; Zarr stack 21.5 s → 17 s. With 2 cores the
computing of the three steps competes with the readers and the writers, so on
Zarr most of the time saved by not reading the stack back is spent there; on a
node with more cores the stacking is limited by reading and the extra
computing is hidden.

**Redoing the subtraction.** `DATA` is always kept. After a look at the
profile — a second line in the channels used for the continuum, say — the
subtraction is redone on the stack with another window, and the statistics
recomputed, without stacking again:

```bash
./run_vista.sh --input input_list.txt --rest-freq 345.7959899 \
               --skip-stack --contsub before --exclude-v 400 700   # or --exclude LO HI
```

The parameters of the subtraction are recorded in the stack (Zarr attributes,
MS keyword `VISTA_CONTSUB`) and in the statistics files, and a new subtraction
says which one it replaces, so it is always known which window produced the
`CORRECTED_DATA` on disk.

### Measuring the line: `measure_line` (`--method joint`)

`measure_line` fits the line on the whole stacked cube, channels × annuli,
at once, with the noise measured from the data; it is the default of
`run_vista.sh` (`--method joint`). The chain of Sec. 5.3 (`extract_flux`,
`--method template|window`) is unchanged and still available.

**The model.** For every cell (channel c, annulus b)

    Re V(c, b) = F φ_c(v₀, FWHM) g(b; θ)   [+ second line] [+ continuum]

with φ a Gaussian of unit integral *averaged over the channel* (a line a few
channels wide is not distorted), F the integrated line flux, g the visibility
of a circular Gaussian of FWHM θ (1 for a point source). The cells are fitted
by weighted least squares, each by the inverse of its variance, and the
correlation of the noise between channels enters every error, S/N and null
realisation; F, the peak, v₀, the FWHM and θ come out of one fit, with their
covariance.

**The noise, four ways** (`vista.extract.noise`), compared in the results and
in `<prefix>_noise.png`:

1. from the weights, `1 / Σw` — right only if the weights are absolute;
2. from the scatter of the visibilities inside each cell (the two new
   accumulators): `Σw(x − x̄)² = (N − 1) κ²`, so κ calibrates the weights for
   every entry and every channel (atmospheric lines included, which `WEIGHT`
   ignores). A single visibility has S/N ≪ 1, so the signal does not inflate
   it. This is the noise used;
3. from the imaginary part of the stack (zero for centred, symmetric sources);
4. from the real part in the line-free channels after the subtraction.

3 and 4 must scatter like 2 (ratio 1; a warning otherwise).

**The correlation between channels** has two causes: the spectral response
and the rebinning (Hanning smoothing: +0.67 at lag 1), and the continuum
subtraction, which removes from every row a level fitted on the line-free
channels — a noisy level, shared by all the channels of the row: a
correlation at *all* lags, of order 1 / (effective number of fit channels).
On a narrow band it can double the error of a flux summed over the line. So
the whole channel × channel matrix is measured, from the imaginary part of
every source in every annulus (same spectral response and same subtraction as
the real part, no line); any sum of channels then has its noise known to
√(2 / N_vectors). With fewer than 30 such vectors, a model of the first lags
is used instead, and the results say so (`channel_correlation_method`).

**Two modes.**
- `mode="detect"`: centre and width fixed in advance (`v0_kms`, `fwhm_kms`,
  e.g. from another tracer): the flux is a matched filter — linear, unbiased,
  and its S/N is the significance. The right choice for a weak line.
- `mode="search"` (default): centre and width free in the line window. The
  best S/N over a grid of centres and widths is the detection statistic, and
  its significance comes from the same search run on noise (1000 realisations,
  extrapolated with the number of effective trials), which accounts for the
  freedom of the fit. A free fit always finds a bump: in a search, F/σ_F is
  the precision of the flux, not its significance.

**Errors.** Thermal, from the fit with the measured noise, and bootstrap over
the sources (noise + scatter of the population); the larger is quoted, and
their ratio is in the results (`bootstrap_over_thermal`). With few sources
the bootstrap underestimates, which the thermal error then covers.

**Size.** With `size="auto"` both a point and a Gaussian source are fitted,
and the total flux (and the profile, and the window fluxes) is their average
with Akaike weights: the spread between the two enters the error. A stack
that is weak, or barely resolved, does not tell its size, and its total flux
depends on it: quoting the flux of whichever model was chosen gave errors up
to 5 times too small in the simulations. The Gaussian is called "resolved"
(`size_test.adopted`, the size shown) when it improves χ² by at least
`resolved_delta_chi2` (9, a 3σ preference; the χ² difference is first divided by the factor by which the channel correlation inflates it, `size_test.correlation_factor`, 4-5 with Hanning smoothing and a continuum subtraction). For a weak resolved line the way
to a meaningful total flux is a size from elsewhere: `theta_prior_arcsec =
(θ_ref, σ_dex)` (e.g. the continuum size of the same sources, `--theta-prior`
or `--theta-prior-from`) ties it softly, `theta_fixed_arcsec` fixes it. The
point-source flux, always given, is a lower limit for an extended source. The detection scan tries a point source and Gaussians of 0.5, 1 and
2 times the size that halves the visibility at the median baseline: for a
resolved source a point template collects only part of the signal (in a test
with a 1″ source, S/N 2.0 instead of 3.4). The source model is averaged over
each annulus (the baselines inside it follow a power law with the measured
mean), not taken at the mean baseline, which made resolved sources look
smaller in the tail of the profile. The window flux of a resolved source
carries the uncertainty of the size too, and the bootstrap refits the size. A lower threshold lets noise make a point source look resolved, and
the extrapolated flux of such a fit scatters high.

**Output**: `<prefix>_results.json` (line, detection, window, size test,
noise), `_profile.txt` (flux density at zero spacing per channel, thermal and
bootstrap errors, imaginary part, model), `_uvamp.txt`, `_noise.txt`, and
with `--plot` three figures: `_profile.png` (the profile with the model, the
±1σ band, the imaginary part, the windows and the residuals; `--plot-rebin N`
averages N channels, with the correlation in the errors), `_uv.png` (line
flux per annulus with the point and Gaussian models), `_noise.png` (noise per
channel from the scatter and from the weights, standardised cells against
N(0, 1), null distribution of the search). `_diagnostics.npz` holds what the
figures need besides the tables (channel correlation matrix, standardised
noise samples, null distributions), so `examples/plot_line.py` — numpy and
matplotlib only, a `STYLE` block at the top, one commented function per
figure — redraws all three from the files, in your own style:
`python examples/plot_line.py <prefix> --unit Jy --rebin 2 --format pdf`.

**Tested on simulations** with a known line (synthetic
statistics built cell by cell as the compression builds them, 20 sources,
weights wrong by κ = 1.3, channel correlation 0.3; 50-100 realisations per
case). *Scatter* is the rms of the flux over the true flux, *coverage* the
fraction of realisations whose quoted ±1σ contains the truth (ideal 0.68),
*noise > 2σ* the fraction of pure-noise stacks with F/σ > 2 (ideal 0.023).

| Method | noise > 2σ | weak (S/N ≈ 3): bias, scatter, coverage | medium (S/N ≈ 6): scatter, coverage | resolved (θ = 1″): bias |
|---|---|---|---|---|
| window ±600 (Sec. 5.3) | 0.07 | −6%, 0.62, 0.59 | 0.32, 0.55 | −61% |
| template, point | 0.09 | −4%, 0.48, 0.65 | 0.24, 0.58 | −69% |
| template, gauss (old default) | 0.03 | +128%, 9.0, 0.90 | 0.27, 0.92 | −4% |
| joint, search | p < 0.05 in 2% | −11%, 0.53, 0.56 | 0.23, 0.73 | +4% |
| **joint, detect** (width known) | 0.00 | −6%, **0.31**, 0.70 | **0.15**, 0.72 | +2% |
| joint, window ±600 | 0.02 | −8%, 0.55, 0.65 | 0.29, 0.62 | +6% |

The errors of the chain of Sec. 5.3 (bootstrap only) are too small for a
point model (noise gives F/σ > 2 three to four times too often) and too large
for a free Gaussian, whose flux also runs away at low S/N. The joint fit with
a known width halves the scatter of the template method; a free search is
reliable from about 4σ (below that its flux follows noise peaks, and the
results warn). On whole Measurement Sets (8 simulated ALMA datasets,
Hanning-smoothed noise, weights wrong by a factor 52, continuum subtracted on
18 line-free channels, 16 noise realisations), the measured scatter of the
flux is 71 against 69 quoted (detect) and 108 against 113 quoted (window);
with only the first lags of the channel correlation, the quoted errors were
50 and 70 — the continuum subtraction shares its noisy level between all the
channels of a row.

**Extended and non-ideal sources** (same simulations, the signal of every
cell averaged over the baselines of its annulus; bias of the total flux and,
in brackets, the coverage of the quoted ±1σ):

| Case (S/N) | template, point | template, gauss (old default) | joint, detect | joint, search |
|---|---|---|---|---|
| Gaussian 1″, 40 sources (8) | −69% | −4% (0.66) | +3% (0.68) | +5% (0.68) |
| the same, weak (4) | −64% | +18%, scatter 1.25 (0.77) | +8%, scatter 0.62 (0.83) | +10% (0.65) |
| weak, size prior from elsewhere (±0.15 dex) | | | +4%, scatter 0.33 (0.65) | +1% (0.63) |
| sizes from 0.5″ to 1.5″ (8) | −62% | −7% (0.66) | −7% (0.74) | −5% (0.72) |
| point sources, 0.3″ position errors (8) | −54% | +16% (0.82) | +4% (0.82) | +9% (0.66) |
| rotating disk, velocity gradient over 1″ (8) | −62% | −16% (0.58) | +1% (0.72) | −7% (0.64) |
| exponential disk (8) | −68% | −10% (0.44) | −4% (0.62) | −1% (0.54) |
| double-horn profile, point source (8) | 0% (0.66) | +4% (0.70) | +9% (0.60) | +3% (0.76) |
| Gaussian 1″, bright (25) | −68% | 0% (0.57) | +1% (0.60) | +2% (0.60) |

A point model loses 55-70% of the flux of anything extended — including point
sources stacked with 0.3″ position errors, which add up to an extended stack.
The joint fit recovers the total flux within a few per cent when the source is
resolved at S/N ≳ 8 (also for an exponential disk or a rotating disk, which
the Gaussian model does not describe exactly). For a weak resolved stack the
total flux is uncertain, because the size is: the averaged model gives an
honest, large error, and a size prior halves it. With a fixed Gaussian shape
(`detect`), a double-horn profile comes out 9% high: there the window flux is
the one to quote. On whole Measurement Sets with a 1″ source (8 datasets, 12
noise realisations): F = 211 ± 18 against 213 true, size 0.95″ ± 0.24″, while
the point-source flux is 78.

**uv plane or image plane** (the same ALMA-like stacks imaged with `tclean`,
details in `docs/extraction.md`): for a point source the dirty-image peak and
the uv point-source flux are the same measurement. For a 1″ source (beam
1.06″ × 0.76″) at S/N ≈ 8, measured / true flux is 1.05 ± 0.12 for the joint
uv fit, against 0.42 for the peak, 0.74-1.09 with scatter 0.10-0.67 for
apertures of 1″-3″, 0.95 ± 0.22 for a 2″ aperture with the JvM correction,
1.20 ± 0.25 for `imfit`, 0.39 ± 0.40 for the CLEAN model; as in Lindroos et
al. (2015), the uv fit is the more robust total flux of a resolved stack.

---

## Extraction parameters

Everything the extraction does is controlled by the configuration objects in
`vista.extract.config`. Each has a command-line counterpart in `run_vista.sh`;
`docs/extraction.md` explains when to change what, this is the reference list.

### `LineConfig` — the spectral setup

| Parameter | Default | Description |
|---|---|---|
| `rest_freq_ghz` | required | Rest frequency of the line, in GHz. The only parameter with no default |
| `v_window_kms` | `(-600, 600)` | Velocity window containing the line. Defines the line-free channels for the continuum fit, and the integration window in `window` mode |
| `second_rest_freq_ghz` | `None` | Rest frequency of a second line in the same band; when set, the two are de-blended analytically |
| `second_v_window_kms` | `(-500, 500)` | Velocity window of the second line, in its own frame |
| `exclude_v_kms` | `()` | Extra velocity intervals kept out of the line-free channels |

### `LineFitConfig` — the joint fit (`measure_line`, `--method joint`)

| Parameter | Default | Description |
|---|---|---|
| `mode` | `"search"` | `search`: centre and width free, significance from the same search on noise. `detect`: centre and width fixed (`v0_kms`, `fwhm_kms`), matched filter |
| `v0_kms`, `fwhm_kms` | `0`, `None` | Fixed in `detect` (`fwhm_kms` required), starting values in `search` |
| `v0_range_kms`, `fwhm_range_kms` | `None` | Allowed centres and widths in `search`; default the line window, and 2 channels to the window width |
| `size` | `"auto"` | `auto`, `point` or `gauss` (see above) |
| `theta_fixed_arcsec`, `theta_max_arcsec` | `None`, `15` | Freeze the size; upper bound of the size |
| `theta_prior_arcsec` | `None` | `(θ_ref, σ_dex)`: soft tie of the size, e.g. to the continuum size; the source is then a Gaussian |
| `resolved_delta_chi2` | `9.0` | χ² improvement needed to call the stack resolved (3σ), after the correction for the channel correlation |
| `continuum` | `False` | Fit a continuum term together with the line (stack not subtracted) |
| `shape_tie` | `"free"` | With a second line: `free`, `centroid`, `centroid+width` |
| `fit_span_kms` | `1500` | Channels farther from the line are not fitted |
| `min_sources_per_bin` | `2` | Annuli with fewer sources are left out |
| `noise` | `"scatter"` | `scatter` (from the visibilities, statistics of format 4) or `weights` |
| `channel_correlation` | `True` | Measure and use the correlation between adjacent channels |
| `n_null` | `1000` | Realisations of the search on noise |
| `n_null_phase` | `200` | Data-driven null: stacks with a random phase per source (same noise, the signal averages out); needs a few tens of sources |
| `window_kms` | `None` | Window of the summed flux; default the line window. The flux in 1.19 FWHM around the fitted centre (best S/N for a Gaussian) is given too |

### `BinningConfig` — the radial binning

| Parameter | Default | Description |
|---|---|---|
| `b_min_klambda` | `3` | Inner edge of the annuli, in kλ |
| `b_max_klambda` | `3000` | Outer edge. Should bracket the baselines actually present; the default is meant for heterogeneous ALMA samples |
| `n_bins` | `18` | Number of logarithmic annuli. Fixed at compression time: changing it means recompressing |

### `WeightingConfig` — how the sources are combined

| Parameter | Default | Description |
|---|---|---|
| `scheme` | `"natural"` | `natural` uses the native visibility weights, so the deepest observations dominate and the formal S/N is highest. `democratic` renormalises each source to total weight 1, giving the population average |
| `redshift_rescaling` | `True` | Apply the factor that transports every source to `z_ref` |
| `z_ref` | `None` | Reference redshift. `None` uses the sample median, which moves with the selection: fix it when comparing stacks |
| `flux_normalisation` | `False` | Rescale amplitudes by `norm_ref / NORM`, with `NORM` the last column of the input list |
| `norm_ref` | `None` | Reference value of that normalisation. `None` uses the sample median |
| `already_applied` | `False` | The amplitudes on disk already carry the redshift factor; do not apply it twice |
| `H0`, `Om0` | `67.4`, `0.315` | Cosmology for the luminosity distances |

### `ProfileConfig` — the flux per annulus

| Parameter | Default | Description |
|---|---|---|
| `method` | `"template"` | `template` fits the shape once and the amplitude per annulus, so the flux follows analytically with no truncation. `window` integrates over a fixed velocity window. `continuum` averages the line-free channels |
| `joint_continuum` | `False` | Fit the line on top of a constant continuum and measure both. Use it when the stack was not continuum subtracted |
| `fit_span_kms` | `1500` | Half-width used by the shape fit and the per-annulus least squares. Cannot usefully exceed the band |
| `shape_tie` | `"free"` | For a blend: `free`, `centroid`, or `centroid+width` |
| `fixed_shape_kms` | `None` | Impose `(centroid, sigma)` and skip the shape fit |
| `continuum_overlap_frac` | `1.0` | In continuum mode, use only channels covered by at least this fraction of the sources |

### `UVFitConfig` — the source model

| Parameter | Default | Description |
|---|---|---|
| `model` | `"gauss"` | Circular Gaussian. `point` for an unresolved stack, `gauss2` for a compact plus extended decomposition |
| `theta_fixed_arcsec` | `None` | Freeze the size and fit the flux alone |
| `theta_max_arcsec` | `15` | Upper bound on the fitted size. Check `theta_at_bound` in the results |
| `theta_prior` | `None` | `(theta_ref, sigma_dex)` soft tie, typically the continuum size of the same band |
| `b_max_klambda` | `None` | Ignore annuli beyond this baseline |
| `min_sources_per_bin` | `2` | Drop annuli with fewer contributing sources |
| `error_floor_frac` | `0.1` | Floor on the bootstrap errors, as a fraction of their median. Set it to 0 on noiseless simulated data |
| `second_line_theta_tie_dex` | `0.2` | Soft tie of the second line size to the target size on the same bootstrap realisation |
| `continuum_theta_tie_dex` | `None` | The same for the continuum in joint mode; free by default |

### `BootstrapConfig` — the uncertainties

| Parameter | Default | Description |
|---|---|---|
| `n_realisations` | `500` | Resamplings of the source sample |
| `seed` | `42` | Random seed, for reproducibility |

### `ContinuumConfig` — the subtraction on the stack

| Parameter | Default | Description |
|---|---|---|
| `order` | `0` | Constant. `1` for a line, `-1` to choose per spectral window by BIC |
| `delta_bic` | `2.0` | Margin required to prefer order 1 |
| `require_positive_slope` | `True` | With `order=-1`, accept order 1 only for a rising continuum |
| `exclude_kms` | `600` | Half-width of the excluded window; overridden by the explicit pair below |
| `exclude_kms_lo`, `exclude_kms_hi` | `None` | Asymmetric exclusion window |
| `data_column` | `"DATA"` | Column read |
| `model_column` | `"MODEL_DATA"` | Where the fitted continuum is written |
| `line_column` | `"CORRECTED_DATA"` | Where the subtracted visibilities are written |
| `chunk_rows` | `50000` | Rows read at a time from an MS, at most: fewer for wide spectra, so that a chunk stays within about 256 MB. A Zarr store is read by its own chunks |
| `bic_rows` | `5000` | With `order=-1`, the order of each entry is chosen on the mean spectrum of its first `bic_rows` rows (same choice in the separate step and during the stacking) |

`subtract_continuum(...)` also takes `n_io` (default 4): threads reading and
writing a Zarr store at the same time.

### `compress_visibilities(...)` — further arguments

| Parameter | Default | Description |
|---|---|---|
| `line_column` | `None` | `None` picks `CORRECTED_DATA` when present, otherwise `DATA` |
| `continuum_column` | `"MODEL_DATA"` | Column holding the continuum |
| `max_amplitude` | `None` | Discard visibilities above this amplitude; NaN and infinities are always discarded |
| `chunk_mb` | `512` | Target size of one read chunk, in MB |
| `resume` | `True` | Reuse the data descriptors already present in the output file |
| `n_io` | `4` | Threads reading a Zarr store at the same time (an MS is read by one) |

### `extract_flux(...)` — further arguments

| Parameter | Default | Description |
|---|---|---|
| `match_tol_arcsec` | `2.0` | Angular tolerance for deciding that two entries are the same physical source. Set it to 0 for simulations that share one field |
| `output_prefix` | `None` | Write the results and the text products with this prefix |

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

## Changelog — v2.5

**One pass** (`run(..., line=, continuum=, stats_line=, stats_continuum=)`; the default of `run_vista.sh` when it stacks and compresses in the same run, `--two-pass` for the separate steps): the continuum subtraction and the compression run on every chunk right after the stacking kernel, so the stack is read once instead of three times. Same stack and statistics as the separate steps (columns bit for bit; statistics bit for bit on Zarr, to 1e-15 on an MS). `DATA` is kept: the subtraction can be redone on the stack with another window, and the parameters of every subtraction are recorded in the stack and in the statistics (a new subtraction says which one it replaces). With `order=-1` the order is chosen on the first `bic_rows` rows of each entry, in both paths.

**The noise, measured** (`vista.extract.noise`): the statistics store `Σw Re²` and `Σw Im²` too (format 4; files of format 3 still work, with the noise from the weights), so the variance of every cell comes from the scatter of its visibilities: κ (scatter / weights) for every entry and every channel, checked against the imaginary part and the line-free channels. The correlation of the noise between channels — spectral response, rebinning, and the level removed by the continuum subtraction, shared by all the channels of a row — is measured as a whole matrix from the imaginary part.

**`measure_line`** (`--method joint`, the new default of `run_vista.sh`): the line fitted on all channels and annuli at once — profile at zero spacing, flux, peak, centre, FWHM, size — with the measured noise; thermal and bootstrap errors; matched filter with a fixed shape (`mode="detect"`) or a search with its significance from the same search on noise; window fluxes with the correlation between channels; three figures (profile, uv, noise). For extended sources: point and Gaussian templates in the detection scan, the model averaged over each annulus, the total flux averaged over the point and the Gaussian model (Akaike weights, the spread in the error), an optional size prior (`theta_prior_arcsec`, `--theta-prior`). Tested on point, Gaussian, exponential, rotating, offset and double-horn sources. The chain of Sec. 5.3 (`extract_flux`, `--method template|window`) is unchanged. See [Measuring the line](#measuring-the-line-measure_line---method-joint).

**Tutorial notebook** (`examples/ViSta_tutorial.ipynb`, runs on Colab) and `examples/simulate.py` (synthetic ALMA-like Measurement Sets with a line, a continuum, a size, primary-beam attenuation, noise with wrong weights and Hanning smoothing; each dataset also without noise, `noiseless=True`, same visibilities and weights). The warnings on the imaginary part and the line-free channels are given only when they scatter more than the noise (less means conservative errors).

**Figures in your own style**: `measure_line` also writes `<prefix>_diagnostics.npz` (channel correlation matrix, standardised noise samples, null distributions), and `examples/plot_line.py` redraws the three figures from the output files alone (numpy + matplotlib, `STYLE` block, `--style file.mplstyle`, one commented function per figure). With `n_null=0` and several templates the p-value is a Bonferroni bound (it stopped with an error). The size test divides the χ² difference between the point and the Gaussian source by the factor by which the channel correlation inflates it (`size_test.correlation_factor`): the fits weight the cells by their noise alone, and with Hanning smoothing and a continuum subtraction the raw difference is 4-5 times too large, which made noisy point sources look resolved.

**Kernel**: `annuli_accumulate` gives the same sums at every run (each thread adds a fixed block of rows, the partial sums are added in thread order; they were added in the order the threads finished, which changed the last bits from run to run), and accumulates the two new sums when asked.

**`run_vista.sh`**: one pass by default, `--two-pass`; `--method joint` (default), `--mode`, `--v0`, `--fwhm`, `--size`, `--n-null`, `--plot-rebin`; exit status 0 with `--no-plot` (it was 1).

## Changelog — v2.4

**The stack can be a Zarr store, written in parallel** (`run(output="stack.zarr")`, the default for any path not ending in `.ms`):
- layout of dask-ms (`MAIN/MAIN_<k>` per input line, subtables in their own groups), readable with `daskms.experimental.zarr.xds_from_zarr`
- written by `n_writers` threads in parallel (default 4): every chunk is a file of its own, so the writers never wait for each other; the rows of each dataset stay in input order
- columns compressed with Blosc lz4 (`compress=True`): flags and empty channels take almost no space; constant columns are not written at all
- `vista.export_ms(store, ms)` / `python -m vista.extract export-ms`: the Measurement Set that `output_format="ms"` writes, with the rows in input order, and with `MODEL_DATA` and `CORRECTED_DATA` when present
- a Measurement Set is still written directly when `output` ends in `.ms` (or with `output_format="ms"`), so existing scripts keep producing an MS; `ms_out=` is still accepted
- needs `zarr>=2.12,<3` (dask-ms reads zarr 2 stores)

**Spectral frame of the inputs** (`run(frame="LSRK")`, `run_vista.sh --frame`): the input frequencies are brought to the frame of the redshifts row by row, from the time of each row, the array position and the source direction, inside the kernel call (no second resampling, no copy of the inputs). The Doppler factor is exact to 0.2 mm/s and reproducible bit for bit (computed once, in a fixed order, by the main process). Without it, a warning gives the size of the effect for TOPO inputs. The output is labelled LSRK. See [Spectral frame of the inputs](#spectral-frame-of-the-inputs)

**Precision**: every computation is in float64, and a value is rounded to float32 once, when it is written. WEIGHT × R and SIGMA / √R are computed in float64 (they were computed in float32, or in float64 depending on the NumPy version); so are the NumPy versions of the continuum subtraction and of the compression, used when the kernel is not built (they rounded the model before subtracting it, and the products weight × visibility, to float32). A NaN or infinite visibility is treated as flagged: it made its output channel NaN, still unflagged, which then spoils an image. Entries with different numbers of correlations (e.g. XX,YY and XX,XY,YX,YY) are refused before anything is read, with the list of entries (the writing failed after the stacking); different correlation types (XX,YY and RR,LL) give a warning

**Extraction** (same results, much faster, see [Performance](#performance)):
- continuum subtraction and compression read Zarr stores and Measurement Sets through one interface (`vista.open_stack`), with python-casacore for MSs: `casatools` is no longer needed (also for the primary-beam correction, which reads the original MSs)
- reading, computing and writing overlap (`n_io` threads on a Zarr store)
- new OpenMP kernels in `ms_ops`: `continuum_subtract` (closed-form least squares per row, order 0 or 1, instead of a `np.polyfit` per row and correlation) and `annuli_accumulate` (one pass over the chunk, per-thread accumulators); NumPy fallbacks when the kernel is not built

**Continuum subtraction, fixes**:
- the rows of each data descriptor are found wherever they are: the stacking of v2.3 writes the chunks of different datasets interleaved, and the subtraction read the rows of a data descriptor as one consecutive block, fitting rows of other datasets
- only the unflagged samples enter the fit: the flagged, empty channels of a dataset that does not cover the whole grid were counted as 0 and pulled the continuum towards zero (on a stack covering 60% of the grid, the continuum came out at 0.43 of its value)
- an MS is read in chunks of at most about 256 MB of visibilities, also for very wide spectra

**Code**: the stacking is split into modules (`spectral_grid`, `ms_input`, `readers`, `compute`, `frames`, `ms_output`, `stackio`, `common`); the shared memory of the readers is released also when a run fails.

**`run_vista.sh`**: `--format`, `--writers`, `--io-threads`, `--export-ms`, `--frame`.

## Changelog — v2.3

**Stacking engine** (same results, faster and lighter):
- **Reader processes**: `n_readers` processes read different MSs at the same time with python-casacore; visibilities and flags reach the main process through shared memory, without copies. `reader_mode="thread"` keeps everything in one process
- **GIL-free kernel**: the C++ kernel releases the Python GIL, so reading, computing and writing overlap; the rebinning plan is computed once per chunk and the work buffers once per thread (output bit-identical to v2.2)
- **Single direct writer**: the output main table is opened once and written with `addrows`/`putcol`, no dask graph and no global lock; the subtables are still written with dask-ms
- **Bounded queues** between the stages (`chunk_queue`, `out_queue`) cap the memory in use
- **New GPU engine** (`device="gpu"`): many chunks of different MSs and shapes in one launch, input→output channel map built on the host with the CPU formulas (decreasing or gapped SPWs handled exactly), automatic split into passes that fit the free GPU memory, persistent device buffers, page-locked reused output buffers; on a GPU error the run continues on the CPU with a warning. Built without `--use_fast_math`
- **One kernel module**: `ms_ops` holds the CPU kernel and, with `-DWITH_CUDA=ON`, the GPU path; CUDA runtime linked statically
- **Fixes**: `OMP_NUM_THREADS` set by the user is respected (it was overwritten at import, and the value set in `run()` had no effect); the GPU is detected by asking the module (`gpu_available()`), not from the presence of a function
- Scripts without an `if __name__ == "__main__":` guard work with the reader processes (they do not re-run the script)

**New `run()` parameters**: `device`, `n_readers`, `n_threads`, `gpu_batch`, `timing`, `log_file`, `reader_mode`, `chunk_queue`, `out_queue`. `run()` now returns a summary dictionary. Default `chunk_rows` in `run_vista.sh` and in the examples: 5000.

**Log** (whole package):
- one format, `[ViSta  elapsed s] message`, for stacking, continuum subtraction, compression and extraction, written immediately (not buffered) through the logger `"vista"`
- three levels: `verbose=True` (progress and results), `False` (warnings only), `"debug"` (details); `vista.set_verbosity()` and `log_file` for a copy on file
- new per-dataset information: rows used and removed by the field / target-scan selection, column read, SPWs and channels used, coverage of the output grid, phase-centre offset, shift of the output window
- new warnings: stacking position outside half the primary beam, window moved off the line, missing `data_column`, skipped datasets with the reason, GPU error; final summary with the flagged fraction of the output
- `timing=True`: time of each stage and the stage that limited the run

**`run_vista.sh`**: new options `--device`, `--readers`, `--threads`, `--gpu-batch`, `--timing`, `--debug`.

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
