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
- **Parallel pipeline**: reader processes, a GIL-free C++/OpenMP kernel and a single writer work at the same time
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

### Python API — stacking only

`vista.ViSta` is self-contained: it reads the input list, builds the stacked
Measurement Set, and stops there. Nothing in it depends on `vista.extract`,
so if all you want is the stack, this is the whole interface.

```python
from vista import ViSta

pipeline = ViSta(
    input_file="input_list.txt",   # one line per MS / field / spw
    chunk_rows=5000,               # rows read per processing chunk
    verbose=True,                  # or "debug"
)

result = pipeline.run(
    ms_out="stacked_output.ms",
    central_freq=345.7959899e9,    # rest frequency of the line [Hz]
    velocity_range_kms=2000.0,     # output bandwidth, ±km/s
    device="cpu",                  # "gpu" with a CUDA build
)
```

A script that calls `run()` does not need an `if __name__ == "__main__":`
guard: the reader processes do not re-run it.

The output is a standard Measurement Set with one spectral window per line of
the input list, which you can image with `tclean`, fit with any
visibility-domain tool, or hand to `vista.extract`.

#### `ViSta(...)` — constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_file` | `str` | required | Path to the input list |
| `chunk_rows` | `int` | `5000` | Rows read and processed per chunk. Bounds the memory footprint; the code lowers it automatically when the input has many channels. Raise it on a machine with plenty of RAM, lower it if the run is killed |
| `verbose` | `bool` or `"debug"` | `True` | `True`: one line per step and per dataset, with the elapsed time, and the warnings. `False`: warnings and errors only. `"debug"`: also metadata, grids per dataset, queues, GPU buffers (see [The log](#the-log)) |

#### `ViSta.run(...)` — the stacking itself

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ms_out` | `str` | required | Output Measurement Set. Anything already at that path is deleted first |
| `central_freq` | `float` | required | Rest-frame frequency, in **Hz**, on which the output grid is centred. This is the rest frequency of the line being stacked |
| `velocity_range_kms` | `float`, `(lo, hi)`, or `None` | `None` | Output bandwidth as a velocity range. A scalar means ±v. Overrides `nchan_out`. Asking for more than the narrowest dataset covers makes the window slide to that dataset's band edge, so the stack is no longer centred on the line for it |
| `nchan_out` | `int` or `None` | `None` | Number of output channels, set directly. `None` takes the widest coverage any single dataset can provide |
| `channel_width_hz` | `float` or `None` | `None` | Common rest-frame channel width, in Hz. `None` uses the widest rest-framed channel of the sample, which is the finest grid every dataset supports; a finer value is refused with an error naming the dataset that sets the floor |
| `data_column` | `str` | `"auto"` | Column read from each input MS. `"auto"` prefers `CORRECTED_DATA` and falls back to `DATA`. Name the column explicitly when `CORRECTED_DATA` holds something you do not want, as in a simulation whose corrupted copy lives there |
| `scratch_dir` | `str` or `None` | `None` | Build the MS here and move it to `ms_out` at the end. Use a local NVMe or `$TMPDIR` when the destination is on a network filesystem |
| `device` | `"cpu"` or `"gpu"` | `"cpu"` | Where the kernel runs. `"gpu"` needs a CUDA build and a visible GPU; if the GPU fails during the run, the run continues on the CPU with a warning |
| `n_readers` | `int` | `2` | Reader processes, each reading a different MS at the same time. More readers help only on fast filesystems: the run is usually limited by the writing |
| `n_threads` | `int` or `None` | `None` | OpenMP threads of the CPU kernel. `None` keeps `OMP_NUM_THREADS`, else `SLURM_CPUS_PER_TASK`, else all cores |
| `gpu_batch` | `int` | `8` | GPU only: maximum chunks per GPU launch. The compute takes the chunks already read, without waiting for more; larger values use more host memory |
| `timing` | `bool` | `False` | At the end, also log how long setup, reading, computing and writing took, and which stage limited the run |
| `log_file` | `str` or `None` | `None` | Also append the log to this file |
| `reader_mode` | `"process"` or `"thread"` | `"process"` | `"thread"` avoids shared memory (`/dev/shm`), e.g. in containers where it is small, at the cost of reads taking turns |
| `chunk_queue`, `out_queue` | `int` | `n_readers+1`, `3` | Advanced: chunks waiting for the compute and for the writer. They bound the memory in use |

`run()` returns a dictionary with `ms_out`, `n_rows`, `n_spw`, `nchan`,
`df_hz`, `flagged_fraction`, `skipped` (indices of the input lines that gave no
data) and `seconds`.

**CPU or GPU?** On the benchmarks below the run is limited by the writing of
the output MS (a table is written by one thread), so the GPU gives the same
time as the CPU. It pays off when the computation dominates: many input
channels and rows per dataset, a fast output filesystem.

See `examples/run_vista.py` for a complete example.

### How the stacking works

```
setup:  metadata of every MS → common output grid → output subtables

        readers (processes) ──► chunks ──► compute (main thread) ──► results ──► writer (thread)
        one MS each,            bounded     C++ kernel, GIL released  bounded     appends to the
        shared memory           queue       OpenMP or CUDA            queue       output MS
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
- **Writer** appends every processed chunk to the output MS. The rows are in
  the order the chunks are processed, not in the input order; the content does
  not depend on it.

### The log

Every line carries the time since the start of the step. A run on four test
datasets on a 2-core machine, with `timing=True` (one input line has wrong
coordinates):

```
[ViSta       0.0s] ViSta 2.3.0 · 4 MS from input_list.txt → stacked.ms · CPU (2 threads) · 2 readers
[ViSta       1.0s] Output grid: 480 channels × 5625.0 kHz (7.31 km/s), ±1756 km/s around 230.5380 GHz
[ViSta       1.0s]   (width: the coarsest rest-frame channel of the sample; channels: automatic, the MS fill 210-480 of them)
[ViSta       1.0s]   4 of 4 MS do not cover the whole window around 230.5380 GHz: their window is moved to the edge of their band (shift in the lines below), so their channel frequencies differ from the others
[ViSta       1.7s] Setup done (48 antennas in total); stacking 4 MS...
[ViSta       4.4s] [2/4] ms_001.ms  z=1.1600  3,960 rows  CORRECTED_DATA  2 SPWs, 480 ch  covers 75% of the grid  offset 1.7"  window +411.1 MHz
[ViSta       4.4s] [1/4] ms_000.ms  z=0.8000  3,960 rows  DATA  1 SPW, 480 ch  covers 62% of the grid  offset 0.3"  window +526.1 MHz
[ViSta       4.5s] [3/4] ms_002.ms  z=1.5200  3,168/3,960 rows (target scans)  DATA  1 SPW, 480 ch  covers 44% of the grid  offset 464.3"  window +755.7 MHz
[ViSta       4.5s] WARNING: ms_002.ms: the stacking position is 464.3" from the phase centre, beyond half the primary beam (FWHM ~ 64" at 91.5 GHz): check RA/Dec in the input list
[ViSta       4.8s] [4/4] ms_003.ms  z=1.8800  3,960 rows  CORRECTED_DATA  1 SPW, 480 ch  covers 100% of the grid  offset 3.5"  window +20.3 MHz
[ViSta       6.2s] Timing: setup 1.7 s · stacking 3.8 s · subtables and closing 0.8 s
[ViSta       6.2s]   during the stacking (overlapping): reading 0.4 s per reader · computing 0.3 s · writing 0.8 s  → no stage busy for half of the time: the first data arrived after 2.7 s (start of the readers)
[ViSta       6.2s] Done in 6.2 s: stacked.ms · 4 SPW × 480 channels · 15,048 rows · 124.3 MB · 28.5% of the output flagged (no data)
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
limited the run (`→ limited by writing (95% busy)` on a long run; on a short
one like this, the start of the reader processes dominates). `verbose="debug"`
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
├── src/                 # C++ / CUDA kernel (Python module ms_ops)
│   ├── ms_ops.cpp       # OpenMP kernel + Python bindings (+ GPU dispatch)
│   ├── ms_ops_cuda.cu   # CUDA engine
│   └── ms_ops_cuda.h    # CPU/GPU interface and shared formulas
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

The three steps write to the same log as the stacking (see [The log](#the-log)):
one line per data descriptor, the results, and warnings for what was skipped
or fell back (too few line-free channels, a profile fit that did not converge,
sources without a normalisation value).

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
| `chunk_rows` | `50000` | Rows read per chunk |

### `compress_visibilities(...)` — further arguments

| Parameter | Default | Description |
|---|---|---|
| `line_column` | `None` | `None` picks `CORRECTED_DATA` when present, otherwise `DATA` |
| `continuum_column` | `"MODEL_DATA"` | Column holding the continuum |
| `max_amplitude` | `None` | Discard visibilities above this amplitude; NaN and infinities are always discarded |
| `chunk_mb` | `512` | Target size of one read chunk, in MB |
| `resume` | `True` | Reuse the data descriptors already present in the output file |

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
