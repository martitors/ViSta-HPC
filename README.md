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

### Python API — stacking only

`vista.ViSta` is self-contained: it reads the input list, builds the stacked
Measurement Set, and stops there. Nothing in it depends on `vista.extract`,
so if all you want is the stack, this is the whole interface.

```python
from vista import ViSta

pipeline = ViSta(
    input_file="input_list.txt",   # one line per MS / field / spw
    chunk_rows=50_000,             # rows read per processing chunk
    verbose=True,
)

pipeline.run(
    ms_out="stacked_output.ms",
    central_freq=345.7959899e9,    # rest frequency of the line [Hz]
    velocity_range_kms=2000.0,     # output bandwidth, ±km/s
)
```

The output is a standard Measurement Set with one spectral window per line of
the input list, which you can image with `tclean`, fit with any
visibility-domain tool, or hand to `vista.extract`.

#### `ViSta(...)` — constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_file` | `str` | required | Path to the input list |
| `chunk_rows` | `int` | `5000` | Rows read and processed per chunk. Bounds the memory footprint; the code lowers it automatically when the input has many channels. Raise it on a machine with plenty of RAM, lower it if the run is killed |
| `verbose` | `bool` | `True` | Progress and diagnostics on stdout |

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

Threading is set from the environment: `SLURM_CPUS_PER_TASK` when present,
otherwise half the logical cores, exported as `OMP_NUM_THREADS`. GPU dispatch,
when the CUDA kernel was built, is automatic; the batch size, the number of
datasets sent per CUDA kernel launch, is set inside the pipeline and defaults
to 20. Larger batches improve the overlap between GPU compute and I/O, but a
batch larger than the sample removes the overlap altogether.

See `examples/run_vista.py` for a complete example.

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
