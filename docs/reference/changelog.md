# Changelog

## v3.0.0

**One object, three calls**: `ViSta(input_list, output_name, output_format, device, ..., verbose, log_file)` holds the input, the output and how the run uses the machine; `.stack(line_freq=, line_window=, velocity_range=, ..., continuum=, continuum_order=, exclude_between=)` what makes the stack; `.extract(annuli=, weighting_scheme=, z_norm=, z_ref=, pb_corr=, flux_norm=, flux_ref=, v0_range=, fwhm_range=, model=, model_prior=, flux_measure=, window=, n_bootstrap=, n_null=, seed=, figures_draw=, figures_unit=)` what is measured on it; their options are given by name. Without `extract()` the run only writes the stack (no statistics); with it, the stack is compressed while it is written and measured. `.help()` lists every option with its value and what it can be, and `.run()` runs them (`run(only="stack" | "continuum" | "measure")`, `tag=` to name a measurement). Only the input list and the rest frequency are needed; a value that does not make sense is refused when it is set, before any data is read. `velocity_range` defaults to the widest band of one dataset. `z_ref` and `flux_ref` default to the median and the value used is always reported; `flux_norm=True` without the `NORM` column is refused. The configuration classes (`LineConfig`, `ContinuumConfig`, `LineFitConfig`, ...) are no longer part of the interface, and `ViSta(...).run(output, central_freq, ...)` is replaced by the above. The command line (`run_vista.sh`, now `python -m vista`) has the same options, one flag each, and measures only with `--extract`; `python -m vista.extract` is gone.

**Outputs**: every file starts with `output_name` (default `stacked`): the stack `stacked.zarr` (default; `output_format="ms"` or a name ending in `.ms` for a Measurement Set), `stacked_line_stats.npy`, `stacked_cont_stats.npy`, `stacked_results.json`, `stacked_continuum_results.json`, the figures.

**Several lines**: `line_freq` and `line_window` take a list (one window per line, or one for all). The continuum is fitted outside all the windows (`exclude_between=True`: also between them); every line is searched in its own window; the lines whose windows overlap (directly or through another line) are fitted together, each with its flux, centre and width, sharing the size, and every other line is measured on its own, with its own size, on the channels near it and outside the other windows. Which lines went together is decided from the windows, with no option (`line_groups` in the results). The results have every line in `lines`. `line_freq` also takes a string with its unit (`"345.796GHz"`, `"345796 MHz"`).

**One mode**: `mode="detect"` / `"search"` is replaced by limits, `v0_range` and `fwhm_range` (km/s, `lo = hi` fixes the value); the fit stays exactly inside them (it allowed 0.5–1.5 times the width range). With the centre and the width fixed the measurement is the former `detect`, reproduced to the last digit; a search with free centre and width moves by less than 0.1% (the bounds of the fit are now the limits themselves).

**Two fluxes**: every line has the integral of the fitted Gaussian (`flux_gauss`) and the sum of the spectrum over its window (`flux_area`), with the error of their difference computed from the same data. `flux_measure="auto"` (default) quotes the Gaussian one unless the two differ by more than 2σ (a line that is not Gaussian: double horn, wings), the window holding the line and no other line overlapping it; `"gauss"` or `"area"` choose. The profile figure draws both. The errors of `v0` and `FWHM` are now, like the flux, the larger of the thermal and the bootstrap one.

**Residual continuum**: the profile outside the windows is fitted with an offset and a slope; a significant one (3σ) is flagged (`continuum_residual`, a warning, the figure), after a subtraction (continuum left) or without one (continuum in the data).

**The continuum is subtracted by default** (`stack(continuum="subtract")`; `"none"`, `--continuum none`: not); its window is that of the lines (`ContinuumConfig.exclude_kms` is gone). Instead of the subtraction it can be fitted together with the lines (`continuum="fit"`, `--continuum fit`; the former `--contsub after`): a constant with the size of the lines, in `continuum_fit`, and the area fluxes are the sums minus it. **Renamed**: `size` → `model`, `size_prior` → `model_prior`, `flux` → `flux_measure`, the binning → `annuli`, the weighting options → `weighting_scheme`, `z_norm`, `pb_corr`, `flux_norm`, `flux_ref`, the order of the subtraction → `continuum_order`. `tie` (`"centroid"`, `"centroid+width"`) is gone: the grouping of the lines follows from their windows. `theta_max_arcsec` becomes `theta_range=(min, max)` (with a minimum), `min_sources_per_bin` becomes `min_sources_per_annulus`, and both, with `shape_nsigma`, are options of `extract()`. `fit_span_kms` becomes `fit_span` (km/s, default 1500; `None`: every channel), an option of `extract()`. Gone: `resolved_delta_chi2` (fixed at 9; the Δχ² found is drawn on the figures), `already_applied` (nothing on disk is ever rescaled).

**Resources**: the readers, writers and kernel threads come from the cores the process may use (CPU affinity, SLURM, container quota), about a quarter each for readers and writers and the rest for the kernel; numbers fixed by the user are taken out first, and more than the cores is refused. `gpu_batch` comes from the free memory of the GPU and of the host. The readers run as threads when `/dev/shm` is too small. `timing`, `reader_mode`, `chunk_queue`, `out_queue`, `chunk_rows`, `compress` and `ms_out` are no longer options; the timing of the stages is in `verbose="debug"`.

**Faster stacking**: `OMP_PROC_BIND=close`, set by default by the stacking, pinned the main thread to one core when the kernel was loaded, and every reader process and writer thread started afterwards inherited that core. The binding is no longer set (and a binding set by the user no longer pins them): on two cores the simulated sample of the tutorial stacks in 6.6 s instead of 9.0 s (with the defaults of each version).

**Quiet by default**: `verbose=False` (warnings only); `True` for the progress, `"debug"` for the details and the timing. The significance stays finite at very high S/N (it was infinite above S/N ≈ 37).

**Figures in a notebook**: `result.plot()` draws the figures of a measurement right away (`"profile"`, `"uv"`, `"noise"`, `"continuum"`), and `vista.compare({label: result, ...}, expected=...)` puts several measurements in one figure (the profiles with their fits, the flux per annulus with the source models, the fluxes with their errors) and prints their numbers side by side. `examples/simulate.py` has `expected(truth)`, what the stack of a sample should give, in the form `compare` takes.

**The example notebook** goes one step at a time, every step explained and with its own figure: what the simulator makes and one galaxy without and with noise (image and spectrum); a heterogeneous sample (different redshifts, fluxes, widths, sizes, depths and arrays); the stack, in an image, against one galaxy; how to read the measurement; then the window, the source model, the normalisations, the weighting, the errors and the continuum, one at a time; and three harder cases (a line with wings, sources that are rings, two blended lines). It no longer needs CASA (whose data tables made it stop on Colab).

**Clearer results**: `print(result)` says in words what was measured and the choices made (which flux is quoted and why, which source model and why, what happened to the continuum). The profile figure names the two fluxes (the area of the Gaussian fit; the area under the spectrum in the window), gives the limits of the window and the value of the continuum, and shows the imaginary part with the residuals; the uv figure says which source model was chosen. `result.plot("image")` and `vista.dirty_image` give a quick-look image of the stack (or of one of its galaxies). The residual-continuum check uses the size of the line, as the spectrum drawn (a resolved continuum left in the data was underestimated). Going back to the subtraction after a measurement without it compresses the stack again instead of stopping. `examples/simulate.py` can make ring-shaped sources (`shape="ring"`) and draws a galaxy (`plot_galaxy`) and a sample (`plot_sample`, `describe`).

**Documentation**: the installation without the cluster section; every page on the new interface; the examples regenerated.

## v2.5.1

**The frame of the redshifts by default**: `run(frame="LSRK")` and `run_vista.sh --frame LSRK` are the default. The input frequencies are always brought to the frame of the redshifts, row by row (TOPO data from ALMA and the VLA have the Earth's motion removed); inputs already in that frame are left as they are, so stacks of inputs labelled LSRK, such as the simulated ones, do not change. `frame="BARY"` for barycentric redshifts. `frame=None`, which used the frequencies as they are, is refused.

**Citation**: the PASP paper has its own title (*The ViSta method for stacking in the Fourier domain and its application to the dusty star-forming galaxies in the ALMA Science Archive*); the optimized version is described in Proc. SPIE 14153, 141531X (2026).

## v2.5

**Documentation**: a Sphinx site for Read the Docs (`docs/`, `.readthedocs.yaml`): installation, every step of the stacking, the measurement, the command line, the worked example as pages with its results (generated from the executed notebook by `docs/_tools/notebook_to_pages.py`), the parameters, the API, performance and changelog. The README is now a short introduction. `examples/simulate.py`: `band_offset_kms` (the line away from the band centre), `t0` and `layout_seed` (other dates and arrays), `noise_seed` (a new noise realisation with the same weights).

**One pass** (`run(..., line=, continuum=, stats_line=, stats_continuum=)`; the default of `run_vista.sh` when it stacks and compresses in the same run, `--two-pass` for the separate steps): the continuum subtraction and the compression run on every chunk right after the stacking kernel, so the stack is read once instead of three times. Same stack and statistics as the separate steps (columns bit for bit; statistics bit for bit on Zarr, to 1e-15 on an MS). `DATA` is kept: the subtraction can be redone on the stack with another window, and the parameters of every subtraction are recorded in the stack and in the statistics (a new subtraction says which one it replaces). With `order=-1` the order is chosen on the first `bic_rows` rows of each entry, in both paths.

**The noise, measured** (`vista.extract.noise`): the statistics store `Σw Re²` and `Σw Im²` too (format 4; files of format 3 still work, with the noise from the weights), so the variance of every cell comes from the scatter of its visibilities: κ (scatter / weights) for every entry and every channel, checked against the imaginary part and the line-free channels. The correlation of the noise between channels — spectral response, rebinning, and the level removed by the continuum subtraction, shared by all the channels of a row — is measured as a whole matrix from the imaginary part.

**`measure_line`** (`--method joint`, the new default of `run_vista.sh`): the line fitted on all channels and annuli at once — profile at zero spacing, flux, peak, centre, FWHM, size — with the measured noise; thermal and bootstrap errors; matched filter with a fixed shape (`mode="detect"`) or a search with its significance from the same search on noise; window fluxes with the correlation between channels; three figures (profile, uv, noise). For extended sources: point and Gaussian templates in the detection scan, the model averaged over each annulus, the total flux averaged over the point and the Gaussian model (Akaike weights, the spread in the error), an optional size prior (`theta_prior_arcsec`, `--theta-prior`). Tested on point, Gaussian, exponential, rotating, offset and double-horn sources. The chain of Sec. 5.3 (`extract_flux`, `--method template|window`) is unchanged. See [Measuring the line](../extraction/measure_line).

**Tutorial notebook** (`examples/ViSta_tutorial.ipynb`, runs on Colab) and `examples/simulate.py` (synthetic ALMA-like Measurement Sets with a line, a continuum, a size, primary-beam attenuation, noise with wrong weights and Hanning smoothing; each dataset also without noise, `noiseless=True`, same visibilities and weights). The warnings on the imaginary part and the line-free channels are given only when they scatter more than the noise (less means conservative errors).

**Figures in your own style**: `measure_line` also writes `<prefix>_diagnostics.npz` (channel correlation matrix, standardised noise samples, null distributions), and `examples/plot_line.py` redraws the three figures from the output files alone (numpy + matplotlib, `STYLE` block, `--style file.mplstyle`, one commented function per figure). With `n_null=0` and several templates the p-value is a Bonferroni bound (it stopped with an error). The size test divides the χ² difference between the point and the Gaussian source by the factor by which the channel correlation inflates it (`size_test.correlation_factor`): the fits weight the cells by their noise alone, and with Hanning smoothing and a continuum subtraction the raw difference is 4-5 times too large, which made noisy point sources look resolved.

**Kernel**: `annuli_accumulate` gives the same sums at every run (each thread adds a fixed block of rows, the partial sums are added in thread order; they were added in the order the threads finished, which changed the last bits from run to run), and accumulates the two new sums when asked.

**`run_vista.sh`**: one pass by default, `--two-pass`; `--method joint` (default), `--mode`, `--v0`, `--fwhm`, `--size`, `--n-null`, `--plot-rebin`; exit status 0 with `--no-plot` (it was 1).

## v2.4

**The stack can be a Zarr store, written in parallel** (`run(output="stack.zarr")`, the default for any path not ending in `.ms`):
- layout of dask-ms (`MAIN/MAIN_<k>` per input line, subtables in their own groups), readable with `daskms.experimental.zarr.xds_from_zarr`
- written by `n_writers` threads in parallel (default 4): every chunk is a file of its own, so the writers never wait for each other; the rows of each dataset stay in input order
- columns compressed with Blosc lz4 (`compress=True`): flags and empty channels take almost no space; constant columns are not written at all
- `vista.export_ms(store, ms)` / `python -m vista.extract export-ms`: the Measurement Set that `output_format="ms"` writes, with the rows in input order, and with `MODEL_DATA` and `CORRECTED_DATA` when present
- a Measurement Set is still written directly when `output` ends in `.ms` (or with `output_format="ms"`), so existing scripts keep producing an MS; `ms_out=` is still accepted
- needs `zarr>=2.12,<3` (dask-ms reads zarr 2 stores)

**Spectral frame of the inputs** (`run(frame="LSRK")`, `run_vista.sh --frame`): the input frequencies are brought to the frame of the redshifts row by row, from the time of each row, the array position and the source direction, inside the kernel call (no second resampling, no copy of the inputs). The Doppler factor is exact to 0.2 mm/s and reproducible bit for bit (computed once, in a fixed order, by the main process). Without it, a warning gives the size of the effect for TOPO inputs. The output is labelled LSRK. See [Rest frame](../stacking/restframing)

**Precision**: every computation is in float64, and a value is rounded to float32 once, when it is written. WEIGHT × R and SIGMA / √R are computed in float64 (they were computed in float32, or in float64 depending on the NumPy version); so are the NumPy versions of the continuum subtraction and of the compression, used when the kernel is not built (they rounded the model before subtracting it, and the products weight × visibility, to float32). A NaN or infinite visibility is treated as flagged: it made its output channel NaN, still unflagged, which then spoils an image. Entries with different numbers of correlations (e.g. XX,YY and XX,XY,YX,YY) are refused before anything is read, with the list of entries (the writing failed after the stacking); different correlation types (XX,YY and RR,LL) give a warning

**Extraction** (same results, much faster, see [Performance](performance)):
- continuum subtraction and compression read Zarr stores and Measurement Sets through one interface (`vista.open_stack`), with python-casacore for MSs: `casatools` is no longer needed (also for the primary-beam correction, which reads the original MSs)
- reading, computing and writing overlap (`n_io` threads on a Zarr store)
- new OpenMP kernels in `ms_ops`: `continuum_subtract` (closed-form least squares per row, order 0 or 1, instead of a `np.polyfit` per row and correlation) and `annuli_accumulate` (one pass over the chunk, per-thread accumulators); NumPy fallbacks when the kernel is not built

**Continuum subtraction, fixes**:
- the rows of each data descriptor are found wherever they are: the stacking of v2.3 writes the chunks of different datasets interleaved, and the subtraction read the rows of a data descriptor as one consecutive block, fitting rows of other datasets
- only the unflagged samples enter the fit: the flagged, empty channels of a dataset that does not cover the whole grid were counted as 0 and pulled the continuum towards zero (on a stack covering 60% of the grid, the continuum came out at 0.43 of its value)
- an MS is read in chunks of at most about 256 MB of visibilities, also for very wide spectra

**Code**: the stacking is split into modules (`spectral_grid`, `ms_input`, `readers`, `compute`, `frames`, `ms_output`, `stackio`, `common`); the shared memory of the readers is released also when a run fails.

**`run_vista.sh`**: `--format`, `--writers`, `--io-threads`, `--export-ms`, `--frame`.

## v2.3

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

## v2.1

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
