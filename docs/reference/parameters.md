# Parameters

Every option of the stacking and of the measurement, with its default. The
pages of the [stacking](../stacking/index) and of the
[measurement](../extraction/index) explain when to change what; the
[command line](../command_line) has a flag for most of them.

## The stacking: ViSta and run

```python
ViSta(input_file, chunk_rows=5000, verbose=True).run(output, central_freq, ...)
```

### ViSta(): the constructor

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input_file` | `str` | required | Path to the input list |
| `chunk_rows` | `int` | `5000` | Rows read and processed per chunk. Bounds the memory footprint; the code lowers it automatically when the input has many channels. Raise it on a machine with plenty of RAM, lower it if the run is killed |
| `verbose` | `bool` or `"debug"` | `True` | `True`: one line per step and per dataset, with the elapsed time, and the warnings. `False`: warnings and errors only. `"debug"`: also metadata, grids per dataset, queues, GPU buffers (see [The log](../stacking/index.md#the-log)) |

### ViSta.run(): the stacking

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output` | `str` | required | Output path: a Zarr store (e.g. `stack.zarr`) or a Measurement Set (a path ending in `.ms`, or `output_format="ms"`). Anything already at that path is deleted first. `ms_out` is accepted as an alias |
| `central_freq` | `float` | required | Rest-frame frequency, in **Hz**, on which the output grid is centred. This is the rest frequency of the line being stacked |
| `velocity_range_kms` | `float`, `(lo, hi)`, or `None` | `None` | Output bandwidth as a velocity range. A scalar means ±v. Overrides `nchan_out`. Asking for more than the narrowest dataset covers makes the window slide to that dataset's band edge, so the stack is no longer centred on the line for it |
| `nchan_out` | `int` or `None` | `None` | Number of output channels, set directly. `None` takes the widest coverage any single dataset can provide |
| `channel_width_hz` | `float` or `None` | `None` | Common rest-frame channel width, in Hz. `None` uses the widest rest-framed channel of the sample, which is the finest grid every dataset supports; a finer value is refused with an error naming the dataset that sets the floor |
| `data_column` | `str` | `"auto"` | Column read from each input MS. `"auto"` prefers `CORRECTED_DATA` and falls back to `DATA`. Name the column explicitly when `CORRECTED_DATA` holds something you do not want, as in a simulation whose corrupted copy lives there |
| `scratch_dir` | `str` or `None` | `None` | Build the stack here and move it to `output` at the end. Use a local NVMe or `$TMPDIR` when the destination is on a network filesystem |
| `frame` | `str` or `None` | `None` | The frame of your redshifts: `"LSRK"` or `"BARY"`. The input frequencies are brought to it row by row before the rest-framing, which removes the motion of the Earth from TOPO data (ALMA, VLA). Inputs already in that frame are left as they are. `None` uses the frequencies as they are and warns when the inputs are TOPO. See [Rest frame](../stacking/restframing) |
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
| `continuum` | `ContinuumConfig` or `None` | `None` | Subtract the continuum **while the stack is written**: the stack gets `MODEL_DATA` and `CORRECTED_DATA` next to `DATA`, bit for bit those of `subtract_continuum` on the finished stack, without reading it back. `DATA` is kept, so the subtraction can be redone later with another window. See [Continuum](../stacking/continuum) |
| `stats_line`, `stats_continuum` | `str` or `None` | `None` | Also **compress** while the stack is written, into these files, as `compress_visibilities` would: the line from `CORRECTED_DATA` (`DATA` without `continuum`), the continuum from `MODEL_DATA` |
| `binning`, `max_amplitude` | | `BinningConfig()`, `None` | Options of that compression |

`run()` returns a dictionary with `output`, `format`, `n_rows`, `n_spw`,
`nchan`, `df_hz`, `flagged_fraction`, `skipped` (indices of the input lines
that gave no data), `bytes_on_disk` and `seconds`; with `continuum`, also
`contsub_orders`, and with `stats_line`, `statistics` (the files written).

## The measurement

Everything the measurement does is set by the configuration objects of
`vista.extract.config`.

### LineConfig: the spectral setup

| Parameter | Default | Description |
|---|---|---|
| `rest_freq_ghz` | required | Rest frequency of the line, in GHz. The only parameter with no default |
| `v_window_kms` | `(-600, 600)` | Velocity window containing the line. Defines the line-free channels for the continuum fit, and the integration window in `window` mode |
| `second_rest_freq_ghz` | `None` | Rest frequency of a second line in the same band; when set, the two are de-blended analytically |
| `second_v_window_kms` | `(-500, 500)` | Velocity window of the second line, in its own frame |
| `exclude_v_kms` | `()` | Extra velocity intervals kept out of the line-free channels |

### LineFitConfig: the joint fit

The options of `measure_line` (`--method joint`).

| Parameter | Default | Description |
|---|---|---|
| `mode` | `"search"` | `search`: centre and width free, significance from the same search on noise. `detect`: centre and width fixed (`v0_kms`, `fwhm_kms`), matched filter |
| `v0_kms`, `fwhm_kms` | `0`, `None` | Fixed in `detect` (`fwhm_kms` required), starting values in `search` |
| `v0_range_kms`, `fwhm_range_kms` | `None` | Allowed centres and widths in `search`; default the line window, and 2 channels to the window width |
| `size` | `"auto"` | `auto`, `point` or `gauss` (see [Measuring the line](../extraction/measure_line.md#the-size)) |
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

### BinningConfig: the radial binning

| Parameter | Default | Description |
|---|---|---|
| `b_min_klambda` | `3` | Inner edge of the annuli, in kλ |
| `b_max_klambda` | `3000` | Outer edge. Should bracket the baselines actually present; the default is meant for heterogeneous ALMA samples |
| `n_bins` | `18` | Number of logarithmic annuli. Fixed at compression time: changing it means recompressing |

### WeightingConfig: how the sources are combined

| Parameter | Default | Description |
|---|---|---|
| `scheme` | `"natural"` | `natural` uses the native visibility weights, so the deepest observations dominate and the formal S/N is highest. `democratic` renormalises each source to total weight 1, giving the population average |
| `redshift_rescaling` | `True` | Apply the factor that transports every source to `z_ref` |
| `z_ref` | `None` | Reference redshift. `None` uses the sample median, which moves with the selection: fix it when comparing stacks |
| `flux_normalisation` | `False` | Rescale amplitudes by `norm_ref / NORM`, with `NORM` the last column of the input list |
| `norm_ref` | `None` | Reference value of that normalisation. `None` uses the sample median |
| `already_applied` | `False` | The amplitudes on disk already carry the redshift factor; do not apply it twice |
| `primary_beam` | `True` | Correct every source for the primary beam at its position: amplitudes / A, weights × A² (A from a Gaussian beam of FWHM 1.2 c / ν D, computed during the compression) |
| `H0`, `Om0` | `67.4`, `0.315` | Cosmology for the luminosity distances |

### ProfileConfig: the flux per annulus

| Parameter | Default | Description |
|---|---|---|
| `method` | `"template"` | `template` fits the shape once and the amplitude per annulus, so the flux follows analytically with no truncation. `window` integrates over a fixed velocity window. `continuum` averages the line-free channels |
| `joint_continuum` | `False` | Fit the line on top of a constant continuum and measure both. Use it when the stack was not continuum subtracted |
| `fit_span_kms` | `1500` | Half-width used by the shape fit and the per-annulus least squares. Cannot usefully exceed the band |
| `shape_tie` | `"free"` | For a blend: `free`, `centroid`, or `centroid+width` |
| `fixed_shape_kms` | `None` | Impose `(centroid, sigma)` and skip the shape fit |
| `continuum_overlap_frac` | `1.0` | In continuum mode, use only channels covered by at least this fraction of the sources |

### UVFitConfig: the source model

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

### BootstrapConfig: the uncertainties

| Parameter | Default | Description |
|---|---|---|
| `n_realisations` | `500` | Resamplings of the source sample |
| `seed` | `42` | Random seed, for reproducibility |

### ContinuumConfig: the subtraction on the stack

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

### compress_visibilities(): further arguments

| Parameter | Default | Description |
|---|---|---|
| `line_column` | `None` | `None` picks `CORRECTED_DATA` when present, otherwise `DATA` |
| `continuum_column` | `"MODEL_DATA"` | Column holding the continuum |
| `max_amplitude` | `None` | Discard visibilities above this amplitude; NaN and infinities are always discarded |
| `chunk_mb` | `512` | Target size of one read chunk, in MB |
| `resume` | `True` | Reuse the data descriptors already present in the output file |
| `n_io` | `4` | Threads reading a Zarr store at the same time (an MS is read by one) |

### extract_flux(): further arguments

| Parameter | Default | Description |
|---|---|---|
| `match_tol_arcsec` | `2.0` | Angular tolerance for deciding that two entries are the same physical source. Set it to 0 for simulations that share one field |
| `output_prefix` | `None` | Write the results and the text products with this prefix |
