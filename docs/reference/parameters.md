# Parameters

Every option of {class}`vista.ViSta`, with its default. A run is described
by three calls:

```python
from vista import ViSta

v = ViSta(input_list, ...)        # input, output, how it runs
v.stack(line_freq=..., ...)           # what makes the stack
v.extract(...)                    # what is measured on it (skip it: run() only stacks)
result = v.run(only=None, tag=None)
```

Only the input list and the rest frequency are needed; the options of
`stack()` and `extract()` are given by name (`v.stack(line_freq=345.796)`);
every other option has a default, and a value that does not make sense is refused when it is
set, before any data is read. `v.help()` prints every option with its value
now, what it can be and what it does (`v.help("stack")` for one step;
`help(v.extract)` for the full description). Each call of `stack()` or
`extract()` replaces all the options of that step: the ones not given go back
to their defaults. The pages of the [stacking](../stacking/index) and of the
[measurement](../extraction/index) explain when to change what; the
[command line](../command_line) has one flag for each.

## ViSta(): input, output, how it runs

| Parameter | Default | Description |
|---|---|---|
| `input_list` | required | Path to the [input list](../getting_started/input_list). The datasets must exist |
| `output_name` | `"stacked"` | Name and path of everything written: the stack `<output_name>.zarr` (or `.ms`), the statistics, the results, the figures ([Output files](../extraction/outputs)). For `run(only="measure")`, where the stack and its statistics are |
| `output_format` | `None` | `"zarr"`: a compressed Zarr store written in parallel ({func}`vista.export_ms` converts it). `"ms"`: a Measurement Set, written by one thread. `None`: `"zarr"`, or `"ms"` when `output_name` ends in `.ms` |
| `device` | `"cpu"` | Where the kernel runs. `"gpu"` needs a CUDA build and a visible GPU; if the GPU fails during the run, the run continues on the CPU with a warning |
| `n_readers`, `n_writers`, `n_threads` | `None` | Reader processes, writer threads (Zarr only; a Measurement Set has one) and kernel threads. `None`: from the cores this process may use (its CPU affinity, a SLURM allocation, a container quota): readers and writers about a quarter each, the kernel the rest. The ones you fix are taken out first and the others share what is left; more than the cores is refused ([Performance](performance)) |
| `gpu_batch` | `None` | GPU: chunks per launch. `None`: as many as fit in the free memory of the GPU and in a quarter of the free host memory (at most 32); a larger value is refused |
| `scratch_dir` | `None` | Write the stack on this (fast, local) disk and move it to its place at the end |
| `verbose` | `False` | `False`: warnings and errors only. `True`: the progress and the results. `"debug"`: also the details (metadata, grids, queues, buffers) and the time of every stage of the stacking ([The log](../stacking/index.md#the-log)) |
| `log_file` | `None` | Also write the log to this file (with the same detail as `verbose`) |

How the readers pass the data (processes through shared memory, or threads
when `/dev/shm` is too small), the size of the chunks and of the queues
between readers, kernel and writers are chosen by the code; `verbose="debug"`
shows them.

## stack(): what makes the stack

| Parameter | Default | Description |
|---|---|---|
| `line_freq` | required | Rest frequency of the line: a number in **GHz**, or a string with its unit (`"345.796GHz"`, `"345796 MHz"`; Hz, kHz, MHz, GHz, THz). A list for several lines: the first is the line of the stack (the output grid is centred on it, the velocities refer to it) |
| `line_window` | `600` | Window of every line in its own velocity, km/s: `500` means −500..+500, `(-400, 600)` an asymmetric one. With several lines one entry per line (a number or a pair), or one for all. The channels inside the windows are left out of the continuum fit, and the area flux of every line is the sum over its window. Keep it wide enough for the whole line. Lines whose windows overlap are fitted together, the others each on its own |
| `velocity_range` | `None` | Band of the stack around the first line, km/s: `2000` means −2000..+2000, or a pair. `None`: the widest band a single dataset covers. Every line window must be inside it; a dataset that does not cover it has its window slid to the edge of its band (its line off-centre) and the channels it misses flagged ([Spectral grid](../stacking/spectral_grid)) |
| `nchan` | `None` | A number of output channels instead of `velocity_range` |
| `channel_width` | `None` | Common rest-frame channel width, Hz. `None`: the widest rest-framed channel of the sample, the finest grid every dataset supports; a finer value is refused |
| `frame` | `"LSRK"` | The frame of your redshifts: `"LSRK"` or `"BARY"` (barycentric or heliocentric, as optical redshifts usually are). The input frequencies are always brought to it, row by row ([Rest frame](../stacking/restframing)) |
| `data_column` | `"auto"` | Column read from each input: `"auto"` prefers `CORRECTED_DATA` and falls back to `DATA` |
| `continuum` | `"subtract"` | `"subtract"`: fitted row by row on the channels outside the line windows and subtracted while the stack is written (`MODEL_DATA` the fit, `CORRECTED_DATA` the line, `DATA` kept), then measured on its own. `"fit"`: left in the data and fitted together with the lines (a constant at zero spacing with their size; its flux density in `continuum_fit`, the area fluxes are the sums minus it). `"none"`: left as it is (data already subtracted, or no continuum); a continuum still present is flagged by the measurement ([Continuum](../stacking/continuum)) |
| `continuum_order` | `0` | With `"subtract"`: `0` a constant, `1` a straight line, `"bic"` the one of the two the BIC prefers in every spectral window |
| `exclude_between` | `False` | With several lines, also keep the channels between them out of the continuum fit |

## extract(): what is measured

Without this call `run()` only writes the stack. With it, the stack is also
compressed into the statistics while it is written, and measured: every
line on every channel and annulus at once. Lines whose windows (`line_window`)
overlap are fitted together, sharing the size; every other line on its own
([Several lines](../extraction/measure_line.md#several-lines)). All the
options are given by name.

**The statistics and the weighting**

| Parameter | Default | Description |
|---|---|---|
| `annuli` | `(3, 3000, 18)` | Annuli of rest-frame baseline length: from `b_min` to `b_max` kλ, how many (logarithmic); they should bracket the baselines present ([Compression](../stacking/compression)). Changing them compresses the stack again at the next measurement |
| `weighting_scheme` | `"natural"` | `natural` uses the native visibility weights: the deepest observations dominate and the formal S/N is highest. `democratic` gives every source the same total weight: the mean of the population ([Weighting](../extraction/weighting)) |
| `z_norm` | `True` | Transport every source to `z_ref` |
| `z_ref` | `None` | Reference redshift. `None`: the median of the sample. The value used is always reported: printed after every measurement (also with `verbose=False`), in the results (`z_ref`, `z_ref_given`) and in the result of `run()`. It moves with the selection: fix it when comparing stacks |
| `pb_corr` | `True` | Correct every source for the primary beam at its position (amplitudes / A, weights × A²) |
| `flux_norm` | `False` | Rescale the amplitudes by `flux_ref / NORM`, with `NORM` the last column of the input list; every line of the list must have it (otherwise `extract()` refuses) |
| `flux_ref` | `None` | Reference of that normalisation. `None`: the median, reported like `z_ref` (`flux_ref`, `flux_ref_given`) |

**The lines**

| Parameter | Default | Description |
|---|---|---|
| `v0_range` | `None` | Allowed centres, km/s, in the velocity of each line: a pair `(lo, hi)` for all the lines, or one per line. `lo = hi` fixes the centre. `None`: anywhere in the window (shrunk by half the smallest width) |
| `fwhm_range` | `None` | Allowed FWHM, as above. `lo = hi` fixes the width; with the centre and the size fixed too, the flux is a matched filter and its S/N is the significance. `None`: two channels to the width of the window. The fit stays exactly inside both ranges |
| `model` | `"auto"` | The source. `"auto"`: a Gaussian only if it improves the χ² by 9 over a point source (a 3σ preference; the Δχ² found is in the results and on the figures), the flux averaged over the two with their Akaike weights. `"point"`, `"gauss"`, or a number: a Gaussian of that FWHM, arcsec |
| `model_prior` | `None` | `(θ, σ_dex)`: soft tie of the size of the Gaussian, e.g. to the continuum size |
| `theta_range` | `(0, 15)` | Allowed size (FWHM) of the Gaussian source, arcsec, for the lines and the continuum. A minimum above 0 leaves the point source out (`"auto"` fits the Gaussian only, `"point"` is refused); a fixed size (`model=θ`) must be inside it |
| `flux_measure` | `"auto"` | The flux quoted for every line; both are always given and drawn. `"gauss"`: the integral of the fitted Gaussian. `"area"`: the sum of the spectrum over the window. `"auto"`: the Gaussian, unless the two differ by more than `shape_nsigma` times the error of their difference (a non-Gaussian line), the window holds the fitted line and no other line overlaps it |
| `shape_nsigma` | `2` | Threshold of that test |
| `window` | `None` | Window of the area flux of every line (same forms as `line_window`); `None`: the line windows. It can be changed and the measurement repeated (`run(only="measure")`) without stacking again |
| `fit_span` | `1500` | Channels farther than this (km/s) from every line, in its own velocity, are not fitted: a wide band (`velocity_range`) then does not slow the fit down, nor bring in a continuum that is no longer flat far from the lines. `None`: every channel of the stack. With several lines measured apart, every one uses the channels within it and outside the windows of the others |
| `min_sources_per_annulus` | `2` | Annuli with fewer sources are left out, so that no point of the fit is set by a single object (a single source: 1) |
| `n_bootstrap` | `500` | Resamplings of the sources, for the population errors (0: none) |
| `n_null` | `1000` | Searches on noise, for the significance (0: a Bonferroni bound) |
| `seed` | `42` | Seed of the random numbers of the bootstrap and of the null searches: the same seed gives the same numbers, another seed numbers that differ within their errors |

**The figures**

| Parameter | Default | Description |
|---|---|---|
| `figures_draw` | `True` | Draw the figures of the run: profile, uv, noise, continuum ([Output files](../extraction/outputs)) |
| `figures_unit` | `"Jy"` | Unit of the data, for the labels (scaled to mJy, μJy as needed) |

Advanced keywords, for diagnostics:

| Keyword | Default | Description |
|---|---|---|
| `noise` | `"scatter"` | `scatter`: the noise of every cell (channel × annulus) measured from the scatter of its visibilities, so it holds whatever is really there (weights off by a factor, residual RFI, calibration). `weights`: from the weights alone, right only when they are right in absolute terms; a check, and the only choice for noiseless simulations ([The noise](../extraction/measure_line.md#the-noise)) |
| `channel_correlation` | `True` | Adjacent channels are not independent (the spectral response of the correlator, and the regridding to the common rest-frame grid, which spreads every input channel over two output ones): their correlation is measured on the line-free channels and used in the fit and the errors. `False` treats them as independent, which underestimates the error of a flux summed over channels; a check |
| `n_null_phase` | `200` | Significance without assuming Gaussian noise: stacks in which every source gets a random phase, so the line no longer adds up while noise, weights, correlations and artefacts stay the same; the same search on each, and `p_value_phase_null` the fraction that reaches the measured S/N. 0 to skip |
| `residual_nsigma` | `3` | Threshold of the residual-continuum check |
| `H0`, `Om0` | 67.4, 0.315 | Cosmology of the redshift rescaling |

Nothing on disk is rescaled: the redshift factor, the primary-beam correction
and the flux normalisation are applied to the statistics of every source
while they are combined, at every measurement.

## run(): what runs

| Parameter | Default | Description |
|---|---|---|
| `only` | `None` | `None`: everything (the stack; with `extract()`, its statistics and the measurement). `"stack"`: the stack (and its statistics). `"continuum"`: the subtraction again on the existing stack (with the current windows and order; `DATA` is kept) and its statistics. `"measure"`: the measurement again, from the statistics, in seconds; the stack is compressed again first when the annuli changed, or when the continuum is no longer subtracted. A list for several |
| `tag` | `None` | Name of this measurement: its results and figures are `<output_name>_<tag>_*`, so variants on the same statistics do not overwrite each other |

`run()` returns a `RunResult`: `.line` (the lines, `.line.summary` holds every
number), `.continuum` (the continuum, when it was subtracted), `.stack` (the
summary of the stacking: rows, channels, flagged fraction, size on disk,
time, resources used) and `.files`. Printed, it shows the lines, `z_ref` and
`flux_ref` (and whether they are the median), and the continuum.
