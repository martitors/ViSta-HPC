# Post-processing and uv-domain signal extraction

`vista.extract` recovers the stacked flux density from the combined
Measurement Set produced by `vista.ViSta`, working directly in the visibility
domain.

The stacked MS is a standard interferometric product, so in principle the flux
could be measured by imaging it with `tclean` or by fitting the visibilities
with a general tool such as UVMultiFit. Both routes become problematic on a
stack of heterogeneous, wide datasets: imaging inherits the ill-defined hybrid
beam of a sample spanning very different angular resolutions and only partially
overlapping *uv* coverages, so forcing a common resolution correlates the noise
and dilutes the compact flux, while a general visibility fitter processes
everything in memory and needs preliminary time averaging on a large
concatenated stack.

This subpackage instead compresses the visibilities once into a compact set of
sufficient statistics and runs every fit on that representation. Compact and
extended configurations sample the same visibility function `V(b)` at
different baseline lengths and jointly constrain the total flux and the source
size without any loss of resolution.

The method is described in Chapter 5 of Torsello (2026); equation numbers
below refer to that text.

## The four steps

| Step | Function | Reads | Needs |
|------|----------|-------|-------|
| 1. Stacking | `ViSta.run` | the input datasets | `dask-ms`, `ms_ops` |
| 2. Continuum subtraction | `subtract_continuum` | the stacked MS, read/write | `casatools` |
| 3. Compression | `compress_visibilities` | the stacked MS, read-only | `casatools` |
| 4. Extraction | `extract_flux` | the compressed statistics | `numpy`, `scipy` |

Only steps 2 and 3 touch the MS. Step 4 works on a file of a few MB, so it
runs on a laptop in seconds and can be repeated as often as needed for
different weighting schemes, normalisations and source models. **Neither the
weighting nor the flux normalisation is ever applied to the visibilities on
disk**: both are computed once and applied to the compressed statistics at
extraction time, so the stacked MS always keeps its native amplitudes.

`run_vista.sh` drives all four from the command line; see `--help`.

```python
from vista.extract import (LineConfig, ContinuumConfig, WeightingConfig,
                           subtract_continuum, compress_visibilities,
                           extract_flux)

line = LineConfig(rest_freq_ghz=345.7959899,   # the line you are stacking, GHz
                  v_window_kms=(-600, 600))

subtract_continuum("stacked.ms", line, ContinuumConfig(order=0))

compress_visibilities("stacked.ms", "input_list.txt", line,
                      out_line="line_stats.npy",
                      out_continuum="cont_stats.npy")

result = extract_flux("line_stats.npy", "input_list.txt", line,
                      weighting=WeightingConfig(scheme="democratic"),
                      output_prefix="stack_democratic")
```

## The output spectral grid

ViSta does not keep a fixed velocity interval around the line. The grid is
built in three steps, in the rest frame.

**The channel width.** Each dataset has its own observed width `df_obs`, which
in the rest frame becomes `df_obs * (1 + z)`. The common width is the
*largest* of these across the sample: it is the finest grid every dataset can
support, since resampling a coarse spectrum onto finer channels would only
correlate the noise between them. `--width HZ` overrides it, and a value finer
than that floor is refused with an error naming the dataset that sets it.

**The number of channels.** With `--velocity-range KMS` (the default, 2000)
the bandwidth is `nu_rest * dv / c` and the number of channels follows from the
width; `--nchan N` sets it directly instead. Without either, ViSta takes the
widest coverage any single dataset can provide. The count is rounded down to
an even number.

**The placement.** The window of that bandwidth is centred on the rest
frequency of the line, the same for every dataset. Where a dataset does not
cover the whole window — its band ends before the edge — the window is slid to
the nearest edge of that dataset's coverage rather than being truncated, and
the log records the shift. So every output spectral window has the same width
and the same number of channels, but possibly a different start frequency.

That last point is why the extraction builds a common frequency grid of its
own when compressing: the spectral windows share a step but not an origin, and
the channels have to be mapped onto a single axis before the sources can be
averaged. It is also why the number of contributing sources varies from
channel to channel, which the figures show on the right-hand axis of the
spectrum: the coverage is complete near the line and thins out in the wings.

---

## The input list

The extraction reads the **same file used for the stacking**, with one optional
extra column:

```
# <ms_path>  <redshift>  <RA>  <Dec>  [FIELD_ID]  [SPW_IDS]  [NORM]
/path/to/obs1.ms  2.369  00:18:02.46  -31:35:05.2  4   27,29  2.91e13
/path/to/obs2.ms  2.561  00:32:07.60  -30:37:35.2  11  25,27  6.05e12
/path/to/obs3.ms  2.561  00:32:07.60  -30:37:35.2  12  23,25  6.05e12
```

The line number is the `DATA_DESC_ID` of that entry in the stacked MS, because
the pipeline creates one output spectral window per input line and tags every
row it writes with the index of that line. Entries skipped at runtime keep
their slot, with zero rows, so the numbering never shifts on its own; lines
removed by hand must be replaced by an `# EXCLUDED` placeholder.

`NORM` is the quantity the amplitudes are divided by when the flux
normalisation is on: a luminosity, a continuum flux density, or any proxy that
correlates with the stacked emission. It is recognised because it is neither a
bare integer nor a comma-separated list of integers, so it cannot be confused
with `FIELD_ID` or `SPW_IDS`; it can also be written as `norm=2.91e13`. The
stacking ignores it.

**Repeated observations.** Several lines may describe the same object, as in
the example above. They are recognised by their coordinates, within
`match_tol_arcsec` (default 2"), and their accumulators are summed before the
population average. This combines the repeated observations with their native
weights, i.e. naturally, preserving the intra-source sensitivity; the
democratic renormalisation is then applied to the resulting single
measurement, as required by Eq. (26). Nothing else identifies a source, so
there is no name column and no separate coordinate file to keep in sync.

---

## The continuum: three cases

This is the first decision, and it decides everything downstream: which
column is read, whether there are one or two fits, and which figures come out.
On the command line it is `--contsub before`, `--contsub after`, or nothing at
all, which is the default.

### `--contsub before` — remove it first, fit the two separately

Use it when the continuum is bright enough to distort the
baseline of the stacked line profile, or when a few sources have a much
brighter continuum than the rest.

A low-order polynomial is fitted to the line-free channels, separately for the
real and the imaginary part, and subtracted from every channel, spectral
window by spectral window. The fitted continuum goes to `MODEL_DATA` and the
line to `CORRECTED_DATA`; `DATA` is left untouched, so the step can be rerun.

Line and continuum are then compressed and fitted as two independent runs, on
two different columns, and each gets its own outputs:
`<tag>_line_*` from `CORRECTED_DATA` and `<tag>_continuum_*` from the
line-free channels of `MODEL_DATA`. Nothing is ever drawn on a shared axis.
The continuum size can be fed back as a prior on the line size with
`--theta-prior-from <tag>_continuum_results.json`.

### `--contsub after` — keep it, one fit for both

Use it when you want both quantities out of the same fit on the same data, or
when subtracting first would be awkward.

`ProfileConfig(joint_continuum=True)` adds a constant term to the shape fit on
the integrated spectrum *and* to the per-annulus least squares. The amplitude
of that term is the continuum flux density of the annulus, so the continuum
gets its own radial profile and its own uv fit, correlated with the line
because they came out of one solution. One run, one set of outputs, one
figure with three panels: line, continuum, spectrum.

### no flag — no continuum to fit

The default. The data are treated as line only, over the velocity window of
interest. Right when the continuum is genuinely absent or far below the noise,
which is the common case for a line stack; wrong, and badly so, when a real
continuum is present.

Leaving a real continuum in **without** the joint term is the one combination
to avoid: the shape fit has no baseline, so the Gaussian stretches to cover
the pedestal and the line flux comes out far too high. On a synthetic test
with a true flux of 4.0 and a true FWHM of 471 km/s, `--contsub after` returns
3.9 and 451 km/s while the default on the same data returns 10.7 and 1227
km/s. If the line FWHM comes out implausibly broad, this is the first thing to
check.

Over a bandwidth spanning the thermal dust emission a polynomial is not the
right function at all: in the Rayleigh-Jeans limit the continuum scales
roughly as `nu^4`, so fit it locally, on narrow windows.

---

## What to choose

### Always

| Parameter | Where | Notes |
|-----------|-------|-------|
| `rest_freq_ghz` | `LineConfig` | rest frequency of the line you are stacking, GHz. The only parameter with no default |
| `v_window_kms` | `LineConfig` | default `(-600, 600)`. Must contain the whole profile: it defines the line-free channels for the continuum fit and, with `method='window'`, the integration window |
| `scheme` | `WeightingConfig` | `natural` (default) or `democratic`. A scientific choice, not a technical one |
| the continuum | — | subtract it or fit it jointly, as above |

### Natural or democratic weighting

**`democratic`** rescales the internal weights of each source so
that they sum to one (Eq. 26), so every object contributes equally regardless
of its observational depth. It buys population representativeness at the cost
of formal sensitivity, because deep observations are no longer allowed to
drive the result. Use it when the question is the *average behaviour of the
ensemble*, and when the line is bright enough to stay well defined at lower
S/N.

**`natural`** (the default) keeps the native visibility weights, so the deepest observations
dominate. It gives the highest formal S/N but a combined estimate that may
fail to represent the population. Use it for emission too faint to be seen in
the individual sources.

Running both is cheap, since only step 4 has to be repeated, and the
difference is itself informative about the depth distribution of the sample.

### Flux normalisation

| Parameter | Default | When to change it |
|-----------|---------|-------------------|
| `redshift_rescaling` | `True` | Applies alpha (Eq. 27), transporting every source to `z_ref`. Essential over a broad redshift range, negligible over a narrow one |
| `z_ref` | `None` = sample median | **Fix it explicitly** when comparing different stacks, otherwise the reference moves with the sample |
| `flux_normalisation` | `False` | Rescales by `norm_ref / NORM` so that a bright minority cannot skew the stack. Needs the `NORM` column. Omit it when you want the total emission weighted by the true intrinsic luminosities |
| `norm_ref` | `None` = sample median | **Fix it explicitly**, same reason as `z_ref` |
| `already_applied` | `False` | `True` if the amplitudes on disk already carry alpha |
| `H0`, `Om0` | 67.4, 0.315 | only for consistency with a different paper |

### Continuum subtraction

| Parameter | Default | Notes |
|-----------|---------|-------|
| `order` | `0` | A constant: a narrow line-free bandwidth cannot constrain a slope, and imposing one introduces a spurious gradient. `1` for a line, `-1` to let the BIC choose per spectral window |
| `delta_bic` | `2.0` | Margin required to prefer order 1 |
| `require_positive_slope` | `True` | With `order=-1`, accept order 1 only for a rising continuum |
| `exclude_kms_lo/hi` | the line window | Widen or skew the mask where a neighbouring line or feature falls inside the fitted range |
| `exclude_v_kms` | `()` | Extra intervals, e.g. a known absorption feature |
| `chunk_rows` | `50 000` | Lower it if memory is tight |

### Radial binning

| Parameter | Default | Notes |
|-----------|---------|-------|
| `b_min_klambda`, `b_max_klambda` | 3, 3000 | Should bracket the shortest and longest rest-frame baselines in the stack |
| `n_bins` | 18 | More bins resolve `V(b)` better but leave fewer visibilities, hence less S/N, per annulus. Fixed at compression time: changing it means recompressing |
| `max_amplitude` | `None` | Discard visibilities above this amplitude; NaN and infinities are always discarded. A stacked MS can carry a few corrupt cells, and one of them poisons a whole annulus |

### Per-annulus flux

| Parameter | Default | Notes |
|-----------|---------|-------|
| `method` | `template` | `template` fits the shape once on the integrated spectrum, freezes centroid and width, and fits only the amplitude per annulus; the flux then follows analytically from Eq. (32), with no truncation and no cross-contamination between blended lines. `window` is the plain boxcar integration. `continuum` averages the line-free channels, for the continuum statistics |
| `joint_continuum` | `False` | Fit the line on top of a constant continuum, and measure both. Switch it on when the stack was not subtracted; also a good robustness test on a subtracted one |
| `fit_span_kms` | `1500` | Half-width used for the shape fit and the amplitude least squares |
| `shape_tie` | `free` | For a blend: `free` (6 parameters), `centroid`, `centroid+width`. Tie more when the second line is weak |
| `fixed_shape_kms` | `None` | Impose `(centroid, sigma)` and skip the shape fit |

### Source model

| Parameter | Default | Notes |
|-----------|---------|-------|
| `model` | `gauss` | Circular Gaussian, Eq. (33): gives `F_tot = V(b=0)` and `theta_FWHM`. Use `point` when the stack is unresolved, so a free Gaussian cannot overfit a size the data cannot constrain. `gauss2` only with many well-populated annuli |
| `theta_fixed_arcsec` | `None` | Freeze the size, fit the flux alone |
| `theta_max_arcsec` | `15` | Lower it to a physical value if the fit converges to the bound: check `theta_at_bound` and `bootstrap_fraction_at_bound` |
| `theta_prior` | `None` | `(theta_ref, sigma_dex)` soft tie, typically the continuum size of the same band with `sigma_dex ~ 0.15`. Breaks the flux-size degeneracy without freezing anything. From the CLI, `--theta-prior-from continuum_results.json` |
| `b_max_klambda` | `None` | Ignore the longest baselines, e.g. where only one configuration contributes |
| `min_sources_per_bin` | `2` | Drop annuli with fewer contributing sources. Raise it (3-5) on heterogeneous samples |
| `error_floor_frac` | `0.1` | Floor on the bootstrap errors as a fraction of their median, so a single annulus cannot dominate the chi square |
| `second_line_theta_tie_dex` | `0.2` | Soft tie of the second line size to the target size on the *same* bootstrap realisation, which propagates the correlation. `None` fits it freely |
| `continuum_theta_tie_dex` | `None` | The same for the continuum in joint mode. Leave it free when the point is to compare the two sizes |

### Uncertainties

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_realisations` | `500` | Enough for 1-sigma errors; increase it if you quote far tail percentiles |
| `seed` | `42` | Reproducibility |

---

## Output

`extract_flux` returns an `ExtractionResult` and, with `output_prefix`, writes:

- `<prefix>_results.json` — every fitted quantity and the configuration that
  produced it;
- `<prefix>_spectrum.txt` — `v_kms  Re_stack  err_bootstrap  n_sources`;
- `<prefix>_uvamp.txt` — `b_klambda  flux  err_bootstrap  flux_imaginary  n_sources`;
- `<prefix>_uvamp_second_line.txt`, `<prefix>_uvamp_continuum.txt` when those
  components are fitted.

### Figures

Which figures you get follows from how the continuum was handled, because that
is what decides whether line and continuum are two measurements or one.
`plot_all(result, line, prefix)` picks the right set, and so does `--plot`.

| Case | Figures |
|------|---------|
| `--contsub before` | `<tag>_line_uvamp.png`, `<tag>_line_spectrum.png`, `<tag>_continuum_uvamp.png` — two independent runs, two independent sets |
| `--contsub after` | `<tag>_joint.png` — one figure, three panels: line profile, continuum profile, spectrum with the line on the fitted continuum level |
| no flag | `<tag>_line_uvamp.png`, `<tag>_line_spectrum.png` |

The uv figures carry a lower panel with the number of sources contributing to
each annulus and the minimum below which an annulus was dropped; annuli
excluded from the fit are drawn open, and the imaginary part is drawn as grey
crosses. The spectrum covers the velocity range the data actually span.

The legends carry the fitted numbers and nothing else: flux with its error,
size with its error, and for the spectrum the centroid and the width.
Everything else is in the JSON, or in `fit_summary_lines(result)` as plain
text for a caption.

Pass `unit="Jy"` (or `--unit Jy`) to name the amplitude unit on the axes; the
default label is "data units", since the flux normalisation can rescale them.
`v_limits` (`--v-limits`) overrides the velocity range.

Fluxes are in `[data units] * km/s` for a line and `[data units]` for the
continuum; the flux normalisation rescales them by `norm_ref / NORM`.

### Things to check in the results

| Key | What it tells you |
|-----|-------------------|
| `null_test_max_imaginary_over_sigma` | The imaginary part must be consistent with zero: a nonzero component signals a source that is not correctly centred |
| `theta_at_bound`, `bootstrap_fraction_at_bound` | The size ran into `theta_max_arcsec`: the profile does not constrain it, switch to `point` or to a prior |
| `chi2_reduced` | Much larger than one: an inadequate source model, underestimated errors, or a residual continuum |
| `n_bins_in_fit`, `n_sources_min_in_window` | How much of the profile actually entered the fit |
| `n_bootstrap_failed` | Realisations whose fit did not converge; a large fraction makes the quoted errors unreliable |
| `flux_point_source` vs `flux_total` | A large ratio between the two is the signature of a resolved source |
| `n_sources` vs `n_entries` | How many lines of the input list were merged into physical sources: check it matches what you expect, and raise or lower `match_tol_arcsec` if not |
