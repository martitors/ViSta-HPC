# Post-processing and uv-domain signal extraction

`vista.extract` recovers the stacked flux density from the stack produced by
`vista.ViSta` (a Zarr store by default, or a Measurement Set), working directly
in the visibility domain.

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
| 1. Stacking | `ViSta.run` | the input datasets | `dask-ms`, `zarr`, `ms_ops` |
| 2. Continuum subtraction | `subtract_continuum` | the stack, read/write | `zarr` or `python-casacore`; `ms_ops` (optional, much faster) |
| 3. Compression | `compress_visibilities` | the stack, read-only | `zarr` or `python-casacore`; `ms_ops` (optional, much faster) |
| 4. Extraction | `measure_line` (joint fit) or `extract_flux` (Sec. 5.3) | the compressed statistics | `numpy`, `scipy` |

Steps 2 and 3 can run **during step 1** (`ViSta.run(..., continuum=...,
stats_line=...)`, the default of `run_vista.sh`): see [One pass](#one-pass).
Only steps 2 and 3 touch the stack. Step 4 works on a file of a few MB, so it
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

subtract_continuum("stack.zarr", line, ContinuumConfig(order=0))

compress_visibilities("stack.zarr", "input_list.txt", line,
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
window by spectral window. Only the unflagged samples enter the fit: the
channels of the common grid that a dataset does not cover are flagged and
hold 0, and counting them would pull the continuum towards zero. The fitted
continuum goes to `MODEL_DATA` and the line to `CORRECTED_DATA`; `DATA` is
left untouched, so the step can be rerun.

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

## One pass

The continuum subtraction and the compression work row by row on the output
grid, so they can run on each chunk right after the stacking kernel, before
it is written:

```python
ViSta("input_list.txt").run("stack.zarr", central_freq=345.7959899e9,
                            line=line, continuum=ContinuumConfig(order=0),
                            stats_line="line_stats.npy",
                            stats_continuum="cont_stats.npy")
```

The stack and the statistics are those of the separate steps (the columns bit
for bit; the statistics bit for bit on Zarr, to 1e-15 on an MS), and the
stack is read once instead of three times. `DATA` is kept: after a look at
the profile the subtraction can be redone with another window
(`subtract_continuum`, or `run_vista.sh --skip-stack ...`) and the statistics
recomputed, without stacking again. The parameters of every subtraction are
recorded in the stack and in the statistics, and a new one says which it
replaces.

A good sequence when the line window is uncertain:

1. stack in one pass with a generous window (`v_window_kms`), e.g. ±1.5 times
   the expected width;
2. look at `<prefix>_profile.png`: the residuals in the channels used for the
   continuum must be noise (the `±1σ` band, `line_free_rms_over_noise` ≈ 1);
3. if another line, or a wing, sits there, add it to `exclude_v_kms` (or
   `second_rest_freq_ghz` if you know what it is: an identified line is
   safer than an interval chosen by eye, which can bias the result), and redo
   steps 2-3 on the stack.

## Measuring the line (`measure_line`, `--method joint`)

`measure_line` fits all channels and annuli of the stacked cube at once with

    Re V(c, b) = F φ_c(v₀, FWHM) g(b; θ)   [+ F₂ φ₂_c g] [+ C g]

φ a Gaussian of unit integral averaged over the channel, g the visibility of a
circular Gaussian (1 for a point source), by weighted least squares with the
noise measured from the data, the correlation between channels entering every
error (it is not inverted: Hanning smoothing makes it singular). Every
quantity — flux F, peak, centre, FWHM, size — comes from the same fit and has
its error from the same covariance; the bootstrap over the sources adds the
population scatter.

Compared with the chain of Sec. 5.3 (shape on the spectrum averaged over the
annuli, amplitude annulus by annulus, uv fit), it uses all the information at
once, with the right weights: the spectrum averaged over the annuli is
dominated by the long baselines, where a resolved source has the least
signal, so the shape was measured on its noisiest version.

### The noise

The compression stores, for every cell, `Σw Re²` and `Σw Im²` too, so the
scatter of the visibilities inside the cell gives its variance directly:

    Σw (x − x̄)² = Σw x² − (Σw x)² / Σw,   E[·] = (N − 1) κ²

κ is the factor by which the weights misjudge the noise. It is measured for
every entry (`kappa_per_entry`: an MS with wrong weights stands out) and every
channel (atmospheric lines, which a `WEIGHT` per row ignores). A visibility
has S/N far below 1, so the signal does not inflate it. Three more estimates
check it, all expected to be 1 in the results:

| Key | Estimate | Off 1 means |
|-----|----------|-------------|
| `kappa_scatter_over_weights` | scatter / weights | the weights are not absolute (not an error: the scatter is used) |
| `imaginary_rms_over_noise` | Im of the stacked cells / noise | sources not centred, asymmetric, or phase errors |
| `line_free_rms_over_noise` | Re in the line-free channels / noise (after a subtraction) | residual continuum, baseline ripples, an unmasked line |
| `bootstrap_over_thermal` (in `line`) | bootstrap / thermal error of F | > 1: scatter of the population (or systematics); < 1 with few sources: the bootstrap underestimates |

The correlation of the noise between channels is measured as a whole matrix
from the imaginary part of every source in every annulus (one noise vector
each, with the spectral response and the continuum subtraction of the real
part, and no line) and used in every error. It has two parts: the spectral
response and the rebinning (Hanning smoothing gives +0.67 at lag 1, +0.17 at
lag 2), and the continuum subtraction, whose fitted level is noisy and shared
by all the channels of a row — a correlation at every lag, about 1 / (the
effective number of fit channels), which on a narrow band can double the
error of a flux summed over the line (`channel_correlation_far_in_line_window`
gives it). `channel_correlation` summarises the first lags,
`channel_correlation_method` says how it was obtained (with fewer than 30
noise vectors, a model of the first lags only).

### Detection and significance

| | `mode="detect"` | `mode="search"` (default) |
|---|---|---|
| centre, width | fixed (`v0_kms`, `fwhm_kms`) | free in the line window |
| flux | matched filter: linear, unbiased | from the free fit |
| significance | its S/N | the best S/N of the search, calibrated on the same search run on noise (`p_value`, `significance_sigma`, `n_effective_trials`) |
| use it | a weak line whose width is known from elsewhere | the line is clear, or its width is unknown |

In a search, the fit always finds a bump: `F / σ_F` is the precision of the
flux, not the evidence that the line exists. That is `significance_sigma`.
Two more checks: `snr_imaginary` (the imaginary part through the same
filter: ≈ 0) and `p_value_phase_null` (the search on stacks with a random
phase per source, which keeps the noise and scrambles the signal; it needs a
few tens of sources).

### Profile, window flux, size

- `<prefix>_profile.txt` / `.png`: flux density at zero spacing per channel
  (for a point source the noise-weighted mean over the annuli; for a resolved
  one each annulus is corrected for the resolution first), with the thermal
  and bootstrap errors, the imaginary part and the model. Its integral is the
  line flux. Plot it rebinned with `--plot-rebin N`.
- `window`: the sum of the profile in the line window (model independent,
  defined in advance, so unbiased) and in 1.19 FWHM around the fitted centre
  (the best S/N for a Gaussian, 84% of the flux; defined on the data). Their
  errors include the correlation between channels.
- resolved sources: the detection scan includes Gaussian templates (0.5, 1
  and 2 times the size that halves the visibility at the median baseline),
  since a point template collects only part of their signal; the model is
  averaged over each annulus rather than taken at its mean baseline; the
  window flux includes the uncertainty of the size.
- with `size="auto"` the flux, the profile and the window fluxes are the
  average of the point and the Gaussian fit, with Akaike weights
  (`size_test.weight_gauss`), and the spread between the two is in the error:
  a weak or barely resolved stack does not tell its size, and its total flux
  depends on it. For a weak resolved line give a size from elsewhere,
  `theta_prior_arcsec=(θ_ref, σ_dex)` (`--theta-prior`, `--theta-prior-from`),
  e.g. the continuum size of the same sources.
- `size_test`: χ² of a point and of a Gaussian source; with `size="auto"` the
  Gaussian is adopted only if it improves χ² by 9 or more (3σ). At low S/N a free
  size scatters high and drags the extrapolated flux with it. The fits weight
  the cells by their noise alone, so their χ² treats correlated channels as
  independent: the difference is divided by `correlation_factor`, the
  variance of the data along the direction that only the Gaussian can follow
  with the measured channel correlation, over the same without it (4-5 on
  the ALMA simulations below, with Hanning smoothing and the continuum
  subtraction; `delta_chi2_uncorrelated` is the raw difference). Without it
  noise made 2 of 6 point sources look resolved at Δχ² = 15-21.

### How well it works

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

These tables were made before the size test was corrected for the channel
correlation (Sec. *Profile, window flux, size*). With the correlation of these
simulations (0.3 at lag 1, a factor of about 1.6 on the χ² difference) the
correction changes them within their noise — on 50 new realisations each,
before → after: point source, bias +0.8% → −0.1%, scatter 0.21 → 0.20;
Gaussian 1″ at S/N 8, +3.5% → +3.2%, coverage 0.68 → 0.70; a very weak 1″
source (S/N ≈ 2), +15% → +9%, coverage 0.78 → 0.68. It matters with Hanning
smoothing and a continuum subtraction, where the factor is 4-5 (below).

### uv plane or image plane

The same stacks measured both ways: 8 simulated ALMA datasets (43 antennas,
baselines up to ~370 m, beam 1.06″ × 0.76″, Hanning-smoothed noise, weights
wrong by 1.3), a line of 200 km/s FWHM, stacked in one pass; in the image
plane, `tclean` of the stacked visibilities over the line window (±300 km/s,
natural weighting, CLEAN to 2σ in a 4″ mask). Mean of measured / true flux
and its scatter over 6 noise realisations per case (so the scatter is known to
about ±30%):

| Method | point source, S/N ≈ 6 | 1″ source, S/N ≈ 8 | 1″ source, S/N ≈ 5 |
|---|---|---|---|
| uv, joint fit, detect | 1.09 ± 0.16 | **1.05 ± 0.12** | 1.16 ± 0.29 |
| uv, joint fit, search | 1.20 ± 0.28 | 1.04 ± 0.14 | 1.12 ± 0.35 |
| uv, window ±300 km/s | 1.20 ± 0.26 | 0.99 ± 0.17 | 1.09 ± 0.41 |
| uv, point-source fit only | **1.00 ± 0.12** | 0.44 ± 0.05 | 0.48 ± 0.11 |
| image, dirty peak | 1.11 ± 0.17 | 0.42 ± 0.07 | 0.52 ± 0.14 |
| image, aperture 1″ / 2″ / 3″ | 1.14 / 0.77 / 0.82 ± 0.37 / 0.63 / 0.78 | 0.74 / 1.09 / 0.87 ± 0.10 / 0.25 / 0.67 | 0.87 / 0.89 / 0.43 ± 0.35 / 0.63 / 1.41 |
| image, aperture 2″, JvM-corrected | 0.78 ± 0.52 | 0.95 ± 0.22 | 0.76 ± 0.57 |
| image, CLEAN model total | 0.58 ± 0.23 | 0.39 ± 0.40 | −0.03 ± 0.67 |
| image, `imfit` Gaussian | 1.41 ± 1.05 | 1.20 ± 0.25 | 4.2 ± 3.1 (2 of 6 failed) |

- **Point source**: the two planes measure the same thing. With natural
  weighting the dirty image at the source position is the weighted mean of
  the real part of the visibilities, i.e. the uv point-source flux (on a
  noiseless stack: 209 against 212). The point-source fit has the smaller
  scatter only because it weights the channels with the line shape instead
  of averaging a ±300 km/s window, which a fit of the spectrum of the peak
  pixel in a cube would do as well. With `size="auto"` the joint fit does
  not know that the source is unresolved and averages in a Gaussian fitted
  to the noise, which costs scatter (0.16 against 0.12; the +9% here is
  within the noise of 6 runs: on 50 simulated point sources the bias is
  0.0% ± 3%). If the source is known to be unresolved, `size="point"`.
- **Resolved source**: the peak and the point-source flux lose 55-60%; in the
  image the total flux needs an aperture, and the result depends on it
  (1″ loses a quarter, 3″ is dominated by noise; the dirty-beam wings make
  the raw aperture sum high, the JvM correction brings it back). CLEAN to 2σ
  of a stack at S/N 5-8 leaves most of the flux in the residual. The joint
  uv fit has no aperture to choose and half the scatter of the best image
  method.
- **Weak resolved source**: every total flux is uncertain because the size
  is; the uv fit gives the honest error, `imfit` fails or runs away. A size
  from elsewhere (`theta_prior_arcsec`) is the way to tighten it.

This agrees with Lindroos et al. (2015, MNRAS 446, 3502), who find
uv-stacking more robust than image stacking, with better size estimates.
The image is still the place to look at the source: a cube of the stacked
visibilities (`export_ms`, then `tclean`) shows where the emission is and
whether it is symmetric, which the uv fit assumes.

### Output files and figures

With `output_prefix` (or `--output`), `measure_line` writes:

| File | Content |
|------|---------|
| `<prefix>_results.json` | every number: `line`, `detection`, `window`, `size_test`, `noise`, the `contsub` record |
| `<prefix>_profile.txt` | per channel: `v_kms freq_GHz flux_density err_thermal err_bootstrap imaginary model fitted line_free n_sources` |
| `<prefix>_uvamp.txt` | per annulus: `b_klambda line_flux err_thermal used n_sources` (line flux at the fitted line shape) |
| `<prefix>_noise.txt` | per channel: noise of the profile from the scatter of the visibilities and from the weights |
| `<prefix>_diagnostics.npz` | `channel_correlation` (the matrix), `z_imaginary`, `z_line_free` (cells over their noise), `null_snr`, `null_snr_phase` (null distributions of the detection S/N) |

`--plot` makes `_profile.png`, `_uv.png`, `_noise.png` (`plots_line.py`).
**To make them in your own style**, `examples/plot_line.py` draws the same
three figures from these files alone, with only numpy and matplotlib (it does
not import ViSta, so it can be copied next to the paper):

    python examples/plot_line.py <prefix> --unit Jy --rebin 2 --vlim -800 800 --format pdf
    python examples/plot_line.py <prefix> --style thesis.mplstyle --only profile

Colours, line styles, sizes and fonts are in the `STYLE` dictionary at the top
(or in a matplotlib style file, `--style`, applied on top); each figure is one
function (`plot_profile`, `plot_uv`, `plot_noise`) that returns the
matplotlib figure, commented on what every element is and how it is computed —
e.g. the rebinned errors use the channel correlation matrix,
`var(mean) = Σᵢⱼ Rᵢⱼ σᵢ σⱼ / n²`, not `σ / √n`. From Python:

    import plot_line as pl
    pl.STYLE["model"]["color"] = "C2"
    pl.apply_style()
    fig = pl.plot_profile(pl.load("<prefix>"), unit="Jy", rebin=2)
    fig.axes[0].set_ylim(-3, 6); fig.savefig("line.pdf")

Without `_diagnostics.npz` (outputs of earlier runs) the noise histograms are
replaced by the rms values of the JSON and the rebinned errors ignore the
correlation.

With `n_null=0` and more than one template (a search, or a size scan in
`detect`), the p-value is a Bonferroni bound over the templates
(`p_value_method`): conservative, no simulation needed.

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
| `primary_beam` | `True` | Corrects every line of the input list for the primary beam at the source position (Gaussian beam, FWHM = 1.2 c / ν D, D read from the original MS): amplitudes / A, weights × A², so an attenuated pointing is corrected and also counts less. A is computed during the compression, which therefore needs the original MSs to be readable. `--no-primary-beam` switches it off |
| `H0`, `Om0` | 67.4, 0.315 | only for consistency with a different paper |

### Continuum subtraction

| Parameter | Default | Notes |
|-----------|---------|-------|
| `order` | `0` | A constant: a narrow line-free bandwidth cannot constrain a slope, and imposing one introduces a spurious gradient. `1` for a line, `-1` to let the BIC choose per spectral window |
| `delta_bic` | `2.0` | Margin required to prefer order 1 |
| `require_positive_slope` | `True` | With `order=-1`, accept order 1 only for a rising continuum |
| `exclude_kms_lo/hi` | the line window | Widen or skew the mask where a neighbouring line or feature falls inside the fitted range |
| `exclude_v_kms` | `()` | Extra intervals, e.g. a known absorption feature |
| `chunk_rows` | `50 000` | Rows read at a time from an MS, at most; fewer for wide spectra (about 256 MB per chunk). A Zarr store is read by its own chunks |

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
- `<prefix>_spectrum.txt` — `v_kms  Re_stack  err_bootstrap  n_sources`, the
  stacked visibility average, whose amplitude is diluted by the long
  baselines and is not a flux;
- `<prefix>_spectrum_zerob.txt` — `v_kms  flux_density  err_bootstrap
  n_sources`, the same line in flux units, extrapolated to zero baseline;
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

`<tag>_lineprofile.png` is the line as an image-plane fit would show it: for
every channel the flux is measured on the whole radial profile with the fitted
size held fixed, so the resolution of each annulus is divided out rather than
averaged in. The y axis is a flux density and the area under the Gaussian is
the total line flux, which makes it the spectrum to publish. The `Re(V)`
spectrum next to it is the input of the shape fit, not a flux.

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

With `measure_line` (`--method joint`):

| Key | What it tells you |
|-----|-------------------|
| `detection.significance_sigma`, `p_value` | Is there a line at all. In a search, this — not `F / σ_F` — is the evidence |
| `noise.imaginary_rms_over_noise`, `line_free_rms_over_noise` | ≈ 1. Above: centring, calibration, residual continuum or an unmasked line; the thermal errors are optimistic there |
| `noise.kappa_per_entry` | One entry far from the others: its weights are off (harmless, the scatter is used) or its data are bad |
| `noise.channel_correlation_method` | Whether the whole correlation matrix was measured (needs ≥ 30 source × annulus vectors) |
| `line.bootstrap_over_thermal` | ≈ 1 for a uniform population; > 1 the population scatters (or systematics); < 1 with few sources |
| `detection.snr_imaginary` | The imaginary part through the line filter: ≈ 0 |
| `size_test` | Point against Gaussian: Δχ² and both fluxes |
| `line.chi2_reduced` | ≈ 1; much larger: the model (one Gaussian, separable in velocity and baseline) does not describe the data |
| `warnings` | e.g. a weak line in a search, whose free-shape flux is unreliable |
