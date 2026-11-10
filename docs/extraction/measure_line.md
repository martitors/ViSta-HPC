# Measuring the line

`ViSta.run()` measures the stacked lines on all channels and all annuli of
baseline length at once, with the noise measured from the data; the options
are those of `v.extract(...)` ([Parameters](../reference/parameters.md#extract-what-is-measured)).
`run(only="measure")` repeats the measurement from the statistics, in
seconds.

## The model

For every cell (channel `c`, annulus `b`) of the stack

```{math}
\mathrm{Re}\,V(c, b) = \Big[\sum_k F_k\,\phi_c(v_{0,k}, \mathrm{FWHM}_k)\Big]\,g(b;\theta)
```

with `φ` a Gaussian of unit integral *averaged over the channel* (a line a
few channels wide is not distorted), `F` the integrated line flux, and `g` the
visibility of a circular Gaussian of FWHM `θ` (1 for a point source), averaged
over the baselines of the annulus rather than taken at its mean. The sum runs
over the lines fitted together (those whose windows overlap, see
[Several lines](#several-lines)): every line `k` has its own flux, centre and
width, in its own velocity, and they share the size. The cells are
fitted by weighted least squares, each by the inverse of its variance; `F`,
the peak, `v₀`, the FWHM and `θ` of every line come out of one fit, with
their covariance.

Compared with measuring the shape on the spectrum averaged over the annuli
and then the amplitude annulus by annulus ([the per-annulus chain](extract_flux.md)),
the joint fit uses all the information at once, with the right weights: the
spectrum averaged over the annuli is dominated by the long baselines, where a
resolved source has the least signal.

## The noise

The noise is measured four ways, and compared in the results and in the
noise figure:

1. **from the weights**, `1 / Σw` — right only if the weights are absolute;
2. **from the scatter of the visibilities inside each cell**, thanks to the
   sums of squares stored by the compression:
   `Σw (x − x̄)² = Σw x² − (Σw x)² / Σw`, whose expectation is `(N − 1) κ²`.
   `κ` is the factor by which the weights misjudge the noise, measured for
   every dataset (`kappa_per_entry`: an MS with wrong weights stands out) and
   every channel (atmospheric lines, which a weight per row ignores). A
   single visibility has S/N far below 1, so the signal does not inflate it.
   **This is the noise used**;
3. from the **imaginary part** of the stack, zero for centred, symmetric
   sources;
4. from the **real part in the line-free channels**, after a subtraction.

3 and 4 must scatter like 2: a ratio above 1 gives a warning (the source is
not centred or not symmetric, a phase error, a residual continuum, an
unmasked line), and the thermal errors are optimistic there.

**The correlation between channels** has two causes: the spectral response
and the rebinning (Hanning smoothing: +0.67 at lag 1, +0.17 at lag 2), and
the continuum subtraction, which removes from every row a level fitted on the
line-free channels — a noisy level, shared by all the channels of the row: a
correlation at *every* lag, about 1 / (effective number of fit channels). On
a narrow band it can double the error of a flux summed over the line. The
whole channel × channel matrix is measured from the imaginary part of every
source in every annulus (same spectral response and same subtraction as the
real part, no line) and enters every error, S/N and null realisation. With
fewer than 30 such vectors a model of the first lags is used instead
(`channel_correlation_method`).

## Centre and width: limits

The centre and the width of every line are searched **within limits**,
`v0_range` and `fwhm_range` (km/s, in the velocity of that line):

| | default | `lo < hi` | `lo = hi` |
|---|---|---|---|
| centre | anywhere in the window | between the limits | fixed |
| width | two channels to the window | between the limits | fixed |

The fit stays exactly inside the limits. Give them when you know something:
a redshift accurate to ±100 km/s, a width from another line of the same
galaxies, a range of plausible widths. Narrower limits mean fewer chances for
noise to look like a line: a lower threshold for the same significance.

**The significance** is the best S/N of the search, calibrated on the same
search run on noise: `n_null` realisations (default 1000) with the measured
noise and channel correlation, extrapolated with the number of effective
trials (`p_value`, `significance_sigma`, `n_effective_trials`; with
`n_null=0`, a Bonferroni bound). In a search the fit always finds a bump:
`F / σ_F` is the precision of the flux, not the evidence that the line
exists — that is `significance_sigma`. With the centre, the width and the
size all fixed there is one template: the flux is a matched filter (linear,
unbiased) and its S/N is the significance (`fixed_shape: true`). Two more
checks: `snr_imaginary` (the imaginary part through the same filter, ≈ 0)
and `p_value_phase_null` (the search on stacks with a random phase per
source, which keeps the noise and scrambles the signal; it needs a few tens
of sources).

A free search is reliable from about 4σ; below that its flux follows noise
peaks, and the results warn: fix the width from another tracer, narrow the
limits, or quote the area flux.

## Several lines

With several rest frequencies (`v.stack(line_freq=[f1, f2, ...], line_window=[...])`,
one window each) every line is searched in its own window — its own detection
and significance. How the lines are then fitted follows from their windows,
with no option to set:

- **windows that overlap** (directly, or through another line: A with B and
  B with C) — the lines are blended, and are fitted together: one Gaussian
  each, with its own flux, centre and width, sharing the size, so the fit
  splits the channels they share. Their area fluxes hold part of each
  other, so `flux_measure="auto"` quotes the Gaussian ones.
- **windows apart** — every line is measured on its own, as if it were alone:
  on the channels near it (within `fit_span_kms`) and outside the windows of
  the other lines, with its own size and source model, its own bootstrap and,
  with `continuum="fit"`, its own continuum.

Windows that only touch at an edge are apart. The log says which lines went
together (`lines 1 + 2 fitted together`), and so do the results
(`line_groups`; `fitted_with_lines` in the block of every line, with the size
of its fit). Every line has its block in `lines`; `line` is the first.

## Errors

Two errors are computed for every quantity:

- **thermal**, from the fit with the measured noise and the channel
  correlation;
- **bootstrap** over the sources (`n_realisations`), which includes the
  scatter of the population.

The larger is quoted; their ratio is `bootstrap_over_thermal` (≈ 1 for a
uniform population, > 1 when the sources differ). With few sources the
bootstrap underestimates, and the thermal error covers it.

## The size

With `model="auto"` (default) both a point and a Gaussian source are fitted,
and the total flux (and the profile and the window fluxes) is their average
with Akaike weights; the spread between the two enters the error. A weak or
barely resolved stack does not tell its size, and its total flux depends on
it: quoting the flux of whichever model was chosen gave errors up to five
times too small in the simulations.

- `size_test` gives `χ²(point) − χ²(Gaussian)` and both fluxes. The
  Gaussian is called resolved (`size_test.adopted`) when it improves χ² by at
  least 9 (a 3σ preference; `resolved_delta_chi2`). The fits weight the cells
  by their noise alone, so their χ² treats correlated channels as
  independent: the difference is first divided by `correlation_factor`, the
  factor by which the channel correlation inflates it (4–5 with Hanning
  smoothing and a continuum subtraction; `delta_chi2_uncorrelated` is the raw
  difference). Without it, noise made point sources look resolved.
- The detection scan tries a point source and Gaussians of 0.5, 1 and 2 times
  the size that halves the visibility at the median baseline: for a resolved
  source a point template collects only part of the signal.
- For a weak resolved line, a size from elsewhere makes the total flux
  meaningful: `model_prior=(θ_ref, σ_dex)` ties it softly (e.g. to the
  continuum size of the same sources), `model=θ` fixes it. The point-source flux, always given, is a
  lower limit for an extended source.
- `model="point"` when the source is known to be unresolved: the joint fit
  then does not average in a Gaussian fitted to the noise.

## Two fluxes for every line

- The **profile** is the flux density at zero spacing in every channel: for a
  point source the noise-weighted mean over the annuli, for a resolved one
  each annulus corrected for the resolution first. It is the spectrum to show.
- The **Gaussian flux** (`flux_gauss`) is the integral of the fitted
  Gaussian: it uses every channel with the weight the line gives it, so it is
  the most precise, and it is right if the line is a Gaussian.
- The **area flux** (`flux_area`) is the sum of the profile over the window
  of the line, times the channel width: no assumption on the shape, but the
  noise of every channel of the window. Its error includes the correlation
  between channels and, for a resolved source, the uncertainty of the size.
  `flux_optimal` is the sum over 1.19 FWHM around the fitted centre (the best
  S/N for a Gaussian, 84% of its flux, not corrected).

**Which one is quoted** (`flux`, `flux_error`, `flux_method`) is set by
`flux_measure=`: `"gauss"`, `"area"`, or `"auto"` (default). `auto` compares the two:
their difference has an error `σ_Δ` computed from the same data (both are
linear in the cells to first order, so `σ_Δ² = (a_area − a_gauss)ᵀ C (a_area −
a_gauss)`, with `a` how each moves with the data and `C` the noise of the
cells; it is about `√(σ_area² − σ_gauss²)`). If `|Δ| > 2 σ_Δ` the line is not
a Gaussian — a double horn, wings, an asymmetry — and the area flux is
quoted, with a warning; otherwise the Gaussian one. The area is not used,
whatever the difference, when the window holds less than 95% of the fitted
Gaussian (too narrow for a flux without a shape) or overlaps the window of
another line (only the joint fit separates them). Everything is in
`shape_check`, and the profile figure shows both: the area of the Gaussian
hatched, the window shaded. The test is in units of the noise: it also
catches a width fixed too narrow (the area does not depend on it), and it
fires wrongly when the noise is wrong (from the weights when they are off,
or without the channel correlation).

On simulated lines at S/N ≈ 10 the Gaussian flux was within 1% for a
Gaussian line, 7% high for a double horn, 6% low with broad wings; the area
over a window of four FWHM was within 2% in all three, with an error 1.3
times larger. With the width free, a window of about 2.3 FWHM gives the area
the same error as the Gaussian.

## Residual continuum

After the fit, the profile outside every window (the channels of the
continuum) is fitted with an offset and a slope, `a + b v`, with the channel
correlation in their errors (`continuum_residual`). If either exceeds 3σ
(`detected: true`) the log warns: after a subtraction, continuum is left (a
slope fitted as a constant, wings of the line in the continuum channels, a
line not listed); without a subtraction (`stack(..., continuum="none")`), there
is continuum in the data and the line fluxes include it: subtract it, or fit
it with the lines (`stack(..., continuum="fit")`, a constant with the
size of the lines, in `continuum_fit`; the check then looks at what it
leaves). The figure draws it.

## The parameters

```python
v.extract(v0_range=(-150, 150),        # km/s; (0, 0): fixed
          fwhm_range=(100, 600),       # km/s; (300, 300): fixed
          model="auto",                # "point", "gauss", or a size in arcsec
          model_prior=None,            # (0.8, 0.15): soft size prior
          flux_measure="auto",         # "gauss" or "area"
          window=None,                 # window of the area flux; default the line window
          n_bootstrap=500, n_null=1000)
```

[Example 4](../examples/04_measuring_the_stacked_line) and
[Example 6](../examples/06_what_each_option_changes) show them on a simulated
sample, against the true signal.
