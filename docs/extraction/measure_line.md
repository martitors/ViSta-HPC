# Measuring the line

`measure_line` (`run_vista.sh --method joint`, the default) fits the stacked
line on all channels and all annuli of baseline length at once, with the
noise measured from the data.

## The model

For every cell (channel `c`, annulus `b`) of the stack

```{math}
\mathrm{Re}\,V(c, b) = F\,\phi_c(v_0, \mathrm{FWHM})\,g(b;\theta)
\;\;[+\,F_2\,\phi_{2,c}\,g]\;\;[+\,C\,g]
```

with `φ` a Gaussian of unit integral *averaged over the channel* (a line a
few channels wide is not distorted), `F` the integrated line flux, and `g` the
visibility of a circular Gaussian of FWHM `θ` (1 for a point source), averaged
over the baselines of the annulus rather than taken at its mean. A second
line and a continuum term are optional. The cells are fitted by weighted
least squares, each by the inverse of its variance; `F`, the peak, `v₀`, the
FWHM and `θ` come out of one fit, with their covariance.

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

## Detect or search

| | `mode="detect"` | `mode="search"` (default) |
|---|---|---|
| centre, width | fixed (`v0_kms`, `fwhm_kms`), e.g. from another tracer | free in the line window |
| flux | matched filter: linear, unbiased | from the free fit |
| significance | its S/N | the best S/N of the search, calibrated on the same search run on noise (`p_value`, `significance_sigma`, `n_effective_trials`) |
| use it | a weak line whose width is known | the line is clear, or its width is unknown |

In a search the fit always finds a bump: `F / σ_F` is the precision of the
flux, not the evidence that the line exists — that is `significance_sigma`.
The null distribution comes from `n_null` realisations of the search on
noise (default 1000), extrapolated with the number of effective trials; with
`n_null=0` and several templates, a Bonferroni bound is used. Two more
checks: `snr_imaginary` (the imaginary part through the same filter, ≈ 0)
and `p_value_phase_null` (the search on stacks with a random phase per
source, which keeps the noise and scrambles the signal; it needs a few tens
of sources).

A free search is reliable from about 4σ; below that its flux follows noise
peaks, and the results warn.

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

With `size="auto"` (default) both a point and a Gaussian source are fitted,
and the total flux (and the profile and the window fluxes) is their average
with Akaike weights; the spread between the two enters the error. A weak or
barely resolved stack does not tell its size, and its total flux depends on
it: quoting the flux of whichever model was chosen gave errors up to five
times too small in the simulations.

- `size_test` gives `χ²(point) − χ²(Gaussian)` and both fluxes. The
  Gaussian is called resolved (`size_test.adopted`) when it improves χ² by at
  least `resolved_delta_chi2` (9, a 3σ preference). The fits weight the cells
  by their noise alone, so their χ² treats correlated channels as
  independent: the difference is first divided by `correlation_factor`, the
  factor by which the channel correlation inflates it (4–5 with Hanning
  smoothing and a continuum subtraction; `delta_chi2_uncorrelated` is the raw
  difference). Without it, noise made point sources look resolved.
- The detection scan tries a point source and Gaussians of 0.5, 1 and 2 times
  the size that halves the visibility at the median baseline: for a resolved
  source a point template collects only part of the signal.
- For a weak resolved line, a size from elsewhere makes the total flux
  meaningful: `theta_prior_arcsec=(θ_ref, σ_dex)` ties it softly (e.g. to the
  continuum size of the same sources, `--theta-prior-from`),
  `theta_fixed_arcsec` fixes it. The point-source flux, always given, is a
  lower limit for an extended source.
- `size="point"` when the source is known to be unresolved: the joint fit
  then does not average in a Gaussian fitted to the noise.

## Profile and window fluxes

- The **profile** is the flux density at zero spacing in every channel: for a
  point source the noise-weighted mean over the annuli, for a resolved one
  each annulus corrected for the resolution first. Its integral is the line
  flux, and it is the spectrum to show.
- The **window fluxes** are the sum of the profile in the line window (model
  independent, defined in advance, unbiased) and in 1.19 FWHM around the
  fitted centre (the best S/N for a Gaussian, 84% of the flux, defined on the
  data). Their errors include the correlation between channels, and for a
  resolved source the uncertainty of the size. For a line that is not a
  Gaussian (a double horn) the window flux is the one to quote.

## The parameters

All the options are in [`LineFitConfig`](../reference/parameters.md#linefitconfig-the-joint-fit);
the most used:

```python
LineFitConfig(mode="detect", v0_kms=0.0, fwhm_kms=300.0,    # matched filter
              size="auto",                                   # point, gauss, auto
              theta_prior_arcsec=None,                       # (0.8, 0.15): soft size prior
              noise="scatter",                               # or "weights"
              channel_correlation=True,
              n_null=1000, n_null_phase=200)
```

[Example 4](../examples/04_measuring_the_stacked_line) and
[Example 6](../examples/06_what_each_option_changes) show them on a simulated
sample, against the true signal.
