# Weighting and normalisation

How the sources are combined is a scientific choice, made when the stack is
measured and never applied to the visibilities on disk: the same stack can be
measured with every scheme, and comparing them is cheap.

```python
WeightingConfig(scheme="natural",          # or "democratic"
                redshift_rescaling=True, z_ref=None,
                primary_beam=True,
                flux_normalisation=False, norm_ref=None)
```

## Natural or democratic

- **`natural`** (default) keeps the native visibility weights: the deepest
  observations dominate, the formal S/N is the highest, and the result may
  not represent the population. Use it for emission too faint to be seen in
  the individual sources.
- **`democratic`** rescales the weights of each source so that they sum to
  one (Eq. 26 of the paper): every source counts the same, whatever its
  depth. It buys the representativeness of the population at the cost of
  sensitivity. Use it when the question is the *average behaviour* of the
  sample and the line is bright enough.

The difference between the two is itself informative about the depth
distribution of the sample.

## Redshift rescaling

With `redshift_rescaling=True` (default) every source is transported to a
common redshift `z_ref` by the factor (Eq. 27)

```{math}
\alpha = \left(\frac{D_L(z)}{D_L(z_\mathrm{ref})}\right)^2 \frac{1 + z_\mathrm{ref}}{1 + z},
```

so that the stack measures the flux a source of the mean luminosity would
have at `z_ref`. It matters over a broad redshift range (at z = 2–3 the
factor spans about a factor of 2), little over a narrow one. `z_ref`
defaults to the median of the sample, which moves with the selection: **fix
it** when comparing stacks. `already_applied=True` if the amplitudes on disk
already carry `α`. Cosmology: `H0`, `Om0` (67.4, 0.315).

## Primary beam

With `primary_beam=True` (default) every line of the input list is
corrected for the primary beam at the position of its source: amplitudes
divided by the gain `A`, weights multiplied by `A²`, so an attenuated
source is corrected and also counts less. `A` is that of a Gaussian beam of
FWHM `1.2 c / (ν D)` at the observed frequency of the line, with the dish
diameter of the original MS, computed during the compression (which
therefore needs the original Measurement Sets).

## Flux normalisation

`flux_normalisation=True` rescales every source by `norm_ref / NORM`, with
`NORM` the last column of the input list (a luminosity, a continuum flux,
any proxy that correlates with the stacked emission), so that a bright
minority cannot dominate the stack. `norm_ref` defaults to the median:
**fix it** for the same reason as `z_ref`. Omit the normalisation when you
want the emission weighted by the true intrinsic luminosities.

## Repeated observations

Lines of the input list with the same coordinates (within
`match_tol_arcsec`, 2″) are the same source: their statistics are added
before the average, which combines them naturally (with their own weights),
and the democratic renormalisation then treats the source as one
measurement. `n_sources` and `n_entries` in the results say how many sources
the lines were merged into.
