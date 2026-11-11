# Weighting and normalisation

**In one sentence.** How the galaxies are combined is a scientific choice;
it is made when the stack is measured, never on the data on disk, so the
same stack can be measured in every way.

```python
v.extract(weighting_scheme="natural",          # or "democratic"
          z_norm=True, z_ref=None,             # None: the median, reported
          pb_corr=True,
          flux_norm=False, flux_ref=None)      # None: the median, reported
```

Changing one of these and running `v.run(only="measure")` again takes
seconds. [Example 3](../examples/03_the_window_the) and
[Example 7](../examples/07_a_few_galaxies_that_count) show each of them.

## Natural or democratic

| | What it does | When |
|---|---|---|
| **`natural`** (default) | keeps the weights of the visibilities: the deepest observations count most | emission too faint to be seen in the single galaxies. The lowest noise |
| **`democratic`** | rescales the weights so that every galaxy counts the same (Eq. 26 of the paper) | the *average* of the sample, when the line is bright enough. More noise |

**What to watch.** With natural weighting a few galaxies observed much
deeper than the others decide the result. If they are also different (a
lensed galaxy, a much brighter one), the stack looks like them and not like
the sample. The difference between the two schemes tells you how uneven the
depths are.

## The redshift correction

**What.** With `z_norm=True` (default) every galaxy is brought to the flux
it would have at one redshift, `z_ref`, by the factor (Eq. 27)

```{math}
\alpha = \left(\frac{D_L(z)}{D_L(z_\mathrm{ref})}\right)^2 \frac{1 + z_\mathrm{ref}}{1 + z}.
```

**Why.** The same galaxy is fainter when it is farther. After the
correction the stack measures the flux that a galaxy of the mean luminosity
would have at `z_ref`. It matters over a broad redshift range (between
z = 2 and 3 the factor changes by about 2), little over a narrow one.

**`z_ref`.** By default the median of the sample, which moves when the
sample changes: **give it** when you compare stacks. The value used is
always printed after the measurement, and it is in the results (`z_ref`,
`z_ref_given`).

Nothing on disk is rescaled: `α` is applied to the statistics of every
galaxy when they are combined. Cosmology: `H0`, `Om0` (67.4, 0.315).

## The primary beam

**What.** An antenna is less sensitive away from the centre of its field.
With `pb_corr=True` (default) every galaxy is corrected for the gain `A` at
its position: amplitudes divided by `A`, weights multiplied by `A²`.

**Why.** A galaxy off-centre is seen fainter than it is. The correction
restores its flux, and gives it less weight, because its noise grows by the
same factor.

`A` is that of a Gaussian beam of FWHM `1.2 c / (ν D)` at the observed
frequency of the line, with the dish diameter of the original Measurement
Set. It is computed during the compression, which therefore needs the
original Measurement Sets.

## The flux normalisation

**What.** `flux_norm=True` rescales every galaxy by `flux_ref / NORM`.
`NORM` is the last column of the input list: a magnification, a luminosity,
a continuum flux, anything that scales with the stacked emission.

**Why.** So that a bright minority does not dominate the stack. With the
magnification of lensed galaxies as `NORM` and `flux_ref=1`, the stack is
that of the unlensed galaxies
([Example 7](../examples/07_a_few_galaxies_that_count)).

Every line of the list must have a `NORM`, otherwise `extract()` refuses.
`flux_ref` defaults to the median and is reported like `z_ref` (`flux_ref`,
`flux_ref_given`): **give it** when you compare stacks. Leave the
normalisation off when you want the emission weighted by the true
luminosities.

## The same galaxy observed several times

Lines of the input list with the same coordinates (within
`match_tol_arcsec`, 2″) are the same galaxy. Their statistics are added
before the average, each with its weights, and the democratic weighting
then counts the galaxy once. `n_sources` and `n_entries` in the results say
how many galaxies the lines were merged into.
