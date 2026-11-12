# The continuum

**In one sentence.** The continuum of the stack is averaged over the
channels without line in every ring of baseline length, and a source model
(a Gaussian or a point) is fitted to those averages: its value at zero
baseline is the flux density, its width the size.

## Where the continuum comes from

- **After a subtraction** (`continuum="subtract"`, the default). While the
  stack is written, a constant (or a slope) is fitted to every visibility on
  the channels **outside the line windows**, and subtracted. What was
  subtracted is kept (`MODEL_DATA`), and that is what is measured here.
  With a constant, it is exactly the mean of the channels without line.
- **Without any line** (`v.stack()` without `line_freq`). Every channel is
  continuum: see [The continuum alone](../stacking/continuum.md#the-continuum-alone).

Measuring what was subtracted, and not the channels again, has two
advantages: with a slope, the continuum is the one *under the line*, not at
the mean frequency of the free channels; and the number measured is exactly
what was taken out of the line.

## What you get

```{image} ../_static/fig_continuum.png
:alt: The stacked continuum: the channels without line and the level measured
:width: 70%
:align: center
```

`result.plot("continuum")`: the channels outside the line window (black),
the window left out (orange) and the level measured (blue), with its S/N.

```{image} ../_static/fig_continuum_uv.png
:alt: The continuum against the baseline length, with the two source models
:width: 60%
:align: center
```

`result.plot("continuum_uv")`: the continuum in every annulus of baseline
length, the Gaussian source (red) and the point source (green, dashed), in
bold the one used. Grey crosses: the imaginary part, a null test.

```python
result["continuum"]
# {'value': 8.9e-05, 'error': 4.5e-05, 'unit': 'Jy', 'snr': 2.0,
#  'treatment': 'subtracted, measured on its own',
#  'size': {'value': 0.95, 'unit': 'arcsec'},
#  'model': {'chosen': 'gauss', 'size_from': 'line', 'delta_chi2': 0.6, ...}}
```

## Which source model it gets

`continuum_model` chooses. With the default, `"auto"`, the rule is the one
of the line, and the continuum never contradicts the line:

1. **Its own Gaussian**, when it fits better than a point source by
   Δχ² ≥ 9: the continuum measures its size.
2. Otherwise **a Gaussian with the size of the line**, when the line is
   extended. The galaxies are the same: the continuum cannot be a point.
   This assumes that gas and dust have the same extent; the report says
   that the size is taken from the line.
3. Otherwise **a point**.

| `continuum_model` | The source |
|---|---|
| `"auto"` (default) | the rule above |
| `"point"`, `"gauss"` | always that one (its own Gaussian) |
| `"line"` | the model and the size measured on the line |
| a number | a Gaussian of that size (FWHM, arcsec) |

**Why not always its own Gaussian?** A weak continuum fits a point and a
Gaussian equally well. The size that comes out is then noise, and the flux
depends on it. A point, on the other hand, is too low for a source that is
extended. [Example 5](../examples/05_a_weak_continuum_under_a) compares the
three choices on a simulated sample.

**The two sizes.** The size measured on the line and on the continuum, each
on its own, are always reported side by side, with their difference in
units of its error:

```text
Sizes      each on its own: line 0.95 ± 0.12 arcsec, continuum 0.76 ± 1.09 arcsec: compatible (0.2 sigma apart)
```

If they differ clearly, gas and dust do not have the same extent, and
neither should borrow the size of the other: ask each for its own
(`model="gauss"`, `continuum_model="gauss"`).

**Only the continuum, again.** `v.extract(continuum_model=..., measure="continuum")`
repeats the continuum alone; `measure="line"` the lines alone.

## How it is measured

1. **The stack**: the statistics of the sources are combined with the same
   weighting, redshift rescaling and primary-beam correction as the line
   ([Weighting](weighting)).
2. **The mean per annulus**: over the channels that every source covers,
   annulus by annulus. Annuli with fewer than two sources are left out.
3. **The source model**: fitted to those means against the baseline length,
   every annulus weighted by its thermal noise.
4. **The errors**: two, and the larger is quoted, as for the line.
   - *thermal*: from the noise measured in the cells (the scatter of their
     visibilities);
   - *bootstrap*: steps 1 to 3 repeated on stacks of galaxies drawn with
     replacement. It includes how much the galaxies differ.

**Why both.** With few galaxies the bootstrap alone is too small, and noisy
from one annulus to the next: an annulus where it happens to be small would
decide the fit. On simulated stacks of 5 to 10 galaxies it gave fluxes off
by several of its own errors ([Validation](validation)). The thermal noise
does not depend on the number of galaxies. Both are in the results
(`flux_total_error_thermal`, `flux_total_error_bootstrap`).

The function is `vista.extract.measure_continuum`; `result.continuum` is its
result (a `ContinuumResult`).

## Files

`<name>_continuum_results.json` (every number: `flux_total`,
`theta_fwhm_arcsec`, their errors, and `size_test`: the Gaussian fitted
freely, its Δχ² over a point, and the model adopted),
`<name>_continuum_spectrum.txt` (the continuum per channel),
`<name>_continuum_uvamp.txt` (per annulus, with the imaginary part), and
the two figures above.

Things to check: `null_test_max_imaginary_over_sigma` (the imaginary part:
are the sources centred), `theta_at_bound` (a size at the edge of
`theta_range` is not a measurement), `chi2_reduced`, `n_bins_in_fit`.
