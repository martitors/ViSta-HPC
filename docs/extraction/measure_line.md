# Measuring the line

**In one sentence.** The line is fitted on all the channels and all the
baselines at once, with a Gaussian in velocity from a source that is a point
or a Gaussian on the sky, and with the noise measured from the data.

The options are those of `v.extract(...)`
([Parameters](../reference/parameters.md#extract-what-is-measured));
`v.run(only="measure")` repeats the measurement from the statistics, in
seconds.

## What you get

```{image} ../_static/fig_line.png
:alt: The stacked line: spectrum, fit, window, the two fluxes
:width: 70%
:align: center
```

- **The spectrum** (black): the flux density the source would show at zero
  baseline, in every channel. It is the real part (Re) of the stacked
  visibilities, corrected for the size of the source.
- **The fit** (red): a Gaussian in velocity.
- **Two fluxes**, the one quoted in bold:
  `F_fit`, the area of the Gaussian, and `F_window`, the area under the
  spectrum in the orange window.
- **μ, FWHM, peak**: centre, width and peak of the Gaussian.
- **S/N**: of the line, through the best filter (centre and width).
- **Below**: the residuals in units of the noise, and the number of sources
  that contribute at every velocity.

```{image} ../_static/fig_line_uv.png
:alt: Line flux against baseline length, with the Gaussian and the point source models
:width: 60%
:align: center
```

- **The line against the baseline length**: the line flux in every ring
  (*annulus*) of baseline length. Red: the Gaussian source. Green, dashed:
  the point source. In bold the one used. Grey crosses: the imaginary part
  (Im), which holds only noise if the sources are where the list says.

## The model

For every cell (channel `c`, annulus `b`) of the stack

```{math}
\mathrm{Re}\,V(c, b) = \Big[\sum_k F_k\,\phi_c(v_{0,k}, \mathrm{FWHM}_k)\Big]\,g(b;\theta)
```

- `F` is the integrated flux of a line;
- `φ` is a Gaussian of unit area, **averaged over the channel**, so a line
  only a few channels wide is not distorted;
- `g` is how a round Gaussian source of size `θ` (FWHM) fades with the
  baseline: 1 for a point source. It is averaged over the baselines of the
  annulus.

All the cells are fitted together, each weighted by its noise. Flux, centre,
width and size come out of one fit, with their covariance.

**Why all at once.** Measuring first the shape on the spectrum averaged over
the baselines, and then the flux baseline by baseline, wastes information:
that average is dominated by the long baselines, where an extended source
has the least signal.

## The two fluxes, and which one is quoted

- **`F_fit`** (`flux_fit`): the area of the fitted Gaussian. It uses every
  channel with the weight the line gives it, so it is the most precise. It
  is right if the line is a Gaussian.
- **`F_window`** (`flux_window`): the sum of the spectrum over the window,
  times the channel width. It assumes no shape, and carries the noise of
  every channel of the window.

`flux_measure` chooses: `"gauss"`, `"area"`, or `"auto"` (default).

**What `"auto"` does.** It compares the two. The error of their difference
is computed from the same data. If they differ by more than `shape_nsigma`
(2) times that error, the line is not a Gaussian (a double horn, wings, an
asymmetry): the flux in the window is quoted, with a warning. Otherwise the
fit. The window flux is never quoted when

- the window holds less than 95 % of the fitted Gaussian (too narrow), or
- it overlaps the window of another line (only the joint fit separates
  them).

The reason of the choice is in the report and in `results["flux"]["reason"]`.

**The window.** `window` sets it (default: the `line_window` of the stack).
Too wide, and `F_window` sums channels of pure noise: a larger error. Too
narrow, and it cuts the line. About 2.3 FWHM gives `F_window` the same error
as `F_fit`. [Example 3](../examples/03_the_window_the) shows both.

On simulated lines at S/N ≈ 10, `F_fit` was within 1 % for a Gaussian line,
7 % high for a double horn, 6 % low with broad wings; `F_window` over four
FWHM was within 2 % in all three, with an error 1.3 times larger.

## The source: a point or a Gaussian

`model` chooses the source of the lines.

| `model` | The source |
|---|---|
| `"auto"` (default) | a Gaussian only if the data ask for it, otherwise a point |
| `"point"`, `"gauss"` | always that one |
| a number | a Gaussian of that size (FWHM, arcsec) |
| `"continuum"` | the model and the size measured on the continuum |

**What `"auto"` does.** It fits both. The Gaussian is kept when it improves
the fit by Δχ² ≥ 9 (a 3σ preference). The Δχ² is first corrected for the
correlation between channels, which would otherwise make noise look like an
extended source.

The flux of the fit is the average of the two fits, each weighted by how
well it fits (Akaike weights), so that an undecided case carries the
uncertainty of the size in its error.

**When the line cannot tell its size.** A weak line fits a point and a
Gaussian equally well. If the continuum of the same stack is extended, the
line is fitted again with **the size of the continuum**: the galaxies are
the same, so they cannot be a point in the line and extended in the
continuum. The report says so, and the legend of the figure reads
`size from cont`:

```{image} ../_static/fig_line_uv_size_from_cont.png
:alt: A weak line fitted with the size measured on the continuum
:width: 60%
:align: center
```

The continuum does the same the other way round
([The continuum](extract_flux)). The two sizes, each measured on its own,
are always reported with their difference in units of its error (`Sizes` in
the report, `results["sizes"]`).
[Examples 5](../examples/05_a_weak_continuum_under_a) and
[6](../examples/06_a_weak_line_on_a_strong) are the two cases.

**Why it matters.** The total flux of an extended source depends on its
size. A point source fitted to an extended one is too low
([Example 8](../examples/08_points_and_extended_sources)).

**What the model assumes: one size for the whole stack.** Galaxies of
somewhat different sizes are fine. A stack of point-like and clearly
extended galaxies together is not: it is a flat level plus something that
falls with the baseline, a single Gaussian settles on a small size, and the
flux comes out low (16 % in the [Validation](validation), with an error
that does not cover it). The figure against the baseline length shows it:
the short baselines lie above the model. If the two kinds can be told
apart, stack them separately.

**A size from elsewhere.** `model=0.8` fixes it at 0.8″;
`model_prior=(0.8, 0.15)` ties it softly (0.15 dex);
`theta_range=(0.3, 2)` keeps it within limits.

## Where the line is searched

The centre and the width of every line are searched **within limits**,
`v0_range` and `fwhm_range` (km/s, in the velocity of that line):

| | default | `lo < hi` | `lo = hi` |
|---|---|---|---|
| centre | anywhere in the window | between the limits | fixed |
| width | two channels to the window | between the limits | fixed |

Give limits when you know something: a redshift good to ±100 km/s, a width
from another line. Narrower limits leave fewer chances for noise to look
like a line.

**S/N and significance are two numbers.**

- The **S/N** is that of the best filter found by the search.
- The **significance** answers "could noise do this?". The same search is
  run on noise `n_null` times (default 1000), with the measured noise and
  its correlation between channels. A search always finds a bump, so the
  significance is a little lower than the S/N.

With centre, width and size all fixed there is one filter: a matched filter,
linear and unbiased, and its S/N is the significance.

A free search is reliable from about 4σ. Below that the flux follows noise
peaks, and the report warns: fix the width from another tracer, narrow the
limits, or quote the flux in the window.

## Several lines

`v.stack(line_freq=[f1, f2, ...], line_window=[...])` lists them, each with
its window. Every line is searched in its own window. How they are then
fitted follows from the windows, with nothing to set:

- **windows that overlap**: the lines blend, and are fitted **together**:
  one Gaussian each, with its own flux, centre and width, and one size for
  all. Their window fluxes hold part of each other, so the fit is quoted.
- **windows apart**: every line is measured **on its own**, as if it were
  alone, with its own size and source model.

```{image} ../_static/fig_two_lines.png
:alt: Two blended lines fitted together
:width: 70%
:align: center
```

The report says which lines went together. `results["lines"]` has one entry
per line. [Examples 9](../examples/09_two_lines_far_apart) and
[10](../examples/10_two_lines_that_blend) show the two cases.

## The errors

Two errors are computed for every quantity, and the larger is quoted:

- **thermal**: from the fit, with the measured noise and its correlation
  between channels;
- **bootstrap**: the fit repeated on samples of the galaxies drawn with
  replacement (`n_bootstrap`, default 500). It includes how much the
  galaxies differ from one another.

Their ratio is in the results (`bootstrap_over_thermal`): about 1 for
galaxies that are alike, above 1 when they differ. With few sources the
bootstrap is too small, and the thermal error covers it.

## The noise

The noise is **measured**, not taken from the weights of the files.

1. **From the scatter of the visibilities** inside each cell. This is the
   noise used. It also gives `κ`, the factor by which the weights misjudge
   the noise, for every dataset and every channel.
2. From the weights: right only if the weights are absolute.
3. From the **imaginary part** of the stack, which is zero for sources that
   are centred and symmetric.
4. From the **real part in the channels without line**, after a
   subtraction.

3 and 4 must scatter like 1. If they scatter more, the report warns: a
source off-centre, a phase error, continuum left, a line not listed.

**Correlated channels.** Adjacent channels share noise, for two reasons: the
spectral response and the rebinning, and the continuum subtraction, which
takes the same noisy level from every channel of a row. The whole channel ×
channel correlation is measured from the imaginary part and enters every
error. On a narrow band it can double the error of a flux summed over the
line.

```{image} ../_static/fig_noise.png
:alt: The noise checks
:width: 95%
:align: center
```

`result.plot("noise")`: the noise per channel from the scatter and from the
weights; the stacked cells away from the line over their noise, against the
bell curve; the search on noise (grey) and the S/N measured (red).

## Continuum left outside the window

After the fit, the spectrum outside the windows is fitted with an offset and
a slope. If either is above 3σ the report warns, and the figure draws it
(blue, dashed):

- after a subtraction: continuum is left (a slope fitted as a constant, the
  wings of the line in the continuum channels, a line not listed);
- without a subtraction (`continuum="none"`): there is continuum in the
  data, and the line flux includes it.

[Continuum](../stacking/continuum) says what to do in each case.

## The options, together

```python
v.extract(v0_range=(-150, 150),        # km/s; (0, 0): fixed
          fwhm_range=(100, 600),       # km/s; (300, 300): fixed
          model="auto",                # "point", "gauss", "continuum", or a size in arcsec
          continuum_model="auto",      # "point", "gauss", "line", or a size in arcsec
          model_prior=None,            # (0.8, 0.15): soft size prior
          flux_measure="auto",         # "gauss" or "area"
          window=None,                 # window of the area flux; default the line window
          measure=None,                # "line" or "continuum": only one of the two
          n_bootstrap=500, n_null=1000)
```

The [Examples](../examples/index) go through them one situation at a time
on simulated samples, against the expected values: how to read the
measurement ([2](../examples/02_the_basic_case_the_same)), the window, the
normalisations and the noise ([3](../examples/03_the_window_the)), whose
size the line and the continuum take
([5](../examples/05_a_weak_continuum_under_a),
[6](../examples/06_a_weak_line_on_a_strong)), the source model
([8](../examples/08_points_and_extended_sources)), two lines
([9](../examples/09_two_lines_far_apart),
[10](../examples/10_two_lines_that_blend)), a line and a source that are
not Gaussians ([11](../examples/11_a_line_that_is_not_a),
[12](../examples/12_a_source_that_is_not_a)).
