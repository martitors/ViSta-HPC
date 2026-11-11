# Compression

**In one sentence.** The measurement does not read the stack: it reads a
compressed version of it, a few MB that hold everything a fit of the
visibilities needs.

**Why it matters.** Reading the visibilities takes minutes; reading the
compressed statistics takes a second. Every choice of the measurement
(weighting, redshift correction, source model, bootstrap) is made on the
statistics, so it can be changed and repeated at will.

## The idea: annuli of baseline length

A source at the phase centre that is round (a point, a circular Gaussian)
has visibilities that are real and depend only on the **length** of the
baseline `b`: `V(b) = F g(b)`, with `g(0) = 1`.

So the direction of a baseline carries no information, and the
visibilities can be summed in rings (**annuli**) of baseline length without
losing anything. By default there are 18 annuli, logarithmically spaced
from 3 to 3000 kλ.

## What is stored

For every cell (dataset, channel, annulus):

| Sum | Used for |
|---|---|
| `Σ w Re V`, `Σ w Im V` | the weighted mean of the real and of the imaginary part |
| `Σ w` | the weight of the cell |
| `Σ w b` | the mean baseline of the cell |
| `N` | the number of visibilities |
| `Σ w (Re V)²`, `Σ w (Im V)²` | the scatter of the visibilities: the noise, measured from the data |

**Why sums.** Sums are additive. Any weighting, rescaling, selection or
bootstrap of the galaxies is a weighted sum of them, computed in
milliseconds. The measurement never reads the visibilities again.

The number of datasets that contribute to each channel and annulus is kept
too.

## The two files

| File | From | Holds |
|---|---|---|
| `<name>_line_stats.npy` | `CORRECTED_DATA` after a subtraction, otherwise `DATA` | the line |
| `<name>_cont_stats.npy` | `MODEL_DATA`, the fitted continuum | the continuum |

They also hold the grid, the annuli, the parameters of the continuum
subtraction, and the primary-beam gain of every dataset. The gain is
computed here from the original Measurement Sets (dish diameter, observed
frequency of the line, offset of the source), which is why the compression
needs them.

## When it runs

- **During the stacking**, on every chunk, when `v.extract(...)` was called
  before `run()`. The stack is read once.
- **After**, when it was not: `run(only="measure")` compresses the stack
  first.
- **Again**, with `run(only="continuum")`: the continuum is subtracted
  again on the stack, which is then compressed again.

The statistics are the same in all three (bit for bit on a Zarr store).

## Choosing the annuli

`v.extract(annuli=(b_min, b_max, n))`, in kλ.

- **`b_min`, `b_max`** should bracket the baselines of the stack (the
  observed ones, see [Rest frame](restframing)).
  `examples/inspect_ms.py` suggests them.
- **`n`**: more annuli follow `V(b)` more finely, but leave fewer
  visibilities, hence less S/N, in each.
- The annuli are fixed here: changing them means compressing again.

NaN and infinities are always discarded.
