# Compression

The measurement does not read the stack: it reads a compressed version of
it, a few MB that hold everything a fit of the visibilities needs. The
compression is the last step that touches the visibilities.

## Annuli of baseline length

A source at the phase centre that is symmetric (a point, a circular
Gaussian) has a real visibility function that depends only on the baseline
length `b`: `V(b) = F g(b)`, with `g(0) = 1`. So, for every dataset and every
channel, the visibilities are summed in **annuli of rest-frame baseline
length**, logarithmically spaced between `b_min_klambda` and `b_max_klambda`
(default 18 annuli from 3 to 3000 kλ), storing in each cell (dataset,
channel, annulus):

| Sum | Used for |
|---|---|
| `Σ w Re V`, `Σ w Im V` | the weighted mean of the real and the imaginary part |
| `Σ w` | the weight of the cell |
| `Σ w b` | the mean baseline of the cell |
| `N` | the number of visibilities |
| `Σ w (Re V)²`, `Σ w (Im V)²` | the scatter of the visibilities: the noise, measured |

These sums are additive: any weighting, rescaling, selection or bootstrap
of the sources is a weighted sum of them, computed in milliseconds. The
measurement never reloads the visibilities.

The cells of a dataset are put on one common frequency axis (the windows of
the stack share the channel width but not always the start), and the number
of datasets contributing to each channel and annulus is kept.

## The two files

- `line_stats.npy`: from `CORRECTED_DATA` after a subtraction, otherwise
  from `DATA`;
- `cont_stats.npy`: from `MODEL_DATA`, the fitted continuum.

They also hold the grid, the annuli, the parameters of the subtraction that
produced the column, and the primary-beam gain `A` of every dataset,
computed here from the original Measurement Sets (dish diameter, observed
frequency of the line, offset of the source).

## During the stacking, or after

```python
ViSta("input_list.txt").run(..., stats_line="line_stats.npy", stats_continuum="cont_stats.npy",
                            binning=BinningConfig(b_min_klambda=3, b_max_klambda=3000, n_bins=18))
```

or on an existing stack (Zarr store or MS):

```python
from vista.extract import compress_visibilities
compress_visibilities("stack.zarr", "input_list.txt", line,
                      out_line="line_stats.npy", out_continuum="cont_stats.npy")
```

The statistics are the same both ways (bit for bit on a Zarr store, to
1e-15 on an MS, read there in larger pieces). The compression reads the
stack with `n_io` threads and does its per-row work in the kernel (OpenMP);
without the kernel it falls back to NumPy.

## Choosing the annuli

- `b_min_klambda`, `b_max_klambda` should bracket the baselines of the stack
  (in kλ: the observed ones, see [Rest frame](restframing)).
- More annuli resolve `V(b)` better but leave fewer visibilities, hence less
  S/N, in each. The number is fixed here: changing it means compressing
  again.
- `max_amplitude` discards visibilities above that amplitude (NaN and
  infinities are always discarded): a few corrupt cells can spoil a whole
  annulus.
