# Continuum

What to do with the continuum is the first decision: it sets which column is
measured, whether line and continuum are one fit or two, and which figures
come out. On the command line it is `--contsub before`, `--contsub after`, or
nothing.

| | `--contsub before` | `--contsub after` | nothing (default of the command line) |
|---|---|---|---|
| the continuum | fitted on the line-free channels and subtracted from the stack | kept, fitted together with the line | ignored |
| the line is measured on | `CORRECTED_DATA` | `DATA` | `DATA` |
| fits | two: the line, and the continuum (from `MODEL_DATA`) | one: line on a constant continuum | one: line only |
| use it | a continuum bright enough to shape the baseline of the line, or a few bright sources | both quantities from one fit | no continuum, or far below the noise |

Leaving a real continuum in the data **without** fitting it is the one
combination to avoid: the line profile has no baseline, so the line
stretches to cover the pedestal. On a synthetic test with a true flux of
4.0 and a true FWHM of 471 km/s, `--contsub after` returns 3.9 and 451 km/s,
the line alone 10.7 and 1227 km/s. A line that comes out implausibly broad
is the sign.

## The subtraction

For every row and every correlation, a polynomial is fitted to the
**line-free channels** — those outside the line window — separately for the
real and the imaginary part, and subtracted from all the channels:

- `order=0` (default): a constant. A narrow line-free band cannot constrain a
  slope, and a spurious one biases the line.
- `order=1`: a slope; `order=-1`: the BIC chooses 0 or 1 for each dataset,
  on the mean spectrum of its first `bic_rows` rows (`delta_bic`, and by
  default only a rising slope is accepted).
- Only the unflagged samples enter the fit: the channels of the grid that a
  dataset does not cover are flagged, and counting them as zeros would pull
  the continuum down.
- The fitted continuum goes to `MODEL_DATA`, the line to `CORRECTED_DATA`;
  `DATA` is never changed.

The line-free channels are those outside `LineConfig.v_window_kms` (and
outside the window of a second line, and the `exclude_v_kms` intervals), or
outside `ContinuumConfig.exclude_kms`, or `exclude_kms_lo/hi` for an
asymmetric window. The window must contain the whole line, wings included.

Over a band that spans the dust emission the continuum is not a low-order
polynomial (it rises roughly as `ν⁴` in the Rayleigh–Jeans limit): fit it on
narrow windows.

## During the stacking, or after

The subtraction works row by row on the output grid, so it runs on each
chunk right after the stacking kernel, before the chunk is written:

```python
ViSta("input_list.txt").run("stack.zarr", central_freq=line.rest_freq_hz,
                            line=line, continuum=ContinuumConfig(order=0),
                            stats_line="line_stats.npy", stats_continuum="cont_stats.npy")
```

The result is the same as subtracting it on the finished stack
(`DATA`, `MODEL_DATA`, `CORRECTED_DATA` bit for bit) and the stack is read
once instead of three times. `run_vista.sh` does this whenever it builds the
stack and compresses it in the same run (`--two-pass` for the separate
steps). On an existing stack:

```python
from vista.extract import subtract_continuum
subtract_continuum("stack.zarr", line, ContinuumConfig(order=0))
```

## Redoing it

`DATA` is always kept. After a look at the profile — a second line or a wing
in the channels used for the continuum — the subtraction is redone on the
stack with another window, and the statistics recomputed, without stacking
again:

```bash
./run_vista.sh --input input_list.txt --rest-freq 345.79599 \
               --skip-stack --contsub before --exclude-v "400 700"   # or --exclude LO HI
```

The parameters of every subtraction are recorded in the stack (Zarr
attributes, MS keyword `VISTA_CONTSUB`) and in the statistics, and a new
subtraction says which one it replaces: it is always known which window
produced the `CORRECTED_DATA` on disk.

A good sequence when the line window is uncertain:

1. stack in one pass with a generous window, e.g. ±1.5 times the expected
   width;
2. look at the profile: the residuals in the line-free channels must be
   noise (`noise.line_free_rms_over_noise` ≈ 1);
3. if another line or a wing sits there, add it to `exclude_v_kms` (or
   `second_rest_freq_ghz` for a known line, safer than an interval chosen by
   eye) and redo the subtraction on the stack.

[Example 8](../examples/08_redoing_the_continuum) does it on the simulated
sample.
