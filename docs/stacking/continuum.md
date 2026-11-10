# Continuum

The continuum is **subtracted by default**, while the stack is written:
fitted row by row on the channels outside the line windows, it goes to
`MODEL_DATA`, the line to `CORRECTED_DATA`, and `DATA` is kept. The line is
then measured on `CORRECTED_DATA`, and the continuum on its own, from
`MODEL_DATA` ([The continuum measurement](../extraction/extract_flux)).

```python
v.stack(line_freq=..., continuum="subtract", continuum_order=0, exclude_between=False)  # defaults
v.stack(line_freq=..., continuum="none")   # data already subtracted, or no continuum
v.stack(line_freq=..., continuum="fit")    # left in the data, fitted with the lines
```

With `continuum="none"` the data are measured as they are. A continuum left in
them would be taken for part of the line — the profile has no baseline and
the line stretches to cover the pedestal (on a synthetic test with a true
flux of 4.0 and FWHM of 471 km/s, the line came out at 10.7 and 1227 km/s) —
so the measurement checks it: the profile outside the windows is fitted with
an offset and a slope, and a significant one is flagged
(`continuum_residual`, [Measuring the line](../extraction/measure_line.md#residual-continuum)).

## Fitting it with the lines

`continuum="fit"` leaves the continuum in the data and
fits it together with the lines: a constant in frequency, at zero spacing,
with the size of the lines,

```{math}
\mathrm{Re}\,V(c, b) = \Big[\sum_k F_k\,\phi_c(v_{0,k}, \mathrm{FWHM}_k) + C\Big]\,g(b;\theta),
```

on all the channels within 1500 km/s of the lines. Its flux density is in
`continuum_fit` of the results, with its errors; the area flux of every line
is the sum over its window minus `C`. It suits a continuum flat over the
band and with the size of the line; a slope, or a continuum much more
extended than the line, is better subtracted (`continuum_order=1` for the slope). On
two simulated lines with a continuum of 0.2 mJy, the fluxes came out at
0.388 ± 0.020 and 0.193 ± 0.016 (true 0.4 and 0.2), as with the subtraction,
and the continuum at 0.194 ± 0.015 mJy; without the fit, at 0.54 and 0.33.

## The subtraction

For every row and every correlation, a polynomial is fitted to the
**line-free channels** — those outside the line window — separately for the
real and the imaginary part, and subtracted from all the channels:

- `continuum_order=0` (default): a constant. A narrow line-free band cannot constrain a
  slope, and a spurious one biases the line.
- `continuum_order=1`: a slope; `"bic"`: the BIC chooses 0 or 1 for each
  dataset, on the mean spectrum of its first 5000 rows (by default only a
  rising slope is accepted).
- Only the unflagged samples enter the fit: the channels of the grid that a
  dataset does not cover are flagged, and counting them as zeros would pull
  the continuum down.
- The fitted continuum goes to `MODEL_DATA`, the line to `CORRECTED_DATA`;
  `DATA` is never changed.

The line-free channels are those outside the windows of **all** the lines
(`line_window`, one per line of `line_freq`). Windows that overlap are
simply joined (and their lines are fitted together by the measurement). With `exclude_between=True` the channels between the lines
are left out too (everything from the lowest to the highest window): for
lines close enough to share wings, or a blend. The windows must contain the
whole lines, wings included.

Over a band that spans the dust emission the continuum is not a low-order
polynomial (it rises roughly as `ν⁴` in the Rayleigh–Jeans limit): fit it on
narrow windows.

## During the stacking

The subtraction works row by row on the output grid, so it runs on each
chunk right after the stacking kernel, before the chunk is written, and the
compression follows on the same chunk: the stack is read once instead of
three times. The result is the same as subtracting it on the finished stack
(`DATA`, `MODEL_DATA`, `CORRECTED_DATA` bit for bit), which is what
`run(only="continuum")` does.

## Redoing it

`DATA` is always kept. After a look at the profile — a second line or a wing
in the channels used for the continuum — the subtraction is redone on the
stack with other windows (or another order), and the statistics recomputed,
without stacking again:

```python
v = ViSta("input_list.txt", output_name="stacked")    # the existing stacked.zarr
v.stack(line_freq=[345.79599, 344.98857], line_window=[650, 300], continuum_order=1)
v.extract()
v.run(only=["continuum", "measure"])
```

(`--only continuum measure` on the command line.)

The parameters of every subtraction are recorded in the stack (Zarr
attributes, MS keyword `VISTA_CONTSUB`) and in the statistics, and a new
subtraction says which one it replaces: it is always known which window
produced the `CORRECTED_DATA` on disk.

A good sequence when the line window is uncertain:

1. stack in one pass with a generous window, e.g. ±1.5 times the expected
   width;
2. look at the profile: the residuals in the line-free channels must be
   noise (`noise.line_free_rms_over_noise` ≈ 1, `continuum_residual` not
   flagged);
3. if another line or a wing sits there, widen the window, or add the line to
   `line_freq` with its own window (it is then also measured), and redo the
   subtraction on the stack;
4. check that `continuum_residual` is not flagged.

[Example 9](../examples/09_the_continuum_subtracted) compares the continuum
subtracted, fitted with the line and left in the data, on the simulated
sample.
