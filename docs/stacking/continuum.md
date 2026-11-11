# Continuum

**In one sentence.** The continuum under a line has to be accounted for, or
it is taken for part of the line; `continuum` in `v.stack` says how.

| `continuum=` | What is done | When |
|---|---|---|
| `"subtract"` (default) | fitted on the channels outside the line windows and subtracted from every visibility; measured on its own | most cases. Line and continuum can have different sizes |
| `"fit"` | left in the data, and fitted together with the lines as a constant level | few channels free of the line (a broad line, a narrow band). One size for both |
| `"none"` | nothing | data already subtracted, or no continuum |

```python
v.stack(line_freq=..., continuum="subtract", continuum_order=0, exclude_between=False)  # defaults
v.stack(line_freq=..., continuum="fit")
v.stack(line_freq=..., continuum="none")
```

The stack always keeps the data before the subtraction (`DATA`), so
changing this option needs no new stack: only the measurement is repeated.
[Example 4](../examples/04_not_subtracting_the) compares the three on a
simulated sample.

## Subtracted (the default)

**What happens.** While the stack is written, the continuum is fitted row
by row on the channels outside the line windows. The fit goes to
`MODEL_DATA`, the line to `CORRECTED_DATA`; `DATA` is kept.

**What is measured.** The line on `CORRECTED_DATA`. The continuum on its
own, from `MODEL_DATA` ([The continuum](../extraction/extract_flux)).

## Left in the data

**What goes wrong.** With `continuum="none"` the data are measured as they
are. If there is a continuum, the spectrum has no zero level, and the fit
stretches the line to cover it. On a test with a true flux of 4.0 and a
FWHM of 471 km/s, the line came out at 10.7 and 1227 km/s.

**The safety net.** The measurement always checks: the spectrum outside the
windows is fitted with an offset and a slope, and a significant one is
flagged (`continuum_residual`,
[Measuring the line](../extraction/measure_line.md#continuum-left-outside-the-window)).

## Fitting it with the lines

**What.** `continuum="fit"` adds a constant level `C` to the model of the
lines, with the same source size:

```{math}
\mathrm{Re}\,V(c, b) = \Big[\sum_k F_k\,\phi_c(v_{0,k}, \mathrm{FWHM}_k) + C\Big]\,g(b;\theta),
```

on all the channels within 1500 km/s of the lines. Its flux density is in
`continuum_fit` of the results. The flux in the window of every line is the
sum over the window minus `C`.

```{image} ../_static/fig_line_cont.png
:alt: The stacked line on top of the continuum fitted with it
:width: 70%
:align: center
```

The spectrum then shows the line **on top of** the continuum (blue,
dashed), and the title says `line + cont`.

**When it suits.** A continuum that is flat over the band and has the size
of the line. A slope, or a continuum much more extended than the line, is
better subtracted (`continuum_order=1` for the slope).

**How well.** On two simulated lines with a continuum of 0.2 mJy, the
fluxes came out at 0.388 ± 0.020 and 0.193 ± 0.016 (true 0.4 and 0.2), as
with the subtraction, and the continuum at 0.194 ± 0.015 mJy. Without
accounting for it: 0.54 and 0.33.

## The continuum alone

Without `line_freq` there is no line, and the stack is that of the
continuum:

```python
v = ViSta("input_list.txt", output_name="dust")
v.stack()            # no line: every channel of every dataset
v.extract()
result = v.run()     # flux density and size of the stacked continuum
```

- **Every channel is kept**, whatever its rest frequency: the bands need
  not overlap in the rest frame (galaxies at different redshifts observed
  in the same band).
- **Nothing is subtracted.** The statistics sum every dataset over its
  channels.
- **The measurement** is the one of a continuum after a subtraction: flux
  density at zero baseline and size, with the same weighting, redshift
  correction, primary beam and source model (`continuum_model`). The
  options of the lines are not used.

From the command line: `python -m vista --input input_list.txt --extract`,
without `--line-freq`.
[Example 13](../examples/13_the_continuum_alone) is a complete case.

## How the subtraction works

**The fit.** For every row and every correlation, a polynomial is fitted to
the **line-free channels** (those outside the line windows), separately for
the real and the imaginary part, and subtracted from all the channels.

**The order.**

| `continuum_order` | The fit | Note |
|---|---|---|
| `0` (default) | a constant | a narrow line-free band cannot constrain a slope, and a spurious one biases the line |
| `1` | a slope | |
| `"bic"` | 0 or 1 for each dataset, chosen by the BIC on its mean spectrum | by default only a rising slope is accepted |

**The line-free channels.** Those outside the windows of **all** the lines
(`line_window`, one per line of `line_freq`). The windows must hold the
whole lines, wings included. Windows that overlap are joined. With
`exclude_between=True` the channels between the lines are left out too: for
lines close enough to share wings.

**Flagged channels are not counted.** The channels of the grid that a
dataset does not cover are flagged; counted as zeros they would pull the
continuum down.

**A wide band.** Over a band that spans the dust emission the continuum is
not a low-order polynomial (it rises roughly as `ν⁴`): fit it on narrow
windows.

**One pass.** The subtraction runs on each chunk right after the stacking
kernel, and the compression follows on the same chunk: the stack is read
once instead of three times. The result is the same, bit for bit, as
subtracting on the finished stack (`run(only="continuum")`).

## Redoing it

`DATA` is always kept. If a look at the spectrum shows a second line or a
wing in the channels used for the continuum, the subtraction is redone on
the stack with other windows (or another order), without stacking again:

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
2. look at the spectrum: the residuals in the line-free channels must be
   noise (`noise.line_free_rms_over_noise` ≈ 1, `continuum_residual` not
   flagged);
3. if another line or a wing sits there, widen the window, or add the line to
   `line_freq` with its own window (it is then also measured), and redo the
   subtraction on the stack;
4. check that `continuum_residual` is not flagged.
