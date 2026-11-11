# 10. The continuum

The galaxies have a continuum of about 0.2 mJy under the line. By default
(`continuum="subtract"` in `v.stack`) it is fitted on the channels outside
the line window and subtracted while stacking, then measured on its own:

```python
r.plot("continuum")
```

```{image} figures/10_01.png
:width: 100%
```

**Left in the data.** `continuum="none"` is for data where it was already
subtracted, or that have none. Here there is one: the line sits on it, the
fit takes it for part of the line and the flux is too high. ViSta notices:
it measures the level outside the window and warns. (The stack keeps the
data before the subtraction, so no new stack is needed: `run(only="measure")`
reads them again.)

```python
v.stack(line_freq=LINE, line_window=500, velocity_range=900, continuum="none")
v.extract(**base)
left = v.run(only="measure", tag="left")
left.plot("profile")
```

```text
[ViSta      13.1s] WARNING: measure: continuum in the data (not subtracted) outside the line windows: offset 0.000149 +- 4.9e-05 (+3.0 sigma, +10% of the line peak), slope 2.48e-05 +- 6.1e-05 per 1000 km/s (+0.4 sigma).  Subtract it (the default) or fit it with the lines (stack(..., continuum='fit')): as it is, the line fluxes include it
```

```{image} figures/10_02.png
:width: 100%
```

**Fitted with the line.** `continuum="fit"` leaves it in the data and fits
it together with the line, as a constant level: the dashed line of the
figure, with its value.

```python
v.stack(line_freq=LINE, line_window=500, velocity_range=900, continuum="fit")
fitted = v.run(only="measure", tag="fitted")
fitted.plot("profile")
```

```{image} figures/10_03.png
:width: 100%
```

The three side by side. Subtracting and fitting agree; leaving the
continuum in does not.

```python
v.stack(line_freq=LINE, line_window=500, velocity_range=900)     # back to the subtraction

vista.compare({"subtracted (default)": r, "fitted with the line": fitted,
               "left in the data": left}, expected=expected)
```

```text
                             flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
subtracted (default)       0.3656   0.054   gauss   0.3656   0.3509    -26   267   0.67  11.8   11.4σ
fitted with the line       0.3927   0.049   gauss   0.3927   0.3769    -25   266   0.79  12.3   12.0σ  fitted 0.000151 ± 4.3e-05
left in the data           0.4789   0.046   gauss   0.4789   0.4664    -25   303   0.76  15.7   15.5σ  residual FLAGGED
expected                 0.3932   continuum 0.000197
```

```{image} figures/10_04.png
:width: 100%
```
