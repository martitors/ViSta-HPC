# 2. Stack and measure

Three calls describe the run:

* `ViSta(...)`: the input list, the name of the outputs, how the run uses the
  machine;
* `v.stack(...)`: the line (its rest frequency and window), the band, the
  continuum (subtracted by default);
* `v.extract(...)`: the measurement, here with its defaults. Without this
  call, `run()` only writes the stack.

`run()` stacks the twelve datasets in the rest frame, subtracts the
continuum, compresses the stack into sums per channel and annulus of
baseline length, and measures the line on them (`ViSta(...,
verbose=True)` shows every step).

```python
v = ViSta(truth["input_list"], output_name="stack")
v.stack(line_freq=LINE, line_window=500, velocity_range=900)
v.extract()
r = v.run()
r
```

```text
[ViSta       9.8s] z_ref = 2.5250 (median of the sample)
```

```text
RunResult(
  stack: stack.zarr (433,440 rows, 90 channels, 5 s)
  line 1 (345.796 GHz): F = 0.3804 ± 0.06 (Gaussian), v0 = +16 km/s, FWHM = 307 km/s, 9.7 sigma
  z_ref = 2.5250 (median of the sample)
  continuum: 0.0002474 ± 8.6e-05, size 1.15"
)
```

`r.plot()` draws the measurement. **Profile**: the stacked flux density at
zero spacing (black), the fitted Gaussian (red, its area hatched), the
window over which the spectrum is summed (orange), the noise (grey band) and
the imaginary part (dotted, pure noise); the box gives the flux quoted, the
two fluxes and the significance; the title, the source size and how much a
Gaussian source improves on a point source (Δχ²). **uv**: the line flux in
every annulus falls with the baseline length, because the galaxies are
extended; the fit extrapolates it to zero length.

```python
r.plot()
```

```{image} figures/02_01.png
:width: 100%
```

```{image} figures/02_02.png
:width: 100%
```
