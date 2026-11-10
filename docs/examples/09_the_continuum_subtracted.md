# 9. The continuum: subtracted, fitted or left

By default the continuum is subtracted while stacking, on the channels
outside the line window, and measured on its own. It can also be left in the
data and fitted together with the line (`continuum="fit"`), or left as it is
(`"none"`, for data already subtracted): here, with a continuum, the line
then sits on it and its flux is overestimated; the measurement flags the
continuum it finds outside the window. The stack keeps the data before the
subtraction, so this needs no new stack, only a new compression (seconds).

```python
continuum = {"subtracted (default)": r}
for mode, name in (("fit", "fitted with the line"), ("none", "left in the data")):
    v.stack(line_freq=LINE, line_window=500, velocity_range=900, continuum=mode)
    continuum[name] = measure()
v.stack(line_freq=LINE, line_window=500, velocity_range=900)     # back to the subtraction

vista.compare(continuum, truth=T)
```

```text
[ViSta      11.3s] WARNING: measure: continuum in the data (not subtracted) outside the line windows: offset 0.000123 +- 3.6e-05 (+3.5 sigma, +9% of the line peak), slope 1.89e-05 +- 5.3e-05 per 1000 km/s (+0.4 sigma).  Subtract it (the default) or fit it with the lines (stack(..., continuum='fit')): as it is, the line fluxes include it
```

```text
                             flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
subtracted (default)       0.3804    0.06   gauss   0.3804   0.3732    +16   307   0.73  10.2    9.7σ
fitted with the line        0.401   0.054   gauss    0.401    0.396    +15   307   0.81  10.8   10.3σ  fitted 0.000212 ± 4.4e-05
left in the data           0.5725   0.074   gauss   0.5725   0.5977    +17   407   0.78  15.1   14.8σ  residual FLAGGED
true                     0.4   continuum 0.0002
```

```{image} figures/09_01.png
:width: 100%
```

The continuum measured on its own (with the subtraction): its flux density
in every annulus, and the fit of its size.

```python
r.plot("continuum")
```

```{image} figures/09_02.png
:width: 100%
```
