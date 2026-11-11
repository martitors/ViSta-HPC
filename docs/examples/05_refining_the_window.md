# 5. Refining: the window

The window of ±500 km/s is much wider than the line (about 300 km/s FWHM, so
all of it within ±300): the area flux sums 400 km/s of channels that hold
only noise, and its error is larger than it needs to be.

`window` sets the window of the area flux. `v.extract(...)` replaces all the
options of the measurement, and `run(only="measure")` repeats it in seconds.
The Gaussian fit does not change; the area flux now has a smaller error, and
the two fluxes are closer.

```python
v.extract(window=320)                  # the area flux: from -320 to +320 km/s
r = v.run(only="measure")
print(r)
```

```text
Line       345.796 GHz, 12 sources: detected at 11.4 sigma (S/N 11.8, searched over centre and width)
  flux     0.3656 ± 0.054 Jy km/s, the area of the Gaussian fit
           because the area agrees with the Gaussian (-0.5 sigma)
           Gaussian fit 0.3656 ± 0.054; area in the window (-320 to +320 km/s) 0.3509 ± 0.056
  Gaussian centre -26 ± 11 km/s, FWHM 267 ± 32 km/s, peak 1.29 ± 0.19 mJy
  source   Gaussian, FWHM 0.67 ± 0.12 arcsec
           because it fits better than a point source: delta chi2 = 18.1 (at least 9 needed to call it extended)
Continuum  subtracted while stacking and measured on its own: 0.177 ± 0.041 mJy, FWHM 1.16 arcsec
           outside the line window: 0.00044 ± 0.0036 mJy (+0.1 sigma): nothing left
Reference  natural weighting, fluxes at z_ref = 2.2867 (median of the sample)
```

```python
r.plot("profile")
```

```{image} figures/05_01.png
:width: 100%
```

A window too narrow would cut the line: the area flux would then be low,
while the Gaussian one would not move. The three windows side by side
(`vista.compare`: the dotted lines are the windows, the triangles the area
fluxes, the squares the Gaussian ones):

```python
v.extract(window=500)
wide = v.run(only="measure", tag="wide")         # tag: its files do not replace the others
v.extract(window=120)
narrow = v.run(only="measure", tag="narrow")

vista.compare({"window ±500": wide, "window ±320": r, "window ±120": narrow}, expected=expected)
```

```text
[ViSta      12.3s] WARNING: measure: the line: the area flux differs from the Gaussian by -2.7 sigma, but the window holds only 70% of the fitted line: widen the window (the Gaussian flux is quoted)
```

```text
                    flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
window ±500       0.3656   0.054   gauss   0.3656   0.3649    -26   267   0.67  11.8   11.4σ
window ±320       0.3656   0.054   gauss   0.3656   0.3509    -26   267   0.67  11.8   11.4σ
window ±120       0.3656   0.054   gauss   0.3656   0.2561    -26   267   0.67  11.8   11.4σ
expected        0.3932   continuum 0.000197
```

```{image} figures/05_02.png
:width: 100%
```

(`line_window` of `v.stack` is another thing: the channels left out of the
continuum fit. It should hold the whole line too; to change it on the stack
that exists, `v.stack(..., line_window=...)` and
`v.run(only=["continuum", "measure"])`.)

From here on the window stays at ±320 km/s: `base` holds the options kept in
every measurement.

```python
base = dict(window=320)
```
