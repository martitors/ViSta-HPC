# 3. The window, the normalisations, the noise

Each cell changes **one option** of the measurement. `v.extract(...)`
replaces all the options, and `run(only="measure")` repeats the measurement
in seconds, without stacking again.

## 3.1 Quoting the flux in the window

`flux_measure="area"` quotes the sum of the spectrum in the window instead
of the Gaussian fit (now in bold on the plot). It assumes nothing about the
shape of the line, and has a larger error: it also sums the channels that
hold only noise.

```python
v.extract(flux_measure="area")
area = v.run(only="measure", tag="area")       # tag: its files do not replace the others
area.plot("line")
```

```{image} figures/03_01.png
:width: 100%
```

## 3.2 A narrower window

The window of ±500 km/s is much wider than the line (300 km/s FWHM, all of
it within ±300). `window=320` sums only where the line is: the flux in the
window keeps its value and its error gets smaller. The fit does not change.

```python
v.extract(window=320)
w320 = v.run(only="measure", tag="w320")
w320.plot("line")
```

```{image} figures/03_02.png
:width: 100%
```

A window **too narrow** cuts the line: the flux in the window drops, the
one of the fit does not move.

```python
v.extract(window=150)
w150 = v.run(only="measure", tag="w150")
w150.plot("line")
```

```text
[ViSta      11.4s] WARNING: measure: the line: the area flux differs from the Gaussian by -3.7 sigma, but the window holds only 78% of the fitted line: widen the window (the Gaussian flux is quoted)
```

```{image} figures/03_03.png
:width: 100%
```

```python
print("window ±500:", flux(r, "flux_window"))
print("window ±320:", flux(w320, "flux_window"))
print("window ±150:", flux(w150, "flux_window"))
print("expected:   ", f"{expected['flux'][0]:.3f} Jy km/s")
```

```text
window ±500: 0.371 ± 0.057 Jy km/s
window ±320: 0.392 ± 0.040 Jy km/s
window ±150: 0.303 ± 0.029 Jy km/s
expected:    0.400 Jy km/s
```

## 3.3 Without the redshift correction

`z_norm=False` stacks the fluxes as observed, each at its own distance. The
number no longer refers to any redshift, and it depends on which redshifts
happen to be in the sample.

```python
v.extract(z_norm=False)
no_z = v.run(only="measure", tag="no_z")
print("with the correction (default):", flux(r))
print("without:                      ", flux(no_z))
print("expected at z_ref:            ", f"{expected['flux'][0]:.3f} Jy km/s")
```

```text
[ViSta       0.2s] no redshift rescaling, no flux normalisation
```

```text
with the correction (default): 0.404 ± 0.040 Jy km/s
without:                       0.436 ± 0.060 Jy km/s
expected at z_ref:             0.400 Jy km/s
```

## 3.4 Without the primary-beam correction

An antenna is less sensitive away from the centre of its field: these
galaxies, up to 20″ away, are seen at 70–100 % of their flux.
`pb_corr=False` leaves that in, and the stack is low.

```python
v.extract(pb_corr=False)
no_pb = v.run(only="measure", tag="no_pb")
print("with the correction (default):", flux(r))
print("without:                      ", flux(no_pb))
print("expected:                     ", f"{expected['flux'][0]:.3f} Jy km/s")
```

```text
[ViSta       0.3s] z_ref = 2.6596 (median of the sample)
```

```text
with the correction (default): 0.404 ± 0.040 Jy km/s
without:                       0.372 ± 0.032 Jy km/s
expected:                      0.400 Jy km/s
```

## 3.5 The noise

The noise is not taken from the weights of the files: it is measured from
the scatter of the visibilities. `r.plot("noise")` shows the checks:

* left: the noise per channel from the scatter and from the weights. Here
  the weights are 30 % too optimistic (we simulated them so);
* centre: the stacked values away from the line, divided by their noise:
  they follow the bell curve if the noise is right;
* right: the same search for a line repeated on pure noise (grey), and the
  S/N measured (red line).

```python
r.plot("noise")
```

```{image} figures/03_04.png
:width: 100%
```
