# 8. Point sources only, extended sources only

The sample of chapter 2 mixes points and extended galaxies. Stacked
separately they show when each source model is the right one.
`simulate.write_list` writes an input list with part of a sample.

```python
points = simulate.write_list("points.txt", [s for s in truth["sources"] if s["size_arcsec"] == 0],
                             z_ref=truth["z_ref"])
extended = simulate.write_list("extended.txt", [s for s in truth["sources"] if s["size_arcsec"] > 0],
                               z_ref=truth["z_ref"])
print(len(points["sources"]), "points,", len(extended["sources"]), "extended")
```

```text
6 points, 6 extended
```

## 8.1 The points

The flux is the same on every baseline. The default (`model="auto"`) picks
the point source.

```python
p = ViSta(points["input_list"], output_name="points")
p.stack(line_freq=LINE, line_window=500, velocity_range=900)
p.extract(z_ref=truth["z_ref"])
pts = p.run()
pts.plot("line_uv")
```

```text
[ViSta       0.2s] z_ref = 2.6596 (given)
```

```{image} figures/08_01.png
:width: 100%
```

Forcing a Gaussian (`model="gauss"`) adds a parameter the data do not need:
the same flux within the errors, with a **larger error**.

```python
p.extract(z_ref=truth["z_ref"], model="gauss", measure="line")
pts_gauss = p.run(only="measure", tag="gauss")
print("point (chosen):  ", flux(pts))
print("Gaussian forced: ", flux(pts_gauss))
print("expected:        ", f"{simulate.expected(points)['flux'][0]:.3f} Jy km/s")
```

```text
point (chosen):   0.381 ± 0.039 Jy km/s
Gaussian forced:  0.395 ± 0.045 Jy km/s
expected:         0.400 Jy km/s
```

## 8.2 The extended ones

The flux falls on the long baselines, and the default picks the Gaussian.

```python
e = ViSta(extended["input_list"], output_name="extended")
e.stack(line_freq=LINE, line_window=500, velocity_range=900)
e.extract(z_ref=truth["z_ref"])
ext = e.run()
ext.plot("line_uv")
```

```text
[ViSta       0.2s] z_ref = 2.6596 (given)
```

```{image} figures/08_02.png
:width: 100%
```

Forcing a point (`model="point"`) misses the flux that the long baselines
do not see: **too low**, and with an error that looks small.

```python
e.extract(z_ref=truth["z_ref"], model="point", measure="line")
ext_point = e.run(only="measure", tag="point")
print("Gaussian (chosen): ", flux(ext))
print("point forced:      ", flux(ext_point))
print("expected:          ", f"{simulate.expected(extended)['flux'][0]:.3f} Jy km/s")
```

```text
Gaussian (chosen):  0.426 ± 0.070 Jy km/s
point forced:       0.210 ± 0.036 Jy km/s
expected:           0.400 Jy km/s
```
