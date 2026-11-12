# 8. Points and extended sources

Six galaxies seen as **points** and six **extended** (0.8″), the same line
in all of them. First each kind on its own, to see which source model is
the right one; then the twelve together.

```python
same = dict(z_range=(2.0, 3.0), rest_freq_ghz=LINE, line_flux=0.4, fwhm_kms=300,
            z_ref=truth["z_ref"], noise=0.05, kappa=1.3, rmax=600)
points = simulate.make_sample("mix_points", n_sources=6, size_arcsec=0.0, seed=31, **same)
extended = simulate.make_sample("mix_extended", n_sources=6, size_arcsec=0.8, seed=32, **same)
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
[ViSta       0.3s] z_ref = 2.6596 (given)
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
point (chosen):   0.371 ± 0.017 Jy km/s
Gaussian forced:  0.372 ± 0.020 Jy km/s
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
[ViSta       0.3s] z_ref = 2.6596 (given)
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
Gaussian (chosen):  0.409 ± 0.059 Jy km/s
point forced:       0.249 ± 0.031 Jy km/s
expected:           0.400 Jy km/s
```

## 8.3 The twelve together

A stack of points and extended sources is **neither**: on the baselines it
is a flat level (the points) plus something that falls (the extended ones).
A single Gaussian cannot follow both. It settles on a small size, and the
flux it finds at zero baseline is **too low**.

Look at the short baselines: the points lie above the model. That is the
sign of a sample whose sources have very different sizes.

```python
mixed = simulate.write_list("mixed.txt", points["sources"] + extended["sources"],
                            z_ref=truth["z_ref"])
m = ViSta(mixed["input_list"], output_name="mixed")
m.stack(line_freq=LINE, line_window=500, velocity_range=900)
m.extract(z_ref=truth["z_ref"])
mix = m.run()
mix.plot("line_uv")
```

```text
[ViSta       0.3s] z_ref = 2.6596 (given)
```

```{image} figures/08_03.png
:width: 100%
```

On 60 simulated stacks like this one the flux came out 16 % low on average,
with an error that held the truth only once in seven
([Validation](https://vista-hpc.readthedocs.io/en/latest/extraction/validation.html)).
A single stack can land closer or farther by chance.

What to do with real data: if you can tell the compact galaxies from the
extended ones (from the continuum, from another line, from their images),
stack them separately, as above. If you cannot, the flux of the stack is a
lower limit of this kind, and the plot against the baseline length is where
to see it.

```python
print("points alone:    ", flux(pts))
print("extended alone:  ", flux(ext))
print("all together:    ", flux(mix))
print("expected (each): ", f"{simulate.expected(mixed)['flux'][0]:.3f} Jy km/s")
```

```text
points alone:     0.371 ± 0.017 Jy km/s
extended alone:   0.409 ± 0.059 Jy km/s
all together:     0.350 ± 0.038 Jy km/s
expected (each):  0.400 Jy km/s
```
