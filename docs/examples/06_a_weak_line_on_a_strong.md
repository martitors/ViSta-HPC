# 6. A weak line on a strong continuum

The opposite case, on a new sample: a bright continuum (1 mJy) and a faint
line (0.2 Jy km/s), all galaxies 0.8″ across.

```python
truth_weak = simulate.make_sample(
    "weak", n_sources=12, z_range=(2.0, 3.0), rest_freq_ghz=LINE,
    line_flux=0.2, fwhm_kms=300,           # a faint line
    cont_flux=1e-3,                        # a bright continuum: 1 mJy
    size_arcsec=0.8, noise=0.1, kappa=1.3, rmax=600, seed=22)
expected_weak = simulate.expected(truth_weak)

k = ViSta(truth_weak["input_list"], output_name="weak")
k.stack(line_freq=LINE, line_window=500, velocity_range=900)
k.extract()
weak = k.run()
print(weak)
```

```text
[ViSta       8.8s] z_ref = 2.5073 (median of the sample)
```

```text
Stack      weak.zarr: 433,440 rows, 90 channels, 5 s
Line       345.796 GHz, 12 sources: detected at 4.9 sigma (S/N 5.6, searched over centre and width)
  flux     0.2049 ± 0.061 Jy km/s, the area of the Gaussian fit
           because the area agrees with the Gaussian (+0.0 sigma)
           Gaussian fit 0.2049 ± 0.061; area in the window (-500 to +500 km/s) 0.2058 ± 0.063
  Gaussian centre -3 ± 34 km/s, FWHM 343 ± 88 km/s, peak 0.562 ± 0.17 mJy
  source   Gaussian, FWHM 0.77 arcsec
           because its own Gaussian improves the fit by delta chi2 = 7.2 only (at least 9 needed to call it extended), and the continuum is extended: the size of the continuum is used
Continuum  subtracted while stacking and measured on its own: 1.01 ± 0.03 mJy (S/N 33.9), Gaussian, FWHM 0.77 ± 0.05 arcsec
           because its own Gaussian improves the fit by delta chi2 = 104.8 (at least 9 needed to call it extended)
           outside the line window: -0.0017 ± 0.0057 mJy (-0.3 sigma): nothing left
Sizes      each on its own: line 0.91 arcsec, continuum 0.77 ± 0.05 arcsec
Reference  natural weighting, fluxes at z_ref = 2.5073 (median of the sample)
```

The continuum measures its size well. The line alone cannot: a Gaussian
source does not fit clearly better than a point. Since the continuum says
that these galaxies are extended, the line is fitted **with the size of the
continuum** (the report says so; `model="continuum"` asks for it).

```python
weak.plot("continuum_uv")
```

```{image} figures/06_01.png
:width: 100%
```

```python
weak.plot("line_uv")
```

```{image} figures/06_02.png
:width: 100%
```

```python
weak.plot("line")
```

```{image} figures/06_03.png
:width: 100%
```

What the line would give as a point source (`model="point"`;
`measure="line"` repeats only the line): too low.

```python
k.extract(model="point", measure="line")
weak_point = k.run(only="measure", tag="point")
print("with the size of the continuum:", flux(weak))
print("as a point:                    ", flux(weak_point))
print("expected:                      ", f"{expected_weak['flux'][0]:.3f} Jy km/s")
```

```text
with the size of the continuum: 0.205 ± 0.061 Jy km/s
as a point:                     0.116 ± 0.035 Jy km/s
expected:                       0.200 Jy km/s
```
