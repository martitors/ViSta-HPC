# 11. A line that is not a Gaussian

Every galaxy of this sample has a line with broad wings (a third of the
flux in a component 800 km/s wide, as an outflow would give).

The Gaussian fit follows the core and misses the wings: its area is too
small. The sum in the window assumes no shape and catches them. ViSta sees
that the two disagree by more than their errors allow, and quotes **the
flux in the window** (in bold): the report says so, and why.

```python
truth_wings = simulate.make_sample(
    "wings", n_sources=12, z_range=(2.0, 3.0), rest_freq_ghz=LINE,
    line_flux=0.4, fwhm_kms=250, size_arcsec=0.5, noise=0.05, rmax=600, seed=1,
    nchan=150,                                                             # a wider band
    extra_lines=[dict(rest_freq_ghz=LINE, flux_ratio=0.5, fwhm_kms=800)])  # the wings

w = ViSta(truth_wings["input_list"], output_name="wings")
w.stack(line_freq=LINE, line_window=1000, velocity_range=1400)
w.extract()
wings = w.run()
print(wings)
```

```text
[ViSta      17.4s] WARNING: measure: the line is not Gaussian (double peak, wings?): the area and the Gaussian flux differ by +2.4 sigma; the area in the window is quoted
```

```text
[ViSta       0.5s] z_ref = 2.5250 (median of the sample)
```

```text
Stack      wings.zarr: 433,440 rows, 140 channels, 8 s
Line       345.796 GHz, 12 sources: detected at 48.7 sigma (S/N 48.8, searched over centre and width)
  flux     0.6295 ± 0.045 Jy km/s, the area under the spectrum in the window
           because the area and the Gaussian fit differ by +2.4 sigma: the line is not a Gaussian
           Gaussian fit 0.5594 ± 0.033; area in the window (-1000 to +1000 km/s) 0.6295 ± 0.045
  Gaussian centre +4 ± 4 km/s, FWHM 307 ± 13 km/s, peak 1.71 ± 0.1 mJy
  source   Gaussian, FWHM 0.53 ± 0.03 arcsec
           because it fits better than a point source: delta chi2 = 155.4 (at least 9 needed to call it extended)
Continuum  subtracted while stacking and measured on its own: none detected (-0.0254 ± 0.017 mJy)
           outside the line window: -1.6e-05 ± 0.003 mJy (-0.0 sigma): nothing left
Reference  natural weighting, fluxes at z_ref = 2.5250 (median of the sample)
```

```python
wings.plot("line")
```

```{image} figures/11_01.png
:width: 100%
```

```python
print("flux in the window (quoted):", flux(wings, "flux_window"))
print("flux of the Gaussian fit:   ", flux(wings, "flux_fit"))
print("expected:                   ", f"{simulate.expected(truth_wings)['flux'][0]:.3f} Jy km/s")
```

```text
flux in the window (quoted): 0.630 ± 0.045 Jy km/s
flux of the Gaussian fit:    0.559 ± 0.033 Jy km/s
expected:                    0.600 Jy km/s
```
