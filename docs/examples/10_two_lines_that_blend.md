# 10. Two lines that blend

The same, with the second line only 400 km/s away: the two overlap. Their
windows overlap too, so ViSta fits them **together**, one Gaussian each
with the same source size, and separates them.

```python
NEAR = LINE * (1 - 400 / 299792.458)           # 400 km/s from the first
truth_blend = simulate.make_sample(
    "blend", n_sources=12, z_range=(2.0, 3.0), rest_freq_ghz=LINE,
    line_flux=0.4, fwhm_kms=250, size_arcsec=0.5, noise=0.05, rmax=600, seed=2,
    nchan=150, extra_lines=[dict(rest_freq_ghz=NEAR, flux_ratio=0.5, fwhm_kms=250)])

b = ViSta(truth_blend["input_list"], output_name="blend")
b.stack(line_freq=[LINE, NEAR], line_window=300, velocity_range=(-900, 1300))
b.extract()
blend = b.run()
print(blend)
```

```text
[ViSta      16.5s] WARNING: measure: line 2: its window overlaps line(s) 1, so its area flux holds part of them (+8.3 sigma from the Gaussian flux); the Gaussian flux of the joint fit separates them, and is quoted
```

```text
[ViSta       0.4s] z_ref = 2.2867 (median of the sample)
```

```text
Stack      blend.zarr: 433,440 rows, 110 channels, 10 s
Line 1     345.796 GHz, 12 sources: detected at 35.4 sigma (S/N 35.5, searched over centre and width)
  flux     0.4076 ± 0.023 Jy km/s, the area of the Gaussian fit
           because the area agrees with the Gaussian (+1.9 sigma)
           Gaussian fit 0.4076 ± 0.023; area in the window (-300 to +300 km/s) 0.4325 ± 0.021
  Gaussian centre +4 ± 4 km/s, FWHM 255 ± 11 km/s, peak 1.5 ± 0.086 mJy
  source   Gaussian, FWHM 0.53 ± 0.05 arcsec
           because it fits better than a point source: delta chi2 = 90.8 (at least 9 needed to call it extended)
Line 2     345.335 GHz, 12 sources: detected at 33.8 sigma (S/N 33.9, searched over centre and width)
  flux     0.1762 ± 0.016 Jy km/s, the area of the Gaussian fit
           because the window overlaps line(s) 1: only the joint Gaussian fit separates them
           Gaussian fit 0.1762 ± 0.016; area in the window (-300 to +300 km/s) 0.26 ± 0.017
  Gaussian centre -9 ± 8 km/s, FWHM 224 ± 19 km/s, peak 0.739 ± 0.065 mJy
Continuum  subtracted while stacking and measured on its own: none detected (0.0162 ± 0.014 mJy)
           outside the line window: -8.3e-05 ± 0.0034 mJy (-0.0 sigma): nothing left
Reference  natural weighting, fluxes at z_ref = 2.2867 (median of the sample)
```

```python
blend.plot("line")
```

```{image} figures/10_01.png
:width: 100%
```

```python
for k, line in enumerate(blend["lines"]):
    print(f"line {k + 1}: {line['flux']['value']:.3f} ± {line['flux']['error']:.3f} Jy km/s"
          f"   expected {simulate.expected(truth_blend)['flux'][k]:.3f}")
```

```text
line 1: 0.408 ± 0.023 Jy km/s   expected 0.400
line 2: 0.176 ± 0.016 Jy km/s   expected 0.200
```
