# 9. Two lines far apart

A new sample: every galaxy has a second line, half as bright, 1500 km/s from
the first. `line_freq` takes a list, and every line has its window. The
windows do not touch, so **each line is measured on its own**: its own
centre, width, flux and size.

```python
FAR = LINE * (1 - 1500 / 299792.458)           # 1500 km/s from the first
truth_far = simulate.make_sample(
    "far", n_sources=12, z_range=(2.0, 3.0), rest_freq_ghz=LINE,
    line_flux=0.4, fwhm_kms=250, size_arcsec=0.5, noise=0.05, rmax=600, seed=2,
    nchan=170, band_offset_kms=750,                                # a band that holds both
    extra_lines=[dict(rest_freq_ghz=FAR, flux_ratio=0.5, fwhm_kms=250)])

f = ViSta(truth_far["input_list"], output_name="far")
f.stack(line_freq=[LINE, FAR], line_window=300, velocity_range=(-700, 2200))
f.extract()
far = f.run()
print(far)
```

```text
[ViSta       0.6s] z_ref = 2.2867 (median of the sample)
```

```text
Stack      far.zarr: 433,440 rows, 144 channels, 9 s
Line 1     345.796 GHz, 12 sources: detected at 39.3 sigma (S/N 39.4, searched over centre and width)
  flux     0.3955 ± 0.015 Jy km/s, the area of the Gaussian fit
           because the area agrees with the Gaussian (-0.1 sigma)
           Gaussian fit 0.3955 ± 0.015; area in the window (-300 to +300 km/s) 0.3946 ± 0.015
  Gaussian centre -1 ± 4 km/s, FWHM 254 ± 8 km/s, peak 1.46 ± 0.057 mJy
  source   Gaussian, FWHM 0.45 ± 0.03 arcsec
           because it fits better than a point source: delta chi2 = 72.3 (at least 9 needed to call it extended)
Line 2     344.066 GHz, 12 sources: detected at 20.3 sigma (S/N 20.4, searched over centre and width)
  flux     0.2072 ± 0.016 Jy km/s, the area of the Gaussian fit
           because the area agrees with the Gaussian (-0.1 sigma)
           Gaussian fit 0.2072 ± 0.016; area in the window (-300 to +300 km/s) 0.2064 ± 0.017
  Gaussian centre -8 ± 6 km/s, FWHM 229 ± 16 km/s, peak 0.85 ± 0.067 mJy
  source   Gaussian, FWHM 0.54 ± 0.06 arcsec
           because it fits better than a point source: delta chi2 = 33.7 (at least 9 needed to call it extended)
Continuum  subtracted while stacking and measured on its own: none detected (-0.00246 ± 0.011 mJy)
           outside the line window: -0.0099 ± 0.005 mJy (-2.0 sigma): nothing left
Reference  natural weighting, fluxes at z_ref = 2.2867 (median of the sample)
```

```python
far.plot("line")
```

```{image} figures/09_01.png
:width: 100%
```

```python
for k, line in enumerate(far["lines"]):
    print(f"line {k + 1}: {line['flux']['value']:.3f} ± {line['flux']['error']:.3f} Jy km/s"
          f"   expected {simulate.expected(truth_far)['flux'][k]:.3f}")
```

```text
line 1: 0.395 ± 0.015 Jy km/s   expected 0.400
line 2: 0.207 ± 0.016 Jy km/s   expected 0.200
```
