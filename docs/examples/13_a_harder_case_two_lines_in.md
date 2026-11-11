# 13. A harder case: two lines in the band

A fourth sample: every galaxy has a second line, half as bright, 400 km/s
from the first. If only the first is listed, part of the second falls in its
window and part where the continuum is fitted: the fit has one Gaussian for
two lines.

```python
LINE2 = LINE * (1 - 400 / 299792.458)          # 400 km/s from the first
truth_two = simulate.make_sample(
    "two", n_sources=12, z_range=(2.0, 3.0), rest_freq_ghz=LINE,
    line_flux=0.4, fwhm_kms=250, size_arcsec=0.5, noise=0.05, rmax=600, seed=2, nchan=150,
    extra_lines=[dict(rest_freq_ghz=LINE2, flux_ratio=0.5, fwhm_kms=250)])

d = ViSta(truth_two["input_list"], output_name="two")
d.stack(line_freq=LINE, line_window=300, velocity_range=1400)      # only the first line
d.extract()
first_only = d.run()
first_only.plot("profile")
```

```text
[ViSta      26.7s] WARNING: measure: the real part in the line-free channels scatters 1.31 x the noise of the visibilities: residual continuum or baseline effects, the thermal errors are optimistic there
```

```text
[ViSta      26.7s] z_ref = 2.2867 (median of the sample)
```

```{image} figures/13_01.png
:width: 100%
```

Listing both (`line_freq=[LINE, LINE2]`, each with its window) makes a new
stack, because the channels kept out of the continuum fit change. The
windows overlap, so the two lines are fitted **together**, one Gaussian
each, and separated. (Lines whose windows do not overlap are measured each
on its own.)

```python
d.stack(line_freq=[LINE, LINE2], line_window=300, velocity_range=1400)
both = d.run()
print(both)
```

```text
[ViSta      17.1s] WARNING: measure: line 2: its window overlaps line(s) 1, so its area flux holds part of them (+8.5 sigma from the Gaussian flux); the Gaussian flux of the joint fit separates them, and is quoted
```

```text
Stack      two.zarr: 433,440 rows, 140 channels, 9 s
Line 1     345.796 GHz, 12 sources: detected at 37.3 sigma (S/N 37.4, searched over centre and width)
  flux     0.4021 ± 0.021 Jy km/s, the area of the Gaussian fit
           because the area agrees with the Gaussian (+1.9 sigma)
           Gaussian fit 0.4021 ± 0.021; area in the window (-300 to +300 km/s) 0.4264 ± 0.019
  Gaussian centre +4 ± 4 km/s, FWHM 254 ± 11 km/s, peak 1.49 ± 0.078 mJy
  source   Gaussian, FWHM 0.51 ± 0.04 arcsec
           because it fits better than a point source: delta chi2 = 96.4 (at least 9 needed to call it extended)
Line 2     345.335 GHz, 12 sources: detected at 36.0 sigma (S/N 36.1, searched over centre and width)
  flux     0.1729 ± 0.015 Jy km/s, the area of the Gaussian fit
           because the window overlaps line(s) 1: only the joint Gaussian fit separates them
           Gaussian fit 0.1729 ± 0.015; area in the window (-300 to +300 km/s) 0.2553 ± 0.016
  Gaussian centre -9 ± 8 km/s, FWHM 223 ± 19 km/s, peak 0.728 ± 0.063 mJy
Continuum  subtracted while stacking and measured on its own: none detected (0.0159 ± 0.021 mJy)
           outside the line window: -0.0018 ± 0.0019 mJy (-1.0 sigma): nothing left
Reference  natural weighting, fluxes at z_ref = 2.2867 (median of the sample)
```

```python
both.plot("profile")
```

```{image} figures/13_02.png
:width: 100%
```
