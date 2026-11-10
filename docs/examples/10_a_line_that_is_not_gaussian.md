# 10. A line that is not Gaussian

A second sample: the same kind of galaxies, but their line has broad wings
(a third of the flux in a component 800 km/s wide), and less noise. The
Gaussian fit follows the core and misses the wings; the sum over a window
wide enough catches them. `flux_measure="auto"` sees that the two disagree
and quotes the area.

```python
truth_w = simulate.make_sample(
    "wings", n_sources=12, z_range=(2.0, 3.0), rest_freq_ghz=LINE,
    line_flux=0.4, fwhm_kms=250, size_arcsec=0.5, noise=0.05, rmax=600, seed=1,
    nchan=150,                                                   # a wider band
    extra_lines=[dict(rest_freq_ghz=LINE, flux_ratio=0.5, fwhm_kms=800)])   # the wings

w = ViSta(truth_w["input_list"], output_name="wings")
w.stack(line_freq=LINE, line_window=1000, velocity_range=1400)
w.extract(flux_measure="auto")
auto = w.run()                                                   # stack and measure
w.extract(flux_measure="gauss")
gauss = w.run(only="measure", tag="gauss")
w.extract(flux_measure="area")
area = w.run(only="measure", tag="area")

vista.compare({"Gaussian": gauss, "area": area, "auto": auto}, truth=simulate.expected(truth_w))
```

```text
[ViSta      14.8s] WARNING: measure: the line is not Gaussian (double peak, wings?): the area and the Gaussian flux differ by +2.4 sigma; the sum over the window is quoted
```

```text
[ViSta      14.8s] z_ref = 2.5250 (median of the sample)
```

```text
[ViSta      15.3s] WARNING: measure: the line is not Gaussian (double peak, wings?): the area and the Gaussian flux differ by +2.4 sigma; quoted: gauss (flux_measure='gauss')
```

```text
[ViSta      14.6s] WARNING: measure: the line is not Gaussian (double peak, wings?): the area and the Gaussian flux differ by +2.4 sigma; quoted: area (flux_measure='area')
```

```text
                 flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
Gaussian       0.5594   0.033   gauss   0.5594   0.6295     +4   307   0.53  48.8   48.7σ
area           0.6295   0.045    area   0.5594   0.6295     +4   307   0.53  48.8   48.7σ
auto           0.6295   0.045    area   0.5594   0.6295     +4   307   0.53  48.8   48.7σ
true         0.6
```

```{image} figures/10_01.png
:width: 100%
```
