# 11. Two lines in the band

A third sample: every galaxy has a second line, half as bright, 400 km/s from
the first. With only the first line listed, the second sits partly in its
window (and its flux is added) and partly where the continuum is fitted.
Listed both, with windows that overlap, they are fitted together and
separated. Lines whose windows do not overlap are measured each on its own.

```python
LINE2 = LINE * (1 - 400 / 299792.458)          # 400 km/s from the first
truth_2 = simulate.make_sample(
    "two", n_sources=12, z_range=(2.0, 3.0), rest_freq_ghz=LINE,
    line_flux=0.4, fwhm_kms=250, size_arcsec=0.5, noise=0.05, rmax=600, seed=2, nchan=150,
    extra_lines=[dict(rest_freq_ghz=LINE2, flux_ratio=0.5, fwhm_kms=250)])

d = ViSta(truth_2["input_list"], output_name="two")
d.extract()
d.stack(line_freq=LINE, line_window=300, velocity_range=1400)
first_only = d.run()
d.stack(line_freq=[LINE, LINE2], line_window=300, velocity_range=1400)
both = d.run()                                  # a new stack: the windows changed

vista.compare({"only the first listed": first_only, "both listed": both},
              truth=simulate.expected(truth_2))
```

```text
[ViSta      23.5s] WARNING: measure: the real part in the line-free channels scatters 1.31 x the noise of the visibilities: residual continuum or baseline effects, the thermal errors are optimistic there
```

```text
[ViSta      23.5s] z_ref = 2.2867 (median of the sample)
```

```text
[ViSta      18.2s] WARNING: measure: line 2: its window overlaps line(s) 1, so its area flux holds part of them (+8.5 sigma from the Gaussian flux); the Gaussian flux of the joint fit separates them, and is quoted
```

```text
                              flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
only the first listed       0.3871   0.038   gauss   0.3871   0.3762    +12   274   0.48  36.1   36.0σ
both listed (1)             0.4021   0.021   gauss   0.4021   0.4264     +4   254   0.51  37.4   37.3σ
both listed (2)             0.1729   0.015   gauss   0.1729   0.2553     -9   223   0.51  36.1   36.0σ
true                      0.4, 0.2
```

```{image} figures/11_01.png
:width: 100%
```
