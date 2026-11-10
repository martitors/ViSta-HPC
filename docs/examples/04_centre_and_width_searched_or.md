# 4. Centre and width: searched or known

By default the centre and the width are searched within the window, and the
significance accounts for the search (the same search on noise). When they
are known, from another line or an optical redshift, fixing them
(`lo = hi`) turns the fit into a matched filter: a higher S/N and smaller
errors. A wrong assumed width biases the flux.

```python
vista.compare({
    "searched (default)": r,
    "centre and width known": measure(v0_range=(0, 0), fwhm_range=(300, 300)),
    "width assumed 200 km/s": measure(v0_range=(0, 0), fwhm_range=(200, 200)),
}, truth=T)
```

```text
                               flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
searched (default)           0.3804    0.06   gauss   0.3804   0.3732    +16   307   0.73  10.2    9.7σ
centre and width known       0.3768    0.05   gauss   0.3768    0.375     +0   300   0.74  10.0    9.9σ
width assumed 200 km/s        0.304    0.04   gauss    0.304   0.3869     +0   200   0.79  10.1   10.0σ
true                       0.4   continuum 0.0002
```

```{image} figures/04_01.png
:width: 100%
```
