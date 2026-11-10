# 6. The window of the area flux

Two fluxes are always given: the integral of the fitted Gaussian, and the sum
of the spectrum over the window. A window narrower than the line loses part
of the area flux, not of the Gaussian one; `flux_measure="auto"` quotes the
Gaussian unless the two disagree and the window holds the whole line. The
window of the area flux can be changed without stacking again.

```python
vista.compare({
    "window ±500 km/s (default)": r,
    "window ±250 km/s": measure(window=250),
    "window ±120 km/s": measure(window=120),
}, truth=T)
```

```text
[ViSta      10.0s] WARNING: measure: the line: the area flux differs from the Gaussian by -2.7 sigma, but the window holds only 64% of the fitted line: widen the window (the Gaussian flux is quoted)
```

```text
                                   flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
window ±500 km/s (default)       0.3804    0.06   gauss   0.3804   0.3732    +16   307   0.73  10.2    9.7σ
window ±250 km/s                 0.3804    0.06   gauss   0.3804    0.355    +16   307   0.73  10.2    9.7σ
window ±120 km/s                 0.3804    0.06   gauss   0.3804   0.2516    +16   307   0.73  10.2    9.7σ
true                           0.4   continuum 0.0002
```

```{image} figures/06_01.png
:width: 100%
```
