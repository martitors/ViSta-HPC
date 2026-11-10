# 5. The source: point or extended

`model="auto"` (default) fits a point source and a Gaussian and averages them
with their weights. A point source misses the flux of an extended one: in the
uv panel the data fall with baseline, the flat model cannot follow them, and
the flux it extrapolates to zero length is low. A size fixed at the true
value is the best case; a range that excludes the true size
(`theta_range`) biases the flux too.

```python
vista.compare({
    "auto (default)": r,
    "point source": measure(model="point"),
    "Gaussian": measure(model="gauss"),
    "size fixed 0.8\"": measure(model=0.8),
    "size kept within 0.2-0.5\"": measure(theta_range=(0.2, 0.5)),
}, truth=T)
```

```text
                                  flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
auto (default)                  0.3804    0.06   gauss   0.3804   0.3732    +16   307   0.73  10.2    9.7σ
point source                    0.2477   0.036   gauss   0.2477   0.2373    +20   327  point   9.3    8.9σ
Gaussian                        0.3806   0.059   gauss   0.3806   0.3733    +16   307   0.73  10.2    9.7σ
size fixed 0.8"                 0.3951   0.051   gauss   0.3951   0.3881    +15   305   0.80  10.3   10.0σ
size kept within 0.2-0.5"       0.3256   0.052   gauss   0.3256    0.317    +18   315   0.50   9.8    9.4σ
true                          0.4   continuum 0.0002
```

```{image} figures/05_01.png
:width: 100%
```
