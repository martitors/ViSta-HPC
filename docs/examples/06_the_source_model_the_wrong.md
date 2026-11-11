# 6. The source model: the wrong one and the right one

What if we assume that the stacked source is a point? `model="point"` fits
the same flux on every baseline. The uv figure shows why it is wrong here:
the points fall with the baseline and the flat line cannot follow them. It
settles on an average, below the short baselines, and the flux is
underestimated.

```python
v.extract(**base, model="point")
point = v.run(only="measure", tag="point")
point.plot("uv")
```

```{image} figures/06_01.png
:width: 100%
```

The default, `model="auto"`, fits both and keeps the Gaussian only if the
data ask for it. Side by side (in the bottom-left panel the two models; on
the right the fluxes against the expected one):

```python
vista.compare({"model='auto'": r, "model='point'": point}, expected=expected)
```

```text
                      flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
model='auto'        0.3656   0.054   gauss   0.3656   0.3509    -26   267   0.67  11.8   11.4σ
model='point'       0.2368   0.031   gauss   0.2368   0.2279    -33   274  point  10.9   10.6σ
expected          0.3932   continuum 0.000197
```

```{image} figures/06_02.png
:width: 100%
```

When the size is known from elsewhere (the continuum, another line), it can
be given: `model=0.8` fixes it at 0.8″, `model_prior=(0.8, 0.15)` ties it
softly, `theta_range=(0.3, 2)` keeps it within limits. A fixed size has the
smallest error on the flux, if the size is right.

```python
v.extract(**base, model=0.8)
fixed = v.run(only="measure", tag="fixed")

vista.compare({"size fitted": r, "size fixed at 0.8\"": fixed}, expected=expected)
```

```text
                           flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
size fitted              0.3656   0.054   gauss   0.3656   0.3509    -26   267   0.67  11.8   11.4σ
size fixed at 0.8"       0.3934   0.047   gauss   0.3934   0.3774    -25   265   0.80  11.8   11.5σ
expected               0.3932   continuum 0.000197
```

```{image} figures/06_03.png
:width: 100%
```
