# 8. The weighting

`weighting_scheme="natural"` (default) weights every visibility by its
noise: the deepest observations dominate, and the noise of the stack is the
lowest possible. `"democratic"` gives every galaxy the same total weight:
the stack is the plain mean of the galaxies, with more noise.

They answer two different questions, so they also have two different
expected values when the galaxies differ, as here.

```python
expected_democratic = simulate.expected(truth, weighting="democratic")
print(f"expected, natural:    {expected['flux'][0]:.3f} Jy km/s")
print(f"expected, democratic: {expected_democratic['flux'][0]:.3f} Jy km/s")

v.extract(**base, weighting_scheme="democratic")
democratic = v.run(only="measure", tag="democratic")

vista.compare({"natural": r, "democratic": democratic}, expected=expected)
```

```text
expected, natural:    0.393 Jy km/s
expected, democratic: 0.403 Jy km/s
```

```text
                   flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
natural          0.3656   0.054   gauss   0.3656   0.3509    -26   267   0.67  11.8   11.4σ
democratic       0.3534    0.06   gauss   0.3534   0.3512    -21   237   0.60  10.1    9.7σ
expected       0.3932   continuum 0.000197
```

```{image} figures/08_01.png
:width: 100%
```
