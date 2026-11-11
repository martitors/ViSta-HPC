# 7. The normalisations

Before they are combined, the galaxies are made comparable in two ways. Each
can be switched off to see what it does.

**The redshift** (`z_norm`). The same galaxy is fainter if it is farther.
With `z_norm=True` every galaxy is brought to the flux it would have at
`z_ref`, so the stack is the mean of the same quantity. Without it the stack
is the mean of the fluxes as observed, at different distances. Here the
number is similar, because `z_ref` is the median redshift of the sample; but
it refers to no redshift, it changes with the redshifts of the galaxies that
happen to be in the sample, and the galaxies differ more from one another
(a larger error).

```python
v.extract(**base, z_norm=False)
no_z = v.run(only="measure", tag="no_z")

vista.compare({"z_norm=True (default)": r, "z_norm=False": no_z}, expected=expected)
```

```text
[ViSta      12.5s] no redshift rescaling, no flux normalisation
```

```text
                              flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
z_norm=True (default)       0.3656   0.054   gauss   0.3656   0.3509    -26   267   0.67  11.8   11.4σ
z_norm=False                0.3566    0.06   gauss   0.3566   0.3436    -23   271   0.69  12.4   12.0σ
expected                  0.3932   continuum 0.000197
```

```{image} figures/07_01.png
:width: 100%
```

**The primary beam** (`pb_corr`). An antenna is less sensitive away from the
centre of its field: these galaxies, up to 20″ away, are seen at 70–100 % of
their flux. With `pb_corr=True` every galaxy is corrected for the response
at its position. Without it the stack is low.

```python
v.extract(**base, pb_corr=False)
no_pb = v.run(only="measure", tag="no_pb")

vista.compare({"pb_corr=True (default)": r, "pb_corr=False": no_pb}, expected=expected)
```

```text
[ViSta      13.5s] z_ref = 2.2867 (median of the sample)
```

```text
                               flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
pb_corr=True (default)       0.3656   0.054   gauss   0.3656   0.3509    -26   267   0.67  11.8   11.4σ
pb_corr=False                0.3184   0.051   gauss   0.3184   0.3049    -26   265   0.66  11.7   11.3σ
expected                   0.3932   continuum 0.000197
```

```{image} figures/07_02.png
:width: 100%
```
