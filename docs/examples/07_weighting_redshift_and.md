# 7. Weighting, redshift and primary beam

* `weighting_scheme="democratic"`: every galaxy counts the same, instead of
  the deepest ones dominating (`"natural"`): the mean of the galaxies rather
  than of the data; here the same within the errors, which are larger.
* `z_norm=False`: the galaxies are not transported to the same redshift; the
  nearer ones are brighter, so the stack measures something else than the
  true value at `z_ref`.
* `pb_corr=False`: no correction for the primary beam: the galaxies, up to
  20″ from the phase centre, where the primary beam is 0.7–0.95, look fainter.

```python
vista.compare({
    "default": r,
    "democratic weighting": measure(weighting_scheme="democratic"),
    "no redshift rescaling": measure(z_norm=False),
    "no primary-beam correction": measure(pb_corr=False),
}, truth=T)
```

```text
[ViSta       9.7s] no redshift rescaling, no flux normalisation
```

```text
[ViSta       9.3s] z_ref = 2.5250 (median of the sample)
```

```text
                                   flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
default                          0.3804    0.06   gauss   0.3804   0.3732    +16   307   0.73  10.2    9.7σ
democratic weighting             0.3447    0.08   gauss   0.3447   0.3581     +4   343  point   7.8    7.2σ
no redshift rescaling            0.3868   0.075   gauss   0.3868   0.3785    +12   313   0.69  11.0   10.6σ
no primary-beam correction       0.3352   0.055   gauss   0.3352   0.3294    +15   305   0.73  10.2    9.7σ
true                           0.4   continuum 0.0002
```

```{image} figures/07_01.png
:width: 100%
```
