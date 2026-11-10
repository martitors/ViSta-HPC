# 5. Searching for the line within limits

Centre and width free, within their limits: by default the centre anywhere in
the window and the width from two channels to the window; `v0_range` and
`fwhm_range` narrow them (from another line, an expected width...). The
significance accounts for having looked at many centres, widths and sizes:
it is the probability that noise alone gives a peak as high, from `n_null`
noise realisations with the measured noise and correlation (and, with
`n_null_phase`, from stacks with random phases).

```python
stack.extract(**MEASURE, n_null=500, n_null_phase=100, n_bootstrap=200, seed=1)
search = stack.run(only="measure", tag="search")
stack0.extract(**MEASURE, **NOISELESS)
search0 = stack0.run(only="measure", tag="search")
print()
compare(search.line, search0.line, "search")
save(add_noiseless(pl.plot_profile(pl.load("stack_search"), unit="Jy", rebin=2, reference=REF_PROFILE),
                   pl.load("stack_noiseless_search"), "profile"), "search_profile")
```

```text
[ViSta       5.0s] z_ref = 2.6218 (median of the sample)
```

```text
[ViSta       0.7s] z_ref = 2.6218 (median of the sample)
```

```text

search   detection S/N 11.1, 10.7 sigma; flux quoted: gauss
                                true  no noise        with noise  (m-t)/err
   line flux [Jy km/s]         0.400     0.400    0.424 ± 0.0623       +0.4
     Gaussian                  0.400     0.400    0.424 ± 0.0623       +0.4
     area (window)             0.400     0.400    0.437 ± 0.0646       +0.6
   peak [mJy]                  1.253     1.254    1.029  ± 0.151       -1.5
   centre v0 [km/s]            0.000    -0.088    0.925   ± 17.3       +0.1
   FWHM [km/s]               300.000   299.683  387.097   ± 52.4       +1.7
   size [arcsec]               0.800     0.800    0.769  ± 0.127       -0.2
```

```{image} figures/05_01.png
:width: 100%
```
