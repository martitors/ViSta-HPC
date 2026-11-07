# 5. Searching for the line (`search`)

Centre and width free, over the velocity window. The significance accounts
for having looked at many centres, widths and sizes: it is the probability
that noise alone gives a peak as high, from `n_null` noise realisations with
the measured noise and correlation (and, with `n_null_phase`, from stacks
with random phases).

```python
search = measure_line("line_stats.npy", LIST, line, weighting,
                      LineFitConfig(mode="search", n_null=500, n_null_phase=100),
                      boot, output_prefix="search")
search0 = measure_line("line_stats_noiseless.npy", LIST0, line, weighting,
                       LineFitConfig(mode="search", **NOISELESS),
                       BootstrapConfig(n_realisations=0), output_prefix="search_noiseless",
                       verbose=False)
print()
compare(search, search0, "search")
save(add_noiseless(pl.plot_profile(pl.load("search"), unit="Jy", rebin=2, reference=REF_PROFILE),
                   pl.load("search_noiseless"), "profile"), "search_profile")
```

```text

search   detection S/N 11.1, 10.7 sigma
                                true  no noise        with noise  (m-t)/err
   line flux [Jy km/s]         0.400     0.400    0.424 ± 0.0623       +0.4
   peak [mJy]                  1.253     1.254    1.029  ± 0.151       -1.5
   centre v0 [km/s]            0.000    -0.088    0.925   ± 17.3       +0.1
   FWHM [km/s]               300.000   299.683  387.099   ± 44.1       +2.0
   size [arcsec]               0.800     0.800    0.769  ± 0.127       -0.2
   window flux [Jy km/s]       0.400     0.400    0.437 ± 0.0646       +0.6
```

```{image} figures/05_01.png
:width: 100%
```
