# 8. Redoing the continuum subtraction

`DATA` stays in the stack, so the subtraction can be redone — for instance
if the first look of §3 shows a feature in the line-free channels, or to see
how much the result depends on the window and the order — and the stack
compressed and measured again, without stacking again. Every subtraction is
recorded in the stack and in the results (`contsub`). Here: a slope instead
of a constant, outside ±600 km/s instead of ±500.

```python
line2 = LineConfig(rest_freq_ghz=REST_GHZ, v_window_kms=(-600, 600))
subtract_continuum("stack.zarr", line2, ContinuumConfig(order=1, exclude_kms=600), verbose=False)
compress_visibilities("stack.zarr", LIST, line2, binning=binning, line_column="CORRECTED_DATA",
                      out_line="line_stats_v2.npy", out_continuum="cont_stats_v2.npy",
                      resume=False, verbose=False)
redo = measure_line("line_stats_v2.npy", LIST, line2, weighting,
                    LineFitConfig(mode="detect", v0_kms=0.0, fwhm_kms=SIM["fwhm_kms"], n_null=200,
                                  n_null_phase=0), boot, output_prefix="redo", verbose=False)
compare(redo, None, "order 1, outside ±600 km/s")
rec = redo.summary["contsub"]
print("\nrecorded:", {k: rec[k] for k in ("mode", "order", "v_window_kms", "exclude_kms")})
save(pl.plot_profile(pl.load("redo"), unit="Jy", reference=REF_PROFILE), "redo_profile")
```

```text
order 1, outside ±600 km/s   detection S/N 10.0, 9.9 sigma
                                true                  with noise  (m-t)/err
   line flux [Jy km/s]         0.400              0.367 ± 0.0477       -0.7
   peak [mJy]                  1.253              1.148  ± 0.149       -0.7
   centre v0 [km/s]            0.000              0.000  (fixed)
   FWHM [km/s]               300.000            300.000  (fixed)
   size [arcsec]               0.800              0.783  ± 0.136       -0.1
   window flux [Jy km/s]       0.400              0.421 ± 0.0822       +0.3

recorded: {'mode': 'separate', 'order': 1, 'v_window_kms': [-600.0, 600.0], 'exclude_kms': 600.0}
```

```{image} figures/08_01.png
:width: 100%
```

A slope fitted on fewer line-free channels adds noise to every channel, so
the error grows; with a flat continuum, as here, order 0 is the better
choice. `order=-1` lets the BIC choose for each dataset.
