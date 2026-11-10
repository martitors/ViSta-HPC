# 8. Redoing the continuum subtraction

`DATA` stays in the stack, so the subtraction can be redone — for instance
if the first look of §3 shows a feature in the line-free channels, or to see
how much the result depends on the window and the order — and the stack
compressed and measured again, without stacking again: `run(only=["continuum",
"measure"])`. Every subtraction is recorded in the stack and in the results
(`contsub`). Here: a slope instead of a constant, outside ±600 km/s instead
of ±500; then back to the first subtraction for §9.

The continuum can also be left in the data and fitted together with the
line, `v.stack(..., continuum="fit")`: a constant at zero spacing with the
size of the line, its flux density in `continuum_fit` (and the area flux is
the sum over the window minus it). That needs a stack without the
subtraction, `stack_nosub` below. With `continuum="none"` the continuum would
be taken for part of the line, and the measurement warns.

```python
redo_v = ViSta(LIST, output_name="stack")                   # the same stack
redo_v.stack(line_freq=REST_GHZ, line_window=600, velocity_range=900, continuum_order=1)
redo_v.extract(**MEASURE, **KNOWN, n_null=200, n_null_phase=0, n_bootstrap=200, seed=1)
redo = redo_v.run(only=["continuum", "measure"], tag="redo")
compare(redo.line, None, "order 1, outside ±600 km/s")
rec = redo.line.summary["contsub"]
print("\nrecorded:", {k: rec[k] for k in ("mode", "order", "lines")})
save(pl.plot_profile(pl.load("stack_redo"), unit="Jy", reference=REF_PROFILE), "redo_profile")
stack.run(only="continuum")                       # back to order 0 outside ±500 km/s

nosub = ViSta(LIST, output_name="stack_nosub")    # a stack without the subtraction
nosub.stack(line_freq=REST_GHZ, line_window=WINDOW, velocity_range=900,
            continuum="fit")                      # the continuum fitted with the line
nosub.extract(**MEASURE, **KNOWN, n_null=0, n_null_phase=0, n_bootstrap=200, seed=1,
              figures_draw=False)
joint = nosub.run()
print()
compare(joint.line, None, "continuum not subtracted, fitted with the line")
cf = joint.line.summary["continuum_fit"]
print(f"   continuum [mJy]          {T['cont'] * 1e3:8.3f}           {cf['flux_density'] * 1e3:8.3f} "
      f"± {cf['flux_density_error'] * 1e3:.3f}")
```

```text
[ViSta       3.1s] z_ref = 2.6218 (median of the sample)
```

```text
order 1, outside ±600 km/s   detection S/N 10.0, 9.9 sigma; flux quoted: gauss
                                true                  with noise  (m-t)/err
   line flux [Jy km/s]         0.400              0.367 ± 0.0477       -0.7
     Gaussian                  0.400              0.367 ± 0.0477       -0.7
     area (window)             0.400              0.421 ± 0.0822       +0.3
   peak [mJy]                  1.253              1.148  ± 0.149       -0.7
   centre v0 [km/s]            0.000              0.000  (fixed)
   FWHM [km/s]               300.000            300.000  (fixed)
   size [arcsec]               0.800              0.783  ± 0.136       -0.1

recorded: {'mode': 'separate', 'order': 1, 'lines': {'exclude_between': False, 'rest_freq_ghz': [345.79599], 'window_kms': [[-600.0, 600.0]]}}
```

```{image} figures/08_01.png
:width: 100%
```

```text
[ViSta       3.6s] z_ref = 2.6218 (median of the sample)
```

```text

continuum not subtracted, fitted with the line   detection S/N 10.7, 10.5 sigma; flux quoted: gauss
                                true                  with noise  (m-t)/err
   line flux [Jy km/s]         0.400              0.342 ± 0.0348       -1.7
     Gaussian                  0.400              0.342 ± 0.0348       -1.7
     area (window)             0.400              0.382  ± 0.055       -0.3
   peak [mJy]                  1.253              1.069  ± 0.109       -1.7
   centre v0 [km/s]            0.000              0.000  (fixed)
   FWHM [km/s]               300.000            300.000  (fixed)
   size [arcsec]               0.800              0.751  ± 0.069       -0.7
   continuum [mJy]             0.200              0.239 ± 0.034
```

A slope fitted on fewer line-free channels adds noise to every channel, so
the error grows; with a flat continuum, as here, order 0 is the better
choice. `order="bic"` lets the BIC choose for each dataset.
