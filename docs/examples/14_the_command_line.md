# 14. The command line

`python -m vista` (or `run_vista.sh`) has the same options as `ViSta()`,
`stack()` and `extract()`, one flag each; `--extract` asks for the
measurement. It writes the same files: the stack, `*_results.json` with
every number, and the figures as PNG.

```bash
PYTHONPATH={REPO} python -m vista --input sample/input_list.txt --output-name cli/stack \
    --line-freq 345.796GHz --line-window 500 --velocity-range 900 \
    --extract --window 320
```

```text
[ViSta      14.2s] z_ref = 2.2867 (median of the sample)
```

```text
Stack      cli/stack.zarr: 433,440 rows, 90 channels, 7 s
Line       345.796 GHz, 12 sources: detected at 11.4 sigma (S/N 11.8, searched over centre and width)
  flux     0.3656 ± 0.054 Jy km/s, the area of the Gaussian fit
           because the area agrees with the Gaussian (-0.5 sigma)
           Gaussian fit 0.3656 ± 0.054; area in the window (-320 to +320 km/s) 0.3508 ± 0.056
  Gaussian centre -26 ± 11 km/s, FWHM 267 ± 32 km/s, peak 1.29 ± 0.19 mJy
  source   Gaussian, FWHM 0.67 ± 0.12 arcsec
           because it fits better than a point source: delta chi2 = 18.1 (at least 9 needed to call it extended)
Continuum  subtracted while stacking and measured on its own: 0.177 ± 0.041 mJy, FWHM 1.16 arcsec
           outside the line window: 0.00044 ± 0.0036 mJy (+0.1 sigma): nothing left
Reference  natural weighting, fluxes at z_ref = 2.2867 (median of the sample)
  stack                  cli/stack.zarr
  line_statistics        cli/stack_line_stats.npy
  continuum_statistics   cli/stack_cont_stats.npy
  results                cli/stack_results.json
  continuum_results      cli/stack_continuum_results.json
```
