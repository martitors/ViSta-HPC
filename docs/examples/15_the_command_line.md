# 15. The command line

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
[ViSta       0.4s] z_ref = 2.6596 (median of the sample)
```

```text
Stack      cli/stack.zarr: 433,440 rows, 90 channels, 10 s
Line       345.796 GHz, 12 sources: detected at 13.8 sigma (S/N 14.1, searched over centre and width)
  flux     0.4314 ± 0.052 Jy km/s, the area of the Gaussian fit
           because the area agrees with the Gaussian (-0.0 sigma)
           Gaussian fit 0.4314 ± 0.052; area in the window (-320 to +320 km/s) 0.4301 ± 0.049
  Gaussian centre -12 ± 11 km/s, FWHM 295 ± 26 km/s, peak 1.37 ± 0.17 mJy
  source   Gaussian, FWHM 0.95 ± 0.12 arcsec
           because it fits better than a point source: delta chi2 = 52.8 (at least 9 needed to call it extended)
Continuum  subtracted while stacking and measured on its own: 0.0895 ± 0.045 mJy (S/N 2.0), Gaussian, FWHM 0.95 arcsec
           because the source measured on the line is used: its own Gaussian improves the fit by delta chi2 = 0.6 (at least 9 needed to call it extended)
           outside the line window: 0.0054 ± 0.0078 mJy (+0.7 sigma): nothing left
Sizes      each on its own: line 0.95 ± 0.12 arcsec, continuum 0.76 ± 1.09 arcsec: compatible (0.2 sigma apart)
Reference  natural weighting, fluxes at z_ref = 2.6596 (median of the sample)
  stack                  cli/stack.zarr
  line_statistics        cli/stack_line_stats.npy
  continuum_statistics   cli/stack_cont_stats.npy
  results                cli/stack_results.json
  continuum_results      cli/stack_continuum_results.json
```
