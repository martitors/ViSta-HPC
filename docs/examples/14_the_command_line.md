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
[ViSta       0.3s] z_ref = 2.6596 (median of the sample)
```

```text
Stack      cli/stack.zarr: 433,440 rows, 90 channels, 8 s
Line       345.796 GHz, 12 sources: detected at 18.5 sigma (S/N 18.8, searched over centre and width)
  flux     0.4035 ± 0.04 Jy km/s, the area of the Gaussian fit
           because the area agrees with the Gaussian (-0.6 sigma)
           Gaussian fit 0.4035 ± 0.04; area in the window (-320 to +320 km/s) 0.3917 ± 0.04
  Gaussian centre +15 ± 10 km/s, FWHM 284 ± 21 km/s, peak 1.34 ± 0.13 mJy
  source   Gaussian, FWHM 0.59 ± 0.07 arcsec
           because it fits better than a point source: delta chi2 = 28.1 (at least 9 needed to call it extended)
Continuum  subtracted while stacking and measured on its own: 0.204 ± 0.032 mJy (S/N 6.4), Gaussian, FWHM 0.59 arcsec
           because the source measured on the line is used: its own Gaussian improves the fit by delta chi2 = 1.0 (at least 9 needed to call it extended)
           outside the line window: 0.0036 ± 0.006 mJy (+0.6 sigma): nothing left
Sizes      each on its own: line 0.59 ± 0.07 arcsec, continuum 0.28 ± 0.15 arcsec: compatible (1.9 sigma apart)
Reference  natural weighting, fluxes at z_ref = 2.6596 (median of the sample)
  stack                  cli/stack.zarr
  line_statistics        cli/stack_line_stats.npy
  continuum_statistics   cli/stack_cont_stats.npy
  results                cli/stack_results.json
  continuum_results      cli/stack_continuum_results.json
```
