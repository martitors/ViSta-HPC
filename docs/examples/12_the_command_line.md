# 12. The command line

`run_vista.sh` (or `python -m vista`) has the same options as `ViSta()`,
`stack()` and `extract()`, one flag each; `--extract` asks for the
measurement. It writes the same files; `--help` lists every option.

```bash
PYTHONPATH={REPO} python -m vista --input sim/input_list.txt --output-name cli/stack \
    --line-freq 345.796GHz --line-window 500 --velocity-range 900 \
    --extract --v0-range 0:0 --fwhm-range 300:300
```

```text
[ViSta       7.2s] z_ref = 2.5250 (median of the sample)
```

```text
RunResult(
  stack: cli/stack.zarr (433,440 rows, 90 channels, 5 s)
  line 1 (345.796 GHz): F = 0.3768 ± 0.05 (Gaussian), v0 = +0 km/s, FWHM = 300 km/s, 9.9 sigma
  z_ref = 2.5250 (median of the sample)
  continuum: 0.0002474 ± 8.6e-05, size 1.15"
)
  stack                  cli/stack.zarr
  line_statistics        cli/stack_line_stats.npy
  continuum_statistics   cli/stack_cont_stats.npy
  results                cli/stack_results.json
  continuum_results      cli/stack_continuum_results.json
```
