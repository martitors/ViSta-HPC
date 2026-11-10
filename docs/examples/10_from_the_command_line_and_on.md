# 10. From the command line, and on your own data

`run_vista.sh` (or `python -m vista`) has the same options as `ViSta()`,
`stack()` and `extract()`, one flag each; `--extract` asks for the
measurement, as `v.extract()` does (the package must be importable:
`PYTHONPATH` points to the repository). It stacks again, so it takes as long
as §3–4; the results go to `cli/`:

```python
fw = SIM["fwhm_kms"]
cmd = (f"PYTHONPATH={REPO} bash {REPO}/run_vista.sh --input {LIST} --line-freq {REST_GHZ}GHz "
       f"--line-window {WINDOW} --velocity-range 900 --output-name cli/stack "
       f"--extract --v0-range 0:0 --fwhm-range {fw:g}:{fw:g} --figures-unit Jy")
print(cmd, "\n")
r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=WORK)
print(r.stdout if r.returncode == 0 else (r.stdout + r.stderr)[-3000:])
print(f"true line flux {T['flux']:.4f}")
```

```text
PYTHONPATH=/content/ViSta-HPC bash /content/ViSta-HPC/run_vista.sh --input sim/input_list.txt --line-freq 345.79599GHz --line-window 500 --velocity-range 900 --output-name cli/stack --extract --v0-range 0:0 --fwhm-range 300:300 --figures-unit Jy
```

```text
[ViSta       8.1s] z_ref = 2.6218 (median of the sample)
RunResult(
  stack: cli/stack.zarr (433,440 rows, 90 channels, 7 s)
  line 1 (345.796 GHz): F = 0.369 ± 0.044 (Gaussian), v0 = +0 km/s, FWHM = 300 km/s, 10.9 sigma
  z_ref = 2.6218 (median of the sample)
  continuum: 0.000216 ± 3.9e-05, size 0.80"
)
  stack                  cli/stack.zarr
  line_statistics        cli/stack_line_stats.npy
  continuum_statistics   cli/stack_cont_stats.npy
  results                cli/stack_results.json
  continuum_results      cli/stack_continuum_results.json

true line flux 0.4000
```

On your own data:

1. write the input list, one line per dataset:
   `<ms_path> <redshift> <RA> <Dec> [FIELD_ID] [SPW_IDS] [NORM]`, with the
   position of the galaxy (hh:mm:ss.s ±dd:mm:ss.s);
   `python examples/inspect_ms.py input_list.txt --line-freq 345.796GHz --all`
   reads the bands of all the datasets and suggests `velocity_range`,
   `line_window` and `annuli`;
2. `v = ViSta(input_list, output_name=...)`, then `v.stack(line_freq=...,
   line_window=...)` with a window wide enough to contain the whole line
   (and one per line if the band holds more than one), `v.extract(...)` and
   `v.run()`: the stack, the continuum subtraction, the compression and the
   measurement;
3. look at the stack (§3), at `line_free_rms_over_noise` and at the
   `continuum_residual` check; if needed, redo the subtraction (§8);
4. measure again as needed (`run(only="measure")`, §4–6): limits on the
   centre and the width when you know them; check `significance_sigma`,
   `imaginary_rms_over_noise`, the size test, the two fluxes and which one is
   quoted, and the ratio of bootstrap to thermal error (much larger than 1:
   the galaxies differ more than the noise);
5. draw the figures with `examples/plot_line.py`: colours, lines and fonts
   are in its `STYLE` dictionary (or a matplotlib style file,
   `pl.apply_style("my.mplstyle")`).

`./run_vista.sh --help` lists every option; the documentation,
<https://vista-hpc.readthedocs.io>, explains them.

```python
# the files written in this run
for f in sorted(os.listdir(WORK)):
    if not os.path.isdir(f):
        print(f"{f:36s} {os.path.getsize(f) / 1e3:10.1f} kB")
```

```text
one_noiseless.txt                           0.1 kB
one_noiseless_cont_stats.npy               93.5 kB
one_noiseless_diagnostics.npz               9.3 kB
one_noiseless_line_stats.npy               93.5 kB
one_noiseless_noise.txt                     6.9 kB
one_noiseless_profile.txt                  20.8 kB
one_noiseless_results.json                  9.6 kB
one_noiseless_uvamp.txt                     2.1 kB
one_noisy.txt                               0.1 kB
one_noisy_cont_stats.npy                   93.4 kB
one_noisy_diagnostics.npz                  54.0 kB
one_noisy_line_stats.npy                   93.4 kB
one_noisy_noise.txt                         6.9 kB
one_noisy_profile.txt                      20.8 kB
one_noisy_results.json                      9.9 kB
one_noisy_uvamp.txt                         2.1 kB
stack_cont_stats.npy                     1101.2 kB
stack_detect_continuum_results.json         2.1 kB
stack_detect_continuum_spectrum.txt         9.1 kB
stack_detect_continuum_uvamp.png           58.4 kB
stack_detect_continuum_uvamp.txt            1.8 kB
stack_detect_diagnostics.npz               58.1 kB
stack_detect_noise.png                     94.7 kB
stack_detect_noise.txt                      6.9 kB
stack_detect_profile.png                  112.2 kB
stack_detect_profile.txt                   22.7 kB
stack_detect_results.json                  12.3 kB
stack_detect_uv.png                        47.3 kB
stack_detect_uvamp.txt                      2.1 kB
stack_line_stats.npy                     1101.2 kB
stack_noiseless_cont_stats.npy           1101.3 kB
stack_noiseless_detect_continuum_results.json        2.1 kB
stack_noiseless_detect_continuum_spectrum.txt        9.1 kB
stack_noiseless_detect_continuum_uvamp.png       69.1 kB
stack_noiseless_detect_continuum_uvamp.txt        1.8 kB
stack_noiseless_detect_diagnostics.npz        9.3 kB
stack_noiseless_detect_noise.png          110.4 kB
stack_noiseless_detect_noise.txt            6.9 kB
stack_noiseless_detect_profile.png         96.3 kB
stack_noiseless_detect_profile.txt         22.7 kB
stack_noiseless_detect_results.json        11.2 kB
stack_noiseless_detect_uv.png              50.5 kB
stack_noiseless_detect_uvamp.txt            2.1 kB
stack_noiseless_line_stats.npy           1101.3 kB
stack_noiseless_plain_continuum_results.json        2.1 kB
stack_noiseless_plain_continuum_spectrum.txt        9.1 kB
stack_noiseless_plain_continuum_uvamp.png       61.2 kB
stack_noiseless_plain_continuum_uvamp.txt        1.8 kB
stack_noiseless_plain_diagnostics.npz        9.3 kB
stack_noiseless_plain_noise.png           109.6 kB
stack_noiseless_plain_noise.txt             6.9 kB
stack_noiseless_plain_profile.png          90.7 kB
stack_noiseless_plain_profile.txt          22.7 kB
stack_noiseless_plain_results.json         11.1 kB
stack_noiseless_plain_uv.png               50.1 kB
stack_noiseless_plain_uvamp.txt             2.1 kB
stack_noiseless_search_continuum_results.json        2.1 kB
stack_noiseless_search_continuum_spectrum.txt        9.1 kB
stack_noiseless_search_continuum_uvamp.png       69.1 kB
stack_noiseless_search_continuum_uvamp.txt        1.8 kB
stack_noiseless_search_diagnostics.npz        9.3 kB
stack_noiseless_search_noise.png          110.4 kB
stack_noiseless_search_noise.txt            6.9 kB
stack_noiseless_search_profile.png         96.3 kB
stack_noiseless_search_profile.txt         22.7 kB
stack_noiseless_search_results.json        12.2 kB
stack_noiseless_search_uv.png              50.5 kB
stack_noiseless_search_uvamp.txt            2.1 kB
stack_nosub_diagnostics.npz                51.5 kB
stack_nosub_line_stats.npy               1100.8 kB
stack_nosub_noise.txt                       6.9 kB
stack_nosub_profile.txt                    22.7 kB
stack_nosub_results.json                   11.8 kB
stack_nosub_uvamp.txt                       2.1 kB
stack_plain_continuum_results.json          2.1 kB
stack_plain_continuum_spectrum.txt          9.1 kB
stack_plain_continuum_uvamp.png            60.9 kB
stack_plain_continuum_uvamp.txt             1.8 kB
stack_plain_diagnostics.npz                55.3 kB
stack_plain_noise.png                     102.9 kB
stack_plain_noise.txt                       6.9 kB
stack_plain_profile.png                   110.8 kB
stack_plain_profile.txt                    22.7 kB
stack_plain_results.json                   11.8 kB
stack_plain_uv.png                         48.3 kB
stack_plain_uvamp.txt                       2.1 kB
stack_redo_continuum_results.json           2.1 kB
stack_redo_continuum_spectrum.txt           9.1 kB
stack_redo_continuum_uvamp.png             59.0 kB
stack_redo_continuum_uvamp.txt              1.8 kB
stack_redo_diagnostics.npz                 54.4 kB
stack_redo_noise.png                       89.7 kB
stack_redo_noise.txt                        6.9 kB
stack_redo_profile.png                    111.7 kB
stack_redo_profile.txt                     22.7 kB
stack_redo_results.json                    12.0 kB
stack_redo_uv.png                          46.7 kB
stack_redo_uvamp.txt                        2.1 kB
stack_search_continuum_results.json         2.1 kB
stack_search_continuum_spectrum.txt         9.1 kB
stack_search_continuum_uvamp.png           58.4 kB
stack_search_continuum_uvamp.txt            1.8 kB
stack_search_diagnostics.npz               57.9 kB
stack_search_noise.png                     95.2 kB
stack_search_noise.txt                      6.9 kB
stack_search_profile.png                  114.9 kB
stack_search_profile.txt                   22.7 kB
stack_search_results.json                  13.3 kB
stack_search_uv.png                        47.7 kB
stack_search_uvamp.txt                      2.1 kB
stack_variant_continuum_results.json        2.1 kB
stack_variant_continuum_spectrum.txt        9.1 kB
stack_variant_continuum_uvamp.txt           1.8 kB
stack_variant_diagnostics.npz              56.2 kB
stack_variant_noise.txt                     6.9 kB
stack_variant_profile.txt                  22.7 kB
stack_variant_results.json                 11.9 kB
stack_variant_uvamp.txt                     2.1 kB
```
