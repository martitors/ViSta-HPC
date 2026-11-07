# 10. From the command line, and on your own data

The shell driver of the repository runs the same steps (the package must be
importable: `PYTHONPATH` points to the repository). It stacks again, so it
takes as long as §3–4; the results go to `cli/`:

```python
cmd = (f"PYTHONPATH={REPO} bash {REPO}/run_vista.sh --input {LIST} --rest-freq {REST_GHZ} "
       f"--v-window '-500 500' --velocity-range 900 --contsub before "
       f"--method joint --mode detect --fwhm {SIM['fwhm_kms']:g} --unit Jy --out-dir cli")
print(cmd, "\n")
r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=WORK)
log = r.stdout + r.stderr
print("\n".join(l for l in log.splitlines() if l.startswith("==") or "Line:" in l)
      if r.returncode == 0 else log[-3000:])
print(f"\ntrue line flux {T['flux']:.4f}")
```

```text
PYTHONPATH=/content/ViSta-HPC bash /content/ViSta-HPC/run_vista.sh --input sim/input_list.txt --rest-freq 345.79599 --v-window '-500 500' --velocity-range 900 --contsub before --method joint --mode detect --fwhm 300 --unit Jy --out-dir cli
```

```text
============================================================
============================================================
== [1/4] stacking, with continuum subtraction and compression in the same pass ==
== [2/4] continuum subtraction: done during the stacking ==
== [3/4] compression: done during the stacking ==
== [4/4] extraction and fit ==
[ViSta       7.2s] Line: F = 0.369 +- 0.0437 (thermal 0.0437, bootstrap 0.0284)  peak = 0.001156  v0 = +0.0 km/s  FWHM = 300 km/s  size gauss 0.77"  chi2r = 0.93
== checks ==

true line flux 0.4000
```

On your own data:

1. write the input list, one line per dataset:
   `<ms_path> <redshift> <RA> <Dec> [FIELD_ID] [SPW_IDS] [NORM]`, with the
   position of the galaxy (hh:mm:ss.s ±dd:mm:ss.s);
2. stack with the continuum subtraction and the compression in one pass
   (§3), with a `v_window_kms` wide enough to contain the whole line;
3. look at the stack (§3) and at `line_free_rms_over_noise`; if needed,
   redo the subtraction (§8);
4. measure (§4–5): `detect` if the width is known, otherwise `search` and
   its `significance_sigma`; check `imaginary_rms_over_noise`, the size test
   and the ratio of bootstrap to thermal error (much larger than 1: the
   galaxies differ more than the noise);
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
cont_stats.npy                           1101.3 kB
cont_stats_noiseless.npy                 1101.4 kB
cont_stats_v2.npy                        1101.3 kB
continuum_noiseless_results.json            2.0 kB
continuum_noiseless_spectrum.txt            9.1 kB
continuum_noiseless_uvamp.txt               1.8 kB
continuum_noisy_results.json                2.0 kB
continuum_noisy_spectrum.txt                9.1 kB
continuum_noisy_uvamp.txt                   1.8 kB
detect_diagnostics.npz                     58.2 kB
detect_noise.txt                            6.9 kB
detect_noiseless_diagnostics.npz            9.3 kB
detect_noiseless_noise.txt                  6.9 kB
detect_noiseless_profile.txt               20.8 kB
detect_noiseless_results.json               4.4 kB
detect_noiseless_uvamp.txt                  2.1 kB
detect_profile.txt                         22.7 kB
detect_results.json                         5.6 kB
detect_uvamp.txt                            2.1 kB
line_stats.npy                           1101.3 kB
line_stats_noiseless.npy                 1101.4 kB
line_stats_v2.npy                        1101.3 kB
one_noiseless.txt                           0.1 kB
one_noiseless_diagnostics.npz               9.3 kB
one_noiseless_noise.txt                     6.9 kB
one_noiseless_profile.txt                  20.8 kB
one_noiseless_results.json                  3.7 kB
one_noiseless_stats.npy                    93.5 kB
one_noiseless_uvamp.txt                     2.1 kB
one_noisy.txt                               0.1 kB
one_noisy_diagnostics.npz                  54.2 kB
one_noisy_noise.txt                         6.9 kB
one_noisy_profile.txt                      20.8 kB
one_noisy_results.json                      4.1 kB
one_noisy_stats.npy                        93.5 kB
one_noisy_uvamp.txt                         2.1 kB
redo_diagnostics.npz                       54.5 kB
redo_noise.txt                              6.9 kB
redo_profile.txt                           22.7 kB
redo_results.json                           5.5 kB
redo_uvamp.txt                              2.1 kB
search_diagnostics.npz                     58.0 kB
search_noise.txt                            6.9 kB
search_noiseless_diagnostics.npz            9.3 kB
search_noiseless_noise.txt                  6.9 kB
search_noiseless_profile.txt               20.8 kB
search_noiseless_results.json               4.5 kB
search_noiseless_uvamp.txt                  2.1 kB
search_profile.txt                         22.7 kB
search_results.json                         6.0 kB
search_uvamp.txt                            2.1 kB
```
