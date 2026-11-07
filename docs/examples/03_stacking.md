# 3. Stacking

`ViSta(...).run(...)` reads every dataset, shifts the phase centre to the
galaxy, brings frequencies and baselines to the rest frame, regrids on a
common velocity grid and writes the stack (a Zarr store; a Measurement Set if
the name ends in `.ms`). In the same pass it

* **subtracts the continuum**, row by row, on the channels outside the line
  window: the fit goes to `MODEL_DATA`, the line to `CORRECTED_DATA`, and
  `DATA` is kept (§8 redoes the subtraction);
* **compresses** the stack: for every dataset, channel and annulus of
  baseline length, the sums the measurement needs (`line_stats.npy` from
  `CORRECTED_DATA`, `cont_stats.npy` from `MODEL_DATA`). The measurements
  then take seconds and can be repeated at will.

| Parameter | Meaning |
|---|---|
| `central_freq` | rest-frame centre of the output grid (the line) |
| `velocity_range_kms` | output channels from −X to +X km/s |
| `channel_width_hz` | default: the coarsest rest-frame channel of the sample |
| `LineConfig.v_window_kms` | where the line is: left out of the continuum fit |
| `ContinuumConfig.order` | 0 constant, 1 slope, −1 chosen by the BIC per dataset |
| `BinningConfig` | the annuli: from `b_min` to `b_max` kλ, `n_bins` log-spaced |
| `frame` | `"LSRK"` / `"BARY"` if the input frequencies are TOPO |
| `n_readers`, `n_threads`, `device` | parallelism (`device="gpu"` with a CUDA build) |

Both samples are stacked the same way; the noiseless stack is the reference.

```python
t0 = time.time()
ViSta(LIST, chunk_rows=5000, verbose=True).run(
    output="stack.zarr",
    central_freq=line.rest_freq_hz,
    velocity_range_kms=900,
    line=line, continuum=contsub,
    stats_line="line_stats.npy", stats_continuum="cont_stats.npy",
    binning=binning)
ViSta(LIST0, verbose=False).run(
    output="stack_noiseless.zarr", central_freq=line.rest_freq_hz, velocity_range_kms=900,
    line=line, continuum=contsub, binning=binning,
    stats_line="line_stats_noiseless.npy", stats_continuum="cont_stats_noiseless.npy")
print(f"\nboth stacked in {time.time() - t0:.0f} s")
```

```text
[ViSta       0.0s] ViSta 2.5.0 · 12 MS from input_list.txt → stack.zarr (Zarr) · CPU (2 threads) · 2 readers · 4 writers
```

```text
[ViSta       0.8s] Output grid: 90 channels × 23069.0 kHz (20.00 km/s), ±900 km/s around 345.7960 GHz
```

```text
[ViSta       0.8s]   (width: the coarsest rest-frame channel of the sample; channels: velocity range -900 to +900 km/s)
```

```text
[ViSta       1.2s] During the stacking: continuum subtracted (order 0) -> MODEL_DATA, CORRECTED_DATA; statistics from CORRECTED_DATA -> line_stats.npy, cont_stats.npy
```

```text
[ViSta       1.3s] Setup done (516 antennas in total); stacking 12 MS...
```

```text
[ViSta       1.3s] [ 2/12] galaxy_01.ms  z=2.0165  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       1.4s] [ 1/12] galaxy_00.ms  z=2.0027  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       2.2s] [ 3/12] galaxy_02.ms  z=2.0410  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       2.3s] [ 4/12] galaxy_03.ms  z=2.2698  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       3.0s] [ 6/12] galaxy_05.ms  z=2.6066  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       3.1s] [ 5/12] galaxy_04.ms  z=2.5436  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       3.8s] [ 7/12] galaxy_06.ms  z=2.6370  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       4.0s] [ 8/12] galaxy_07.ms  z=2.7295  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       4.6s] [ 9/12] galaxy_08.ms  z=2.8133  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       5.0s] [10/12] galaxy_09.ms  z=2.8159  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       5.5s] [11/12] galaxy_10.ms  z=2.9128  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       5.8s] [12/12] galaxy_11.ms  z=2.9351  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       7.0s] Done in 7.0 s: stack.zarr · 12 SPW × 90 channels · 433,440 rows · 679.6 MB of visibilities, 1.1 GB on disk · 0.0% of the output flagged (no data)
```

```text
[ViSta       7.0s]   (Zarr store: vista.export_ms() writes it as a Measurement Set)
```

```text

both stacked in 15 s
```

**A first look at the stack**: all visibilities of all galaxies averaged
over the baselines, before (`DATA`) and after (`CORRECTED_DATA`) the
continuum subtraction, and the fitted continuum (`MODEL_DATA`), with noise
(solid) and without (dashed). It checks the stacking and the continuum
subtraction, and that the line sits in its window. It is not yet the flux:
resolution, primary beam and redshift rescaling come in §4.

```python
cols = ("DATA", "MODEL_DATA", "CORRECTED_DATA")
vel, noisy = baseline_average("stack.zarr", cols)
vel0, clean = baseline_average("stack_noiseless.zarr", cols)
fig, ax = plt.subplots(figsize=(8.5, 3.8))
for col, c in zip(cols, ("0.55", "C0", "k")):
    ax.step(vel, noisy[col] * 1e3, where="mid", color=c, lw=1, label=col)
    ax.plot(vel0, clean[col] * 1e3, color=c, ls="--", lw=1.6)
ax.plot([], [], color="0.3", ls="--", label="without noise")
ax.axvspan(*line.v_window_kms, color="C1", alpha=0.08, label="line window")
ax.axhline(0, color="0.5", lw=0.5)
ax.set_xlabel("velocity [km s$^{-1}$]"); ax.set_ylabel("mean Re V over all baselines [mJy]")
lo, hi = ax.get_ylim()
ax.set_ylim(lo, hi + 0.45 * (hi - lo))                 # room for the legend
ax.legend(ncol=3, loc="upper left")
save(fig, "stack_first_look")
```

```{image} figures/03_01.png
:width: 100%
```
