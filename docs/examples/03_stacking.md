# 3. Stacking

Three calls describe a run:

* `ViSta(input_list, output_name=..., ...)`: what is read and written, and
  how the run uses the machine (device, readers, writers, threads), the log;
* `v.stack(line_freq=..., ...)`: what makes the stack — the line and its window,
  the band, the frame, the continuum;
* `v.extract(...)`: what is measured on it. Without this call, `run()` only
  writes the stack.

`run()` reads every dataset, shifts the phase centre to the galaxy, brings
frequencies and baselines to the rest frame, regrids on a common velocity
grid and writes the stack (a Zarr store; `output_format="ms"` for a
Measurement Set). In the same pass it

* **subtracts the continuum** (`continuum="subtract"`, the default), row by
  row, on the channels outside the line window: the fit goes to
  `MODEL_DATA`, the line to `CORRECTED_DATA`, and `DATA` is kept (§8 redoes
  the subtraction);
* with `v.extract(...)`, **compresses** the stack: for every dataset,
  channel and annulus of baseline length (`annuli`), the sums the
  measurement needs (`stack_line_stats.npy` from `CORRECTED_DATA`,
  `stack_cont_stats.npy` from `MODEL_DATA`). The measurements then take
  seconds and can be repeated at will.

| Option | Meaning |
|---|---|
| `ViSta(output_name=)` | name of everything written: `stack.zarr`, `stack_*` |
| `ViSta(device=, n_readers=, n_threads=)` | where and how parallel (default: from the cores) |
| `.stack(line_freq=, line_window=)` | rest frequency: GHz, or a string with the unit (`"345.796GHz"`); a list for several lines. Where the line is: ±W km/s or (lo, hi) |
| `.stack(velocity_range=)` | output channels from −X to +X km/s (default: the widest band of one dataset) |
| `.stack(frame=)` | `"LSRK"` (default) / `"BARY"`: the frame of the redshifts |
| `.stack(continuum=, continuum_order=)` | `"subtract"` (default), `"fit"`, `"none"`; 0 constant, 1 slope, `"bic"` |
| `.extract(annuli=)` | the annuli: from `b_min` to `b_max` kλ, how many |

`v.help()` prints every option with its value now and what it can be
(`v.help("stack")` one step); `run(only="stack")` stops after the stack and
its statistics. Both samples are stacked the same way; the noiseless stack is
the reference.

```python
t0 = time.time()
stack = vista_for(LIST, "stack", verbose=True)           # verbose: the progress
stack.extract(annuli=ANNULI)                             # so the statistics are written too
stack.help("vista", "stack")                             # the options of the run and of the stack
stack.run(only="stack")
stack0 = vista_for(LIST0, "stack_noiseless")
stack0.extract(annuli=ANNULI)
stack0.run(only="stack")
print(f"\nboth stacked in {time.time() - t0:.0f} s")
```

```text
ViSta(input_list, ...)  —  input, output, how it runs (input_list is required)
  option         now                   can be                  what it does
  input_list     'sim/input_list.txt'  path                    one dataset per line: <ms> <z> <RA> <Dec>
                                                               [FIELD_ID] [SPW_IDS] [NORM]
  output_name    'stack'               name or path            name of everything written: <name>.zarr (or
                                                               .ms), <name>_*
  output_format  'zarr'                'zarr' | 'ms' | None    the stack: Zarr (parallel, compressed) or a
                                                               Measurement Set; None: zarr (ms if the name
                                                               ends in .ms)
  device         'cpu'                 'cpu' | 'gpu'           where the kernel runs ('gpu': a CUDA build)
  n_readers      None                  int | None              reader processes; None: from the cores
  n_writers      None                  int | None              writer threads (zarr only); None: from the
                                                               cores
  n_threads      None                  int | None              kernel threads; None: the cores left
  gpu_batch      None                  int | None              GPU: chunks per launch; None: as many as fit in
                                                               memory
  scratch_dir    None                  path | None             write the stack on this disk, then move it
  verbose        True                  False | True | 'debug'  warnings only | progress and results | also the
                                                               details and the timing
  log_file       None                  path | None             also write the log here

v.stack(line_freq=..., ...)  —  what makes the stack (line_freq is required)
  option           now          can be                       what it does
  line_freq        345.796      GHz | '345.8GHz' | list      rest frequency (a number: GHz; a string: with its
                                                             unit); several lines: the first is the one of the
                                                             stack
  line_window      (-500, 500)  W | (lo, hi) | list          km/s, window of every line (500: -500..500): out
                                                             of the continuum fit; lines whose windows overlap
                                                             are fitted together
  velocity_range   900          km/s | (lo, hi) | None       band of the stack around the first line; None:
                                                             the widest coverage of one dataset
  nchan            None         int | None                   a number of channels instead of velocity_range
  channel_width    None         Hz | None                    rest-frame channel width; None: the coarsest
                                                             channel of the sample
  frame            'LSRK'       'LSRK' | 'BARY' | ...        frame of the redshifts: the inputs are brought to
                                                             it
  data_column      'auto'       'auto' | column              column read; auto: CORRECTED_DATA if present,
                                                             else DATA
  continuum        'subtract'   'subtract' | 'fit' | 'none'  subtracted while stacking (then measured on its
                                                             own) | left, fitted with the lines | left as it
                                                             is
  continuum_order  0            0 | 1 | 'bic'                subtract: constant, straight line, or chosen by
                                                             the BIC per spectral window
  exclude_between  False        True | False                 several lines: also keep the channels between
                                                             them out of the continuum fit

help(ViSta), help(v.stack), help(v.extract): the full description.
[ViSta       0.0s] ViSta 3.0.0 · 12 MS from input_list.txt → stack.zarr (Zarr) · CPU · 2 cores: 1 reader, 1 kernel thread, 1 writer
```

```text
[ViSta       0.5s] Frame: inputs 12 LSRK, already LSRK
```

```text
[ViSta       0.5s] Output grid: 90 channels × 23069.0 kHz (20.00 km/s), ±900 km/s around 345.7960 GHz
```

```text
[ViSta       0.5s]   (width: the coarsest rest-frame channel of the sample; channels: velocity range -900 to +900 km/s)
```

```text
[ViSta       0.9s] During the stacking: continuum subtracted (order 0) -> MODEL_DATA, CORRECTED_DATA; statistics from CORRECTED_DATA -> stack_line_stats.npy, stack_cont_stats.npy
```

```text
[ViSta       0.9s] Setup done (516 antennas in total); stacking 12 MS...
```

```text
[ViSta       1.0s] [ 1/12] galaxy_00.ms  z=2.0027  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       1.3s] [ 2/12] galaxy_01.ms  z=2.0165  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       1.5s] [ 3/12] galaxy_02.ms  z=2.0410  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       1.8s] [ 4/12] galaxy_03.ms  z=2.2698  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       2.1s] [ 5/12] galaxy_04.ms  z=2.5436  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       2.3s] [ 6/12] galaxy_05.ms  z=2.6066  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       2.6s] [ 7/12] galaxy_06.ms  z=2.6370  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       2.9s] [ 8/12] galaxy_07.ms  z=2.7295  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       3.3s] [ 9/12] galaxy_08.ms  z=2.8133  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       3.5s] [10/12] galaxy_09.ms  z=2.8159  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       3.9s] [11/12] galaxy_10.ms  z=2.9128  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       4.2s] [12/12] galaxy_11.ms  z=2.9351  36,120 rows  DATA  1 SPW, 91 ch  covers 100% of the grid  offset 2.2"
```

```text
[ViSta       5.5s] Done in 5.5 s: stack.zarr · 12 SPW × 90 channels · 433,440 rows · 679.6 MB of visibilities, 1.1 GB on disk · 0.0% of the output flagged (no data)
```

```text
[ViSta       5.5s]   (Zarr store: vista.export_ms() writes it as a Measurement Set)
```

```text

both stacked in 11 s
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
ax.axvspan(-WINDOW, WINDOW, color="C1", alpha=0.08, label="line window")
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
