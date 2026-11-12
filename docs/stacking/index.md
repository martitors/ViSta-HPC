# How the stacking works

**In one sentence.** Stacking in the uv plane averages the visibilities of
many galaxies as if they were one observation of one galaxy; for that,
every dataset is first brought to the same position, the same frame and the
same channels.

**What `run()` does.** It reads every dataset once and writes the stack:

```text
input list ──► for every dataset, chunk by chunk:
                 1. select the rows      the field and the target scans, the channels near the line
                 2. re-centre            phase shift to the source              (Centring)
                 3. rest-frame           frequencies × (1+z), baselines / (1+z) (Rest frame)
                 4. regrid               onto the common channel grid           (Spectral grid)
                 5. subtract the continuum, row by row          (default)       (Continuum)
                 6. compress into statistics per annulus                        (Compression)
                 7. write                one spectral window per dataset        (The stack on disk)
```

**What it does not do.** Nothing is averaged across the datasets here. The
stack keeps every visibility with its weight, one spectral window per line
of the input list. The average over the galaxies (with the weighting, the
redshift correction, the primary beam) is taken when the stack is
**measured**, on the statistics. So it can be changed at will, without
stacking again ([Measuring the stack](../extraction/index)).

## The pages

| Step | Page | What you learn |
|---|---|---|
| 2 | [Centring](centring) | where the source is put; what a wrong position does |
| 3 | [Rest frame](restframing) | the frame of the redshifts; the motion of the Earth |
| 4 | [Spectral grid](spectral_grid) | the common channel width, the number of channels, the coverage |
|  | [Companions](companions.md) | other sources in the fields of your galaxies: how they are removed |
| 5 | [Continuum](continuum.md) | subtract it, fit it with the line, or leave it; the continuum alone |
| 6 | [Compression](compression) | the annuli of baseline length; what the statistics hold |
| 7 | [The stack on disk](combination) | Zarr or Measurement Set; what is in it |

## The call

```python
from vista import ViSta

v = ViSta("input_list.txt",
          output_name="stack",                # stack.zarr (output_format="ms": stack.ms)
          verbose=True)                       # (device, n_readers, n_threads...: here too)
v.stack(line_freq=345.79599,                            # rest frequency of the line, GHz
        line_window=500,                      # where the line is: out of the continuum fit
        velocity_range=900,                   # output channels from -900 to +900 km/s
        frame="LSRK",                         # frame of the redshifts
        continuum="subtract",                 # step 5 ('fit', 'none': not subtracted)
        continuum_order=0)
v.help("vista", "stack")                      # all the options of the stacking
v.extract(annuli=(3, 3000, 18))               # step 6 (and the measurement)
summary = v.run(only="stack").stack           # without only: also the measurement
```

All the options are in [Parameters](../reference/parameters.md#stack-what-makes-the-stack).
Without `extract()` the run writes the stack only, without the statistics.
The summary of the stacking (`result.stack`) has `output`, `format`,
`n_rows`, `n_spw`, `nchan`, `df_hz`, `flagged_fraction`, `skipped` (input
lines that gave no data), `bytes_on_disk`, `seconds`, `resources` (cores,
readers, writers, threads), `contsub_orders` and `statistics`.

## The engine

Steps 2 to 4 run in a compiled kernel (`ms_ops`: C++ with OpenMP, or CUDA);
steps 5 and 6 in the same pass, on each chunk.

```text
readers (processes) ──► chunks ──► compute (main thread) ──► results ──► writers (threads)
one MS each,            bounded     C++ kernel, GIL released  bounded     Zarr: n_writers in
shared memory           queue       OpenMP or CUDA            queue       parallel; MS: one
```

- **Readers** read each Measurement Set chunk by chunk: only the rows of
  the field and of the target scans, only the channels in the output
  window. They are separate processes, and hand the data over in shared
  memory, without copies.
- **Compute** runs the kernel on each chunk. Reading and writing go on
  meanwhile. On a GPU (`device="gpu"`) all the chunks already read, up to
  `gpu_batch`, go in one launch.
- **Writers** store each chunk at its place. In a Zarr store every chunk is
  a file of its own, so several writers work at once; a Measurement Set is
  written by one thread.

**You do not have to tune it.** The numbers of readers, writers and threads
come from the cores the process may use (`n_readers`, `n_writers`,
`n_threads` fix them; see [Performance](../reference/performance)). The
queues between the stages are bounded, which bounds the memory.
`verbose="debug"` shows the choices, and reports at the end how long each
stage took and which one limited the run.

## The log

**What it is for.** The log tells you what was read from every dataset, and
warns when something is off. Every line carries the time since the start of
the step.

| `verbose` | What is printed |
|---|---|
| `False` (default) | the warnings only |
| `True` | the progress and the results |
| `"debug"` | also the metadata and the grid of every dataset, the queues, the timing of the stages |

`log_file` writes the same to a file. The measurement writes to the same
log.

An example: four simulated ALMA observations (labelled TOPO, three months
apart) on a 2-core machine, `v.stack(..., nchan=200)`, with
`verbose="debug"` (some detail lines left out):

```text
[ViSta       0.0s] ViSta 3.0.0 · 4 MS from input_list.txt → stack.zarr (Zarr) · CPU · 2 cores: 1 reader, 1 kernel thread, 1 writer
[ViSta       0.5s] Frame: inputs 4 TOPO -> LSRK, corrected row by row (without the correction the line moves by -18.6 .. +34.1 km/s from one observation to another)
[ViSta       0.5s] Output grid: 200 channels × 9228.7 kHz (8.00 km/s), ±800 km/s around 345.7960 GHz
[ViSta       0.5s]   (width: the coarsest rest-frame channel of the sample; channels: nchan, the MS can fill 119-120 of them)
[ViSta       0.5s]   4 of 4 MS do not cover the whole window around 345.7960 GHz: their window is moved to the edge of their band (shift in the lines below), so their channel frequencies differ from the others
[ViSta       0.7s] During the stacking: continuum subtracted (order 0) -> MODEL_DATA, CORRECTED_DATA; statistics from CORRECTED_DATA -> stack_line_stats.npy, stack_cont_stats.npy
[ViSta       0.7s] Setup done (40 antennas in total); stacking 4 MS...
[ViSta       0.7s] [1/4] obs0_20190322.ms  z=0.5000  2,700 rows  DATA  1 SPW, 119 ch  covers 60% of the grid  offset 2.2"  window +175.3 MHz
[ViSta       0.8s] [2/4] obs1_20190620.ms  z=0.5000  2,700 rows  DATA  1 SPW, 120 ch  covers 60% of the grid  offset 2.2"  window +184.6 MHz
[ViSta       0.9s] [3/4] obs2_20190918.ms  z=0.5000  2,700 rows  DATA  1 SPW, 119 ch  covers 60% of the grid  offset 2.2"  window +147.7 MHz
[ViSta       0.9s] [4/4] obs3_20191217.ms  z=0.5000  2,700 rows  DATA  1 SPW, 119 ch  covers 60% of the grid  offset 2.2"  window +129.2 MHz
[ViSta       1.4s] Timing: setup 0.7 s · stacking 0.3 s · subtables and closing 0.4 s
[ViSta       1.4s]   during the stacking (overlapping): reading 0.1 s per reader · computing 0.1 s · writing 0.1 s  → limited by writing (56% busy)
[ViSta       1.4s]   of the computing: continuum subtraction 0.0 s · statistics 0.0 s
[ViSta       1.4s] Done in 1.4 s: stack.zarr · 4 SPW × 200 channels · 10,800 rows · 37.3 MB of visibilities, 38.8 MB on disk · 40.4% of the output flagged (no data)
[ViSta       1.4s]   (Zarr store: vista.export_ms() writes it as a Measurement Set)
```

**For each dataset** the log gives:

- the rows used (and how many the selection of field and scans removed);
- the column read;
- the spectral windows and the channels used;
- the fraction of the output grid it covers (the rest is flagged);
- the distance between its phase centre and the source (`offset`);
- the shift of its window, when it does not cover the one centred on the
  line.

**Warnings** to act on: a source beyond half the primary beam, a missing
`data_column`, spectral windows dropped, datasets skipped (and why), a GPU
error.
