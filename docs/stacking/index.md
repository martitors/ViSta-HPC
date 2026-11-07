# How the stacking works

Stacking in the uv plane averages the visibilities of many sources as if
they were one observation of one source. For that, every dataset is brought
to the same reference: the same position, the same frame, the same
channels. ViSta does it in one call, `ViSta(...).run(...)`, which reads every
dataset once and writes the stack:

```text
input list ──► for every dataset, chunk by chunk:
                 1. select the rows      the field and the target scans, the channels near the line
                 2. re-centre            phase shift to the source            (Centring)
                 3. rest-frame           frequencies × (1+z), baselines / (1+z) (Rest frame)
                 4. regrid               onto the common channel grid          (Spectral grid)
                 5. subtract the continuum, row by row          (optional)    (Continuum)
                 6. compress into statistics per annulus        (optional)    (Compression)
                 7. write                one spectral window per dataset       (The stack on disk)
```

Steps 2–4 run in the compiled kernel (`ms_ops`, C++/OpenMP or CUDA), steps
5–6 in the same pass when `run()` is given a `line` and a `continuum` and the
names of the statistics files. Nothing is averaged across datasets at this
stage: the stack keeps every visibility with its weight, one spectral window
per line of the input list. The average over the sources — with the
weighting, the redshift rescaling and the primary-beam correction — is done
when the stack is measured, on the statistics, so it can be changed at will
([Measuring the stack](../extraction/index)).

The pages of this section follow the steps:

| Step | Page | What it decides |
|---|---|---|
| 2 | [Centring](centring) | where the source is; what an offset or a wrong position does |
| 3 | [Rest frame](restframing) | the frame of the redshifts; the Earth's motion in TOPO data |
| 4 | [Spectral grid](spectral_grid) | the common channel width, the number of channels, the coverage |
| 7 | [The stack on disk](combination) | Zarr or Measurement Set; parallelism, memory, the log |
| 5 | [Continuum](continuum) | subtract it or fit it; the line-free channels; redoing it |
| 6 | [Compression](compression) | the annuli of baseline length; what the statistics hold |

## The call

```python
from vista import ViSta
from vista.extract import LineConfig, ContinuumConfig

line = LineConfig(rest_freq_ghz=345.79599, v_window_kms=(-500, 500))
summary = ViSta("input_list.txt", chunk_rows=5000, verbose=True).run(
    output="stack.zarr",                  # Zarr store; a path ending in .ms gives an MS
    central_freq=line.rest_freq_hz,       # rest frequency of the line, Hz
    velocity_range_kms=900,               # output channels from -900 to +900 km/s
    frame="LSRK",                         # frame of the redshifts
    line=line, continuum=ContinuumConfig(order=0),          # step 5
    stats_line="line_stats.npy", stats_continuum="cont_stats.npy")   # step 6
```

Without `line`, `continuum` and `stats_*`, `run()` only stacks; the continuum
subtraction and the compression can then be run on the stack afterwards
(`subtract_continuum`, `compress_visibilities`), with the same result. All
the parameters are in [Parameters](../reference/parameters.md#the-stacking-vista-and-run).

`run()` returns a dictionary: `output`, `format`, `n_rows`, `n_spw`, `nchan`,
`df_hz`, `flagged_fraction`, `skipped` (input lines that gave no data),
`bytes_on_disk`, `seconds`, and with the continuum and the compression
`contsub_orders` and `statistics`.

## The engine

```text
readers (processes) ──► chunks ──► compute (main thread) ──► results ──► writers (threads)
one MS each,            bounded     C++ kernel, GIL released  bounded     Zarr: n_writers in
shared memory           queue       OpenMP or CUDA            queue       parallel; MS: one
```

- **Readers** read each Measurement Set chunk by chunk with
  python-casacore: only the rows of the requested field and of the target
  scans, only the channels that fall in the output window. They are
  processes (python-casacore holds the GIL while reading), and hand the data
  over in shared memory, without copies.
- **Compute** runs the kernel on each chunk with the GIL released, so
  reading and writing go on meanwhile. On the GPU (`device="gpu"`) all the
  chunks already read, up to `gpu_batch`, go in one launch.
- **Writers** store each processed chunk at its place: in a Zarr store every
  chunk is a file of its own, so several writers work at once; a Measurement
  Set is written by one thread.

The queues between the stages are bounded, which bounds the memory
(`chunk_rows`, `chunk_queue`, `out_queue`). `timing=True` reports at the end
how long each stage took and which one limited the run.

## The log

Every line carries the time since the start of the step; warnings are always
shown. A run on four simulated ALMA observations (labelled TOPO, three
months apart) on a 2-core machine, with `frame="LSRK"` and `timing=True`:

```text
[ViSta       0.0s] ViSta 2.4.0 · 4 MS from input_list.txt → stack.zarr (Zarr) · CPU (2 threads) · 2 readers · 4 writers
[ViSta       2.1s] Frame: inputs 4 TOPO -> LSRK, corrected row by row (without the correction the line moves by -10.3 .. +35.3 km/s from one observation to another)
[ViSta       2.1s] Output grid: 200 channels × 5860.1 kHz (5.08 km/s), ±508 km/s around 345.7960 GHz
[ViSta       2.1s]   (width: the coarsest rest-frame channel of the sample; channels: nchan_out, the MS can fill 119-120 of them)
[ViSta       2.1s]   4 of 4 MS do not cover the whole window around 345.7960 GHz: their window is moved to the edge of their band (shift in the lines below), so their channel frequencies differ from the others
[ViSta       2.2s] Setup done (40 antennas in total); stacking 4 MS...
[ViSta       2.4s] [2/4] obs1_20230530.ms  z=0.5000  2,700 rows  DATA  1 SPW, 120 ch  covers 60% of the grid  offset 1.8"  window +240.3 MHz
[ViSta       2.5s] [1/4] obs0_20230301.ms  z=0.5000  2,700 rows  DATA  1 SPW, 120 ch  covers 60% of the grid  offset 1.8"  window +240.3 MHz
[ViSta       2.8s] [3/4] obs2_20230828.ms  z=0.5000  2,700 rows  DATA  1 SPW, 120 ch  covers 60% of the grid  offset 1.8"  window +240.3 MHz
[ViSta       2.8s] [4/4] obs3_20231126.ms  z=0.5000  2,700 rows  DATA  1 SPW, 120 ch  covers 60% of the grid  offset 1.8"  window +240.3 MHz
[ViSta       3.8s] Timing: setup 2.2 s · stacking 1.1 s · subtables and closing 0.5 s
[ViSta       3.8s]   during the stacking (overlapping): reading 0.5 s per reader · computing 0.5 s · writing 0.1 s per writer  → limited by reading (51% busy)
[ViSta       3.8s] Done in 3.8 s: stack.zarr · 4 SPW × 200 channels · 10,800 rows · 37.3 MB of visibilities, 16.8 MB on disk · 40.0% of the output flagged (no data)
[ViSta       3.8s]   (Zarr store: vista.export_ms() writes it as a Measurement Set)
```

For each dataset: the rows used (and how many the field and target-scan
selection removed), the column read, the spectral windows and channels used,
the fraction of the output grid it covers (the rest is flagged), the
distance between its phase centre and the source, and the shift of its
output window when it does not cover the window centred on the line.
Warnings: source beyond half the primary beam, a missing `data_column`,
spectral windows dropped, datasets skipped and why, a GPU error.

`verbose=False` keeps the warnings only, `verbose="debug"` adds the metadata
and grid of every dataset, the weight scale, the queues and the GPU buffers.
The measurement writes to the same log; `vista.set_verbosity(verbose,
log_file)` sets the level, and a log file, for the whole package.
