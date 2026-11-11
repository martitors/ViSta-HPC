# The stack on disk

**In one sentence.** The stack holds every visibility of every dataset,
re-centred, rest-framed and regridded, with its weight: nothing has been
averaged yet.

**How it is laid out.** One spectral window (`DATA_DESC_ID`) per line of
the input list, in the order of the list. The average over the galaxies is
taken when the stack is measured, with the weighting of your choice.

## Zarr store or Measurement Set

| | Zarr store (default) | Measurement Set |
|---|---|---|
| chosen by | `ViSta(output_name="stack")` | `output_format="ms"`, or an `output_name` ending in `.ms` |
| written by | several threads at once | one thread |
| on disk | compressed: flags and empty channels take almost no space | uncompressed |
| layout | that of [dask-ms](https://dask-ms.readthedocs.io) | a standard MS |
| read by | ViSta, `daskms.experimental.zarr.xds_from_zarr` | any tool |

**Which one.** The Zarr store: it is faster to write and smaller. A
Measurement Set is needed only by other tools (`tclean`, CARTA, ...), and
it can be made from the Zarr store at any time:

```python
from vista import export_ms
export_ms("stack.zarr", "stack.ms")
```

or `python -m vista export stack.zarr stack.ms`. It is the same MS that
`output_format="ms"` would have written, with `MODEL_DATA` and
`CORRECTED_DATA` when the continuum was subtracted.

## What is in it

- **The data**: `DATA`, and after a continuum subtraction `MODEL_DATA` (the
  continuum) and `CORRECTED_DATA` (the line); `FLAG`, `WEIGHT`, `SIGMA`,
  `UVW` (rest-framed), times and antennas of every row.
- **The subtables**: one spectral window per input line; a single field at
  (0, 0), because every galaxy sits at the phase centre; the antennas of
  every dataset, numbered one after the other so that datasets never share
  an antenna; polarization, observation, feed, source (with the rest
  frequency).

## Speed and memory

**Cores.** Readers, writers and kernel threads share the cores the process
may use (its CPU affinity, a SLURM allocation, a container quota): readers
and writers about a quarter each (at most 8), the kernel the rest.
`ViSta(..., n_readers=, n_writers=, n_threads=)` fixes any of them; asking
for more than the cores is refused.

**Memory.** A chunk is at most about 512 MB of visibilities (less for wide
spectra), and the queues between the stages hold a few chunks each.

**Options.**

- `scratch_dir=`: build the stack on a local disk and move it at the end.
- `device="gpu"`: the kernel on the GPU (CUDA build). On a GPU error the
  run goes on on the CPU, with a warning.

**What limits a run.** With a Measurement Set, the single writer; with the
Zarr store, the reading. `verbose="debug"` says which stage it was.
[Performance](../reference/performance) has measured times.

**Two practical notes.** A script that runs the stacking needs no
`if __name__ == "__main__":` guard. When `/dev/shm` is too small for the
chunks in flight (some containers), the readers run as threads, and the log
says so.
