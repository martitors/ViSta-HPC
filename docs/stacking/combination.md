# The stack on disk

The stack holds every visibility of every dataset, re-centred, rest-framed
and regridded, with its weight: one spectral window (`DATA_DESC_ID`) per line
of the input list, in the order of the list. The average over the sources is
not taken here: it is taken when the stack is measured, with the weighting of
your choice.

## Zarr store or Measurement Set

| | Zarr store (default) | Measurement Set |
|---|---|---|
| chosen by | `ViSta(output_name="stack")` (default `output_format`) | `output_format="ms"`, or an `output_name` ending in `.ms` |
| written by | `n_writers` threads at once (default: from the cores) | one thread |
| on disk | compressed (Blosc lz4): flags and empty channels take almost no space | uncompressed |
| layout | that of [dask-ms](https://dask-ms.readthedocs.io): `MAIN/MAIN_<k>` per input line, subtables in their own groups | a standard MS |
| read by | `vista.extract`, `daskms.experimental.zarr.xds_from_zarr` | any tool |

To use a tool that needs a Measurement Set (`tclean`, CARTA, ...):

```python
from vista import export_ms
export_ms("stack.zarr", "stack.ms")
```

or `python -m vista export stack.zarr stack.ms` (`run_vista.sh export ...`).
It writes the same MS that `output_format="ms"`
would, with the rows of every dataset in input order, and with `MODEL_DATA`
and `CORRECTED_DATA` when the continuum was subtracted.

## What the stack contains

- `DATA` (and `MODEL_DATA`, `CORRECTED_DATA` after a continuum subtraction),
  `FLAG`, `WEIGHT`, `SIGMA`, `UVW` (rest-framed), times and antennas of every
  row;
- the subtables: one spectral window per input line, a single field at
  (0, 0) (every source sits at the phase centre), the antennas of every dataset (rest-framed, numbered
  after each other, so the datasets never share an antenna), polarization,
  observation, feed, source (with the rest frequency).

## Speed and memory

- The **readers, writers and kernel threads** share the cores the process
  may use (its CPU affinity, a SLURM allocation, a container quota): readers
  and writers about a quarter of them each (at most 8), the kernel the rest.
  `ViSta(..., n_readers=, n_writers=, n_threads=)` fixes any of them; the others
  share what is left, and more than the cores is refused. On two cores one of
  each was the fastest: oversubscribing does not pay.
- The size of a chunk is bounded (about 512 MB of visibilities) and lowered
  for wide spectra; the queues between the stages hold a few chunks each.
- `ViSta(..., scratch_dir=)`: build the stack on local disk and move it at the end.
- `ViSta(..., device="gpu")`: the kernel on the GPU (CUDA build); `gpu_batch`, the chunks
  per launch, comes from the free memory of the GPU and of the host. On a GPU
  error the run continues on the CPU, with a warning.

With a Measurement Set the single writer sets the pace; with the Zarr store
the limit moves to reading. `verbose="debug"` says which stage limited the
run. See [Performance](../reference/performance) for measured times.

A script that runs the stacking needs no `if __name__ == "__main__":` guard:
the reader processes do not re-run it. When `/dev/shm` is too small for the
chunks in flight (some containers), the readers run as threads, and the log
says so.
