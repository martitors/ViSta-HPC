# The stack on disk

The stack holds every visibility of every dataset, re-centred, rest-framed
and regridded, with its weight: one spectral window (`DATA_DESC_ID`) per line
of the input list, in the order of the list. The average over the sources is
not taken here: it is taken when the stack is measured, with the weighting of
your choice.

## Zarr store or Measurement Set

| | Zarr store (default) | Measurement Set |
|---|---|---|
| chosen by | any `output` not ending in `.ms` | `output="stack.ms"`, or `output_format="ms"` |
| written by | `n_writers` threads at once (default 4) | one thread |
| on disk | compressed (Blosc lz4): flags and empty channels take almost no space | uncompressed |
| layout | that of [dask-ms](https://dask-ms.readthedocs.io): `MAIN/MAIN_<k>` per input line, subtables in their own groups | a standard MS |
| read by | `vista.extract`, `daskms.experimental.zarr.xds_from_zarr` | any tool |

To use a tool that needs a Measurement Set (`tclean`, CARTA, ...):

```python
from vista import export_ms
export_ms("stack.zarr", "stack.ms")
```

or `python -m vista.extract export-ms stack.zarr stack.ms`, or
`run_vista.sh --export-ms`. It writes the same MS that `output_format="ms"`
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

- `chunk_rows` (default 5000) bounds the memory of a chunk; it is lowered
  automatically for wide spectra.
- `n_readers` (default 2) reader processes: more help only on fast
  filesystems.
- `n_threads`: OpenMP threads of the kernel (default `OMP_NUM_THREADS`, else
  `SLURM_CPUS_PER_TASK`, else all cores).
- `n_writers` (default 4): Zarr writers.
- `scratch_dir`: build the stack on local disk and move it at the end.
- `device="gpu"`: the kernel on the GPU (CUDA build); on a GPU error the run
  continues on the CPU, with a warning.

With a Measurement Set the single writer sets the pace; with the Zarr store
the limit moves to reading. `timing=True` says which stage limited the run.
See [Performance](../reference/performance) for measured times.

A script that calls `run()` needs no `if __name__ == "__main__":` guard: the
reader processes do not re-run it. In a container with a small `/dev/shm`,
`reader_mode="thread"` avoids the shared memory.
