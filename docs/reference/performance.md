# Performance

**In one sentence.** The stacking is limited by reading the inputs and
writing the stack, not by the computing: a Zarr stack, written in parallel,
is the fast path.

## Against ViSta 1.0 (CASA)

The numbers of Torsello et al. (2026, Proc. SPIE 14153), with the stack
written as a Measurement Set. Leonardo (CINECA), one node: 32-core Intel
Xeon Platinum 8358 and an NVIDIA A100; input and output on Lustre. 50
datasets, 2 readers, 32 OpenMP threads (GPU: 20 chunks per launch); medians
of 3 runs. ViSta 1.0 is the CASA pipeline (restframing, centring,
rebinning, stacking) on the same data.

| Sample | ViSta 1.0 (CASA) | ViSta-HPC, CPU | ViSta-HPC, GPU |
|---|---|---|---|
| 50 MS × 200 channels, one setup | 308.5 s | 18.5 s (×16.7) | 18.9 s (×16.3) |
| 50 MS × 2048 channels, mixed telescopes, redshifts and channel widths | 833.5 s | 27.4 s (×30) | 28.6 s (×29) |

More OpenMP threads, readers or a larger GPU batch did not shorten these
runs: the single writer of the Measurement Set set the pace, and for the
same reason the GPU gave the time of the CPU.

## Zarr or Measurement Set

On a 2-core test machine, inputs in the page cache: two simulated ALMA
datasets of 541,800 rows × 120 channels stacked on 160 channels (30% of the
output empty, flagged).

| Step | `output_format="ms"` | Zarr (default) |
|---|---|---|
| stacking, whole run | 13.5 s | 10–11 s |
| of which writing (per writer) | 9.6 s | 2.7–4.1 s (4 writers) |
| size of the stack | 2.8 GB | 0.86 GB |
| continuum subtraction, on its own | 28 s | 8.6 s |
| compression into statistics, on its own | 22 s | 4.8 s |
| stack after the subtraction | 8.0 GB | 1.5 GB |

`export_ms` of this store takes 14 s. The emptier the grid, the larger the
gain on disk: with 88% of the grid empty (10 datasets × 200 channels on
1000 channels) the store was 182 MB against 1.6 GB. The two formats hold
the same numbers.

In a run with `extract()`, the subtraction and the compression are done
while the stack is written (one pass), so the stack is not read back:
17 s instead of 21.5 s on this machine, and more on a node with more cores.

## Readers, writers, threads

The readers, the writers and the kernel threads share the cores the
process may use (its CPU affinity, a SLURM allocation, a container quota):
readers and writers about a quarter each (at most 8), the kernel the rest;
on two cores one of each. More workers than cores only make them compete:
on two cores, the twelve datasets of the [Examples](../examples/index)
stack in 6.6 s with one of each and in 7.8 s with two of each.

## CPU or GPU?

`ViSta(..., device="gpu")` runs the kernel on the GPU;
`benchmarks/gpu_check.py` checks that it gives what the CPU gives, to the
last digits of single precision. The GPU helps when the computing is
what limits the run: many channels, fast disks, the Zarr stack.
