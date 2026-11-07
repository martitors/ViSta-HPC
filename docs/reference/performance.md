# Performance

## Stacking: ViSta 1.0 (CASA) and ViSta-HPC

ViSta-HPC 2.3, the stack written as a Measurement Set. Leonardo (CINECA), one node: 32-core Intel Xeon Platinum 8358 and an NVIDIA
A100; input and output on Lustre. 50 datasets, 1000 output channels,
2 readers, 32 OpenMP threads (GPU: 20 chunks per launch); medians of 3 runs.
ViSta 1.0 is the CASA pipeline of the paper (restframing, centering,
rebinning, stacking) on the same data: median of 3 runs (200 channels) and
mean of 2 runs (2048 channels).

| Sample | ViSta 1.0 (CASA) | ViSta-HPC, CPU | ViSta-HPC, GPU |
|---|---|---|---|
| 50 MS × 200 channels, one setup | 308.5 s | 18.5 s (×16.7) | 18.9 s (×16.3) |
| 50 MS × 2048 channels, mixed telescopes, redshifts and channel widths | 833.5 s | 27.4 s (×30) | 28.6 s (×29) |

Adding OpenMP threads, readers or a larger GPU batch does not shorten these
runs: the single writer sets the pace.

The output is the same as the previous version: bit-identical on the CPU
(compared column by column, subtables included, independently of the row
order); on the GPU a few values differ in the last float32 digit, because the
phase is computed per channel instead of by recurrence.

## The Zarr stack and the extraction (v2.4)

Measured on a 2-core test machine (not
Leonardo), with the inputs in the page cache: two simulated ALMA datasets of
541,800 rows × 120 channels, labelled TOPO, stacked with `frame="LSRK"` on 160
channels (30% of the output empty, flagged). v2.3 is the previous version
(MS output, extraction with casatools); "v2.4, MS" is `output_format="ms"`
and the new extraction run on that MS.

| Step | v2.3 (MS, casatools) | v2.4, MS | v2.4, Zarr |
|---|---|---|---|
| stacking, whole run | 13–21 s | 13.5 s | 10–11 s |
| of which writing (per writer) | 9–17 s | 9.6 s | 2.7–4.1 s (4 writers) |
| size of the stack | 2.8 GB | 2.8 GB | 0.86 GB |
| continuum subtraction (order 0) | 224 s | 28 s | 8.6 s |
| compression into statistics | 225 s | 22 s | 4.8 s |
| peak memory of the extraction | 1.9 GB | 1.0 GB | 0.35 GB |
| stack after the subtraction | 8.0 GB | 8.0 GB | 1.5 GB |

`export_ms` of this store takes 14 s. The emptier the grid, the larger the
gain of the store on disk: with 88% of the grid empty (10 datasets × 200
channels on 1000 channels) it was 182 MB against 1.6 GB.

The results are the same: `output_format="ms"` and `export_ms` give the MS
of v2.3 bit for bit (1,083,600 rows, every column and subtable); the
continuum and the subtracted visibilities differ from v2.3 only in the last
float32 digit (closed-form least squares instead of `np.polyfit`), the line
statistics by less than 10⁻⁹ in relative terms, with identical counts. The
continuum statistics change only where they should: the flagged, empty
channels no longer enter the fit of the continuum (see the changelog).

## One pass (v2.5)

On the 2-core test machine (1,083,600 rows, 160 channels, inputs in the page
cache), stacking, continuum subtraction and compression in one pass instead
of three: MS stack 51 s → 26 s; Zarr stack 21.5 s → 17 s. With 2 cores the
computing of the three steps competes with the readers and the writers, so on
Zarr most of the time saved by not reading the stack back is spent there; on
a node with more cores the stacking is limited by reading and the extra
computing is hidden.

## CPU or GPU?

On the Leonardo runs above (MS output) the stacking was limited by the single
writer of the Measurement Set, so the GPU gave the same time as the CPU. With
the parallel Zarr writers the limit moves towards reading and computing,
where the GPU can help.
