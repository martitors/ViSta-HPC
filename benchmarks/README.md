# Checks and benchmarks

| File | What it does |
|---|---|
| `validation.py` | bias, scatter and coverage of the measured fluxes on synthetic stacks; writes `reference/validation.json`, the numbers of the *Validation* page of the documentation |
| `gpu_check.py` | the GPU kernel against the CPU kernel, and a whole stack on the two devices; prints PASS or FAIL for every check |
| `simobserve_sample.py` | the sample of the benchmark: ALMA observations made with CASA `simobserve`, thermal noise, a seed |
| `bench.py` | the benchmark of the three versions (ViSta 1.0, ViSta-HPC v1, ViSta-HPC v2): time of the stack, of the continuum subtraction and of the extraction, memory, disk |
| `report.py` | its tables (medians of the repeats) |
| `monitor.py` | runs a command and measures time per step, memory, disk |
| `run_vista_casa.py`, `uvmultifit.patch` | ViSta 1.0: the CASA stacking of github.com/martitors/ViSta, then `uvcontsub` and UVMultiFit |
| `run_hpc_v2.py` | ViSta-HPC v2 one step after the other, or in one pass |
| `leonardo/` | the jobs on Leonardo (CINECA): `config.sh` (account, partition, environment), `build.sh`, `gpu_check.sh`, `bench_setup.sh`, `bench_sample.sh`, `bench.sh` |

```bash
python benchmarks/validation.py --help
python benchmarks/gpu_check.py --work /tmp/gpu_check --dry-run     # no GPU: tests the script
```

On Leonardo, from a login node:

```bash
bash benchmarks/leonardo/build.sh        # builds the kernel with CUDA
bash benchmarks/leonardo/gpu_check.sh    # submits the job
tail -40 $SCRATCH/vista_gpu_check/gpu_check_*.out
```

The last line of the output is `RESULT: PASS` or `RESULT: FAIL`.

## The benchmark of the three versions

| Name | Version | Code |
|---|---|---|
| `vista1` | ViSta 1.0 (CASA), CPU only | github.com/martitors/ViSta, then `uvcontsub` and UVMultiFit |
| `hpc_v1` | ViSta-HPC v1 | branch `extract` |
| `hpc_v2_ms`, `hpc_v2_zarr` | ViSta-HPC v2, one step after the other | branch `extract_opt` |
| `hpc_v2_onepass` | ViSta-HPC v2 as it is meant to be used: continuum and statistics while the stack is written | branch `extract_opt` |

Every run is timed in three steps: `stack` (until the stacked file is
written), `contsub`, `extraction`; with CPU time, peak memory (PSS of the
process tree), disk and, on the GPU, GPU memory.  For ViSta 1.0 the time
average and the single spectral window that UVMultiFit needs are counted in
the extraction, and every intermediate copy is kept (the disk it takes is
part of the comparison).

The same work for everyone:

* **the band**: every version stacks the widest velocity range around the
  line that every dataset covers (about +-2200 km/s), on the grid of the
  widest rest-frame channel;
* **the cache**: every run reads its inputs from the page cache, filled just
  before it;
* **the cores**: a run is limited to the cores of its configuration and
  divides them without exceeding them: readers, writers (1 for a Measurement
  Set; for Zarr the number ViSta-HPC v2 itself chooses, a quarter of the
  cores at most 8, unless the scan `writers` fixes it), and the rest as kernel
  threads.  With 32 cores and 2 readers: v1 and v2 MS 2/1/29, v2 Zarr 2/8/22
  (readers/writers/threads).

The standard configuration is 50 datasets of 1000 channels, 32 cores, 2
readers, GPU batch 20 (in the units of each version: datasets for v1, chunks
of at most 5000 rows for v2).  A scan changes one thing:

| Scan | Values | Versions |
|---|---|---|
| `nms` | 10, 25, 50, 100 datasets | all |
| `nchan` | 500, 1000, 2000 channels (the same galaxies) | all |
| `readers` | 1, 2, 3, 4 | ViSta-HPC |
| `cores` | 8, 16, 32 | ViSta-HPC |
| `batch` | 1, 5, 10, 20 per GPU launch | ViSta-HPC, GPU |
| `writers` | 1, 2, 4, 8 Zarr writers | ViSta-HPC v2 Zarr |

ViSta-HPC is run 3 times per configuration, ViSta 1.0 once.  The standard
configuration is run only by the scan `standard`; the other scans take it
from there.  The results of the three versions on the standard
configuration are compared with each other (and with the truth).

The sample: every galaxy is one `simobserve` run (43 antennas, 6 minutes,
one window of about 1.875 GHz around CO(3-2) at z = 2-3, thermal noise), with
the weights set from the noise that was added.  All galaxies have the same
luminosity, line width and size (`truth.json`).  Every galaxy is drawn from
its own random numbers: the samples with 500, 1000 and 2000 channels hold
the same galaxies.  A dataset of 1000 channels is 0.2 GB; the three samples
together 45 GB.

### On a workstation first

```bash
export VISTA_HPC_V1=/path/to/ViSta-HPC-extract                  # built (optional)
export VISTA1=/path/to/ViSta UVMULTIFIT=/path/to/UVMultiFit     # optional
bash benchmarks/local_test.sh /disk/vista_local 4               # 4 cores
```

A sample of 6 galaxies, the GPU check of ViSta-HPC v1 with its batched
launch (if there is a GPU), one run of each version on the standard
configuration scaled down, and the tables.

### On Leonardo

From a login node.  The jobs without a GPU (the samples, ViSta 1.0,
ViSta-HPC on the CPU) go to the CPU partition `CPU_PARTITION` (default
`dcgp_usr_prod`, account `CPU_ACCOUNT`); those on the GPU to the Booster.

```bash
bash benchmarks/leonardo/build.sh                # ViSta-HPC v2 (this checkout)
bash benchmarks/leonardo/bench_setup.sh          # ViSta-HPC v1, ViSta 1.0, UVMultiFit, CASA
bash benchmarks/leonardo/bench_sample.sh         # three jobs: the samples
# when the samples are there, one job per scan and device:
for s in standard nms nchan readers cores writers; do
    bash benchmarks/leonardo/bench.sh $s cpu
done
for s in standard nms nchan readers cores batch writers; do
    bash benchmarks/leonardo/bench.sh $s gpu
done
bash benchmarks/leonardo/bench.sh report         # the tables: $BENCH/results/tables.md
```

`bench_setup.sh` downloads the CASA data on the login node into
`$CASA_DATA` (default `~/.casa/data`) and writes `~/.casa/config.py` so that
CASA never tries to update them: the compute nodes have no internet (a
previous `config.py` is kept as `config.py.before_vista_bench`).

`METHODS=...`, `REPEATS=...`, `REPEATS_VISTA1=...`, `BENCH=...`,
`CPU_PARTITION=...`, `CPU_ACCOUNT=...`, `CPU_EXCLUSIVE=1` (the whole node, no
other job beside) in front of a command change its defaults
(`leonardo/config.sh`).  Every job writes its own results file,
`$BENCH/results/runs_<scan>_<job>.json`, and works in its own directory, so
the jobs can run at the same time; a run already in any results file is not
repeated: a job that ran out of time is submitted again with the same
command.  `python benchmarks/bench.py ... --dry-run` prints the commands
without running them.

UVMultiFit (commit 9aa59b7) needs `uvmultifit.patch` and GSL and FFTW to
build (`git apply .../uvmultifit.patch && python setup.py build_ext
--inplace`): the patch lets it build with NumPy 2 and finds the baseline of a
visibility from its two antennas instead of scanning the ~500 000 baselines
of a stack.
