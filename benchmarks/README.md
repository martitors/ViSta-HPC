# Benchmark: ViSta 1.0 (CASA), ViSta-HPC `extract`, ViSta-HPC `extract_opt`

The three versions of ViSta on the same simulated sample: 24 heterogeneous
ALMA Band 3 observations of CO(3–2) at z = 2–3 (different arrays, channel
widths, bandwidths, depths, sizes, line widths, luminosities, positions in
the primary beam). Measured:

- **time** per step, CPU time, **peak memory** (PSS of the process tree) and
  **disk**, from the input Measurement Sets to the measured line;
- **accuracy without noise** (the noiseless twin of the sample), against the
  truth and against the ideal measurement on this coverage;
- **accuracy with noise**: scatter, bias and calibration of the errors over
  many noise realisations (Monte Carlo).

| File | |
|---|---|
| `sample.py` | the sample and its true values (built on `examples/simulate.py`) |
| `monitor.py` | runs a command and measures time per step, memory, disk |
| `run_vista_casa.py` | ViSta 1.0: the CASA stacking of github.com/martitors/ViSta, then `uvcontsub` and UVMultiFit as in Ch. 4.2 of the thesis |
| `bench.py` | runs the three versions: `sample`, `perf`, `noiseless`, `montecarlo` |
| `ideal.py` | what an exact one-Gaussian measurement gives on this sample (reference of the noiseless runs) |
| `report.py` | tables (`report/tables.md`) and figures |
| `mc_tutorial.py` | Monte Carlo of the tutorial notebook (12 identical galaxies, `extract_opt`) |
| `uvmultifit.patch` | the changes UVMultiFit 3.1 needs here (NumPy 2, speed) |
| `reference/` | the results of the run of 8 Nov 2026 (2-core machine, 3 timed runs, 20 noise realisations): `tables.md`, `summary.json`; `extract_opt` was then version 2.5.1 (2 readers, 4 writers, its main thread pinned to one core by `OMP_PROC_BIND`); from 3.0 the scripts use the new interface, which takes the readers, writers and threads from the cores and no longer pins |

## What is needed

- three checkouts: this repository on `extract_opt` (built), a second one on
  `extract` (built: `cmake -S . -B build && cmake --build build && cmake --install build`),
  and [ViSta](https://github.com/martitors/ViSta) (the CASA version);
- `casatools`, `casatasks` (`pip install casatools casatasks`) for ViSta 1.0;
- [UVMultiFit](https://github.com/marti-vidal-i/UVMultiFit) (commit 9aa59b7)
  with `uvmultifit.patch`, built in place:

  ```bash
  git -C $UVMULTIFIT apply $PWD/benchmarks/uvmultifit.patch
  (cd $UVMULTIFIT && python setup.py build_ext --inplace)   # needs GSL and FFTW
  ```

  The patch lets it build with NumPy 2 (`PyArray_*` casts, `numpy.asfarray`)
  and makes it usable on a stack: its C++ looked for the baseline of every
  visibility by scanning all the baselines, and a stack of 24 datasets has
  ~1000 antennas (each dataset keeps its own), so ~500 000 baselines and
  about 3 hours per fit. The patch computes the index from the two antennas;
  the results do not change (`run_vista_casa.py` has the numbers);
- `psutil`, `scipy`, `matplotlib`; about 12 GB of disk: three copies of the
  sample (noisy, noiseless, Monte Carlo; 1.7 GB each) and a ViSta 1.0 run
  with `--keep` (6 GB: it keeps a copy of the data for every step).

## Running

```bash
export VISTA_CASA=/path/to/ViSta UVMULTIFIT=/path/to/UVMultiFit
export VISTA_EXTRACT=/path/to/ViSta-HPC-extract VISTA_EXTRACT_OPT=/path/to/ViSta-HPC
W=/scratch/vista_bench

python benchmarks/bench.py sample     --work $W              # 24 + 24 datasets, 1.7 GB each
python benchmarks/bench.py noiseless  --work $W
python benchmarks/ideal.py            --work $W --extract-opt $VISTA_EXTRACT_OPT
python benchmarks/bench.py perf       --work $W --repeats 3 --keep
python benchmarks/bench.py montecarlo --work $W --n 20
python benchmarks/report.py           --work $W
```

`--threads` sets the OpenMP threads (default: all cores); `--pipelines
extract,extract_opt` runs only some versions; `montecarlo --first K` continues
a run from realisation K (results already in `results/montecarlo.json` are
not repeated). `--keep` keeps the intermediate Measurement Sets of ViSta 1.0
(its default), so that the disk it needs is measured.

On a cluster, one node, nothing else running on it, e.g. with Slurm:

```bash
#!/bin/bash
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G --time=24:00:00 --exclusive
module load python   # and whatever gives casatools
python benchmarks/bench.py perf --work $SCRATCH/vista_bench --threads 32 --repeats 3 --keep
```

The sample is drawn from `SampleConfig.seed`: the same seed gives the same
galaxies everywhere, so results from different machines can be compared.
