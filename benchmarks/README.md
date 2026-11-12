# Checks and benchmarks

| File | What it does |
|---|---|
| `validation.py` | bias, scatter and coverage of the measured fluxes on synthetic stacks; writes `reference/validation.json`, the numbers of the *Validation* page of the documentation |
| `gpu_check.py` | the GPU kernel against the CPU kernel, and a whole stack on the two devices; prints PASS or FAIL for every check |
| `leonardo/` | the same check as a job on Leonardo (CINECA): `config.sh` (account, partition, environment), `build.sh`, `gpu_check.sh` |

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
