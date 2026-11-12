#!/bin/bash
# Check the GPU kernel on one node of Leonardo, from a login node:
#
#     bash benchmarks/leonardo/build.sh          # once
#     bash benchmarks/leonardo/gpu_check.sh      # submits the job
#
# The job runs benchmarks/gpu_check.py: about 5 minutes.  Its output is
# $WORK/gpu_check_<jobid>.out; the last line is "RESULT: PASS" or
# "RESULT: FAIL".
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
mkdir -p "$WORK"
sbatch --job-name=vista_gpu_check --account="$ACCOUNT" --partition="$PARTITION" \
       ${QOS:+--qos="$QOS"} --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
       --time=00:20:00 --output="$WORK/gpu_check_%j.out" \
       --export=ALL,ACCOUNT,PARTITION,QOS,CONDA_BASE,ENV_NAME,MODULES,CUDA_ARCH,WORK \
       --wrap "source '$REPO/benchmarks/leonardo/config.sh' && activate_env && \
               nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && \
               export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK && \
               python '$REPO/benchmarks/gpu_check.py' --work '$WORK'"
echo "When it has run:   tail -40 $WORK/gpu_check_*.out"
