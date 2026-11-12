#!/bin/bash
# Check the GPU kernel of this branch on one node of Leonardo, from a login node:
#
#     bash benchmarks/leonardo/build.sh
#     INPUT=/path/to/sample/input_list.txt bash benchmarks/leonardo/gpu_check.sh
#
# INPUT: the datasets to stack; by default the sample written by the same
# check of the extract_opt branch ($SCRATCH/vista_gpu_check/sample).  The
# output is $WORK/gpu_check_<jobid>.out; its last line is "RESULT: PASS" or
# "RESULT: FAIL".
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
INPUT=${INPUT:-${SCRATCH:-$HOME/scratch}/vista_gpu_check/sample/input_list.txt}
LINE_FREQ=${LINE_FREQ:-345.79599}
[ -f "$INPUT" ] || { echo "no input list at $INPUT (set INPUT=...)"; exit 1; }
mkdir -p "$WORK"
sbatch --job-name=vista_gpu_check_extract --account="$ACCOUNT" --partition="$PARTITION" \
       ${QOS:+--qos="$QOS"} --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 \
       --time=00:20:00 --output="$WORK/gpu_check_%j.out" \
       --export=ALL,ACCOUNT,PARTITION,QOS,CONDA_BASE,ENV_NAME,MODULES,CUDA_ARCH,WORK \
       --wrap "source '$REPO/benchmarks/leonardo/config.sh' && activate_env && \
               nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && \
               export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK && \
               cd \"\$(dirname '$INPUT')/..\" && \
               python '$REPO/benchmarks/gpu_check.py' --input '$INPUT' --line-freq $LINE_FREQ --work '$WORK'"
echo "When it has run:   tail -30 $WORK/gpu_check_*.out"
