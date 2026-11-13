#!/bin/bash
# One scan of the benchmark as a job on one node (from a login node):
#
#     bash benchmarks/leonardo/bench.sh standard cpu   # CPU partition: ViSta 1.0 and ViSta-HPC on the CPU
#     bash benchmarks/leonardo/bench.sh standard gpu   # Booster: ViSta-HPC on the GPU
#     bash benchmarks/leonardo/bench.sh nms gpu        # scans: standard, nms, nchan, readers,
#                                                      # cores, batch (gpu), writers
#     METHODS=hpc_v1,hpc_v2_zarr bash benchmarks/leonardo/bench.sh readers gpu
#
# A job has 32 cores (and, on the Booster, one GPU); every run is limited to
# the cores of its configuration.  Its results go to its own file,
# $BENCH/results/runs_<scan>_<job>.json, and it works in its own directory,
# so several jobs can run at the same time; runs already in any results file
# are not repeated, so the same command continues a scan that ran out of
# time.  Then:
#
#     bash benchmarks/leonardo/bench.sh report         # the tables, on the login node
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
SCAN=${1:?standard, nms, nchan, readers, cores, batch, writers or report}
DEVICE=${2:-gpu}
if [ "$SCAN" = report ]; then
    activate_env
    exec python "$REPO/benchmarks/report.py" --work "$BENCH"
fi
case "$DEVICE" in
    gpu) where=(--account="$ACCOUNT" --partition="$PARTITION" ${BENCH_QOS:+--qos="$BENCH_QOS"}
                --gres=gpu:1)
         check="nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && " ;;
    cpu) where=(--account="$CPU_ACCOUNT" --partition="$CPU_PARTITION" ${CPU_QOS:+--qos="$CPU_QOS"})
         [ "$CPU_EXCLUSIVE" = 1 ] && where+=(--exclusive)
         check="" ;;
    *)   echo "device: cpu or gpu"; exit 1 ;;
esac
mkdir -p "$BENCH/logs"
sbatch --job-name="vista_bench_${SCAN}_$DEVICE" "${where[@]}" \
       --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=240G --time="$BENCH_TIME" \
       --output="$BENCH/logs/bench_${SCAN}_${DEVICE}_%j.out" \
       --wrap "source '$REPO/benchmarks/leonardo/config.sh' && activate_env && \
               ${check}lscpu | grep 'Model name' && \
               python '$REPO/benchmarks/bench.py' --work '$BENCH' --scan $SCAN \
                      --devices $DEVICE --repeats ${REPEATS:-3} \
                      --repeats-vista1 ${REPEATS_VISTA1:-1} --warm-cache \
                      ${METHODS:+--methods $METHODS}"
echo "Follow it with:   tail -f $BENCH/logs/bench_${SCAN}_${DEVICE}_*.out"
