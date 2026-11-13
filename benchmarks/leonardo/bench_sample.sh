#!/bin/bash
# The three simulated samples of the benchmark, as three jobs on the CPU
# partition (from a login node):
#
#     bash benchmarks/leonardo/bench_sample.sh
#
# $BENCH/data/n1000 (100 datasets of 1000 channels), n500 and n2000 (50
# each, the same galaxies as the first 50 of n1000): about 45 GB.  A job that
# ran out of time continues where it stopped when submitted again.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
mkdir -p "$BENCH/logs"
for spec in "1000 100" "500 50" "2000 50"; do
    set -- $spec
    sbatch --job-name="vista_sample_n$1" --account="$CPU_ACCOUNT" --partition="$CPU_PARTITION" \
           ${CPU_QOS:+--qos="$CPU_QOS"} --nodes=1 --ntasks=1 --cpus-per-task=8 \
           --mem=64G --time=04:00:00 --output="$BENCH/logs/sample_n$1_%j.out" \
           --wrap "source '$REPO/benchmarks/leonardo/config.sh' && activate_env && \
                   python '$REPO/benchmarks/simobserve_sample.py' --out '$BENCH/data/n$1' \
                          --nchan $1 --n $2 --seed $SEED --jobs 8"
done
echo "When they have run:   tail -2 $BENCH/logs/sample_n*.out"
