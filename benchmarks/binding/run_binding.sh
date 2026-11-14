#!/bin/bash
# What the Python interface costs: the same kernel called from C++ and from
# Python, on chunks of growing size (from a login node):
#
#     bash benchmarks/binding/run_binding.sh          # a job in the debug queue
#     bash benchmarks/binding/run_binding.sh here     # here and now, for a try
#
# It needs no data and reads nothing of the sample, so it can run while the
# benchmark of the three versions is going on: it only wants a node of its own.
# The result is $BENCH/results/binding.jsonl and the table on the screen.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../leonardo/config.sh"
HERE="$REPO/benchmarks/binding"
OUT=${OUT:-$BENCH/results/binding.jsonl}
THREADS=${THREADS:-"1 32"}
CHANS=${CHANS:-"16 1000"}      # a tiny chunk, where the interface shows, and a real one

if [ "${1:-job}" != here ]; then
    mkdir -p "$BENCH/logs" "$BENCH/results"
    sbatch --job-name=vista_binding --account="$CPU_ACCOUNT" \
           --partition="${CPU_PARTITION:-boost_usr_prod}" --qos="${BINDING_QOS:-boost_qos_dbg}" \
           --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=64G --time="${BINDING_TIME:-00:30:00}" \
           --output="$BENCH/logs/binding_%j.out" \
           --wrap "bash '$HERE/run_binding.sh' here"
    echo "Then:   cat \$BENCH/logs/binding_*.out"
    exit 0
fi

activate_env
mkdir -p "$BENCH/results"
cd "$HERE"
echo "== building the C++ caller, the flags of the module =="
g++ -O3 -fopenmp -std=c++17 $(python -m pybind11 --includes) \
    bench_binding.cpp -o "$BENCH/bench_binding" $(python3-config --ldflags --embed)
lscpu | grep "Model name"

: > "$OUT"
for t in $THREADS; do
    for nc in $CHANS; do
        new=$(( nc * 4 / 5 ))
        echo "== $t thread(s), $nc channels =="
        OMP_NUM_THREADS=$t "$BENCH/bench_binding" --threads "$t" --nchan "$nc" \
                          --nchan-new "$new" | tee -a "$OUT"
        OMP_NUM_THREADS=$t python bench_binding.py --threads "$t" --nchan "$nc" \
                          --nchan-new "$new" | tee -a "$OUT"
    done
done

python - "$OUT" <<TABLE
import json, sys
cpp, py = {}, {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    (cpp if r["side"] == "cpp" else py)[(r["threads"], r["nchan_old"], r["nrow"])] = r
print("\n| threads | channels | rows | C++ kernel | C++ kernel+alloc | Python | interface | of the call |")
print("|--:|--:|--:|--:|--:|--:|--:|--:|")
bad = []
for key in sorted(cpp):
    c, q = cpp[key], py.get(key)
    if q is None:
        continue
    for field in ("sum_vis", "sum_uvw", "n_flag"):
        if abs(c[field] - q[field]) > 1e-5 * max(1.0, abs(c[field])):
            bad.append(f"{key} {field}: {c[field]} (C++) != {q[field]} (Python)")
    over = q["python_min"] - c["kernel_alloc_min"]
    us = lambda v: f"{v * 1e6:,.1f} us"
    print(f"| {key[0]} | {key[1]} | {key[2]} | {us(c['kernel_min'])} | "
          f"{us(c['kernel_alloc_min'])} | {us(q['python_min'])} | {us(over)} | "
          f"{100 * over / q['python_min']:+.2f}% |")
print("\nMinimum of the repetitions, 2 correlations, output grid 4/5 of the "
      "input one.  ViSta-HPC v2 reads in chunks of 5000 rows of 1000 channels: "
      "the last rows are the real ones, the first are there to make the fixed "
      "cost of the call visible.")
if bad:
    print("\nThe two sides did NOT compute the same thing:")
    for b in bad:
        print(" ", b)
    sys.exit(1)
print("Both sides wrote the same result.")
TABLE
