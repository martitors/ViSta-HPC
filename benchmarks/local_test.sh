#!/bin/bash
# A small run of the whole benchmark on a workstation, to try everything
# before Leonardo (about 1.2 GB of data):
#
#     export VISTA_HPC_V1=/path/to/ViSta-HPC-extract                  # built (optional)
#     export VISTA1=/path/to/ViSta UVMULTIFIT=/path/to/UVMultiFit     # optional
#     bash benchmarks/local_test.sh /disk/vista_local [cores]
#
# 1. a sample of 6 galaxies of 1000 channels (CASA simobserve);
# 2. with a GPU and ViSta-HPC v1: its GPU check, batched launch included,
#    with 2 datasets per launch;
# 3. the commands of the standard configuration, scaled down (6 datasets,
#    CORES cores, 1 reader, GPU batch 2), then one run of each;
# 4. the tables: WORK/results/tables.md.
# A version whose path is not set is left out.  N=10 in front: 10 galaxies.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
W=${1:?usage: local_test.sh WORK [cores]}
CORES=${2:-4}
N=${N:-6}
SMALL=(--n-ms "$N" --cores "$CORES" --readers 1 --gpu-batch 2)

methods=hpc_v2_ms,hpc_v2_zarr,hpc_v2_onepass
[ -n "${VISTA_HPC_V1:-}" ] && methods=hpc_v1,$methods
[ -n "${VISTA1:-}" ] && [ -n "${UVMULTIFIT:-}" ] && methods=vista1,$methods
devices=cpu
if command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1; then
    devices=cpu,gpu
fi
echo "versions: $methods   devices: $devices   cores: $CORES"

echo; echo "== 1. the sample"
python "$HERE/simobserve_sample.py" --out "$W/data/n1000" --n "$N" --nchan 1000 --seed 1 --jobs 2

if [ "$devices" = cpu,gpu ] && [ -n "${VISTA_HPC_V1:-}" ]; then
    echo; echo "== 2. ViSta-HPC v1: GPU against CPU, batched launch included"
    PYTHONPATH="$VISTA_HPC_V1" VISTA_OMP_THREADS="$CORES" \
        python "$VISTA_HPC_V1/benchmarks/gpu_check.py" --input "$W/data/n1000/input_list.txt" \
               --line-freq 345.79599 --work "$W/gpu_check_v1" --gpu-batch 2
fi

echo; echo "== 3. the runs"
python "$HERE/bench.py" --work "$W" --scan standard --methods "$methods" --devices "$devices" \
       "${SMALL[@]}" --dry-run
python "$HERE/bench.py" --work "$W" --scan standard --methods "$methods" --devices "$devices" \
       "${SMALL[@]}" --repeats 1 --warm-cache --keep-runs

echo; echo "== 4. the tables"
python "$HERE/report.py" --work "$W" "${SMALL[@]}"
