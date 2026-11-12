# Settings of the Leonardo scripts.  Every value can be given in the
# environment instead:   ACCOUNT=... bash benchmarks/leonardo/gpu_check.sh
#
# Sourced by the other scripts of this directory.

ACCOUNT=${ACCOUNT:-ICT26_MHPC_0}                 # SLURM account (saldo -b)
PARTITION=${PARTITION:-boost_usr_prod}           # Booster: 32 cores and 4 A100 per node
QOS=${QOS:-boost_qos_dbg}                        # debug queue: 30 minutes, 2 jobs; "" = normal
CONDA_BASE=${CONDA_BASE:-$HOME/opt/miniconda3}
ENV_NAME=${ENV_NAME:-vista38}                    # conda environment with the Python packages
MODULES=${MODULES:-cuda/12.2}                    # modules for the CUDA build
CUDA_ARCH=${CUDA_ARCH:-80}                       # compute capability: 80 = A100
WORK=${WORK:-${SCRATCH:-$HOME/scratch}/vista_gpu_check}   # where samples and stacks go

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # this checkout

activate_env() {
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    for m in $MODULES; do module load "$m"; done
    export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
}
