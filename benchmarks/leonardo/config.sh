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
# UVMultiFit needs GSL and FFTW: the modules of the machine, not conda.  The
# names are those of "module avail gsl fftw"; give them in the environment if
# they differ:  GSL_MODULE=gsl/2.7.1--gcc--12.2.0 bash .../bench_setup.sh
GSL_MODULE=${GSL_MODULE:-gsl}
FFTW_MODULE=${FFTW_MODULE:-fftw}
CUDA_ARCH=${CUDA_ARCH:-80}                       # compute capability: 80 = A100
WORK=${WORK:-${SCRATCH:-$HOME/scratch}/vista_gpu_check}   # where samples and stacks go

# the benchmark of the three versions (bench_*.sh)
BENCH=${BENCH:-${SCRATCH:-$HOME/scratch}/vista_bench}     # its samples, runs and results
BENCH_QOS=${BENCH_QOS-}                          # normal queue ("boost_qos_dbg": 30 minutes)
BENCH_TIME=${BENCH_TIME:-12:00:00}
SEED=${SEED:-1}                                  # of the simulated sample
# The jobs without a GPU (the samples, ViSta 1.0, ViSta-HPC on the CPU) go to
# a CPU partition, usually less busy.  Its account can differ from the
# Booster one: check with  saldo -b  and  sinfo -s.
CPU_PARTITION=${CPU_PARTITION:-dcgp_usr_prod}    # DCGP: CPU-only nodes
CPU_ACCOUNT=${CPU_ACCOUNT:-$ACCOUNT}
CPU_QOS=${CPU_QOS-}
CPU_EXCLUSIVE=${CPU_EXCLUSIVE:-0}                # 1: the whole node, no other job beside
# CASA on the compute nodes, which have no internet: its data are downloaded
# once on the login node (bench_setup.sh) and never updated during a job
CASA_DATA=${CASA_DATA:-$HOME/.casa/data}           # not on $SCRATCH, which is purged

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # this checkout
CODES=${CODES:-$(dirname "$REPO")}               # the other codes are cloned next to it
export VISTA_HPC_V1=${VISTA_HPC_V1:-$CODES/ViSta-HPC-extract}   # branch extract
export VISTA1=${VISTA1:-$CODES/ViSta}                           # ViSta 1.0 (CASA)
export UVMULTIFIT=${UVMULTIFIT:-$CODES/UVMultiFit}

# "module" is a shell function: in a script that does not come from a login
# shell it may not be there yet.
if ! type module > /dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh "${LMOD_PKG:-/usr/share/lmod/lmod}/init/bash"; do
        # shellcheck disable=SC1090
        [ -r "$f" ] && source "$f" && break
    done
fi

# Include and library flags of a dependency loaded as a module: from
# pkg-config if the module sets it, else from <NAME>_HOME or <NAME>_ROOT.
# Usage:  dep_flags gsl "${GSL_HOME:-${GSL_ROOT:-}}"
dep_flags() {
    local pc=$1 home=${2-} libs
    if pkg-config --exists "$pc" 2>/dev/null; then
        libs=$(pkg-config --libs-only-L "$pc")
        echo "$(pkg-config --cflags "$pc") $libs ${libs//-L/-Wl,-rpath,}"
    elif [ -n "$home" ] && [ -d "$home/include" ]; then
        echo "-I$home/include -L$home/lib -Wl,-rpath,$home/lib"
    fi
}

activate_env() {
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    for m in $MODULES; do module load "$m"; done
    export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
}
