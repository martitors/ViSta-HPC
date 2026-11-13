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
# FFTW one without MPI is enough and pulls in far less.  Another machine:
# give the names of "module avail gsl fftw" in the environment, or set
# GSL_HOME and FFTW_HOME directly.
GSL_MODULE=${GSL_MODULE:-gsl/2.7.1--gcc--12.2.0-spack0.22}
FFTW_MODULE=${FFTW_MODULE:-fftw/3.3.10--gcc--12.2.0-spack0.22}
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

# Where some modules install, without loading them in this shell: these are
# spack modules and bring their own glibc and libstdc++, which in
# LD_LIBRARY_PATH would break the Python of the environment.  The build of
# UVMultiFit only needs their paths, and an rpath makes them found later.
#    module_homes "gsl fftw"   ->   GSL_HOME=/...   FFTW_HOME=/...
module_homes() {
    bash -lc 'module load '"$1"' > /dev/null 2>&1
              for v in GSL_HOME FFTW_HOME OPENBLAS_HOME; do
                  [ -n "${!v-}" ] && printf "%s=%s\n" "$v" "${!v}"
              done' 2>/dev/null
}

# Include and library flags of a dependency installed in $1.
dep_flags() {
    local home=${1-} out=""
    [ -n "$home" ] && [ -d "$home/include" ] || { echo ""; return; }
    out="-I$home/include"
    for d in lib lib64; do
        [ -d "$home/$d" ] && out="$out -L$home/$d -Wl,-rpath,$home/$d"
    done
    echo "$out"
}

activate_env() {
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    for m in $MODULES; do module load "$m"; done
    export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
}
