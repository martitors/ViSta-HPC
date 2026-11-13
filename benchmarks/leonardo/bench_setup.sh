#!/bin/bash
# What the benchmark needs besides this checkout, on a login node (once):
#
#     bash benchmarks/leonardo/build.sh          # this checkout (ViSta-HPC v2)
#     bash benchmarks/leonardo/bench_setup.sh
#
# It clones next to this checkout and builds: the branch extract (ViSta-HPC
# v1, with CUDA), ViSta 1.0 and UVMultiFit; and installs CASA and psutil in
# the environment.  UVMultiFit needs GSL and FFTW: they come from the modules
# of the machine ($GSL_MODULE, $FFTW_MODULE in config.sh), nothing is
# installed beside the Python packages.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
activate_env

pip install -q casatools casatasks psutil nvidia-ml-py "zarr>=2.12,<3"

# CASA data.  A recent CASA downloads and updates them by itself, which on a
# compute node (no internet) would fail or hang: they are downloaded here,
# on the login node, into $CASA_DATA, and ~/.casa/config.py turns the updates
# off (a previous config.py is kept as config.py.before_vista_bench).  An
# older CASA has no casaconfig and takes its data from the casadata package:
# then nothing is written, and the ~/.casa it already uses is left alone.
if python -c "import casaconfig" 2> /dev/null; then
    mkdir -p "$HOME/.casa" "$CASA_DATA"
    CFG="$HOME/.casa/config.py"
    if [ -f "$CFG" ] && ! grep -q "bench_setup.sh" "$CFG"; then
        cp "$CFG" "$CFG.before_vista_bench"
    fi
    cat > "$CFG" <<CASACFG
# written by benchmarks/leonardo/bench_setup.sh: the CASA data in one place,
# downloaded on the login node and never updated during a job
measurespath = "$CASA_DATA"
measures_auto_update = False
data_auto_update = False
CASACFG
    python -c "import casaconfig; casaconfig.update_all('$CASA_DATA')"
else
    echo "this CASA has no casaconfig: the data of the casadata package are used as they are"
fi
python -c "import casatasks, casatools; print('CASA', casatools.version_string())"

if [ ! -d "$VISTA_HPC_V1" ]; then
    git clone -q --branch extract "$(git -C "$REPO" remote get-url origin)" "$VISTA_HPC_V1"
fi
git -C "$VISTA_HPC_V1" pull -q --ff-only
bash "$VISTA_HPC_V1/benchmarks/leonardo/build.sh"

[ -d "$VISTA1" ] || git clone -q https://github.com/martitors/ViSta.git "$VISTA1"

if [ ! -d "$UVMULTIFIT" ]; then
    git clone -q https://github.com/marti-vidal-i/UVMultiFit.git "$UVMULTIFIT"
    git -C "$UVMULTIFIT" checkout -q 9aa59b7
    git -C "$UVMULTIFIT" apply "$REPO/benchmarks/uvmultifit.patch"
fi
while IFS="=" read -r key value; do
    [ -n "$key" ] && export "$key=$value"
done < <(module_homes "$GSL_MODULE $FFTW_MODULE")
GSL_FLAGS=$(dep_flags "${GSL_HOME-}")
FFTW_FLAGS=$(dep_flags "${FFTW_HOME-}")
BLAS_FLAGS=$(dep_flags "${OPENBLAS_HOME-}")
[ -n "$GSL_FLAGS" ]  || { echo "GSL not found: 'module avail gsl' and set GSL_MODULE, "\
                               "or set GSL_HOME to where it is installed"; exit 1; }
[ -n "$FFTW_FLAGS" ] || { echo "FFTW not found: 'module avail fftw' and set FFTW_MODULE, "\
                               "or set FFTW_HOME"; exit 1; }
# a GSL built against OpenBLAS (as the one of Leonardo) has no libgslcblas
if ! ls "$GSL_HOME"/lib*/libgslcblas.* > /dev/null 2>&1; then
    echo "this GSL has no libgslcblas: UVMultiFit is linked against OpenBLAS instead"
    sed -i "s/'gslcblas'/'openblas'/g" "$UVMULTIFIT/setup.py"
fi
echo "GSL:  $GSL_FLAGS"
echo "FFTW: $FFTW_FLAGS"
[ -n "$BLAS_FLAGS" ] && echo "BLAS: $BLAS_FLAGS" || true
(cd "$UVMULTIFIT" && CFLAGS="$GSL_FLAGS $FFTW_FLAGS $BLAS_FLAGS" \
     LDFLAGS="$GSL_FLAGS $FFTW_FLAGS $BLAS_FLAGS" \
     python setup.py build_ext --inplace > build.log 2>&1) \
    || { tail -20 "$UVMULTIFIT/build.log"; echo "UVMultiFit did not build"; exit 1; }
PYTHONPATH="$UVMULTIFIT" python -c "import uvmultifit; print('UVMultiFit: ok')"
echo "Done: $VISTA_HPC_V1, $VISTA1, $UVMULTIFIT"
