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

pip install -q casatools casatasks psutil nvidia-ml-py

# CASA data: downloaded here, on the login node, into $CASA_DATA, and never
# updated by CASA itself: the compute nodes have no internet, and an import
# that tries to update would fail or hang there.  ~/.casa/config.py says so
# (a previous one is kept as config.py.before_vista_bench).  To refresh the
# data later, from a login node:
#     python -c "import casaconfig; casaconfig.update_all('$CASA_DATA')"
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
python - "$CASA_DATA" <<'PY'
import sys
try:
    from casaconfig import update_all
except ImportError:          # older CASA: its data come with the casatools wheel
    print("this casatools has no casaconfig: the data it ships are used as they are")
else:
    update_all(sys.argv[1])
PY
python -c "import casatasks, casatools; print('CASA', casatools.version_string(), 'with the data in $CASA_DATA')"

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
for m in $GSL_MODULE $FFTW_MODULE; do
    module load "$m" || { echo "module load $m failed: look for the right name with "\
                               "'module avail gsl fftw' and set GSL_MODULE / FFTW_MODULE"; exit 1; }
done
GSL_FLAGS=$(dep_flags gsl "${GSL_HOME:-${GSL_ROOT:-}}")
FFTW_FLAGS=$(dep_flags fftw3 "${FFTW_HOME:-${FFTW_ROOT:-}}")
[ -n "$GSL_FLAGS" ]  || { echo "GSL not found after 'module load $GSL_MODULE': set GSL_HOME";  exit 1; }
[ -n "$FFTW_FLAGS" ] || { echo "FFTW not found after 'module load $FFTW_MODULE': set FFTW_HOME"; exit 1; }
echo "GSL:  $GSL_FLAGS"
echo "FFTW: $FFTW_FLAGS"
(cd "$UVMULTIFIT" && CFLAGS="$GSL_FLAGS $FFTW_FLAGS" LDFLAGS="$GSL_FLAGS $FFTW_FLAGS" \
     python setup.py build_ext --inplace > build.log 2>&1) \
    || { tail -20 "$UVMULTIFIT/build.log"; echo "UVMultiFit did not build"; exit 1; }
PYTHONPATH="$UVMULTIFIT" python -c "import uvmultifit; print('UVMultiFit: ok')"
echo "Done: $VISTA_HPC_V1, $VISTA1, $UVMULTIFIT"
