#!/bin/bash
# Build the kernel with CUDA, on a login node (a couple of minutes):
#
#     bash benchmarks/leonardo/build.sh
#
# The module goes into vista/ of this checkout.  It also runs on nodes
# without a GPU (device="cpu").
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
activate_env
cd "$REPO"
rm -rf build_gpu
cmake -S . -B build_gpu -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" > build_gpu.log 2>&1 \
    || { tail -40 build_gpu.log; echo "cmake failed: see build_gpu.log"; exit 1; }
cmake --build build_gpu -j 8 >> build_gpu.log 2>&1 \
    || { tail -40 build_gpu.log; echo "the build failed: see build_gpu.log"; exit 1; }
cmake --install build_gpu >> build_gpu.log 2>&1
python - <<'PY'
from vista import ms_ops
print("kernel:", ms_ops.__file__)
from vista import pipeline; print("sees a GPU (only on a GPU node):", pipeline._gpu_available)
PY
echo "Done (log: $REPO/build_gpu.log)"
