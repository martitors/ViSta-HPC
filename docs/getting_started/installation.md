# Installation

ViSta is a Python package with a compiled kernel, `ms_ops` (C++/OpenMP,
optionally CUDA). It is installed from the repository: clone it, install the
Python dependencies, build the kernel.

## Requirements

**Python** ≥ 3.8 and

```text
numpy >= 1.21
dask >= 2022.1
dask-ms >= 0.2.18
xarray >= 0.19
python-casacore >= 3.5      # reading and writing Measurement Sets
zarr >= 2.12, < 3           # the stack (dask-ms reads zarr 2 stores)
numcodecs >= 0.10
astropy >= 5.0              # coordinates, luminosity distances
scipy >= 1.7                # the measurement
matplotlib >= 3.4           # optional: figures
```

**To build the kernel**: CMake ≥ 3.18, a C++ compiler with OpenMP (GCC),
`pybind11`; for the GPU, the CUDA Toolkit ≥ 11.0.

CASA is **not** needed. `casatasks` and `casatools` are used only by the
example notebook, to image a stack with `tclean`.

## 1. Clone the repository

```bash
git clone https://github.com/martitors/ViSta-HPC.git
cd ViSta-HPC
```

## 2. Install the Python dependencies

```bash
pip install -r requirements.txt pybind11
```

## 3. Build the kernel

CPU only:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
cmake --install build
```

With the GPU (CUDA):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
cmake --install build
```

`cmake --install` puts the module (`ms_ops*.so`) in the `vista/` directory.
Set `CMAKE_CUDA_ARCHITECTURES` to the compute capability of the GPU: 70 V100,
80 A100 (the default), 86 RTX 30xx, 89 L40, 90 H100. The CUDA runtime is
linked statically, so the module runs on compute nodes without
`LD_LIBRARY_PATH` settings, and a CUDA build also runs on machines without a
GPU (`device="cpu"`).

## 4. Check

```bash
python -c "import vista; from vista import ms_ops; print(vista.__version__, ms_ops.__file__)"
./run_vista.sh --help
```

Use the package from the repository directory, or put it on the path:
`export PYTHONPATH=/path/to/ViSta-HPC:$PYTHONPATH` (or `pip install -e .`).

Without the kernel, the stacking does not run; the continuum subtraction and
the compression fall back to NumPy (much slower), and the measurement needs
only NumPy and SciPy.

## On an HPC cluster

- Build on the node type that will run the jobs (same CPU and GPU).
- Threads: the kernel uses `OMP_NUM_THREADS`, else `SLURM_CPUS_PER_TASK`,
  else all cores; `run(n_threads=...)` overrides them.
- Put the stack on fast local disk while it is written,
  `run(scratch_dir=os.environ["TMPDIR"])` (`--scratch` on the command line):
  it is moved to its destination at the end.
- In containers with a small `/dev/shm`, `run(reader_mode="thread")` avoids
  the shared memory of the reader processes.

## On Google Colab

The [example notebook](../examples/index) installs and builds everything by
itself: open it in Colab and run all the cells.

## Building this documentation

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```
