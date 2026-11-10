# 0. Setup

The code is cloned from GitHub, the C++/OpenMP kernel (`ms_ops`) is built
with CMake, and CASA's `casatasks` and `casatools` are installed for the
images. For a private repository, add a GitHub token to the Colab secrets
(key icon on the left) as `GITHUB_TOKEN`; or upload a zip of the repository
and unzip it in `/content`: the cell then uses it. On Colab the first import
of CASA downloads its data tables (a few minutes).

```python
REPO_URL = "https://github.com/martitors/ViSta-HPC.git"
BRANCH = "extract_opt"

import os, sys, glob, subprocess
IN_COLAB = "google.colab" in sys.modules
REPO = os.environ.get("VISTA_REPO")            # a local checkout, to run this outside Colab
if REPO is None:
    REPO = "/content/ViSta-HPC" if IN_COLAB else os.path.abspath("ViSta-HPC")
    if not os.path.isdir(REPO):
        url = REPO_URL
        if IN_COLAB:
            try:                                 # private repository: token from the Colab secrets
                from google.colab import userdata
                url = REPO_URL.replace("https://", f"https://{userdata.get('GITHUB_TOKEN')}@")
            except Exception:
                pass
        subprocess.run(["git", "clone", "-q", "-b", BRANCH, url, REPO], check=True)
print("ViSta source:", REPO)


def sh(cmd):
    print("$", cmd)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError(cmd)


if IN_COLAB:
    sh(f"pip install -q -r {REPO}/requirements.txt pybind11 casatasks casatools")
if not glob.glob(f"{REPO}/vista/ms_ops*.so"):          # build the kernel once
    sh(f"cmake -S {REPO} -B {REPO}/build -DCMAKE_BUILD_TYPE=Release > /dev/null"
       f" && cmake --build {REPO}/build -j 2 > /dev/null && cmake --install {REPO}/build > /dev/null")

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "examples"))   # simulate.py, plot_line.py
import vista
from vista import ms_ops                               # the compiled kernel
print("ViSta", vista.__version__, "· kernel", os.path.basename(ms_ops.__file__))
```

```text
ViSta source: /content/ViSta-HPC
ViSta 3.0.0 · kernel ms_ops.cpython-313-x86_64-linux-gnu.so
```

```python
import json, math, time, shutil, logging, contextlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from daskms.experimental.zarr import xds_from_zarr
from casatasks import tclean
from casatools import image as iatool

import simulate                                        # the simulated observations
import plot_line as pl                                 # the figures of a line measurement
from vista import ViSta, export_ms

logging.getLogger("daskms").setLevel(logging.ERROR)    # quiet about empty optional columns
WORK = "/content/vista_demo" if IN_COLAB else os.path.abspath("vista_demo")
os.makedirs(f"{WORK}/figures", exist_ok=True)
os.chdir(WORK)
pl.apply_style()                                       # one style for every figure
plt.rcParams.update({"figure.dpi": 110})
C_KMS = 299792.458

TRUE = dict(color="#1e8449", ls="--", lw=1.8)          # the true signal
CLEAN = dict(color="#2471a3", lw=1.3)                  # the data without noise
NOISY = dict(color="k", lw=0.9)                        # the data with noise


def save(fig, name):
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    plt.show()
```
