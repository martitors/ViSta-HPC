# 0. Setup

The code is cloned from GitHub and its C++ kernel is built (a couple of
minutes, once). For a private repository, add a GitHub token to the Colab
secrets (key icon on the left) as `GITHUB_TOKEN`. Outside Colab, set
`VISTA_REPO` to a built checkout.

```python
REPO_URL = "https://github.com/martitors/ViSta-HPC.git"
BRANCH = "extract_opt"

import os, sys, glob, subprocess, logging
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
    else:                                        # a clone of an earlier session: update it
        head = lambda: subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout
        before = head()
        subprocess.run(["git", "-C", REPO, "pull", "-q", "origin", BRANCH])
        if head() != before:                     # new code: build the kernel again, and
            for so in glob.glob(f"{REPO}/vista/ms_ops*.so"):     # forget what was imported
                os.remove(so)
            for name in [m for m in sys.modules if m.split(".")[0] in ("vista", "simulate")]:
                del sys.modules[name]
            if "ms_ops" in sys.modules:
                print("The code was updated: after this cell, Runtime > Restart session, "
                      "then Run all")

if IN_COLAB:
    subprocess.run(f"pip install -q -r {REPO}/requirements.txt pybind11", shell=True, check=True)
if not glob.glob(f"{REPO}/vista/ms_ops*.so"):          # build the kernel once
    subprocess.run(f"cmake -S {REPO} -B {REPO}/build -DCMAKE_BUILD_TYPE=Release > /dev/null"
                   f" && cmake --build {REPO}/build -j 2 > /dev/null"
                   f" && cmake --install {REPO}/build > /dev/null", shell=True, check=True)
sys.path[:0] = [REPO, os.path.join(REPO, "examples")]

import numpy as np
import matplotlib.pyplot as plt
import simulate                                        # examples/simulate.py: the simulator
import vista
from vista import ViSta

logging.getLogger("daskms").setLevel(logging.ERROR)    # quiet about empty optional columns
WORK = "/content/vista_demo" if IN_COLAB else os.path.abspath("vista_demo")
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
print("ViSta", vista.__version__, "from", REPO, "· working in", WORK)
```

```text
ViSta 3.0.0 from /content/ViSta-HPC · working in /content/vista_demo
```

A one-line helper for this notebook, to print a flux with its error:

```python
def flux(result, key="flux"):
    """'0.400 ± 0.050 Jy km/s' from result[key] (small values in mJy)."""
    f = result[key]
    k, unit = (1e3, "m" + f["unit"]) if abs(f["value"]) < 0.01 else (1, f["unit"])   # Jy -> mJy
    return f"{f['value'] * k:.3f} ± {f['error'] * k:.3f} {unit}"
```
