"""Run a command and measure it: wall time per stage, memory, disk.

* **Stages**: the command prints marker lines ``@@stage <name>`` (the
  pipelines of this benchmark do; the ``== [k/4]`` headers printed by the
  ``run_vista.sh`` of the extract branch are recognised too).  A stage lasts
  until the next marker or the end of the run.
* **Memory**: the proportional set size (PSS) of the whole process tree,
  sampled every ``interval`` s.  PSS counts a page shared by several processes
  once in total (each gets its share), so the sum over the tree is the memory
  the run really occupies, shared memory between reader processes included.
  The peak of the run and of every stage are reported.
* **Disk**: the bytes under the output directory, sampled every
  ``disk_interval`` s (peak and final).
* **CPU**: user + system time of the tree, to see how many cores were used:
  the resource usage of the command and of every descendant it waited for
  (``wait4``), so short-lived processes count too.
* **GPU memory** (with ``gpu=True``): the memory used on the GPU, sampled
  every ``gpu_interval`` s with NVML (``nvidia-ml-py``) or ``nvidia-smi``.
  The job has the GPU to itself, so this is the memory of the run.

Usage::

    python monitor.py --out metrics.json --watch OUTDIR -- command args...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import psutil

STAGE_PATTERNS = [
    re.compile(r"^@@stage\s+(?P<name>\S+)"),
    re.compile(r"^== \[(?P<k>\d)/4\]\s+(?P<name>[a-z]+)"),
]


def tree(proc):
    try:
        return [proc] + proc.children(recursive=True)
    except psutil.Error:
        return []


def pss_of(procs):
    total = 0
    for p in procs:
        try:
            total += p.memory_full_info().pss
        except psutil.Error:
            pass
    return total


def cpu_of(procs, seen):
    for p in procs:
        try:
            t = p.cpu_times()
            seen[p.pid] = t.user + t.system + getattr(t, "children_user", 0) \
                + getattr(t, "children_system", 0)
        except psutil.Error:
            pass


def gpu_memory_reader():
    """A function that returns the bytes used on GPU 0, or None."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return lambda: pynvml.nvmlDeviceGetMemoryInfo(handle).used
    except Exception:
        pass
    if shutil.which("nvidia-smi") is None:
        return None

    def query():
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                  "--format=csv,noheader,nounits", "-i", "0"],
                                 capture_output=True, text=True, timeout=10).stdout
            return int(out.split()[0]) * 2**20
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
            return None
    return query


def du(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return total


def run(cmd, watch=None, interval=0.5, disk_interval=2.0, log=None, env=None, cwd=None,
        gpu=False, gpu_interval=1.0, cpus=None):
    """Run ``cmd`` and follow it.  ``cpus``: the cores it may use (its CPU
    affinity, inherited by every process it starts); None: those of this
    process."""
    gpu_used = gpu_memory_reader() if gpu else None
    gpu_base = (gpu_used() or 0) if gpu_used else 0      # memory used before the run
    t_start = time.time()
    pin = (lambda: os.sched_setaffinity(0, cpus)) if cpus else None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1, env=env, cwd=cwd, preexec_fn=pin)
    ps = psutil.Process(proc.pid)
    stages = [("start", 0.0)]
    mem = []                         # (t, pss)
    disk = []                        # (t, bytes)
    gmem = []                        # (t, bytes on the GPU)
    cpu = {}
    stop = threading.Event()

    def sampler():
        last_disk = last_gpu = -1e9
        while not stop.is_set():
            t = time.time() - t_start
            procs = tree(ps)
            mem.append((t, pss_of(procs)))
            cpu_of(procs, cpu)
            if watch and t - last_disk >= disk_interval:
                disk.append((t, du(watch)))
                last_disk = t
            if gpu_used and t - last_gpu >= gpu_interval:
                g = gpu_used()
                if g is not None:
                    gmem.append((t, max(0, g - gpu_base)))
                last_gpu = t
            stop.wait(interval)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    out = open(log, "w") if log else None
    for line in proc.stdout:
        t = time.time() - t_start
        if out:
            out.write(f"[{t:9.2f}] {line}")
            out.flush()
        for pat in STAGE_PATTERNS:
            m = pat.match(line.strip())
            if m:
                stages.append((m.group("name"), t))
                break
    # wait4: the exit status and the resources of the command and of every
    # descendant it waited for
    _, status, usage = os.wait4(proc.pid, 0)
    rc = proc.returncode = os.waitstatus_to_exitcode(status)
    t_end = time.time() - t_start
    stop.set()
    th.join()
    if watch:
        disk.append((t_end, du(watch)))
    if out:
        out.close()

    # stage durations and peaks
    marks = stages + [("end", t_end)]
    per_stage = []
    for (name, a), (_, b) in zip(marks[:-1], marks[1:]):
        if b - a <= 0 and name == "start":
            continue
        m = [v for t, v in mem if a <= t <= b]
        d = [v for t, v in disk if a <= t <= b]
        g = [v for t, v in gmem if a <= t <= b]
        per_stage.append({"stage": name, "seconds": b - a,
                          "peak_memory_gb": (max(m) if m else 0) / 1e9,
                          "peak_disk_gb": (max(d) if d else None) and max(d) / 1e9,
                          "peak_gpu_memory_gb": max(g) / 1e9 if g else None})
    return {
        "command": cmd, "returncode": rc, "wall_seconds": t_end,
        "cpu_seconds": max(usage.ru_utime + usage.ru_stime, sum(cpu.values())),
        "peak_memory_gb": max((v for _, v in mem), default=0) / 1e9,
        "peak_disk_gb": max((v for _, v in disk), default=0) / 1e9 if watch else None,
        "final_disk_gb": disk[-1][1] / 1e9 if disk else None,
        "peak_gpu_memory_gb": max(v for _, v in gmem) / 1e9 if gmem else None,
        "stages": per_stage,
        "memory_trace": [(round(t, 2), round(v / 1e9, 4)) for t, v in mem[::5]],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--watch", default=None, help="directory whose size is followed")
    ap.add_argument("--log", default=None)
    ap.add_argument("--gpu", action="store_true", help="also follow the memory used on the GPU")
    ap.add_argument("--cores", type=int, default=None,
                    help="run the command on the first CORES cores of this process")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    cpus = sorted(os.sched_getaffinity(0))[:a.cores] if a.cores else None
    res = run(cmd, watch=a.watch, log=a.log, gpu=a.gpu, cpus=cpus)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("memory_trace",)}, indent=1))
    sys.exit(res["returncode"])


if __name__ == "__main__":
    main()
