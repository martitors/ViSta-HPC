"""The benchmark of the three versions of ViSta on the same sample.

* ``vista1``: ViSta 1.0, the CASA version (github.com/martitors/ViSta), with
  ``uvcontsub`` and UVMultiFit after it (``run_vista_casa.py``); CPU only;
* ``hpc_v1``: ViSta-HPC v1, the branch ``extract`` (its ``run_vista.sh``);
* ``hpc_v2_ms``, ``hpc_v2_zarr``: ViSta-HPC v2, the branch ``extract_opt``,
  writing a Measurement Set or Zarr, one step after the other;
* ``hpc_v2_onepass``: the same (Zarr) as it is meant to be used: continuum
  subtracted and statistics made while the stack is written.

Every run is timed in three steps -- ``stack`` (until the stacked file is
written), ``contsub`` (the continuum subtraction) and ``extraction`` (the
measurement in the uv plane) -- with CPU time, peak memory, disk and, on the
GPU, GPU memory (``monitor.py``).  For ViSta 1.0 the time average and the
single spectral window that UVMultiFit needs are counted in the extraction,
and every intermediate copy is kept: the disk it takes is part of the
comparison.

**The same work for everyone.**  Every version stacks the same band: the
widest velocity range around the line that every dataset of the sample
covers (``common_velocity_range``), on the grid of the widest rest-frame
channel.  Every run reads its inputs from the page cache, filled just before
it (``--warm-cache``).

**The cores.**  A run is given ``cores`` cores (its CPU affinity is limited
to them) and divides them without exceeding them: ViSta-HPC uses
``readers`` reader threads or processes, its writers (1 for a Measurement
Set; for Zarr ``writers``, or with 0 the number ViSta-HPC v2 itself chooses,
a quarter of the cores at most 8) and the rest as kernel threads.  ViSta
1.0 and UVMultiFit use what they use, within the same cores.

The standard configuration is 50 datasets of 1000 channels, 32 cores, 2
readers, GPU batch 20 (in the units of each version: datasets for v1, chunks
of at most 5000 rows for v2).  Every scan changes one thing:

    standard   the standard configuration
    nms        10, 25, 100 datasets                       (and 50: standard)
    nchan      500, 2000 channels                         (and 1000: standard)
    readers    1, 3, 4 readers                  (ViSta-HPC; and 2: standard)
    cores      8, 16 cores                      (ViSta-HPC; and 32: standard)
    batch      1, 5, 10 per GPU launch      (ViSta-HPC, GPU; and 20: standard)
    writers    1, 2, 4, 8 Zarr writers                  (ViSta-HPC v2 Zarr only)

ViSta 1.0 runs in the scans standard, nms and nchan, once per configuration
(``--repeats-vista1``), and only when ``cpu`` is among the devices.  The
standard configuration is run only by the scan ``standard``: the other scans
take it from there, so that jobs running at the same time do not repeat it.
Every job writes its own file, WORK/results/runs_<scan>_<job>.json, and works
in its own directory, WORK/runs/<job>/; a run already in any results file is
not repeated, so a job that ran out of time is simply submitted again.

Usage::

    python bench.py --work WORK --scan nms --devices gpu --repeats 3
    python bench.py --work WORK --scan standard --devices cpu --dry-run   # the commands only

The samples are WORK/data/n500, n1000, n2000 (``simobserve_sample.py``).
``report.py`` makes the tables (medians of the repeats).
"""
import argparse
import glob
import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

METHODS = ("vista1", "hpc_v1", "hpc_v2_ms", "hpc_v2_zarr", "hpc_v2_onepass")
HPC = METHODS[1:]
# writers = 0: the Zarr writers are chosen as ViSta-HPC v2 would choose them
STANDARD = {"n_ms": 50, "nchan": 1000, "cores": 32, "readers": 2, "gpu_batch": 20,
            "writers": 0}
SCANS = {
    "standard": [{}],
    "nms": [{"n_ms": n} for n in (10, 25, 50, 100)],
    "nchan": [{"nchan": n} for n in (500, 1000, 2000)],
    "readers": [{"readers": n} for n in (1, 2, 3, 4, 8)],
    "cores": [{"cores": n} for n in (8, 16, 32)],
    "batch": [{"gpu_batch": n} for n in (1, 5, 10, 20)],
    "writers": [{"writers": n} for n in (1, 2, 4, 8)],
}
VISTA1_SCANS = ("standard", "nms", "nchan")     # ViSta 1.0 has no readers, threads, GPU
GPU_ONLY = ("batch",)
ZARR_ONLY = ("writers",)                        # only the Zarr store has several writers
TAG = "line345796"                              # the name run_vista.sh gives to CO(3-2)
LINE_WINDOW = 550                               # km/s, half-width of the line window
C_KMS = 299792.458

# the stages of every version -> the three steps of the benchmark
STEPS = {
    # ViSta 1.0 (run_vista_casa.py)
    "restframing": "stack", "centering": "stack", "rebinning": "stack", "stacking": "stack",
    "contsub": "contsub",
    "averaging": "extraction", "unify": "extraction",          # what UVMultiFit needs
    "fit_size": "extraction", "fit_channels": "extraction", "fit_profile": "extraction",
    "fit_continuum": "extraction",
    # ViSta-HPC v1 (the "== [k/4] name" headers of run_vista.sh)
    "continuum": "contsub", "compression": "extraction",
    # ViSta-HPC v2 (run_hpc_v2.py)
    "stack": "stack", "extraction": "extraction",
}


def job_id():
    return os.environ.get("SLURM_JOB_ID") or f"local{os.getpid()}_{int(time.time())}"


def load_runs(work):
    """Every run of every job (and of the single file of earlier versions)."""
    runs = []
    for path in sorted(glob.glob(os.path.join(work, "results", "runs*.json"))):
        try:
            runs += json.load(open(path))
        except (OSError, ValueError):
            print(f"warning: {path} could not be read", flush=True)
    return runs


def value(run, k):
    """A parameter of a recorded run (runs recorded before it existed: its default)."""
    return run.get(k, STANDARD.get(k))


def job_cpus():
    return sorted(os.sched_getaffinity(0))


def common_velocity_range(work, margin_kms=60.0):
    """The widest range +-v around the line that every galaxy of the sample
    covers, minus a margin for the change of frame (km/s, a multiple of 10).
    It depends only on the galaxies: the samples with 500, 1000 and 2000
    channels hold the same galaxies with the same bands."""
    for name in ("n1000", "n500", "n2000"):
        path = os.path.join(work, "data", name, "truth.json")
        if os.path.exists(path):
            t = json.load(open(path))
            break
    else:
        raise SystemExit(f"no sample in {work}/data: run simobserve_sample.py first")
    rest = t["rest_freq_ghz"] * 1e9
    half = min(C_KMS * 0.5 * g["nchan"] * g["channel_width_hz"] / (rest / (1 + g["z"]))
               for g in t["galaxies"])
    v = 10 * math.floor((half - margin_kms) / 10)
    if v <= LINE_WINDOW:
        raise SystemExit(f"the band every dataset covers, +-{half:.0f} km/s, does not hold "
                         f"the line window +-{LINE_WINDOW} km/s")
    return v


def input_list(a, cfg, workdir):
    """The list of the first n_ms datasets of the sample."""
    data = os.path.join(a.work, "data", f"n{cfg['nchan']}")
    rows = [r for r in open(os.path.join(data, "input_list.txt")) if r.strip() and r[0] != "#"]
    if len(rows) < cfg["n_ms"]:
        raise SystemExit(f"{data} has {len(rows)} datasets, {cfg['n_ms']} needed: "
                         f"simobserve_sample.py --out {data} --n {cfg['n_ms']} "
                         f"--nchan {cfg['nchan']}")
    path = os.path.join(workdir, f"input_n{cfg['nchan']}_{cfg['n_ms']}.txt")
    with open(path, "w") as fh:
        fh.writelines(rows[:cfg["n_ms"]])
    truth = json.load(open(os.path.join(data, "truth.json")))
    return path, truth


def split(method, cfg):
    """How a run divides its cores: readers, writers and kernel threads; None
    if they do not fit (the method is then skipped, the rest goes on)."""
    cores, readers = cfg["cores"], cfg["readers"]
    if method == "vista1":
        return {"cores": cores}
    if method in ("hpc_v1", "hpc_v2_ms"):
        writers = 1                                      # a Measurement Set: one writer
    elif cfg["writers"]:
        writers = cfg["writers"]
    else:                                                # the rule of ViSta-HPC v2
        left = cores - readers
        writers = max(1, min(max(1, round(cores / 4)), 8, max(1, left - 1)))
    threads = cores - readers - writers
    if threads < 1:
        return None
    return {"cores": cores, "readers": readers, "writers": writers, "threads": threads}


def command(method, device, a, cfg, lst, truth, out):
    """The command of one run, its environment and how it divides its cores."""
    used = split(method, cfg)
    threads = used.get("threads", cfg["cores"])        # ViSta 1.0: all of them
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1",
               SLURM_CPUS_PER_TASK=str(cfg["cores"]))
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    z_ref = f"{truth['z_ref']:.6f}"
    rest = str(truth["rest_freq_ghz"])
    vrange = str(a.velocity_range)
    if method == "vista1":
        return [sys.executable, os.path.join(HERE, "run_vista_casa.py"), "--input", lst,
                "--vista-casa", a.vista1, "--uvmultifit", a.uvmultifit, "--rest-freq", rest,
                "--z-ref", z_ref, "--v-window", str(-LINE_WINDOW), str(LINE_WINDOW),
                "--velocity-range", vrange, "--out-dir", out, "--keep"], env, used
    if method == "hpc_v1":
        # readers, threads and GPU batch of this version are set in the environment;
        # --scratch "": the stack is written in place, as by the other versions
        env.update(PYTHONPATH=a.hpc_v1, VISTA_N_READERS=str(used["readers"]),
                   VISTA_OMP_THREADS=str(threads), VISTA_GPU_BATCH=str(cfg["gpu_batch"]))
        return ["bash", os.path.join(a.hpc_v1, "run_vista.sh"), "--input", lst,
                "--rest-freq", rest, "--velocity-range", vrange,
                "--v-window", f"{-LINE_WINDOW} {LINE_WINDOW}", "--z-ref", z_ref,
                "--contsub", "before", "--method", "template", "--model", "gauss",
                "--scratch", "", "--no-plot", "--out-dir", out], env, used
    env["PYTHONPATH"] = a.hpc_v2
    zarr = method != "hpc_v2_ms"
    cmd = [sys.executable, os.path.join(HERE, "run_hpc_v2.py"), "--input", lst,
           "--output-name", os.path.join(out, TAG), "--rest-freq", rest,
           "--line-window", str(LINE_WINDOW), "--velocity-range", vrange,
           "--z-ref", z_ref, "--device", device, "--readers", str(used["readers"]),
           "--threads", str(threads), "--gpu-batch", str(cfg["gpu_batch"]),
           "--output-format", "zarr" if zarr else "ms",
           "--mode", "onepass" if method == "hpc_v2_onepass" else "steps"]
    if zarr:
        cmd += ["--writers", str(used["writers"])]
    return cmd, env, used


def collect(method, out):
    """What a run measured, in the same form for the three versions."""
    if method == "vista1":
        r = json.load(open(os.path.join(out, "results.json")))
        return {k: r[k] for k in ("line_flux", "line_flux_error", "v0_kms", "fwhm_kms",
                                  "size_arcsec", "cont_flux", "cont_flux_error")}
    path = os.path.join(out, f"{TAG}_line_results.json")              # ViSta-HPC v1
    if not os.path.exists(path):
        path = os.path.join(out, f"{TAG}_results.json")               # ViSta-HPC v2
    line = json.load(open(path))
    cont = json.load(open(os.path.join(out, f"{TAG}_continuum_results.json")))
    res = {"cont_flux": cont["flux_total"], "cont_flux_error": cont["flux_total_error"]}
    if "line" in line:                                                # v2
        ln = line["line"]
        res.update(line_flux=ln["flux"], line_flux_error=ln["flux_error"], v0_kms=ln["v0_kms"],
                   fwhm_kms=ln["fwhm_kms"], size_arcsec=ln["theta_fwhm_arcsec"])
    else:                                                             # v1
        shape = line.get("line_shape") or {}
        res.update(line_flux=line["flux_total"], line_flux_error=line["flux_total_error"],
                   v0_kms=shape.get("centroid_kms"), fwhm_kms=shape.get("fwhm_kms"),
                   size_arcsec=line["theta_fwhm_arcsec"])
    return res


def git_rev(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def job_meta(a):
    """Where and with what the job ran: written with every run."""
    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                  "--format=csv,noheader", "-i", "0"], capture_output=True,
                                 text=True, timeout=10).stdout.strip() or None
        except (OSError, subprocess.TimeoutExpired):
            pass
    cpu = None
    try:
        cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")
                    if l.startswith("model name")), None)
    except OSError:
        pass
    return {"job": job_id(), "host": socket.gethostname(), "job_cores": len(job_cpus()),
            "cpu": cpu, "gpu": gpu, "velocity_range_kms": a.velocity_range,
            "commits": {"hpc_v2": git_rev(a.hpc_v2),
                        "hpc_v1": git_rev(a.hpc_v1) if a.hpc_v1 else None,
                        "vista1": git_rev(a.vista1) if a.vista1 else None,
                        "uvmultifit": git_rev(a.uvmultifit) if a.uvmultifit else None}}


def seconds_in(text):
    """The seconds of a duration written by ViSta-HPC v2 ("4.5 s", "1.2 min")."""
    m = re.search(r"([\d.]+)\s*(s|min)", text)
    return None if not m else float(m.group(1)) * (60.0 if m.group(2) == "min" else 1.0)


def inside(path):
    """Where the time of the stacking goes, from the log of ViSta-HPC v2
    (``verbose="debug"``): reading (Python and casacore), computing (the
    kernel, on the CPU or on the GPU) and writing, which overlap; and, of the
    computing, the continuum subtraction and the statistics.  Of a log with
    several runs, the first stacking: the one of the step that is timed."""
    out = {}

    def fill(line, fields):
        got = {}
        for key, field in fields:
            if field not in line:
                return
            value = seconds_in(line.split(field, 1)[1])
            if value is None:
                return
            got[key] = value
        out.update(got)

    for line in open(path, errors="replace"):
        if "Timing: setup" in line and "setup" not in out:
            fill(line, (("setup", "setup"), ("stacking", "stacking"),
                        ("closing", "subtables and closing")))
        elif "(overlapping)" in line and "reading" not in out:
            fill(line, (("reading", "reading"), ("computing", "computing"),
                        ("writing", "writing")))
            limit = re.search(r"limited by (\w+)", line)
            if limit and "reading" in out:
                out["limited_by"] = limit.group(1)
        elif "of the computing:" in line and "inline_contsub" not in out:
            fill(line, (("inline_contsub", "continuum subtraction"),
                        ("inline_stats", "statistics")))
    return out or None


def run_one(method, device, a, cfg, lst, truth, workdir):
    import monitor
    out = os.path.join(workdir, f"{method}_{device}")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    if a.warm_cache:                       # the inputs in the page cache for every run
        files = " ".join(shlex.quote(r.split()[0]) for r in open(lst))
        subprocess.run(f"find {files} -type f -exec cat {{}} + > /dev/null", shell=True)
    cmd, env, used = command(method, device, a, cfg, lst, truth, out)
    m = monitor.run(cmd, watch=out, env=env, log=os.path.join(out, "run.log"),
                    gpu=(device == "gpu"), cpus=job_cpus()[:cfg["cores"]])
    try:
        if m["returncode"] != 0:
            raise ValueError(f"exit code {m['returncode']}")
        results = collect(method, out)
    except (OSError, KeyError, ValueError) as e:
        keep = out + "_failed"
        shutil.rmtree(keep, ignore_errors=True)
        os.rename(out, keep)
        raise RuntimeError(f"{method} ({device}) failed ({e}): see {keep}/run.log")
    seconds = {"stack": 0.0, "contsub": 0.0, "extraction": 0.0, "other": 0.0}
    for s in m["stages"]:
        seconds[STEPS.get(s["stage"], "other")] += s["seconds"]
    within = inside(os.path.join(out, "run.log")) if method.startswith("hpc_v2") else None
    if not a.keep_runs:
        shutil.rmtree(out, ignore_errors=True)
    return {"seconds": seconds, "wall_seconds": m["wall_seconds"],
            "cpu_seconds": m["cpu_seconds"], "peak_memory_gb": m["peak_memory_gb"],
            "peak_disk_gb": m["peak_disk_gb"], "final_disk_gb": m["final_disk_gb"],
            "peak_gpu_memory_gb": m.get("peak_gpu_memory_gb"), "stages": m["stages"],
            "inside": within, "used": used, "results": results}


def configurations(scan, standard):
    """The configurations of a scan, without the standard one (except in 'standard')."""
    out = []
    for change in SCANS[scan]:
        cfg = dict(standard, **change)
        if scan != "standard" and cfg == standard:
            continue
        out.append(cfg)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True, help="working directory: data, runs, results")
    ap.add_argument("--scan", default="standard", help="one or more of: " + ", ".join(SCANS))
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--devices", default="cpu",
                    help="cpu, gpu or cpu,gpu (ViSta 1.0 runs only with cpu)")
    ap.add_argument("--repeats", type=int, default=3, help="of every configuration")
    ap.add_argument("--repeats-vista1", type=int, default=1, help="of ViSta 1.0")
    ap.add_argument("--velocity-range", type=float, default=None,
                    help="half-width of the stacked band, km/s (default: the widest range "
                         "every dataset covers)")
    ap.add_argument("--vista1", default=os.environ.get("VISTA1", ""),
                    help="checkout of github.com/martitors/ViSta")
    ap.add_argument("--uvmultifit", default=os.environ.get("UVMULTIFIT", ""),
                    help="directory of UVMultiFit, built")
    ap.add_argument("--hpc-v1", default=os.environ.get("VISTA_HPC_V1", ""),
                    help="built checkout of the branch extract")
    ap.add_argument("--hpc-v2", default=os.environ.get("VISTA_HPC_V2", REPO),
                    help="built checkout of the branch extract_opt (default: this one)")
    ap.add_argument("--warm-cache", action="store_true",
                    help="read the inputs before every run (same page cache for everyone)")
    ap.add_argument("--keep-runs", action="store_true", help="keep the output of every run")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands that would run, and run nothing")
    for k, v in STANDARD.items():
        ap.add_argument("--" + k.replace("_", "-"), type=int, default=v,
                        help=f"of the standard configuration (default {v})")
    a = ap.parse_args()
    methods = [m for m in a.methods.split(",") if m]
    bad = [m for m in methods if m not in METHODS]
    if bad:
        ap.error(f"--methods: {bad} not among {METHODS}")
    scans = [s for s in a.scan.split(",") if s]
    bad = [s for s in scans if s not in SCANS]
    if bad:
        ap.error(f"--scan: {bad} not among {list(SCANS)}")
    devices = [d for d in a.devices.split(",") if d]
    standard = {k: getattr(a, k) for k in STANDARD}
    most = max([standard["cores"]] + [c.get("cores", 0) for s in scans for c in SCANS[s]])
    if most > len(job_cpus()):
        ap.error(f"{most} cores asked, {len(job_cpus())} available: lower --cores")
    if a.velocity_range is None:
        a.velocity_range = common_velocity_range(a.work)
    print(f"band: +-{a.velocity_range:.0f} km/s around the line, for every version", flush=True)
    job = job_id()
    workdir = os.path.join(a.work, "runs", job)
    os.makedirs(os.path.join(a.work, "results"), exist_ok=True)
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(a.work, "results", f"runs_{'-'.join(scans)}_{job}.json")
    mine = []
    meta = None if a.dry_run else job_meta(a)

    for scan in scans:
        for cfg in configurations(scan, standard):
            lst, truth = input_list(a, cfg, workdir)
            for method in methods:
                if method == "vista1":
                    if scan not in VISTA1_SCANS or "cpu" not in devices:
                        continue
                    run_devices, repeats = ["cpu"], a.repeats_vista1
                else:
                    run_devices, repeats = devices, a.repeats
                if scan in ZARR_ONLY and method not in ("hpc_v2_zarr", "hpc_v2_onepass"):
                    continue
                if split(method, cfg) is None:
                    print(f"[{scan}] {method}: {cfg['cores']} cores cannot hold "
                          f"{cfg['readers']} readers, the writers and a kernel thread: "
                          f"skipped", flush=True)
                    continue
                for device in run_devices:
                    if scan in GPU_ONLY and device != "gpu":
                        continue
                    key = dict(cfg, method=method, device=device)
                    done = sum(all(value(r, k) == v for k, v in key.items())
                               for r in load_runs(a.work))      # also those of other jobs
                    label = f"[{scan}] {method} {device} " + " ".join(
                        f"{k}={v}" for k, v in cfg.items())
                    if a.dry_run:
                        cmd, env, used = command(method, device, a, cfg, lst, truth,
                                                 os.path.join(workdir, f"{method}_{device}"))
                        extra = {k: env[k] for k in env if k.startswith("VISTA_")
                                 or k in ("OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}
                        print(f"{label}: {max(0, repeats - done)} to run, {used}\n  {extra}\n  "
                              + shlex.join(cmd), flush=True)
                        continue
                    for rep in range(done, repeats):
                        try:
                            r = run_one(method, device, a, cfg, lst, truth, workdir)
                        except RuntimeError as e:
                            print(f"{label}: {e}", flush=True)
                            break
                        mine.append(dict(key, scan=scan, repeat=rep, meta=meta, **r))
                        with open(path, "w") as fh:
                            json.dump(mine, fh, indent=1)
                        s = r["seconds"]
                        print(f"{label} #{rep}: stack {s['stack']:.1f} s, contsub "
                              f"{s['contsub']:.1f} s, extraction {s['extraction']:.1f} s, "
                              f"memory {r['peak_memory_gb']:.2f} GB, "
                              f"F = {r['results']['line_flux']:.4f}", flush=True)


if __name__ == "__main__":
    main()
