"""The benchmark: ViSta 1.0 (CASA), ViSta-HPC ``extract`` and ViSta-HPC
``extract_opt`` on the same heterogeneous sample at z = 2-3.

Three measurements:

* ``perf``: every pipeline on the noisy sample, ``--repeats`` times, from the
  input Measurement Sets to the measured line and continuum, under
  ``monitor.py`` (wall time per stage, CPU time, peak memory, disk);
* ``noiseless``: every pipeline on the noiseless twin of the sample (same
  weights, no noise): what it measures without noise, against the truth --
  its systematic errors;
* ``montecarlo``: every pipeline on ``--n`` new realisations of the noise:
  the scatter of the results, their bias, and whether the quoted errors are
  right (the pulls).

``report.py`` turns the results into tables and figures.  Usage::

    python bench.py sample     --work WORK
    python bench.py perf       --work WORK --repeats 3
    python bench.py noiseless  --work WORK
    python bench.py montecarlo --work WORK --n 20
    python report.py --work WORK

The three codes are separate checkouts (``--vista-casa``, ``--extract``,
``--extract-opt``); ViSta 1.0 runs with UVMultiFit (``--uvmultifit``).
"""
from __future__ import annotations

import argparse
import glob
import json
import shlex
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sample as sample_mod  # noqa: E402

PIPELINES = ("vista_casa", "extract", "extract_opt")
TAG = "line345796"


# ---------------------------------------------------------------------------
# the settings shared by the three pipelines
# ---------------------------------------------------------------------------
def settings(a):
    return {
        "rest_freq_ghz": a.rest_freq,
        "v_window_kms": (-550.0, 550.0),       # the line: left out of the continuum fit
        "velocity_range_kms": 900.0,           # HPC output grid, +-900 km/s
        "contsub": "before stacking measurement, order 0",
        "weighting": "natural, redshift rescaling to the median z, primary beam corrected",
        "extract_method": "template, Gaussian source (defaults of run_vista.sh)",
        "extract_opt_method": "joint fit, centre and width free in the window, Gaussian "
                              "flux (defaults of run_vista.sh --extract, --flux-measure gauss)",
        "vista_casa_extraction": "UVMultiFit circular Gaussian: size on |v|<300 km/s, then channel by channel",
        "casa_timebin": a.timebin,
    }


def command(name, a, input_list, out, noiseless=False):
    """The command line of one pipeline, and its environment."""
    env = dict(os.environ, OMP_NUM_THREADS=str(a.threads))
    vw = "-550 550"
    if name == "vista_casa":
        z_ref = json.load(open(os.path.join(os.path.dirname(input_list), "truth.json")))["z_ref"]
        cmd = [sys.executable, os.path.join(HERE, "run_vista_casa.py"), "--input", input_list,
               "--vista-casa", a.vista_casa, "--uvmultifit", a.uvmultifit,
               "--rest-freq", str(a.rest_freq), "--z-ref", f"{z_ref:.6f}",
               "--timebin", a.timebin, "--out-dir", out]
        if a.keep:
            cmd.append("--keep")
        if noiseless:
            cmd.append("--noiseless")
        return cmd, env
    repo = a.extract if name == "extract" else a.extract_opt
    env["PYTHONPATH"] = repo
    if name == "extract":
        cmd = ["bash", os.path.join(repo, "run_vista.sh"), "--input", input_list,
               "--rest-freq", str(a.rest_freq), "--velocity-range", "900", "--v-window", vw,
               "--contsub", "before", "--no-plot", "--out-dir", out,
               "--method", "template", "--model", "gauss"]
        return cmd, env
    # extract_opt (ViSta 3): the stacking (continuum subtraction and compression in
    # the same pass), then the measurement; the readers, writers and threads come
    # from the cores; the Gaussian flux is the one compared
    base = ["bash", os.path.join(repo, "run_vista.sh"), "--input", input_list,
            "--line-freq", str(a.rest_freq), "--line-window", "550", "--velocity-range", "900",
            "--output-name", os.path.join(out, TAG), "--extract", "--no-figures",
            "--flux-measure", "gauss"]
    script = "echo @@stage stacking && " + shlex.join(base + ["--only", "stack"])
    if not noiseless:                    # noiseless: measured below with the noise of the weights
        script += " && echo @@stage fit && " + shlex.join(base + ["--only", "measure"])
    return ["bash", "-c", script], env


NOISELESS_FIT = """
import sys
from vista import ViSta
v = ViSta(sys.argv[1], output_name=sys.argv[3])
v.stack(line_freq=float(sys.argv[2]), line_window=550, velocity_range=900)
v.extract(flux_measure="gauss", noise="weights", channel_correlation=False, n_null=0,
          n_null_phase=0, n_bootstrap=0, figures_draw=False)
v.run(only="measure")
"""


def noiseless_fit_opt(a, input_list, out):
    """extract_opt on noiseless data: the measured noise is zero, so the
    measurement uses the noise of the weights (the values are what count)."""
    env = dict(os.environ, PYTHONPATH=a.extract_opt)
    subprocess.run([sys.executable, "-c", NOISELESS_FIT, input_list, str(a.rest_freq),
                    os.path.join(out, TAG)], env=env, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# reading the results of each pipeline into one format
# ---------------------------------------------------------------------------
def collect(name, out):
    """line_flux, v0, fwhm, size, cont_flux and their errors, from the outputs."""
    if name == "vista_casa":
        return json.load(open(os.path.join(out, "results.json")))
    path = os.path.join(out, f"{TAG}_line_results.json")         # extract
    if not os.path.exists(path):
        path = os.path.join(out, f"{TAG}_results.json")          # extract_opt (ViSta 3)
    line = json.load(open(path))
    cont = json.load(open(os.path.join(out, f"{TAG}_continuum_results.json")))
    res = {"pipeline": name, "cont_flux": cont["flux_total"],
           "cont_flux_error": cont["flux_total_error"],
           "cont_size_arcsec": cont.get("theta_fwhm_arcsec")}
    if "line" in line:                                    # measure_line
        ln = line["line"]
        res.update(line_flux=ln["flux"], line_flux_error=ln["flux_error"],
                   v0_kms=ln["v0_kms"], v0_error_kms=ln.get("v0_error_kms"),
                   fwhm_kms=ln["fwhm_kms"], fwhm_error_kms=ln.get("fwhm_error_kms"),
                   size_arcsec=ln["theta_fwhm_arcsec"],
                   size_error_arcsec=ln.get("theta_error_arcsec"),
                   significance_sigma=line.get("detection", {}).get("significance_sigma"),
                   kappa=line.get("noise", {}).get("kappa_scatter_over_weights"))
    else:                                                 # extract_flux
        shape = line.get("line_shape", {}) or {}
        res.update(line_flux=line["flux_total"], line_flux_error=line["flux_total_error"],
                   size_arcsec=line["theta_fwhm_arcsec"],
                   size_error_arcsec=line["theta_error_arcsec"],
                   v0_kms=shape.get("centroid_kms"), v0_error_kms=shape.get("centroid_error_kms"),
                   fwhm_kms=shape.get("fwhm_kms"), fwhm_error_kms=shape.get("fwhm_error_kms"))
    return res


def run_one(name, a, input_list, out, metrics, noiseless=False):
    """One pipeline, measured; returns (metrics, results)."""
    import monitor
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    subprocess.run(f"cat $(find {os.path.dirname(input_list)} -type f) > /dev/null",
                   shell=True)                            # same page cache for everyone
    cmd, env = command(name, a, input_list, out, noiseless)
    m = monitor.run(cmd, watch=out, env=env, log=os.path.join(out, "run.log"))
    if m["returncode"] != 0:
        # run_vista.sh of the extract branch can end with a non-zero status after
        # writing everything (its last check); what counts is that the results exist
        try:
            collect(name, out)
        except (OSError, KeyError, ValueError):
            raise RuntimeError(f"{name} failed, see {out}/run.log")
    if noiseless and name == "extract_opt":
        t0 = time.time()
        noiseless_fit_opt(a, input_list, out)
        m["stages"].append({"stage": "extraction (noise of the weights)",
                            "seconds": time.time() - t0, "peak_memory_gb": None,
                            "peak_disk_gb": None})
    res = collect(name, out)
    if "_perf_" not in os.path.basename(metrics):
        m.pop("memory_trace", None)
    json.dump(m, open(metrics, "w"), indent=1)
    return m, res


def append(path, record):
    data = json.load(open(path)) if os.path.exists(path) else []
    data.append(record)
    json.dump(data, open(path, "w"), indent=1)


# ---------------------------------------------------------------------------
# the commands
# ---------------------------------------------------------------------------
def cmd_sample(a):
    cfg = sample_mod.SampleConfig(n_galaxies=a.n_galaxies, seed=a.seed)
    sample_mod.write(os.path.join(a.work, "data", "noisy"), cfg)
    sample_mod.write(os.path.join(a.work, "data", "noiseless"), cfg, noiseless=True)
    json.dump({"sample": vars(cfg) if hasattr(cfg, "__dict__") else {}, "settings": settings(a)},
              open(os.path.join(a.work, "settings.json"), "w"), indent=1, default=str)


def cmd_perf(a):
    lst = os.path.join(a.work, "data", "noisy", "input_list.txt")
    for rep in range(a.repeats):
        for name in a.pipelines:
            out = os.path.join(a.work, "runs", f"perf_{name}")
            m, res = run_one(name, a, lst, out, os.path.join(a.work, "results", f"{name}_perf_{rep}.json"))
            append(os.path.join(a.work, "results", "perf.json"),
                   {"pipeline": name, "repeat": rep, "metrics": m, "results": res})
            print(f"[perf {rep}] {name}: {m['wall_seconds']:.0f} s, {m['peak_memory_gb']:.2f} GB, "
                  f"disk {m['peak_disk_gb']:.2f} GB, F = {res['line_flux']:.4f}", flush=True)
            if not a.keep_runs:
                shutil.rmtree(out, ignore_errors=True)


def cmd_noiseless(a):
    lst = os.path.join(a.work, "data", "noiseless", "input_list.txt")
    for name in a.pipelines:
        out = os.path.join(a.work, "runs", f"noiseless_{name}")
        m, res = run_one(name, a, lst, out, os.path.join(a.work, "results", f"{name}_noiseless_metrics.json"),
                         noiseless=True)
        json.dump(res, open(os.path.join(a.work, "results", f"{name}_noiseless.json"), "w"), indent=1)
        print(f"[noiseless] {name}: F = {res['line_flux']:.4f}", flush=True)
        if not a.keep_runs:
            shutil.rmtree(out, ignore_errors=True)


def cmd_montecarlo(a):
    cfg = sample_mod.SampleConfig(n_galaxies=a.n_galaxies, seed=a.seed)
    data = os.path.join(a.work, "data", "mc")
    path = os.path.join(a.work, "results", "montecarlo.json")
    done = {(r["pipeline"], r["realisation"]) for r in (json.load(open(path)) if os.path.exists(path) else [])}
    for k in range(a.first, a.first + a.n):
        todo = [p for p in a.pipelines if (p, k) not in done]
        if not todo:
            continue
        sample_mod.write(data, cfg, realisation=k, verbose=False)
        lst = os.path.join(data, "input_list.txt")
        for name in todo:
            out = os.path.join(a.work, "runs", f"mc_{name}")
            try:
                m, res = run_one(name, a, lst, out, os.path.join(a.work, "runs", f"mc_{name}.json"))
            except RuntimeError as e:
                print(f"[mc {k}] {name}: {e}", flush=True)
                continue
            append(path, {"pipeline": name, "realisation": k, "results": res,
                          "wall_seconds": m["wall_seconds"]})
            print(f"[mc {k}] {name}: F = {res['line_flux']:.4f} +- {res['line_flux_error']:.4f}",
                  flush=True)
            shutil.rmtree(out, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("what", choices=["sample", "perf", "noiseless", "montecarlo"])
    ap.add_argument("--work", required=True, help="working directory (data, runs, results)")
    ap.add_argument("--vista-casa", default=os.environ.get("VISTA_CASA", ""),
                    help="checkout of github.com/martitors/ViSta")
    ap.add_argument("--uvmultifit", default=os.environ.get("UVMULTIFIT", ""),
                    help="directory with uvmultifit.py and its compiled module")
    ap.add_argument("--extract", default=os.environ.get("VISTA_EXTRACT", ""),
                    help="built checkout of ViSta-HPC, branch extract")
    ap.add_argument("--extract-opt", default=os.environ.get("VISTA_EXTRACT_OPT", REPO),
                    help="built checkout of ViSta-HPC, branch extract_opt (default: this one)")
    ap.add_argument("--pipelines", default=",".join(PIPELINES))
    ap.add_argument("--rest-freq", type=float, default=345.79599)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--timebin", default="120s", help="time average before UVMultiFit")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--n", type=int, default=20, help="Monte Carlo realisations")
    ap.add_argument("--first", type=int, default=1, help="first Monte Carlo realisation")
    ap.add_argument("--n-galaxies", type=int, default=sample_mod.SampleConfig.n_galaxies)
    ap.add_argument("--seed", type=int, default=sample_mod.SampleConfig.seed)
    ap.add_argument("--keep", action="store_true",
                    help="ViSta 1.0 keeps its intermediate Measurement Sets (its default)")
    ap.add_argument("--keep-runs", action="store_true", help="keep the outputs of every run")
    a = ap.parse_args()
    a.pipelines = [p.strip() for p in a.pipelines.split(",") if p.strip()]
    os.makedirs(os.path.join(a.work, "results"), exist_ok=True)
    os.makedirs(os.path.join(a.work, "runs"), exist_ok=True)
    {"sample": cmd_sample, "perf": cmd_perf, "noiseless": cmd_noiseless,
     "montecarlo": cmd_montecarlo}[a.what](a)


if __name__ == "__main__":
    main()
