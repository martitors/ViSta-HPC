"""The tables of the benchmark, from WORK/results/runs*.json (``bench.py``).

Every number is the median of the repeats of one configuration.  Written to
WORK/results/tables.md (and summary.json, the same numbers):

* the standard configuration: time of every step, CPU time, memory, disk and
  GPU memory, with the resources every version was given;
* the stages of ViSta 1.0 on the standard configuration;
* one table per scan: the time of every step against what the scan changes,
  and the memory (RAM, disk, GPU);
* what every version measured on the standard configuration, with the
  truth, and how far each is from ViSta 1.0.

Usage::

    python report.py --work WORK
"""
import argparse
import json
import os
import statistics

from bench import METHODS, SCANS, STANDARD, load_runs, value

PARAMS = tuple(STANDARD)
STEPS = ("stack", "contsub", "extraction", "other")
LABEL = {"n_ms": "N_MS", "nchan": "N_chan", "readers": "readers", "cores": "cores",
         "gpu_batch": "GPU batch", "writers": "writers"}


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarise(runs):
    """One record per configuration: the medians of its repeats."""
    groups = {}
    for r in runs:
        key = (r["method"], r["device"]) + tuple(value(r, p) for p in PARAMS)
        groups.setdefault(key, []).append(r)
    out = []
    for key, rs in groups.items():
        rec = dict(zip(("method", "device") + PARAMS, key), repeats=len(rs))
        for step in STEPS:
            rec[step] = median([r["seconds"][step] for r in rs])
        for k in ("wall_seconds", "cpu_seconds", "peak_memory_gb", "peak_disk_gb",
                  "final_disk_gb", "peak_gpu_memory_gb"):
            rec[k] = median([r.get(k) for r in rs])
        rec["results"] = {k: median([r["results"].get(k) for r in rs]) for k in rs[0]["results"]}
        rec["used"] = rs[-1].get("used")
        within = [r["inside"] for r in rs if r.get("inside")]
        rec["inside"] = ({k: median([w.get(k) for w in within if isinstance(w.get(k), (int, float))])
                          for k in ("setup", "stacking", "closing", "reading", "computing",
                                    "writing", "inline_contsub", "inline_stats")}
                         if within else None)
        if within:
            rec["inside"]["limited_by"] = within[-1].get("limited_by")
        stages = {}
        for r in rs:
            for s in r["stages"]:
                stages.setdefault(s["stage"], []).append(s["seconds"])
        rec["stages"] = {k: median(v) for k, v in stages.items()}
        rec["runs"] = rs
        out.append(rec)
    return out


def fmt(x, digits=1):
    return "--" if x is None else f"{x:.{digits}f}"


def order(rec):
    return (METHODS.index(rec["method"]), rec["device"])


def truth_of(work, sample, cache={}):
    if sample not in cache:
        path = os.path.join(work, "data", sample, "truth.json")
        cache[sample] = json.load(open(path)) if os.path.exists(path) else None
    return cache[sample]


def used_of(rec):
    u = rec.get("used") or {}
    return "/".join(str(u.get(k, "-")) for k in ("readers", "writers", "threads")) if u else "--"


def time_table(recs, first=None):
    head = ([first] if first else []) + ["version", "device", "stack", "contsub",
                                         "extraction", "other", "whole run", "CPU time"]
    txt = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    for lead, r in recs:
        cells = ([str(lead)] if first else []) + [
            r["method"], r["device"], fmt(r["stack"]), fmt(r["contsub"]), fmt(r["extraction"]),
            fmt(r["other"]), fmt(r["wall_seconds"]), fmt(r["cpu_seconds"])]
        txt.append("| " + " | ".join(cells) + " |")
    return txt


def memory_table(recs, first=None):
    head = ([first] if first else []) + ["version", "device", "RAM peak [GB]", "disk peak [GB]",
                                         "disk at the end [GB]", "GPU memory [GB]",
                                         "readers/writers/threads", "repeats"]
    txt = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    for lead, r in recs:
        cells = ([str(lead)] if first else []) + [
            r["method"], r["device"], fmt(r["peak_memory_gb"], 2), fmt(r["peak_disk_gb"], 2),
            fmt(r["final_disk_gb"], 2), fmt(r["peak_gpu_memory_gb"], 2), used_of(r),
            str(r["repeats"])]
        txt.append("| " + " | ".join(cells) + " |")
    return txt


def accuracy_table(work, recs, first=None):
    head = ([first] if first else []) + ["version", "device", "line flux [Jy km/s]",
                                         "v0 [km/s]", "FWHM [km/s]", "size [arcsec]",
                                         "continuum [mJy]"]
    txt = ["| " + " | ".join(head) + " |", "|---" * len(head) + "|"]
    shown = set()
    for lead, r in recs:
        sample = f"n{r['nchan']}"
        t = truth_of(work, sample)
        if t is not None and (lead, sample) not in shown:
            shown.add((lead, sample))
            txt.append("| " + " | ".join(([str(lead)] if first else []) + [
                "truth", "", f"{t['line_flux']:.4f}", f"{t['v0_kms']:.1f}",
                f"{t['fwhm_kms']:.1f}", f"{t['size_arcsec']:.3f}",
                f"{t['cont_flux'] * 1e3:.4f}"]) + " |")
        m = r["results"]
        cont = None if m.get("cont_flux") is None else m["cont_flux"] * 1e3
        cerr = None if m.get("cont_flux_error") is None else m["cont_flux_error"] * 1e3
        txt.append("| " + " | ".join(([str(lead)] if first else []) + [
            r["method"], r["device"],
            f"{fmt(m.get('line_flux'), 4)} +- {fmt(m.get('line_flux_error'), 4)}",
            fmt(m.get("v0_kms")), fmt(m.get("fwhm_kms")), fmt(m.get("size_arcsec"), 3),
            f"{fmt(cont, 4)} +- {fmt(cerr, 4)}"]) + " |")
    return txt


def agreement_table(recs):
    """How far every version is from ViSta 1.0 (or, without it, from the first
    version measured): the difference, in % and in units of the error of the
    reference."""
    if not recs:
        return []
    ref = next((r for r in recs if r["method"] == "vista1"), recs[0])
    txt = [f"Difference from {ref['method']} ({ref['device']}): in % of its value, and in "
           f"units of its error.", "",
           "| version | device | line flux | line flux / error | size | continuum "
           "| continuum / error |", "|---" * 7 + "|"]

    def d(r, k, err=None):
        a, b = r["results"].get(k), ref["results"].get(k)
        if a is None or b is None or b == 0:
            return "--"
        if err:
            e = ref["results"].get(err)
            return "--" if not e else f"{(a - b) / e:+.2f}"
        return f"{100 * (a - b) / b:+.2f}%"
    for r in recs:
        if r is ref:
            continue
        txt.append(f"| {r['method']} | {r['device']} | {d(r, 'line_flux')} "
                   f"| {d(r, 'line_flux', 'line_flux_error')} | {d(r, 'size_arcsec')} "
                   f"| {d(r, 'cont_flux')} | {d(r, 'cont_flux', 'cont_flux_error')} |")
    return txt


def inside_table(rows):
    """Where the time of the stacking goes, for ViSta-HPC v2: reading (Python
    and casacore), the kernel, writing."""
    rows = [(lead, r) for lead, r in rows if r.get("inside")]
    if not rows:
        return []
    lead = [] if rows[0][0] is None else [LABEL.get(rows[0][0], "")]
    out = ["| " + " | ".join(lead + ["version", "device", "setup", "stacking", "closing",
                                     "reading", "kernel", "writing", "limited by",
                                     "continuum in the kernel", "statistics in the kernel"])
           + " |", "|---" * (len(lead) + 11) + "|"]
    for value_, r in rows:
        i = r["inside"]
        out.append("| " + " | ".join(
            ([] if value_ is None else [str(value_)])
            + [r["method"], r["device"]]
            + [fmt(i.get(k)) for k in ("setup", "stacking", "closing",
                                       "reading", "computing", "writing")]
            + [str(i.get("limited_by") or "--")]
            + [fmt(i.get(k)) for k in ("inline_contsub", "inline_stats")]) + " |")
    out += ["", "Reading, computing and writing overlap, so they do not add up to the "
                "stacking: each is the time one reader, the kernel and one writer were busy."]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True)
    for k, v in STANDARD.items():
        ap.add_argument("--" + k.replace("_", "-"), type=int, default=v,
                        help=f"of the standard configuration (default {v})")
    a = ap.parse_args()
    res = os.path.join(a.work, "results")
    summary = summarise(load_runs(a.work))
    standard = {k: getattr(a, k) for k in STANDARD}

    def at(**change):
        cfg = dict(standard, **change)
        return sorted((r for r in summary if all(r[p] == cfg[p] for p in PARAMS)), key=order)

    txt = ["# Benchmark of ViSta 1.0, ViSta-HPC v1 and ViSta-HPC v2", "",
           "Medians of the repeats; times in seconds.", "",
           "## Standard configuration", "",
           ", ".join(f"{k} = {v}" for k, v in standard.items()), ""]
    txt += time_table([(None, r) for r in at()])
    txt += [""] + memory_table([(None, r) for r in at()])
    txt += ["", "For ViSta 1.0 the extraction includes the time average and the single "
                "spectral window that UVMultiFit needs; `other`: start-up."]
    within = inside_table([(None, r) for r in at()])
    if within:
        txt += ["", "### Inside the stacking of ViSta-HPC v2", ""] + within

    v1 = [r for r in at() if r["method"] == "vista1"]
    if v1:
        st = v1[0]["stages"]
        names = [s for s in st if s != "start"]
        txt += ["", "## ViSta 1.0, stage by stage (standard configuration)", "",
                "| " + " | ".join(names) + " |", "|---" * len(names) + "|",
                "| " + " | ".join(fmt(st[s]) for s in names) + " |"]

    txt += ["", "## What is measured (standard configuration)", ""]
    txt += accuracy_table(a.work, [(None, r) for r in at()])
    txt += [""] + agreement_table(at())

    for scan, changes in SCANS.items():
        if scan == "standard":
            continue
        rows = [(", ".join(str(v) for v in c.values()), r) for c in changes for r in at(**c)]
        if len({lead for lead, _ in rows}) < 2:           # the scan was not run
            continue
        name = ", ".join(LABEL[k] for k in changes[0])
        txt += ["", f"## Scan: {scan}", "", "### Time [s]", ""] + time_table(rows, name)
        txt += ["", "### Memory and disk", ""] + memory_table(rows, name)

    with open(os.path.join(res, "tables.md"), "w") as fh:
        fh.write("\n".join(txt) + "\n")
    with open(os.path.join(res, "summary.json"), "w") as fh:
        json.dump([{k: v for k, v in r.items() if k != "runs"} for r in summary], fh, indent=1)
    print("\n".join(txt))


if __name__ == "__main__":
    main()
