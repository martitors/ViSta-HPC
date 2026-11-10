"""Tables and figures of the benchmark (after bench.py).

    python report.py --work WORK

writes ``WORK/report/``: ``tables.md`` (time, memory and disk per version
and per step; accuracy without noise; Monte Carlo), ``summary.json`` and the
figures ``time.png``, ``memory.png``, ``montecarlo.png``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sample as sample_mod  # noqa: E402

NAMES = {"vista_casa": "ViSta 1.0 (CASA)", "extract": "ViSta-HPC extract",
         "extract_opt": "ViSta-HPC extract_opt"}
ORDER = ["vista_casa", "extract", "extract_opt"]
QUANT = [("line_flux", "line_flux_error", "line flux", "line_flux", 1.0, "Jy km/s"),
         ("fwhm_kms", "fwhm_error_kms", "FWHM", "fwhm_kms", 1.0, "km/s"),
         ("v0_kms", "v0_error_kms", "centre", "v0_kms", 1.0, "km/s"),
         ("size_arcsec", "size_error_arcsec", "size", "size_arcsec", 1.0, "arcsec"),
         ("cont_flux", "cont_flux_error", "continuum", "cont_flux", 1e3, "mJy")]


def load(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default


def truth_for(work):
    t = load(os.path.join(work, "data", "noisy", "truth.json"))
    return t, t["targets"]


def perf_tables(perf):
    lines = []
    by = {}
    for r in perf:
        by.setdefault(r["pipeline"], []).append(r["metrics"])
    summary = {}
    lines.append("| Version | wall time [s] | CPU time [s] | cores used | peak memory [GB] | peak disk [GB] | disk at the end [GB] | runs |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for p in ORDER:
        if p not in by:
            continue
        ms = by[p]
        wall = np.median([m["wall_seconds"] for m in ms])
        cpu = np.median([m["cpu_seconds"] for m in ms])
        mem = np.median([m["peak_memory_gb"] for m in ms])
        disk = np.median([m["peak_disk_gb"] for m in ms])
        fin = np.median([m["final_disk_gb"] for m in ms])
        summary[p] = {"wall_seconds": wall, "cpu_seconds": cpu, "peak_memory_gb": mem,
                      "peak_disk_gb": disk, "final_disk_gb": fin, "runs": len(ms),
                      "wall_all": [m["wall_seconds"] for m in ms]}
        lines.append(f"| {NAMES[p]} | {wall:.0f} | {cpu:.0f} | {cpu / wall:.1f} | {mem:.2f} | "
                     f"{disk:.2f} | {fin:.2f} | {len(ms)} |")
    lines.append("")
    for p in ORDER:
        if p not in by:
            continue
        lines.append(f"**{NAMES[p]}**, per step (median of {len(by[p])} runs)\n")
        lines.append("| step | time [s] | peak memory [GB] | peak disk [GB] |")
        lines.append("|---|---|---|---|")
        steps = [s["stage"] for s in by[p][0]["stages"]]
        summary[p]["steps"] = {}
        for i, name in enumerate(steps):
            vals = [m["stages"][i] for m in by[p] if i < len(m["stages"])]
            sec = np.median([v["seconds"] for v in vals])
            mem = np.median([v["peak_memory_gb"] or 0 for v in vals])
            dk = [v["peak_disk_gb"] for v in vals if v["peak_disk_gb"] is not None]
            summary[p]["steps"][name] = {"seconds": sec, "peak_memory_gb": mem,
                                         "peak_disk_gb": float(np.median(dk)) if dk else None}
            lines.append(f"| {name} | {sec:.1f} | {mem:.2f} | {np.median(dk) if dk else 0:.2f} |")
        lines.append("")
    return lines, summary


def ideal_for(work, p):
    ideal = load(os.path.join(work, "results", "ideal.json"), {})
    return ideal.get("vista_casa" if p == "vista_casa" else "vista_hpc", {})


def accuracy_noiseless(work, targets):
    """Each version without noise, against the truth and against the ideal
    measurement (ideal.py: an exact one-Gaussian fit on this coverage)."""
    lines = ["| Version | " + " | ".join(f"{q[2]} [{q[5]}]" for q in QUANT) + " |",
             "|---|" + "---|" * len(QUANT)]
    out = {}
    for p in ORDER:
        r = load(os.path.join(work, "results", f"{p}_noiseless.json"))
        if r is None:
            continue
        t = targets[p]
        cells, out[p] = [], {}
        for key, _, label, tkey, scale, _ in QUANT:
            m = r.get(key)
            tv = t[tkey]
            if m is None:
                cells.append("—")
                continue
            iv = ideal_for(work, p).get(key)
            out[p][key] = {"measured": m, "true": tv, "ideal": iv}
            if key == "v0_kms":
                cells.append(f"{m:+.1f} (true {tv:+.1f}, ideal {iv:+.1f})" if iv is not None
                             else f"{m:+.1f} (true {tv:+.1f})")
            else:
                txt = f"{m * scale:.4g} ({100 * (m / tv - 1):+.1f}% true"
                txt += f", {100 * (m / iv - 1):+.1f}% ideal)" if iv else ")"
                cells.append(txt)
        lines.append(f"| {NAMES[p]} | " + " | ".join(cells) + " |")
    tr = targets["extract_opt"]
    lines.append(f"| true (ViSta-HPC weights) | {tr['line_flux']:.4g} | {tr['fwhm_kms']:.0f} | "
                 f"{tr['v0_kms']:+.1f} | {tr['size_arcsec']:.3f} | {tr['cont_flux'] * 1e3:.4g} |")
    iv = ideal_for(work, "extract_opt")
    if iv:
        lines.append(f"| ideal one-Gaussian fit (ViSta-HPC weights) | {iv['line_flux']:.4g} | "
                     f"{iv['fwhm_kms']:.0f} | {iv['v0_kms']:+.1f} | {iv['size_arcsec']:.3f} | "
                     f"{iv['cont_flux'] * 1e3:.4g} |")
    return lines, out


def montecarlo(work, targets):
    rows = load(os.path.join(work, "results", "montecarlo.json"), [])
    by = {}
    for r in rows:
        by.setdefault(r["pipeline"], []).append(r["results"])
    lines = ["| Version | quantity | N | true | without noise | mean | bias (vs true) | scatter | "
             "mean quoted error | pull mean | pull rms | within ±1σ |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    noiseless = {p: load(os.path.join(work, "results", f"{p}_noiseless.json"), {}) for p in ORDER}
    for p in ORDER:
        if p not in by:
            continue
        out[p] = {}
        for key, ekey, label, tkey, scale, unit in QUANT:
            m = np.array([r.get(key) if r.get(key) is not None else np.nan for r in by[p]], float)
            e = np.array([r.get(ekey) if r.get(ekey) is not None else np.nan for r in by[p]], float)
            ok = np.isfinite(m)
            if ok.sum() < 2:
                continue
            m, e = m[ok], e[ok]
            # an error of zero is a fit at the boundary of its range (a size of
            # zero): no error to test, so no pull
            at_bound = int(np.sum(e <= 0))
            e = np.where(e > 0, e, np.nan)
            tv = targets[p][tkey]
            # the pulls about what the same version measures without noise: the
            # noise alone, so they test the quoted errors (the systematic part is
            # in the noiseless table)
            ref = noiseless[p].get(key)
            pull = (m - (ref if ref is not None else tv)) / e
            n = len(m)
            bias = m.mean() - tv
            sd = m.std(ddof=1)
            has_err = bool(np.isfinite(pull).sum() >= 2)
            d = {"n": n, "true": tv, "noiseless": ref, "mean": m.mean(), "bias": bias,
                 "bias_error": sd / np.sqrt(n), "scatter": sd, "at_bound": at_bound,
                 "mean_error": float(np.nanmean(e)) if has_err else None,
                 "pull_mean": float(np.nanmean(pull)) if has_err else None,
                 "pull_rms": float(np.sqrt(np.nanmean(pull ** 2))) if has_err else None,
                 "coverage": float(np.mean(np.abs(pull[np.isfinite(pull)]) <= 1)) if has_err else None,
                 "values": m.tolist(), "errors": [None if not np.isfinite(x) else float(x) for x in e]}
            out[p][key] = d
            bias_txt = (f"{bias:+.1f} ± {sd / np.sqrt(n):.1f}" if key == "v0_kms" else
                        f"{100 * bias / tv:+.1f}% ± {100 * sd / np.sqrt(n) / tv:.1f}%")
            ref_txt = "—" if ref is None else f"{ref * scale:.4g}"
            err_txt = (f"{d['mean_error'] * scale:.3g} | {d['pull_mean']:+.2f} | {d['pull_rms']:.2f} | "
                       f"{d['coverage']:.2f}" if has_err else "— | — | — | —")
            if at_bound:
                err_txt += f" ({at_bound} at the boundary, no error)"
            lines.append(f"| {NAMES[p]} | {label} | {n} | {tv * scale:.4g} | {ref_txt} | "
                         f"{m.mean() * scale:.4g} | {bias_txt} | {sd * scale:.3g} | {err_txt} |")
    return lines, out


def figures(work, perf_summary, perf, mc, targets, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pal = {"vista_casa": "#7f7f7f", "extract": "#9ecae1", "extract_opt": "#2171b5"}
    # time per step
    if perf_summary:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        for i, p in enumerate([q for q in ORDER if q in perf_summary]):
            left = 0
            for name, s in perf_summary[p]["steps"].items():
                ax.barh(i, s["seconds"], left=left, color=pal[p], edgecolor="white", lw=0.8)
                if s["seconds"] > 0.06 * perf_summary[ORDER[0] if ORDER[0] in perf_summary else p]["wall_seconds"]:
                    ax.text(left + s["seconds"] / 2, i, name, ha="center", va="center", fontsize=7,
                            color="white" if p != "extract" else "black", rotation=0)
                left += s["seconds"]
            ax.text(left + 5, i, f"{left:.0f} s", va="center", fontsize=8)
        ax.set_yticks(range(len(perf_summary)))
        ax.set_yticklabels([NAMES[p] for p in ORDER if p in perf_summary])
        ax.invert_yaxis()
        ax.set_xlabel("wall time [s], median of the runs")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "time.png"), dpi=150)
        plt.close(fig)
    # memory against time
    if perf:
        fig, ax = plt.subplots(figsize=(8, 3.2))
        seen = set()
        for r in perf:
            p = r["pipeline"]
            if p in seen or "memory_trace" not in r["metrics"]:
                continue
            seen.add(p)
            tr = np.array(r["metrics"]["memory_trace"])
            ax.plot(tr[:, 0], tr[:, 1], color=pal[p], label=NAMES[p])
        ax.set_xlabel("time since the start [s]")
        ax.set_ylabel("memory (PSS) [GB]")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "memory.png"), dpi=150)
        plt.close(fig)
    # Monte Carlo: line flux / truth
    if mc:
        keys = [("line_flux", "line flux"), ("fwhm_kms", "FWHM"), ("size_arcsec", "size")]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        for ax, (k, label) in zip(axes, keys):
            for i, p in enumerate([q for q in ORDER if q in mc and k in mc[q]]):
                d = mc[p][k]
                vals = np.array(d["values"]) / d["true"]
                ax.scatter(np.full(len(vals), i) + np.random.default_rng(1).uniform(-0.12, 0.12, len(vals)),
                           vals, s=8, color=pal[p], alpha=0.7)
                ax.errorbar(i + 0.25, vals.mean(), yerr=vals.std(ddof=1), fmt="o", color="black", ms=4)
            ax.axhline(1, color="#1e8449", ls="--", lw=1)
            ps = [q for q in ORDER if q in mc and k in mc[q]]
            ax.set_xticks(range(len(ps)))
            ax.set_xticklabels([NAMES[p].replace("ViSta-HPC ", "") for p in ps], fontsize=7)
            ax.set_title(f"{label}: measured / true", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "montecarlo.png"), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    outdir = os.path.join(a.work, "report")
    os.makedirs(outdir, exist_ok=True)
    truth, targets = truth_for(a.work)
    perf = load(os.path.join(a.work, "results", "perf.json"), [])
    lines = ["# ViSta benchmark", ""]
    pl, ps = perf_tables(perf) if perf else ([], {})
    lines += ["## Time, memory, disk", ""] + pl
    al, ao = accuracy_noiseless(a.work, targets)
    lines += ["## Without noise", ""] + al + [""]
    ml, mo = montecarlo(a.work, targets)
    lines += ["## Monte Carlo", ""] + ml + [""]
    open(os.path.join(outdir, "tables.md"), "w").write("\n".join(lines) + "\n")
    json.dump({"perf": ps, "noiseless": ao, "montecarlo": mo, "targets": targets},
              open(os.path.join(outdir, "summary.json"), "w"), indent=1, default=float)
    figures(a.work, ps, perf, mo, targets, outdir)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
