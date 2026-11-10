"""ViSta 1.0: the CASA pipeline, from the stack to the measured line.

The stacking is the code of https://github.com/martitors/ViSta (class
``ViSta``: ``restframing``, ``centering``, ``rebinning``, ``stacking``), run as
it is.  The steps after it follow Chapter 4.2 of the thesis, with CASA tasks
and UVMultiFit:

1. the spectral windows and the observations of the concatenated MS are
   unified (``mstransform(combinespws=True)``, one observation and one scan),
   as in ``ViSta.averaging``;
2. the continuum is subtracted after the stacking with ``uvcontsub``
   (order 0, on the channels outside the line window);
3. the data are averaged in time (``mstransform(timeaverage=True)``) so
   that UVMultiFit can hold them in memory -- over ``--timebin`` (120 s, four
   integrations), not over the whole track as ``ViSta.averaging`` does,
   because a whole-track average leaves a resolved source no baselines to be
   fitted on.  Each dataset is averaged after the rebinning, before the
   concatenation (``--average before-concat``, the default): averaging the
   concatenated MS (``--average after``, as in ``ViSta.averaging``) did not
   finish in 80 minutes on 24 datasets, since every dataset keeps its own
   antennas and ``mstransform`` then handles ~1000 antennas at once;
4. UVMultiFit fits a circular Gaussian at the phase centre: the size on the
   channels of the line averaged together, then the flux channel by channel
   with that size fixed (``OneFitPerChannel=True``).  Only the channels that
   every dataset covers are fitted: the common grid is wider than some
   datasets, and UVMultiFit 3.1 does not honour channel flags (it zeroes
   the weight of a row only when all its channels are flagged), so the
   partly covered edge channels would enter its fits as zeros -- on this
   sample the continuum came out 25% low;
5. a Gaussian fitted to the channel fluxes gives the line flux, centre and
   width, with errors from Monte Carlo resampling of the channel fluxes
   within their errors (as in the thesis);
6. the continuum: the same Gaussian model fitted to the line-free channels of
   the unified MS before the subtraction.

UVMultiFit needs ``uvmultifit.patch`` (in this folder) on a stacked MS.  Two
of its changes only let it build with NumPy 2.  The third is for speed: for
every visibility its C++ module looked for the baseline index by scanning
all Nants (Nants - 1) / 2 baselines.  After the concatenation each dataset
keeps its own antennas (~1000 in all, ~500 000 baselines), so one model
evaluation took 44 s and one stack about 3 hours.  The patch computes the
index from the two antenna numbers, in the order UVMultiFit builds its list,
and gives the same results (a size fit: 300 s before, 1 s after, same flux
and size to every printed digit).

Usage::

    python run_vista_casa.py --input input_list.txt --vista-casa /path/to/ViSta \\
        --uvmultifit /path/to/UVMultiFit --rest-freq 345.79599 --z-ref 2.49 \\
        --out-dir casa_run

Every stage prints a ``@@stage <name>`` line (read by ``monitor.py``) and the
results go to ``<out-dir>/results.json``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import time

import numpy as np

C_KMS = 299792.458


def stage(name):
    print(f"@@stage {name}", flush=True)


def load_vista_casa(repo):
    """The ViSta class of the CASA repository.

    The file is compiled with its tabs expanded: the version on GitHub mixes
    tabs and spaces in ``rebinning`` (since commit 4d86e39), which Python 3
    refuses (TabError).  Nothing else is changed.
    """
    path = os.path.join(repo, "vista", "vista.py")
    src = open(path).read().replace("\t", "    ")
    spec = importlib.util.spec_from_loader("vista_casa", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod.ViSta


def channel_velocities(ms, rest_hz):
    from casatools import table
    tb = table()
    tb.open(f"{ms}/SPECTRAL_WINDOW")
    freqs = [tb.getcell("CHAN_FREQ", i) for i in range(tb.nrows())]
    tb.close()
    return [C_KMS * (rest_hz - f) / rest_hz for f in freqs]


def ranges(mask):
    """'a~b;c~d' for the True runs of a boolean array."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append(f"{i}~{j}")
            i = j + 1
        else:
            i += 1
    return ";".join(out)


def covered_channels(ms, max_flagged=0.01):
    """Per spectral window, the channels that (almost) every visibility covers.

    The common grid of the stack is wider than some datasets, so its edge
    channels are flagged in the rows of those datasets.  UVMultiFit 3.1 does
    not honour channel flags (it zeroes only the weights of rows flagged in
    every channel): those channels would enter its fits as zeros.
    """
    from casatools import table
    tb = table()
    tb.open(ms)
    out = {}
    for k in np.unique(tb.getcol("DATA_DESC_ID")):
        sub = tb.query(f"DATA_DESC_ID=={k}", columns="FLAG")
        frac = sub.getcol("FLAG").mean(axis=(0, 2))
        sub.close()
        out[int(k)] = frac <= max_flagged
    tb.close()
    return out


def spw_selection(ms, rest_hz, keep, covered=None):
    """CASA spw:channel selection of the channels where ``keep(v)`` is true
    (and, with ``covered``, that every visibility covers)."""
    parts = []
    for k, v in enumerate(channel_velocities(ms, rest_hz)):
        m = keep(np.asarray(v))
        if covered is not None:
            m = m & covered.get(k, np.zeros_like(m))
        r = ranges(m)
        if r:
            parts.append(f"{k}:{r}")
    return ",".join(parts)


def unify(ms_in, ms_out):
    """One spectral window, one observation, one scan (as ViSta.averaging)."""
    from casatasks import mstransform
    from casatools import table
    tb = table()
    tb.open(f"{ms_in}/SPECTRAL_WINDOW")
    nspw = tb.nrows()
    tb.close()
    if nspw > 1:
        mstransform(vis=ms_in, outputvis=ms_out, datacolumn="data", combinespws=True)
    else:
        shutil.copytree(ms_in, ms_out)
    tb.open(f"{ms_out}/OBSERVATION", nomodify=False)
    tr = tb.getcol("TIME_RANGE")
    tr[0, 0], tr[1, 0] = tr[0].min(), tr[1].max()
    tb.putcol("TIME_RANGE", tr)
    tb.close()
    tb.open(ms_out, nomodify=False)
    n = tb.nrows()
    tb.putcol("SCAN_NUMBER", np.zeros(n, np.int32))
    tb.putcol("OBSERVATION_ID", np.zeros(n, np.int32))
    tb.close()


def average(ms_in, ms_out, timebin):
    from casatasks import mstransform
    mstransform(vis=ms_in, outputvis=ms_out, datacolumn="data", timeaverage=True,
                timebin=timebin)


def uvmf_gauss(uvm, ms, spw, size=None, per_channel=False, outfile="fit.dat"):
    """Circular Gaussian at the phase centre: flux (and size unless given)."""
    if size is None:
        var, p_ini, bounds = "0,0,p[0],p[1],1.0,0.0", [1e-3, 0.5], [[-1.0, 1.0], [0.01, 10.0]]
    else:
        var, p_ini, bounds = f"0,0,p[0],{size:.6f},1.0,0.0", [0.0], [[-1.0, 1.0]]
    fit = uvm.uvmultifit(vis=ms, spw=spw, column="data", model=["Gaussian"], var=[var],
                         p_ini=p_ini, bounds=bounds, OneFitPerChannel=per_channel,
                         outfile=outfile, NCPU=os.cpu_count() or 1)
    return fit.result


def gauss(v, area, v0, fwhm):
    s = fwhm / 2.3548200450309493
    return area * np.exp(-0.5 * ((v - v0) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def fit_spectrum(v, f, e, n_mc=500, seed=1):
    """Gaussian fit of the channel fluxes; errors from Monte Carlo resampling."""
    from scipy.optimize import curve_fit
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0)
    v, f, e = v[ok], f[ok], e[ok]
    p0 = [np.trapezoid(np.clip(f, 0, None), v) or 1e-3, v[np.argmax(f)], 300.0]
    bounds = ([-np.inf, -1500.0, 2 * abs(np.median(np.diff(v)))], [np.inf, 1500.0, 3000.0])
    p, _ = curve_fit(gauss, v, f, p0=p0, sigma=e, absolute_sigma=True, bounds=bounds, maxfev=20000)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_mc):
        try:
            q, _ = curve_fit(gauss, v, f + rng.normal(0, e), p0=p, sigma=e, absolute_sigma=True,
                             bounds=bounds, maxfev=20000)
            draws.append(q)
        except RuntimeError:
            pass
    d = np.array(draws)
    return p, (d.std(axis=0) if len(draws) > 1 else np.zeros(3)), len(draws)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--vista-casa", required=True, help="checkout of github.com/martitors/ViSta")
    ap.add_argument("--uvmultifit", required=True, help="directory with uvmultifit.py and its module")
    ap.add_argument("--rest-freq", type=float, required=True, help="GHz")
    ap.add_argument("--z-ref", type=float, required=True)
    ap.add_argument("--dish", type=float, default=12.0)
    ap.add_argument("--v-window", type=float, nargs=2, default=(-550.0, 550.0))
    ap.add_argument("--size-window", type=float, nargs=2, default=(-300.0, 300.0),
                    help="channels averaged for the size fit, km/s")
    ap.add_argument("--timebin", default="120s")
    ap.add_argument("--average", choices=["before-concat", "after"], default="before-concat")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--keep", action="store_true", help="keep the intermediate Measurement Sets")
    ap.add_argument("--noiseless", action="store_true",
                    help="noiseless data: UVMultiFit errors are zero, the spectrum is fitted "
                         "with equal weights and without Monte Carlo errors")
    a = ap.parse_args()

    rest_hz = a.rest_freq * 1e9
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.abspath(a.out_dir)
    t_all = time.time()

    # the inputs are linked into the output directory, where ViSta writes its copies
    lines = []
    for line in open(a.input):
        if not line.strip() or line.startswith("#"):
            continue
        p = line.split()
        link = os.path.join(out, os.path.basename(p[0].rstrip("/")))
        if not os.path.exists(link):
            os.symlink(os.path.abspath(p[0]), link)
        lines.append(" ".join([link] + p[1:4]))
    lst = os.path.join(out, "input_casa.txt")
    open(lst, "w").write("\n".join(lines) + "\n")

    sys.path.insert(0, a.uvmultifit)
    ViSta = load_vista_casa(a.vista_casa)
    v = ViSta(lst, clean_previous=not a.keep)

    stage("restframing")
    v.restframing()
    stage("centering")
    v.centering()
    stage("rebinning")
    v.rebinning(central_freq=rest_hz)
    if a.average == "before-concat":
        stage("averaging")
        averaged = []
        for ms in v.ms_list:
            average(ms, ms + ".avg", a.timebin)
            if not a.keep:
                shutil.rmtree(ms)
            averaged.append(ms + ".avg")
        v.ms_list = averaged
    stage("stacking")
    stacked = os.path.join(out, "stacked.ms")
    v.stacking(output_ms=stacked, dl_weighting=True, z_common=a.z_ref,
               pb_weighting=True, nu_pb=rest_hz, dish_diameter=a.dish)
    if not a.keep:
        for ms in v.ms_list:
            v._safe_remove(ms)

    stage("unify")
    unified = stacked + ".unified"
    unify(stacked, unified)
    if not a.keep:
        shutil.rmtree(stacked)

    stage("contsub")
    from casatasks import uvcontsub
    lo, hi = a.v_window
    linefree = spw_selection(unified, rest_hz, lambda vv: (vv < lo) | (vv > hi))
    contsub = unified + ".contsub"
    uvcontsub(vis=unified, outputvis=contsub, fitspec=linefree, fitorder=0, datacolumn="data")

    if a.average == "after":
        stage("averaging")
        avg_line = contsub + ".avg"
        average(contsub, avg_line, a.timebin)
        avg_cont = unified + ".avg"
        average(unified, avg_cont, a.timebin)
        if not a.keep:
            shutil.rmtree(contsub)
            shutil.rmtree(unified)
    else:
        avg_line, avg_cont = contsub, unified

    import uvmultifit as uvm
    stage("fit_size")
    cov_line = covered_channels(avg_line)
    s_lo, s_hi = a.size_window
    core = spw_selection(avg_line, rest_hz, lambda vv: (vv >= s_lo) & (vv <= s_hi), cov_line)
    r = uvmf_gauss(uvm, avg_line, core, outfile=os.path.join(out, "size.dat"))
    par = np.asarray(r["Parameters"], float).ravel()
    err = np.asarray(r["Uncertainties"], float).ravel()
    size, size_err = abs(par[1]), err[1]

    stage("fit_channels")
    every = spw_selection(avg_line, rest_hz, lambda vv: np.ones(vv.shape, bool), cov_line)
    rc = uvmf_gauss(uvm, avg_line, every, size=size, per_channel=True,
                    outfile=os.path.join(out, "channels.dat"))
    freq = np.concatenate([np.asarray(x, float).ravel() for x in np.atleast_1d(rc["Frequency"])]) \
        if isinstance(rc["Frequency"], (list, tuple)) else np.asarray(rc["Frequency"], float).ravel()
    fl = np.asarray(rc["Parameters"], float).reshape(len(freq), -1)[:, 0]
    fe = np.asarray(rc["Uncertainties"], float).reshape(len(freq), -1)[:, 0]
    vel = C_KMS * (rest_hz - freq) / rest_hz
    order = np.argsort(vel)
    vel, fl, fe = vel[order], fl[order], fe[order]
    np.savetxt(os.path.join(out, "spectrum.txt"), np.c_[vel, fl, fe],
               header="v_kms flux_density_Jy error_Jy")

    stage("fit_profile")
    if a.noiseless:
        fe = np.full_like(fl, max(np.nanmax(np.abs(fl)), 1e-12) * 1e-3)
    p, pe, n_ok = fit_spectrum(vel, fl, fe, n_mc=0 if a.noiseless else 500)

    stage("continuum")
    cont_sel = spw_selection(avg_cont, rest_hz, lambda vv: (vv < lo) | (vv > hi),
                             covered_channels(avg_cont))
    rcont = uvmf_gauss(uvm, avg_cont, cont_sel, outfile=os.path.join(out, "continuum.dat"))
    cpar = np.asarray(rcont["Parameters"], float).ravel()
    cerr = np.asarray(rcont["Uncertainties"], float).ravel()
    if not a.keep:
        shutil.rmtree(avg_cont)
        shutil.rmtree(avg_line)

    res = {"pipeline": "vista_casa",
           "line_flux": float(p[0]), "line_flux_error": float(pe[0]),
           "v0_kms": float(p[1]), "v0_error_kms": float(pe[1]),
           "fwhm_kms": float(p[2]), "fwhm_error_kms": float(pe[2]),
           "size_arcsec": float(size), "size_error_arcsec": float(size_err),
           "cont_flux": float(cpar[0]), "cont_flux_error": float(cerr[0]),
           "cont_size_arcsec": float(abs(cpar[1])),
           "n_mc": int(n_ok), "seconds": time.time() - t_all,
           "settings": {"timebin": a.timebin, "average": a.average, "v_window_kms": list(a.v_window),
                        "size_window_kms": list(a.size_window), "z_ref": a.z_ref}}
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
