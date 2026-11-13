"""The sample of the benchmark, made with CASA ``simobserve``.

Every galaxy is one ALMA Band 3 observation of CO(3-2) at z = 2-3: a
Gaussian source with a Gaussian line on a flat continuum, a few arcsec off
the phase centre, observed with one of the Cycle 8 configurations and with
thermal noise (``thermalnoise="tsys-atm"``).  The band has ``--nchan``
channels of 1.875 GHz / nchan, each dataset with its own width within
+-10%, so the bands go from 1.69 to 2.06 GHz and the rebinning is tested.

Every galaxy has the same luminosity: its flux is the one that gives
``--line-flux`` (and ``--cont-flux``) at the reference redshift ``--z-ref``
(2.5), to which the extraction of the three versions brings the sample.  The
stack should then give back those values, with the line width and the size
of the sample.

Every galaxy is drawn from its own random numbers, seeded by ``--seed`` and
its index: the first 50 galaxies of a sample of 100 are the galaxies of a
sample of 50, and the samples with 500, 1000 and 2000 channels hold the same
galaxies.  ``--seed`` draws the galaxies (redshifts, positions, offsets,
arrays); ``--noise-seed`` (default: ``--seed``) the noise, so the same
galaxies can be observed again with other noise.  Usage::

    python simobserve_sample.py --out /media/martina/Elements/vista_bench/data/n1000 \\
        --n 100 --nchan 1000 --seed 1 --jobs 4

writes ``galaxy_000.ms ...``, ``input_list.txt`` and ``truth.json`` in
``--out``.  Galaxies already there are not made again, so a run that was
interrupted continues.
"""
import argparse
import json
import multiprocessing
import os
import shutil
import sys

import numpy as np

C_KMS = 299792.458
REST_GHZ = 345.79599                       # CO(3-2)
BAND_HZ = 1.875e9                          # one ALMA spectral window
CONFIGS = ("alma.cycle8.1.cfg", "alma.cycle8.2.cfg", "alma.cycle8.3.cfg")
NPIX, CELL = 128, 0.1                      # sky model: pixels, arcsec


def alpha_factor(z, z_ref, H0=67.4, Om0=0.315):
    """Flux at z_ref of a source of flux 1 at z, at fixed luminosity (as the
    extraction of ViSta): (D_L(z) / D_L(z_ref))^2 (1 + z_ref) / (1 + z)."""
    from astropy.cosmology import FlatLambdaCDM
    d = FlatLambdaCDM(H0=H0, Om0=Om0).luminosity_distance([z, z_ref]).value
    return (d[0] / d[1]) ** 2 * (1 + z_ref) / (1 + z)


def fmt_ra(deg):
    h = deg / 15.0
    m = (h - int(h)) * 60
    return f"{int(h):02d}:{int(m):02d}:{(m - int(m)) * 60:09.6f}"


def fmt_dec(deg):
    s, a = ("-" if deg < 0 else "+"), abs(deg)
    m = (a - int(a)) * 60
    return f"{s}{int(a):02d}:{int(m):02d}:{(m - int(m)) * 60:08.5f}"


def draw(n, nchan, seed, noise_seed, z_ref, line_flux, cont_flux, fwhm_kms, size_arcsec):
    """The galaxies: what changes from one to the other, and their fluxes.
    Galaxy i is drawn from its own generator, seeded by (seed, i)."""
    out = []
    for i in range(n):
        rng = np.random.default_rng([seed, i])
        z = round(float(rng.uniform(2.0, 3.0)), 4)
        dx, dy = rng.uniform(-2.0, 2.0, 2)                 # source - phase centre, arcsec
        dec0 = rng.uniform(-40.0, -5.0)
        ra0 = rng.uniform(0.0, 360.0)
        scale = 1.0 / alpha_factor(z, z_ref)               # same luminosity for everyone
        out.append({
            "name": f"galaxy_{i:03d}", "z": z,
            "phase_centre_deg": [ra0, dec0], "offset_arcsec": [float(dx), float(dy)],
            "ra_deg": ra0 + dx / 3600.0 / np.cos(np.radians(dec0)), "dec_deg": dec0 + dy / 3600.0,
            "line_flux": line_flux * scale, "cont_flux": cont_flux * scale,
            "fwhm_kms": fwhm_kms, "size_arcsec": size_arcsec,
            "nchan": nchan, "channel_width_hz": BAND_HZ / nchan * rng.uniform(0.9, 1.1),
            "config": CONFIGS[rng.integers(len(CONFIGS))],
            "noise_seed": int(noise_seed) * 100000 + i + 1,
        })
    return out


def sky_model(g, path):
    """The cube given to simobserve [Jy/pixel]: a Gaussian source with a
    Gaussian line on a flat continuum, in the LSRK frame."""
    from casatools import coordsys, image
    nchan, dnu = g["nchan"], g["channel_width_hz"]
    nu0 = REST_GHZ * 1e9 / (1.0 + g["z"])                  # the line, observed
    freq = nu0 + (np.arange(nchan) - (nchan - 1) / 2.0) * dnu
    v = C_KMS * (nu0 - freq) / nu0
    sigma_v = g["fwhm_kms"] / 2.3548200450309493
    line = g["line_flux"] / (sigma_v * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (v / sigma_v) ** 2)
    spectrum = g["cont_flux"] + line                       # Jy
    # the source: +x is east, so it decreases with the pixel index
    x = (np.arange(NPIX) - NPIX // 2) * CELL
    sig = g["size_arcsec"] / 2.3548200450309493
    dx, dy = g["offset_arcsec"]
    shape = np.exp(-0.5 * ((-x[:, None] - dx) ** 2 + (x[None, :] - dy) ** 2) / sig ** 2)
    shape /= shape.sum()

    cs = coordsys().newcoordsys(direction=True, spectral=True, stokes=["I"])
    ra0, dec0 = g["phase_centre_deg"]
    cs.setreferencecode("J2000", "direction")
    cs.setreferencevalue(f"{ra0:.9f}deg {dec0:.9f}deg", "direction")
    cs.setreferencepixel([NPIX // 2, NPIX // 2], "direction")
    cs.setincrement(f"{-CELL}arcsec {CELL}arcsec", "direction")
    cs.setreferencecode("LSRK", "spectral")
    cs.setrestfrequency(f"{nu0:.3f}Hz")
    cs.setreferencevalue(f"{freq[0]:.3f}Hz", "spectral")
    cs.setreferencepixel(0, "spectral")
    cs.setincrement(f"{dnu:.6f}Hz", "spectral")
    ia = image()
    ia.fromshape(path, [NPIX, NPIX, 1, nchan], csys=cs.torecord(), overwrite=True)
    ia.setbrightnessunit("Jy/pixel")
    for c0 in range(0, nchan, 256):                        # written in blocks of channels
        block = shape[:, :, None, None] * spectrum[None, None, None, c0:c0 + 256]
        ia.putchunk(block.astype(np.float32), blc=[0, 0, 0, c0])
    ia.close()
    cs.done()


def set_weights(noisy, clean):
    """SIGMA and WEIGHT (of one channel, per polarisation) of the noisy MS from
    the noise that was added, i.e. its difference from the noiseless one:
    simobserve leaves them at 1."""
    from casatools import table
    tb = table()
    tb.open(clean)
    model = tb.getcol("DATA")
    tb.close()
    tb.open(noisy, nomodify=False)
    noise = tb.getcol("DATA") - model                      # (pol, channel, row)
    sigma = np.sqrt(0.5 * (noise.real ** 2 + noise.imag ** 2).mean(axis=(1, 2)))   # one channel
    n = tb.nrows()
    tb.putcol("SIGMA", np.repeat(sigma[:, None], n, axis=1).astype(np.float32))
    tb.putcol("WEIGHT", np.repeat(1.0 / sigma[:, None] ** 2, n, axis=1).astype(np.float32))
    tb.close()


def make_galaxy(args):
    """One galaxy: its sky model, simobserve, and the noisy MS kept."""
    g, out, totaltime, integration, pwv = args
    final = os.path.join(out, g["name"] + ".ms")
    if os.path.exists(os.path.join(final, "table.dat")):
        return g["name"], "already there"
    work = os.path.join(out, "_work_" + g["name"])
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    cwd = os.getcwd()
    os.chdir(work)                                         # simobserve writes where it runs
    try:
        sys.argv = sys.argv[:1]                            # casaconfig reads the command line
        from casatasks import simobserve, split
        sky_model(g, "sky.im")
        simobserve(project="sim", skymodel="sky.im", direction="",    # the centre of the model
                   obsmode="int", antennalist=g["config"], totaltime=f"{totaltime}s",
                   integration=f"{integration}s", mapsize=["", ""], maptype="ALMA",
                   pointingspacing="", setpointings=True, hourangle="transit",
                   thermalnoise="tsys-atm", user_pwv=pwv, seed=g["noise_seed"],
                   graphics="none", overwrite=True, verbose=False)
        cfg = g["config"][:-4]
        clean, noisy = (os.path.join("sim", f"sim.{cfg}{k}.ms") for k in ("", ".noisy"))
        set_weights(noisy, clean)
        split(vis=noisy, outputvis=final, datacolumn="data")       # DATA only
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
    return g["name"], "made"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="directory of the sample")
    ap.add_argument("--n", type=int, default=100, help="galaxies")
    ap.add_argument("--nchan", type=int, default=1000, help="channels of every dataset")
    ap.add_argument("--seed", type=int, default=1, help="of the galaxies")
    ap.add_argument("--noise-seed", type=int, default=None,
                    help="of the noise (default: --seed); another value: the same galaxies, "
                         "other noise")
    ap.add_argument("--z-ref", type=float, default=2.5,
                    help="redshift at which every galaxy has --line-flux and --cont-flux")
    ap.add_argument("--jobs", type=int, default=1, help="galaxies made at the same time")
    ap.add_argument("--totaltime", type=float, default=360.0, help="on source, seconds")
    ap.add_argument("--integration", type=float, default=30.0, help="seconds")
    ap.add_argument("--pwv", type=float, default=1.0, help="water vapour of the noise, mm")
    ap.add_argument("--line-flux", type=float, default=0.5, help="Jy km/s at the median z")
    ap.add_argument("--cont-flux", type=float, default=2e-4, help="Jy at the median z")
    ap.add_argument("--fwhm", type=float, default=300.0, help="line width, km/s")
    ap.add_argument("--size", type=float, default=0.6, help="source FWHM, arcsec")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    noise_seed = a.seed if a.noise_seed is None else a.noise_seed
    galaxies = draw(a.n, a.nchan, a.seed, noise_seed, a.z_ref, a.line_flux, a.cont_flux,
                    a.fwhm, a.size)
    z_ref = a.z_ref
    old = os.path.join(out, "truth.json")
    if os.path.exists(old):                                # a sample is there: the same one?
        prev = {g["name"]: g for g in json.load(open(old))["galaxies"]}
        for g in galaxies:
            p = prev.get(g["name"])
            if p is not None and (p["z"] != g["z"] or p["noise_seed"] != g["noise_seed"]
                                  or p["nchan"] != g["nchan"]):
                raise SystemExit(f"{out} holds another sample ({g['name']} differs): "
                                 f"use another --out")
    jobs = [(g, out, a.totaltime, a.integration, a.pwv) for g in galaxies]
    if a.jobs > 1:
        with multiprocessing.get_context("spawn").Pool(a.jobs) as pool:
            for name, what in pool.imap_unordered(make_galaxy, jobs):
                print(f"{name}: {what}", flush=True)
    else:
        for j in jobs:
            print("%s: %s" % make_galaxy(j), flush=True)

    with open(os.path.join(out, "input_list.txt"), "w") as fh:
        fh.write("# <ms_path>  <redshift>  <RA>  <Dec>\n")
        for g in galaxies:
            fh.write(f"{os.path.join(out, g['name'] + '.ms')}  {g['z']:.4f}  "
                     f"{fmt_ra(g['ra_deg'])}  {fmt_dec(g['dec_deg'])}\n")
    truth = {"seed": a.seed, "noise_seed": noise_seed, "z_ref": z_ref, "rest_freq_ghz": REST_GHZ,
             "line_flux": a.line_flux, "cont_flux": a.cont_flux, "fwhm_kms": a.fwhm,
             "size_arcsec": a.size, "v0_kms": 0.0, "totaltime_s": a.totaltime,
             "integration_s": a.integration, "pwv_mm": a.pwv, "galaxies": galaxies}
    with open(os.path.join(out, "truth.json"), "w") as fh:
        json.dump(truth, fh, indent=1)
    print(f"{a.n} galaxies, {a.nchan} channels, seed {a.seed}, noise seed {noise_seed}: "
          f"{out}/input_list.txt")


if __name__ == "__main__":
    sys.exit(main())
