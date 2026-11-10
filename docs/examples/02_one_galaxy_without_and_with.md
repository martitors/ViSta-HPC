# 2. One galaxy, without and with noise

Before stacking, one galaxy (at the median redshift) measured in the two ways
an interferometer allows, on its noiseless and its noisy dataset.

ViSta works on a single dataset as well: a "stack" of one galaxy shifts its
phase centre to the galaxy, brings it to the rest frame and subtracts the
continuum (`CORRECTED_DATA`); the measurement then gives the line flux in
every annulus of baseline length and extrapolates it to zero length. The same
continuum-subtracted visibilities are exported (`export_ms`) and imaged with
CASA's `tclean`: a CLEANed cube, natural weighting.

Some helpers first: the baseline average of a stack, and the image cube with
its spectra (at the peak pixel, in Jy/beam, and summed in an aperture, in
Jy), its moment 0, its beam and the errors of the line flux, measured on the
image itself away from the galaxy.

```python
REST_HZ = REST_GHZ * 1e9
WINDOW = 500           # km/s: the line is within |v| < WINDOW (out of the continuum fit; area flux)


ANNULI = (3, 3000, 18)  # annuli of baseline length: from 3 to 3000 kλ, 18 (logarithmic)


def vista_for(input_list, name, verbose=False):
    # the ViSta object of this notebook: input and output (the name of the files);
    # the stacking: the line and its window, the band (-900..+900 km/s); the rest
    # has defaults (continuum subtracted with a constant, ...)
    v = ViSta(input_list, output_name=name, verbose=verbose)
    v.stack(line_freq=REST_GHZ, line_window=WINDOW, velocity_range=900)
    return v


def baseline_average(store, columns=("CORRECTED_DATA",)):
    # every visibility of a stack averaged over the baselines with its weight, per channel
    datasets = xds_from_zarr(store)
    chan_freq = xds_from_zarr(f"{store}::SPECTRAL_WINDOW")[0].CHAN_FREQ.values
    acc = {}
    for ds in datasets:
        freq = np.round(chan_freq[int(ds.DATA_DESC_ID[0])])
        w = ds.WEIGHT.values[:, None, :] * ~ds.FLAG.values
        wsum = w.sum(axis=(0, 2))
        for col in columns:
            num = (w * ds[col].values.real).sum(axis=(0, 2))
            for f, a, b in zip(freq, num, wsum):
                n0, d0 = acc.setdefault(col, {}).get(f, (0.0, 0.0))
                acc[col][f] = (n0 + a, d0 + b)
    freq = np.array(sorted(acc[columns[0]]))
    vel = C_KMS * (REST_HZ - freq) / REST_HZ
    return vel, {col: np.array([acc[col][f][0] / acc[col][f][1] for f in freq]) for col in columns}


CELL, NPIX, R_AP = 0.15, 192, 2.0                 # arcsec, pixels, aperture radius (arcsec)


@contextlib.contextmanager
def quiet():
    # tclean writes a progress meter and its log straight to the terminal
    sys.stdout.flush(); sys.stderr.flush()
    saved, null = (os.dup(1), os.dup(2)), os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1); os.dup2(null, 2)
    try:
        yield
    finally:
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        for fd in (*saved, null):
            os.close(fd)


def image_line(vis, name, noiseless=False):
    # CLEANed cube of the continuum-subtracted visibilities, its spectra and their errors
    for p in glob.glob(name + ".*"):
        shutil.rmtree(p, ignore_errors=True)
    with quiet():
        tclean(vis=vis, imagename=name, datacolumn="corrected", specmode="cube",
               imsize=NPIX, cell=f"{CELL}arcsec", weighting="natural",
               niter=3000 if noiseless else 100000, nsigma=0.0 if noiseless else 2.0,
               threshold="1e-7Jy" if noiseless else "0Jy", usemask="user",
               mask=f"circle[[{NPIX // 2}pix,{NPIX // 2}pix],{3 / CELL:.0f}pix]",
               restoringbeam="common", interpolation="nearest")
    ia = iatool(); ia.open(f"{name}.image")
    cube = ia.getchunk()[:, :, 0, :]                        # (RA, Dec, channel), Jy/beam
    freq = np.array([ia.toworld([0, 0, 0, k])["numeric"][3] for k in range(cube.shape[-1])])
    beam = ia.restoringbeam(); ia.close()
    bmaj, bmin = beam["major"]["value"], beam["minor"]["value"]
    omega = math.pi * bmaj * bmin / (4 * math.log(2)) / CELL ** 2   # beam area, pixels
    ix, iy = np.mgrid[:NPIX, :NPIX]
    dx, dy = (ix - NPIX // 2) * CELL, (iy - NPIX // 2) * CELL     # arcsec from the galaxy
    r = np.hypot(dx, dy)
    vel = C_KMS * (REST_HZ - freq) / REST_HZ
    dv = abs(np.median(np.diff(vel)))
    flux_map = cube[:, :, np.abs(vel) <= WINDOW].sum(axis=2) * dv    # Jy/beam km/s
    # the errors, from the image itself away from the galaxy (r > 5"): the rms of the
    # line-flux map (for the peak pixel) and the scatter of the same aperture on empty sky
    sky = [flux_map[np.hypot(dx - a, dy - b) <= R_AP].sum() / omega
           for a in range(-10, 11, 4) for b in range(-10, 11, 4) if math.hypot(a, b) > 5]
    return {"v": vel, "dv": dv, "peak": cube[NPIX // 2, NPIX // 2, :],
            "aperture": cube[r <= R_AP].sum(axis=0) / omega,
            "peak_error": float(np.std(flux_map[r > 5])), "aperture_error": float(np.std(sky)),
            "mom0": cube[:, :, np.abs(vel) <= SIM["fwhm_kms"]].sum(axis=2) * dv,
            "beam": (bmaj, bmin, beam["positionangle"]["value"])}


def integrate(v, spectrum):
    # line flux: the spectrum integrated over |v| < WINDOW
    sel = np.abs(v) <= WINDOW
    return np.nansum(spectrum[sel]) * abs(np.median(np.diff(v)))
```

```python
K = SIM["n_sources"] // 2                          # the galaxy at the median redshift
gal, gal0 = truth["sources"][K], truth0["sources"][K]
KNOWN = dict(v0_range=(0, 0), fwhm_range=(SIM["fwhm_kms"], SIM["fwhm_kms"]))   # centre and width known
one = {}
for tag, lst in (("noisy", LIST), ("noiseless", LIST0)):
    entry = [l for l in open(lst) if not l.startswith("#")][K]
    open(f"one_{tag}.txt", "w").write(entry)
    v = vista_for(f"one_{tag}.txt", f"one_{tag}")
    # one galaxy: no redshift rescaling, no primary-beam correction (compared with the image)
    noise = dict(noise="weights", channel_correlation=False) if tag == "noiseless" else {}
    v.extract(annuli=ANNULI, z_norm=False, pb_corr=False, model="gauss", n_null=0,
              n_null_phase=0, n_bootstrap=0, figures_draw=False, **KNOWN, **noise)
    r = v.run()
    vel, avg = baseline_average(f"one_{tag}.zarr")
    export_ms(f"one_{tag}.zarr", f"one_{tag}.ms", verbose=False)
    one[tag] = {"fit": r.line, "uv": pl.load(f"one_{tag}")["uv"], "v": vel,
                "average": avg["CORRECTED_DATA"],
                "image": image_line(f"one_{tag}.ms", f"img_one_{tag}", noiseless=tag == "noiseless")}
print(f"{os.path.basename(gal['path'])}: z = {gal['z']:.3f}, true line flux "
      f"{gal['line_flux'] * gal['primary_beam']:.3f} Jy km/s, size {gal['size_arcsec']}\", "
      f"beam {one['noisy']['image']['beam'][0]:.2f}\" x {one['noisy']['image']['beam'][1]:.2f}\"")
```

```text
[ViSta       0.0s] no redshift rescaling, no flux normalisation
```

```text
[ViSta       0.0s] no redshift rescaling, no flux normalisation
```

```text
galaxy_06.ms: z = 2.637, true line flux 0.395 Jy km/s, size 0.8", beam 1.41" x 0.81"
```

**In the uv plane.** Left: the visibilities averaged over all baselines,
without noise (blue) and with noise (black), and the true total flux density
of the line (green). Right: the line flux in every annulus of baseline
length, with the true curve F·g(b) and the fit of ViSta to the noisy data.

Even without noise the average over the baselines is well below the total:
the long baselines resolve the galaxy and see only part of it. The flux per
annulus shows why — it falls with baseline length, from the total flux at
zero length — and the extrapolation to zero recovers it.

```python
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.2))
vv = np.linspace(-900, 900, 721)
a1.step(one["noisy"]["v"], one["noisy"]["average"] * 1e3, where="mid", label="with noise", **NOISY)
a1.step(one["noiseless"]["v"], one["noiseless"]["average"] * 1e3, where="mid",
        label="without noise", **CLEAN)
a1.plot(vv, simulate.line_profile(gal, vv) * 1e3, label="true total (zero spacing)", **TRUE)
a1.axhline(0, color="0.5", lw=0.5)
a1.set_xlabel("velocity [km s$^{-1}$]"); a1.set_ylabel("mean Re V over all baselines [mJy]")
a1.legend(loc="upper left")

for tag, sty, mk in (("noisy", NOISY, "o"), ("noiseless", CLEAN, "s")):
    u = one[tag]["uv"]
    ok = (u["used"] == 1) & np.isfinite(u["line_flux"])
    a2.errorbar(u["b_klambda"][ok], u["line_flux"][ok], u["err_thermal"][ok] if tag == "noisy" else None,
                fmt=mk, color=sty["color"], ms=5, capsize=2, mfc="none" if tag == "noiseless" else None,
                label="with noise" if tag == "noisy" else "without noise")
bb = np.geomspace(5, 800, 200)
g = simulate.shape_visibility(bb * 1e3, gal["size_arcsec"])
a2.plot(bb, gal["line_flux"] * gal["primary_beam"] * g, label="true", **TRUE)
ln = one["noisy"]["fit"].summary["line"]
a2.plot(bb, ln["flux"] * simulate.shape_visibility(bb * 1e3, ln["theta_fwhm_arcsec"]),
        color="#c0392b", lw=1.4, label="ViSta fit (with noise)")
a2.axhline(0, color="0.5", lw=0.5)
a2.set_xscale("log"); a2.set_xlabel(r"baseline [k$\lambda$]"); a2.set_ylabel("line flux [Jy km s$^{-1}$]")
lo, hi = a2.get_ylim(); a2.set_ylim(max(lo, -0.5), min(hi, 1.2))
a2.legend()
fig.tight_layout()
save(fig, "one_galaxy_uv")
```

```{image} figures/02_01.png
:width: 100%
```

**In the image.** Top: the line (moment 0 over ±FWHM) without and with
noise, with the true size of the galaxy (green circle, FWHM) and the beam.
Bottom: the spectrum at the peak pixel (Jy/beam) and in an aperture of
radius 2″ (Jy), with the true total.

The peak is in Jy/beam: for a source not much smaller than the beam it is
less than the flux, and its spectrum is, channel by channel, the mean over
the baselines of the uv plane (left, above). The aperture collects the flux
if it is large enough; with noise it also collects the noise of every pixel
in it.

```python
fig = plt.figure(figsize=(12.5, 8))
ext = NPIX / 2 * CELL
vmax = np.nanmax(one["noiseless"]["image"]["mom0"])
for j, tag in enumerate(("noiseless", "noisy")):
    im_ = one[tag]["image"]
    ax = fig.add_subplot(2, 2, j + 1)
    im = ax.imshow(im_["mom0"].T, origin="lower", extent=(ext, -ext, -ext, ext), cmap="inferno",
                   vmin=-0.3 * vmax, vmax=1.1 * vmax)
    fig.colorbar(im, ax=ax, label="Jy/beam km s$^{-1}$")
    ax.add_patch(Circle((0, 0), gal["size_arcsec"] / 2, fill=False, ec="#2ecc71", lw=1.5, ls="--",
                        label="true size (FWHM)"))
    bmaj, bmin, pa = im_["beam"]
    ax.add_patch(Ellipse((-3.0, -3.0), bmin, bmaj, angle=-pa, fill=False, ec="w", label="beam"))
    ax.add_patch(Circle((0, 0), R_AP, fill=False, ec="w", lw=0.8, ls=":", label=f"aperture {R_AP:g}\""))
    ax.set_xlim(4, -4); ax.set_ylim(-4, 4)
    ax.set_xlabel(r"$\Delta$RA [arcsec]"); ax.set_ylabel(r"$\Delta$Dec [arcsec]")
    ax.set_title("without noise" if tag == "noiseless" else "with noise")
    if j == 0:
        ax.legend(loc="upper left", fontsize=8)
for j, (key, unit) in enumerate((("peak", "Jy/beam"), ("aperture", "Jy"))):
    ax = fig.add_subplot(2, 2, 3 + j)
    for tag, sty in (("noisy", NOISY), ("noiseless", CLEAN)):
        im_ = one[tag]["image"]
        ax.step(im_["v"], im_[key] * 1e3, where="mid",
                label="with noise" if tag == "noisy" else "without noise", **sty)
    ax.plot(vv, simulate.line_profile(gal, vv) * 1e3, label="true total", **TRUE)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlim(-900, 900)
    ax.set_xlabel("velocity [km s$^{-1}$]")
    ax.set_ylabel(f"{'peak pixel' if key == 'peak' else f'aperture {R_AP:g}″'} [m{unit}]")
    ax.legend(loc="upper left")
fig.tight_layout()
save(fig, "one_galaxy_image")
```

```{image} figures/02_02.png
:width: 100%
```

```python
def flux_table(rows, title):
    # rows: (label, without noise, with noise, error of the one with noise)
    print(f"{title}\n   {'line flux [Jy km/s]':31s} {'without noise':>13s}   {'with noise':>15s}")
    for lab, a, b, e in rows:
        err = f" ± {e:.3f}" if e is not None else ""
        print(f"   {lab:31s} {a:13.3f}   {b:7.3f}{err}")


u = {tag: one[tag]["fit"].summary["line"] for tag in one}
im = {tag: one[tag]["image"] for tag in one}
F_TRUE = gal["line_flux"] * gal["primary_beam"]
flux_table([
    ("true", F_TRUE, F_TRUE, None),
    ("uv: mean over the baselines", integrate(one["noiseless"]["v"], one["noiseless"]["average"]),
     integrate(one["noisy"]["v"], one["noisy"]["average"]), im["noisy"]["peak_error"]),
    ("uv: ViSta, extrapolated to b=0", u["noiseless"]["flux"], u["noisy"]["flux"],
     u["noisy"]["flux_error"]),
    ("image: peak pixel (Jy/beam)", integrate(im["noiseless"]["v"], im["noiseless"]["peak"]),
     integrate(im["noisy"]["v"], im["noisy"]["peak"]), im["noisy"]["peak_error"]),
    (f"image: aperture {R_AP:g}\"", integrate(im["noiseless"]["v"], im["noiseless"]["aperture"]),
     integrate(im["noisy"]["v"], im["noisy"]["aperture"]), im["noisy"]["aperture_error"]),
], f"{os.path.basename(gal['path'])} (one galaxy)")
```

```text
galaxy_06.ms (one galaxy)
   line flux [Jy km/s]             without noise        with noise
   true                                    0.395     0.395
   uv: mean over the baselines             0.244     0.281 ± 0.106
   uv: ViSta, extrapolated to b=0          0.395     0.329 ± 0.091
   image: peak pixel (Jy/beam)             0.243     0.282 ± 0.106
   image: aperture 2"                      0.395     0.040 ± 0.325
```

Without noise, the flux extrapolated to zero baseline and a large enough
aperture give the true total; the mean over the baselines and the peak pixel
give less, because the long baselines resolve the galaxy. These two are the
same number: every pixel of a naturally weighted image is a weighted mean of
the visibilities, phased to that pixel, and the peak pixel is the mean over
the baselines phased to the galaxy (compare the spectra above).

With noise, every measurement scatters about its noiseless value by about
its error. The errors of the image are measured on the image itself, away
from the galaxy: the rms of the line-flux map for a pixel, and the scatter of
the same aperture on empty sky. The aperture sums the noise of many beams and
has by far the largest error; the uv fit uses every baseline with its weight.
One galaxy is at a few sigma at best: hence the stack.
