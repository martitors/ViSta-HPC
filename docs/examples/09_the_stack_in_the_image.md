# 9. The stack in the image, without and with noise

The stack exported as a Measurement Set (`export_ms`) and imaged as in §2.
The stacked visibilities are the plain weighted mean of the galaxies: no
redshift rescaling and no primary-beam correction, which ViSta applies only
when it measures. For a like-for-like comparison, ViSta is run here without
them too (`extract(z_norm=False, pb_corr=False)`). The true signal is the noiseless stack: the same visibilities
without noise.

```python
stack_img, plain = {}, {}
for tag, v in (("noisy", stack), ("noiseless", stack0)):
    store = v.files["stack"]
    export_ms(store, f"{store[:-5]}.ms", verbose=False)
    stack_img[tag] = image_line(f"{store[:-5]}.ms", f"img_{tag}", noiseless=tag == "noiseless")
    extra = NOISELESS if tag == "noiseless" else dict(n_null=0, n_null_phase=0, n_bootstrap=100)
    v.extract(annuli=ANNULI, z_norm=False, pb_corr=False, **KNOWN, **extra)
    plain[tag] = v.run(only="measure", tag="plain").line.summary["line"]

fig = plt.figure(figsize=(12.5, 8))
vmax = np.nanmax(stack_img["noiseless"]["mom0"])
for j, tag in enumerate(("noiseless", "noisy")):
    im_ = stack_img[tag]
    ax = fig.add_subplot(2, 2, j + 1)
    im = ax.imshow(im_["mom0"].T, origin="lower", extent=(ext, -ext, -ext, ext), cmap="inferno",
                   vmin=-0.3 * vmax, vmax=1.1 * vmax)
    fig.colorbar(im, ax=ax, label="Jy/beam km s$^{-1}$")
    ax.add_patch(Circle((0, 0), SIM["size_arcsec"] / 2, fill=False, ec="#2ecc71", lw=1.5, ls="--",
                        label="true size (FWHM)"))
    bmaj, bmin, pa = im_["beam"]
    ax.add_patch(Ellipse((-3.0, -3.0), bmin, bmaj, angle=-pa, fill=False, ec="w", label="beam"))
    ax.add_patch(Circle((0, 0), R_AP, fill=False, ec="w", lw=0.8, ls=":", label=f"aperture {R_AP:g}\""))
    ax.set_xlim(4, -4); ax.set_ylim(-4, 4)
    ax.set_xlabel(r"$\Delta$RA [arcsec]"); ax.set_ylabel(r"$\Delta$Dec [arcsec]")
    ax.set_title(("without noise" if tag == "noiseless" else "with noise") + " (stack)")
    if j == 0:
        ax.legend(loc="upper left", fontsize=8)
for j, (key, unit) in enumerate((("peak", "Jy/beam"), ("aperture", "Jy"))):
    ax = fig.add_subplot(2, 2, 3 + j)
    for tag, sty in (("noisy", NOISY), ("noiseless", CLEAN)):
        im_ = stack_img[tag]
        ax.step(im_["v"], im_[key] * 1e3, where="mid",
                label="with noise" if tag == "noisy" else "without noise (the true stack)", **sty)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlim(-900, 900)
    ax.set_xlabel("velocity [km s$^{-1}$]")
    ax.set_ylabel(f"{'peak pixel' if key == 'peak' else f'aperture {R_AP:g}″'} [m{unit}]")
    ax.legend(loc="upper left")
fig.tight_layout()
save(fig, "stack_image")

im = stack_img
avg = {tag: baseline_average(store) for tag, store in (("noisy", "stack.zarr"),
                                                       ("noiseless", "stack_noiseless.zarr"))}
flux_table([
    ("uv: mean over the baselines", integrate(avg["noiseless"][0], avg["noiseless"][1]["CORRECTED_DATA"]),
     integrate(avg["noisy"][0], avg["noisy"][1]["CORRECTED_DATA"]), im["noisy"]["peak_error"]),
    ("uv: ViSta, extrapolated to b=0", plain["noiseless"]["flux"],
     plain["noisy"]["flux"], plain["noisy"]["flux_error"]),
    ("image: peak pixel (Jy/beam)", integrate(im["noiseless"]["v"], im["noiseless"]["peak"]),
     integrate(im["noisy"]["v"], im["noisy"]["peak"]), im["noisy"]["peak_error"]),
    (f"image: aperture {R_AP:g}\"", integrate(im["noiseless"]["v"], im["noiseless"]["aperture"]),
     integrate(im["noisy"]["v"], im["noisy"]["aperture"]), im["noisy"]["aperture_error"]),
], "the stack, no rescaling (without noise = the true stack)")
```

```text
[ViSta       1.5s] no redshift rescaling, no flux normalisation
```

```text
[ViSta       0.3s] no redshift rescaling, no flux normalisation
```

```{image} figures/09_01.png
:width: 100%
```

```text
the stack, no rescaling (without noise = the true stack)
   line flux [Jy km/s]             without noise        with noise
   uv: mean over the baselines             0.267     0.312 ± 0.042
   uv: ViSta, extrapolated to b=0          0.438     0.394 ± 0.043
   image: peak pixel (Jy/beam)             0.266     0.309 ± 0.042
   image: aperture 2"                      0.435     0.412 ± 0.143
```

As for one galaxy: the mean over the baselines and the peak pixel are the
same number, below the total because the stack is resolved; the flux
extrapolated to zero baseline and the aperture recover the total of the
noiseless stack. With noise, the uv fit and the aperture find about the same
flux, but the error of the aperture is several times larger: it sums the
noise of every pixel in it, while the fit weighs every baseline by what it
knows about the source.

The image shows where the emission is and whether it is symmetric, which the
uv fit assumes: the two are complementary. For the flux of a resolved stack
the uv measurement needs no aperture and no deconvolution.
