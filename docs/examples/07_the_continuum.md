# 7. The continuum

The continuum removed during the stacking (`MODEL_DATA`) was compressed into
`cont_stats.npy`. `extract_flux` with `method="continuum"` averages it over
the line-free channels and fits its uv profile: flux at zero spacing and
size.

```python
conts = {}
for tag, stats, lst in (("noisy", "cont_stats.npy", LIST),
                        ("noiseless", "cont_stats_noiseless.npy", LIST0)):
    conts[tag] = extract_flux(stats, lst, line, weighting, profile=ProfileConfig(method="continuum"),
                              uvfit=UVFitConfig(model="gauss"), bootstrap=boot,
                              output_prefix=f"continuum_{tag}", verbose=False)
cs, cs0 = conts["noisy"].summary, conts["noiseless"].summary
print(f"continuum flux [mJy]   true {T['cont'] * 1e3:.3f}   without noise {cs0['flux_total'] * 1e3:.3f}"
      f"   with noise {cs['flux_total'] * 1e3:.3f} ± {cs['flux_total_error'] * 1e3:.3f}")
print(f"continuum size [\"]     true {T['size']:.2f}    without noise {cs0['theta_fwhm_arcsec']:.2f}"
      f"    with noise {cs['theta_fwhm_arcsec']:.2f} ± {cs['theta_error_arcsec']:.2f}")

fig, ax = plt.subplots(figsize=(6.5, 4.2))
for tag, mk in (("noisy", "o"), ("noiseless", "s")):
    c = conts[tag]
    ok = np.isfinite(c.flux) & c.bin_ok
    ax.errorbar(c.baseline_klambda[ok], c.flux[ok] * 1e3, c.flux_error[ok] * 1e3 if tag == "noisy" else None,
                fmt=mk, color=(NOISY if tag == "noisy" else CLEAN)["color"], ms=5, capsize=2,
                mfc="none" if tag == "noiseless" else None,
                label="with noise" if tag == "noisy" else "without noise")
g = simulate.shape_visibility(B_TRUE * 1e3, cs["theta_fwhm_arcsec"])
ax.plot(B_TRUE, cs["flux_total"] * g * 1e3, color="#c0392b", lw=1.4, label="fit (with noise)")
ax.plot(B_TRUE, simulate.true_uv(truth, B_TRUE, "continuum") * 1e3, label="true signal", **TRUE)
ax.axhline(0, color="0.5", lw=0.5)
ax.set_xscale("log"); ax.set_xlabel(r"baseline [k$\lambda$]"); ax.set_ylabel("continuum [mJy]")
ax.legend()
save(fig, "continuum_uv")
```

```text
continuum flux [mJy]   true 0.200   without noise 0.200   with noise 0.218 ± 0.040
continuum size ["]     true 0.80    without noise 0.80    with noise 0.80 ± 0.27
```

```{image} figures/07_01.png
:width: 100%
```
