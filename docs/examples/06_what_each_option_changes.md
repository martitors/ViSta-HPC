# 6. What each option changes

The statistics are a few MB: every variant takes seconds. Each one changes
**one** thing with respect to the `detect` measurement of §4.

| Option | What it does | When |
|---|---|---|
| `size="point"` | point source only | the source is known to be unresolved |
| `size="gauss"` | Gaussian only | clearly resolved |
| `theta_prior_arcsec=(θ, σ_dex)` | soft prior on the size | a weak line, with a size from the continuum or another line |
| `theta_fixed_arcsec=θ` | size fixed | the same, a hard version |
| `noise="weights"` | noise from the weights, not the scatter | weights known to be absolute |
| `channel_correlation=False` | channels treated as independent | (shown to see the effect) |
| `scheme="democratic"` | every galaxy counts the same | the mean of the galaxies rather than of the data |
| `redshift_rescaling=False` | no transport to z_ref | a narrow redshift range |
| `primary_beam=False` | no primary-beam correction | data already corrected |
| `fwhm_kms` in `detect` | the assumed width | |

```python
def variant(weighting=weighting, **fit):
    cfg = dict(mode="detect", v0_kms=0.0, fwhm_kms=SIM["fwhm_kms"], n_null=200, n_null_phase=0)
    cfg.update(fit)
    return measure_line("line_stats.npy", LIST, line, weighting, LineFitConfig(**cfg),
                        BootstrapConfig(n_realisations=100, seed=1), verbose=False)


variants = {
    "reference (size auto)": variant(),
    "size = point": variant(size="point"),
    "size = gauss": variant(size="gauss"),
    "size prior 0.8\" ± 0.15 dex": variant(theta_prior_arcsec=(0.8, 0.15)),
    "size fixed 0.8\"": variant(theta_fixed_arcsec=0.8),
    "noise from the weights": variant(noise="weights"),
    "no channel correlation": variant(channel_correlation=False),
    "democratic weighting": variant(WeightingConfig(scheme="democratic")),
    "no redshift rescaling": variant(WeightingConfig(redshift_rescaling=False)),
    "no primary-beam correction": variant(WeightingConfig(primary_beam=False)),
    "assumed FWHM 200 km/s": variant(fwhm_kms=200.0),
    "assumed FWHM 400 km/s": variant(fwhm_kms=400.0),
}

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True,
                                 gridspec_kw={"width_ratios": [1.3, 1, 0.8]})
names = list(variants)
y = np.arange(len(names))[::-1]
for yi, name in zip(y, names):
    s = variants[name].summary
    ln = s["line"]
    a1.errorbar(ln["flux"], yi, xerr=ln["flux_error"], fmt="o", color="k", ms=5, capsize=2)
    te = s.get("theta_error_arcsec") or ln.get("theta_error_arcsec") or 0
    a2.errorbar(ln["theta_fwhm_arcsec"], yi, xerr=te, fmt="o", color="k", ms=5, capsize=2)
    a3.barh(yi, s["detection"]["snr"], color="0.6", height=0.6)
a1.axvline(T["flux"], label="true signal", **TRUE)
a2.axvline(T["size"], **TRUE)
a1.set_yticks(y); a1.set_yticklabels(names)
a1.set_xlabel("line flux [Jy km s$^{-1}$]"); a2.set_xlabel("size [arcsec]")
a3.set_xlabel("detection S/N")
a1.legend()
fig.tight_layout()
save(fig, "options")
```

```text
[ViSta       1.8s] WARNING: measure: the imaginary part scatters 1.30 x the noise of the visibilities: centring or calibration effects, the thermal errors are optimistic there
```

```text
[ViSta       1.8s] WARNING: measure: the real part in the line-free channels scatters 1.17 x the noise of the visibilities: residual continuum or baseline effects, the thermal errors are optimistic there
```

```{image} figures/06_01.png
:width: 100%
```

What the figure shows (the numbers are those of the default simulation):

* a **point-source model** keeps only part of the flux of this 0.8″ source —
  any extended emission is lost the same way. A size prior or a fixed size
  helps when the line is weak and the size known from elsewhere;
* with the **noise from the weights** (wrong by κ = 1.3) the flux error is
  rescaled by √χ²ᵣ and survives, but the detection S/N is overstated by κ —
  and ViSta warns that the imaginary part and the line-free channels scatter
  more than that noise;
* **without the channel correlation** the errors are too small and the S/N
  much too high: never on Hanning-smoothed data;
* **democratic** weighting gives the shallow datasets the weight of the deep
  ones: a larger error;
* **without the redshift rescaling** the stack is the mean of galaxies at
  different distances, not the flux at `z_ref`; **without the primary-beam
  correction** it is a little low;
* a **wrong width** in `detect` mode biases the flux: when the width is not
  known, use `search` (§5).
