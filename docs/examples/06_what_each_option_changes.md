# 6. What each option changes

The statistics are a few MB: every variant takes seconds. Each one changes
**one** option of `extract()` with respect to the measurement of §4 (centre
and width known).

| Option | What it does | When |
|---|---|---|
| `model="point"` | point source only | the source is known to be unresolved |
| `model="gauss"` | Gaussian only | clearly resolved |
| `model_prior=(θ, σ_dex)` | soft prior on the size | a weak line, with a size from the continuum or another line |
| `model=θ` | size fixed | the same, a hard version |
| `noise="weights"` | noise from the weights, not the scatter | weights known to be absolute |
| `channel_correlation=False` | channels treated as independent | (shown to see the effect) |
| `weighting_scheme="democratic"` | every galaxy counts the same | the mean of the galaxies rather than of the data |
| `z_norm=False` | no transport to z_ref | a narrow redshift range |
| `pb_corr=False` | no primary-beam correction | data already corrected |
| `fwhm_range=(w, w)` | the assumed width | |

```python
def variant(**options):
    # one option changed with respect to §4; the Gaussian flux, to see how it changes the fit
    stack.extract(**{**MEASURE, **KNOWN, "flux_measure": "gauss", "n_null": 200,
                     "n_null_phase": 0, "n_bootstrap": 100, "seed": 1, "figures_draw": False,
                     **options})
    return stack.run(only="measure", tag="variant").line


variants = {
    "reference (model auto)": variant(),
    "model = point": variant(model="point"),
    "model = gauss": variant(model="gauss"),
    "model prior 0.8\" ± 0.15 dex": variant(model_prior=(0.8, 0.15)),
    "model: size fixed 0.8\"": variant(model=0.8),
    "noise from the weights": variant(noise="weights"),
    "no channel correlation": variant(channel_correlation=False),
    "democratic weighting": variant(weighting_scheme="democratic"),
    "no redshift rescaling": variant(z_norm=False),
    "no primary-beam correction": variant(pb_corr=False),
    "assumed FWHM 200 km/s": variant(fwhm_range=(200, 200)),
    "assumed FWHM 400 km/s": variant(fwhm_range=(400, 400)),
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
[ViSta       1.5s] z_ref = 2.6218 (median of the sample)
```

```text
[ViSta       0.4s] WARNING: measure: the line is not Gaussian (double peak, wings?): the area and the Gaussian flux differ by +2.1 sigma; quoted: gauss (flux_measure='gauss')
```

```text
[ViSta       0.4s] z_ref = 2.6218 (median of the sample)
```

```text
[ViSta       1.3s] z_ref = 2.6218 (median of the sample)
```

```text
[ViSta       1.3s] z_ref = 2.6218 (median of the sample)
```

```text
[ViSta       1.3s] z_ref = 2.6218 (median of the sample)
```

```text
[ViSta       2.1s] WARNING: measure: the imaginary part scatters 1.30 x the noise of the visibilities: centring or calibration effects, the thermal errors are optimistic there
```

```text
[ViSta       2.1s] WARNING: measure: the real part in the line-free channels scatters 1.17 x the noise of the visibilities: residual continuum or baseline effects, the thermal errors are optimistic there
```

```text
[ViSta       2.1s] WARNING: measure: the line is not Gaussian (double peak, wings?): the area and the Gaussian flux differ by +2.2 sigma; quoted: gauss (flux_measure='gauss')
```

```text
[ViSta       2.1s] z_ref = 2.6218 (median of the sample)
```

```text
[ViSta       1.7s] WARNING: measure: the line is not Gaussian (double peak, wings?): the area and the Gaussian flux differ by +2.9 sigma; quoted: gauss (flux_measure='gauss')
```

```text
[ViSta       1.7s] z_ref = 2.6218 (median of the sample)
```

```text
[ViSta       1.8s] z_ref = 2.6218 (median of the sample)
```

```text
[ViSta       1.5s] no redshift rescaling, no flux normalisation
```

```text
[ViSta       1.6s] z_ref = 2.6218 (median of the sample)
```

```text
[ViSta       1.6s] WARNING: measure: the line is not Gaussian (double peak, wings?): the area and the Gaussian flux differ by +3.3 sigma; quoted: gauss (flux_measure='gauss')
```

```text
[ViSta       1.6s] z_ref = 2.6218 (median of the sample)
```

```text
[ViSta       1.6s] z_ref = 2.6218 (median of the sample)
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
* a **wrong width** fixed in advance biases the Gaussian flux: when the width
  is not known, search it within limits (§5). With `flux_measure="auto"`
  (the default) a width too narrow is caught: the area and the Gaussian flux then
  differ by more than 2σ, and the area flux, which does not depend on the
  width, is quoted. The same test fires, wrongly, when the noise is wrong (from
  the weights, or without the channel correlation): its threshold is in units
  of the noise.
