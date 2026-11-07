# 4. Measuring the stacked line at a known width (`detect`)

`measure_line` fits **all channels and all annuli at once** with

    Re V(channel, b) = F · φ(v; v₀, FWHM) · g(b; θ)

φ a Gaussian line profile, g the visibility of a circular Gaussian source of
FWHM θ (1 for a point source). The noise of every cell is measured from the
scatter of its visibilities (not from the weights), the correlation between
channels from the imaginary part; both enter every error.

* `mode="detect"`: centre and width given (from another line, an optical
  redshift...): a matched filter, the most sensitive measurement.
* `mode="search"` (§5): centre and width free; the significance comes from
  running the same search on noise (`n_null` realisations).
* `size="auto"` (default): a point source and a Gaussian are both fitted and
  averaged with their Akaike weights; the error includes the difference.

The galaxies are transported to `z_ref` (redshift rescaling) and corrected
for the primary beam, so the result compares directly with the true values.
The noiseless stack has no scatter to measure the noise from: it is measured
with the noise of the weights, and its errors mean nothing — only its values.

```python
weighting = WeightingConfig(scheme="natural", redshift_rescaling=True, primary_beam=True)
boot = BootstrapConfig(n_realisations=200, seed=1)
NOISELESS = dict(noise="weights", channel_correlation=False, n_null=0, n_null_phase=0)


def compare(r, r0=None, name=""):
    # measured (without and with noise) against true, for the main quantities
    def rows(s):
        ln, win = s["line"], s["window"]
        return [("line flux [Jy km/s]", ln["flux"], ln["flux_error"], T["flux"]),
                ("peak [mJy]", ln["peak"] * 1e3, (ln.get("peak_error") or np.nan) * 1e3, T["peak"] * 1e3),
                ("centre v0 [km/s]", ln["v0_kms"], ln.get("v0_error_kms") or np.nan, T["v0"]),
                ("FWHM [km/s]", ln["fwhm_kms"], ln.get("fwhm_error_kms") or np.nan, T["fwhm"]),
                ("size [arcsec]", ln["theta_fwhm_arcsec"],
                 s.get("theta_error_arcsec") or ln.get("theta_error_arcsec") or np.nan, T["size"]),
                ("window flux [Jy km/s]", win["flux"], win["flux_error"], T["flux"])]
    det = r.summary["detection"]
    print(f"{name}   detection S/N {det['snr']:.1f}"
          + (f", {det['significance_sigma']:.1f} sigma" if np.isfinite(det["significance_sigma"]) else ""))
    print(f"   {'':24s} {'true':>8s} {'no noise' if r0 else '':>9s} {'with noise':>17s} {'(m-t)/err':>10s}")
    clean_rows = rows(r0.summary) if r0 else [None] * 6
    for (lab, m, e, t), c in zip(rows(r.summary), clean_rows):
        err = f"± {e:.3g}" if np.isfinite(e) and e > 0 else "(fixed)"
        pull = f"{(m - t) / e:+.1f}" if np.isfinite(e) and e > 0 else ""
        c0 = f"{c[1]:9.3f}" if c else f"{'':9s}"
        print(f"   {lab:24s} {t:8.3f} {c0} {m:8.3f} {err:>8s} {pull:>10s}")


det = measure_line("line_stats.npy", LIST, line, weighting,
                   LineFitConfig(mode="detect", v0_kms=0.0, fwhm_kms=SIM["fwhm_kms"],
                                 n_null=500, n_null_phase=100),
                   boot, output_prefix="detect")
det0 = measure_line("line_stats_noiseless.npy", LIST0, line, weighting,
                    LineFitConfig(mode="detect", v0_kms=0.0, fwhm_kms=SIM["fwhm_kms"], **NOISELESS),
                    BootstrapConfig(n_realisations=0), output_prefix="detect_noiseless", verbose=False)
print()
compare(det, det0, "detect")
```

```text

detect   detection S/N 11.0, 10.9 sigma
                                true  no noise        with noise  (m-t)/err
   line flux [Jy km/s]         0.400     0.400    0.369 ± 0.0437       -0.7
   peak [mJy]                  1.253     1.253    1.156  ± 0.137       -0.7
   centre v0 [km/s]            0.000     0.000    0.000  (fixed)
   FWHM [km/s]               300.000   300.000  300.000  (fixed)
   size [arcsec]               0.800     0.800    0.771  ± 0.123       -0.2
   window flux [Jy km/s]       0.400     0.400    0.438 ± 0.0641       +0.6
```

`measure_line` wrote `detect_results.json` (every number), `_profile.txt`,
`_uvamp.txt`, `_noise.txt` and `_diagnostics.npz`; `examples/plot_line.py`
draws the figures from them, with the true signal as a `reference` curve,
and the noiseless measurement is added on top.

* **Profile**: the flux density at zero spacing in every channel (the total
  flux density of the stack), the fit, ±1σ noise, the imaginary part (a null
  test: noise only) and the residuals.
* **Line flux per annulus** against baseline length: flat for a point
  source, falling for a resolved one; its value at zero is the total flux.
* **Noise**: per channel from the scatter of the visibilities and from the
  weights (their ratio, κ, was 1.3 in the simulation), the standardised cells
  against N(0,1), and the null distribution of the detection S/N.

```python
V_TRUE = np.linspace(-900, 900, 721)
B_TRUE = np.geomspace(5, 800, 200)
REF_PROFILE = (V_TRUE, simulate.true_profile(truth, V_TRUE))
REF_UV = (B_TRUE, simulate.true_uv(truth, B_TRUE))


def add_noiseless(fig, d0, kind):
    # the noiseless measurement on top of a plot_line figure
    f, _ = pl.flux_scale("Jy", d0)
    ax = fig.axes[0]
    if kind == "profile":
        p = d0["prof"]
        o = np.argsort(p["v_kms"])
        ax.step(p["v_kms"][o], p["flux_density"][o] * f, where="mid", label="without noise",
                **dict(CLEAN, lw=2.2, alpha=0.8, zorder=1.5))      # under the true curve
    else:
        u = d0["uv"]
        ok = (u["used"] == 1) & np.isfinite(u["line_flux"])
        ax.plot(u["b_klambda"][ok], u["line_flux"][ok] * f, "s", mfc="none", color=CLEAN["color"],
                ms=6, label="without noise")
    ax.legend(loc="upper right")
    return fig


d, d0 = pl.load("detect"), pl.load("detect_noiseless")
save(add_noiseless(pl.plot_profile(d, unit="Jy", reference=REF_PROFILE), d0, "profile"),
     "detect_profile")
save(add_noiseless(pl.plot_uv(d, unit="Jy", reference=REF_UV), d0, "uv"), "detect_uv")
save(pl.plot_noise(d, unit="Jy"), "detect_noise")
```

```{image} figures/04_01.png
:width: 100%
```

```{image} figures/04_02.png
:width: 100%
```

```{image} figures/04_03.png
:width: 100%
```

```python
n = det.summary["noise"]
print(f"noise / weights (kappa):  measured {n['kappa_scatter_over_weights']:.3f}   true {SIM['kappa']}")
rho = n["channel_correlation"]
print(f"channel correlation:      measured {rho[0]:+.2f} {rho[1]:+.2f} {rho[2]:+.2f}   "
      f"true (Hanning) {'+0.67 +0.17 +0.00' if SIM['hanning'] else '0 0 0'}")
print(f"imaginary rms / noise:    {n['imaginary_rms_over_noise']:.3f}   (1 = noise only)")
print(f"line-free rms / noise:    {n['line_free_rms_over_noise']:.3f}   (1 = no residual continuum)")
st = det.summary["size_test"]
print(f"size test: chi2(point) - chi2(Gaussian) = {st['delta_chi2_point_minus_gauss']:.1f} "
      f"(correlation factor {st['correlation_factor']:.2f}): adopted {st['adopted']}, "
      f"Gaussian weight {st['weight_gauss']:.2f}")
```

```text
noise / weights (kappa):  measured 1.303   true 1.3
channel correlation:      measured +0.64 +0.10 -0.08   true (Hanning) +0.67 +0.17 +0.00
imaginary rms / noise:    1.038   (1 = noise only)
line-free rms / noise:    0.932   (1 = no residual continuum)
size test: chi2(point) - chi2(Gaussian) = 23.0 (correlation factor 3.84): adopted gauss, Gaussian weight 1.00
```

Without noise the stack returns the true values: the stacking, the
rescaling, the primary-beam correction and the fit add no bias. With noise
the measurement scatters around them by about its error. The noise from the
scatter recovers the factor by which the weights are wrong, and the channel
correlation that of Hanning smoothing; the imaginary part and the line-free
channels are at the noise level.
