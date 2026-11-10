# 4. Measuring the stacked line at a known width

`v.extract(...)` sets the measurement, `run(only="measure")` runs it: it fits **all channels (within `fit_span`, 1500 km/s, of the line) and all annuli at once** with

    Re V(channel, b) = F · φ(v; v₀, FWHM) · g(b; θ)

φ a Gaussian line profile, g the visibility of a circular Gaussian source of
FWHM θ (1 for a point source). The noise of every cell is measured from the
scatter of its visibilities (not from the weights), the correlation between
channels from the imaginary part; both enter every error.

* `v0_range`, `fwhm_range`: the limits of the centre and of the width. With
  `lo = hi` they are fixed — here the true ones, as if known from another
  line or an optical redshift: a matched filter, the most sensitive
  measurement. Without them (§5) they are searched within the window, and
  the significance comes from running the same search on noise.
* `model="auto"` (default): a point source and a Gaussian are both fitted and
  averaged with their Akaike weights; the error includes the difference.
  How much the Gaussian improves on the point source (Δχ²) is in the title
  of the profile; `theta_range` (0 to 15″) limits its size.
* `flux_measure="auto"` (default): two fluxes are always given, the integral
  of the Gaussian and the sum of the spectrum over the window; the Gaussian
  one is quoted unless the two differ by more than `shape_nsigma` (2σ: the
  line is not Gaussian).

The galaxies are transported to `z_ref` (`z_norm=True`; `z_ref` is the
median of the sample unless given, and the results say which) and corrected
for the primary beam (`pb_corr=True`), so the result compares directly with
the true values.
The noiseless stack has no scatter to measure the noise from: it is measured
with the noise of the weights, and its errors mean nothing — only its values.
`tag` names the measurement: its files are `stack_detect_*`.

```python
# the noiseless stack: the noise of the weights (there is no scatter to measure), no
# null realisations; a few bootstrap realisations, for the continuum (§7)
NOISELESS = dict(noise="weights", channel_correlation=False, n_null=0, n_null_phase=0,
                 n_bootstrap=20)
# the weighting (the defaults, written out) and the annuli of the statistics
MEASURE = dict(annuli=ANNULI, weighting_scheme="natural", z_norm=True, pb_corr=True)


def compare(r, r0=None, name=""):
    # measured (without and with noise) against true, for the main quantities
    def rows(s):
        ln = s["line"]
        return [("line flux [Jy km/s]", ln["flux"], ln["flux_error"], T["flux"]),
                ("  Gaussian", ln["flux_gauss"], ln["flux_gauss_error"], T["flux"]),
                ("  area (window)", ln["flux_area"], ln["flux_area_error"], T["flux"]),
                ("peak [mJy]", ln["peak"] * 1e3, (ln.get("peak_error") or np.nan) * 1e3, T["peak"] * 1e3),
                ("centre v0 [km/s]", ln["v0_kms"], ln.get("v0_error_kms") or np.nan, T["v0"]),
                ("FWHM [km/s]", ln["fwhm_kms"], ln.get("fwhm_error_kms") or np.nan, T["fwhm"]),
                ("size [arcsec]", ln["theta_fwhm_arcsec"],
                 s.get("theta_error_arcsec") or ln.get("theta_error_arcsec") or np.nan, T["size"])]
    det = r.summary["detection"]
    print(f"{name}   detection S/N {det['snr']:.1f}"
          + (f", {det['significance_sigma']:.1f} sigma" if np.isfinite(det["significance_sigma"]) else "")
          + f"; flux quoted: {r.summary['line']['flux_method']}")
    print(f"   {'':24s} {'true':>8s} {'no noise' if r0 else '':>9s} {'with noise':>17s} {'(m-t)/err':>10s}")
    clean_rows = rows(r0.summary) if r0 else [None] * 7
    for (lab, m, e, t), c in zip(rows(r.summary), clean_rows):
        err = f"± {e:.3g}" if np.isfinite(e) and e > 0 else "(fixed)"
        pull = f"{(m - t) / e:+.1f}" if np.isfinite(e) and e > 0 else ""
        c0 = f"{c[1]:9.3f}" if c else f"{'':9s}"
        print(f"   {lab:24s} {t:8.3f} {c0} {m:8.3f} {err:>8s} {pull:>10s}")


stack.extract(**MEASURE, **KNOWN, n_null=500, n_null_phase=100, n_bootstrap=200, seed=1)
det = stack.run(only="measure", tag="detect")
stack0.extract(**MEASURE, **KNOWN, **NOISELESS)
det0 = stack0.run(only="measure", tag="detect")
stack.verbose = False                  # from here on, only the warnings
print()
compare(det.line, det0.line, "centre and width known")
```

```text
[ViSta       0.0s]   redshift reference z_ref = 2.6218 (median of the sample)
```

```text
[ViSta       0.3s]   primary beam correction: A from 0.995 to 0.997
```

```text
[ViSta       0.3s]   12 sources from 12 entries (0 observed more than once), natural weighting, flux normalisation off
```

```text
[ViSta       0.3s]   cube: 1,080 cells (90 channels x 12 annuli), noise from the scatter of the visibilities; channel correlation +0.64, +0.10, -0.08 (measured, whole matrix (imaginary part, 142 noise vectors))
```

```text
[ViSta       0.3s]   search: S/N = 10.99 at v0 = +0 km/s, FWHM = 300 km/s, size 0.57"  ->  p = 3.6e-28 (10.9 sigma, 2 effective trials)
```

```text
[ViSta       3.1s]   noise: scatter/weights kappa = 1.303; imaginary 1.038, line-free 0.932 x the noise (expected 1)
```

```text
[ViSta       3.1s]   written: stack_detect_results.json, stack_detect_profile.txt, stack_detect_uvamp.txt, stack_detect_noise.txt, stack_detect_diagnostics.npz
```

```text
[ViSta       3.1s] Line: F = 0.369 +- 0.0437 (Gaussian; Gaussian 0.369, area 0.4378)  peak = 0.001156  v0 = +0.0 km/s  FWHM = 300 km/s  size gauss 0.77"  chi2r = 0.93
```

```text
[ViSta       3.1s] z_ref = 2.6218 (median of the sample)
```

```text
[ViSta       4.3s]   figures: stack_detect_profile.png, stack_detect_uv.png, stack_detect_noise.png
```

```text
[ViSta       4.3s] Continuum, from the channels outside the line windows (MODEL_DATA):
```

```text
[ViSta       0.0s] Extraction: stack_cont_stats.npy, continuum line profile, gauss uv model, 200 bootstrap realisations
```

```text
[ViSta       0.0s]   redshift reference z_ref = 2.6218 (median of the sample)
```

```text
[ViSta       0.2s]   primary beam correction: A from 0.995 to 0.997
```

```text
[ViSta       0.2s]   12 sources from 12 entries (0 observed more than once), natural weighting, flux normalisation off
```

```text
[ViSta       0.2s]   continuum: 90 channels, effective frequency 345.80 GHz
```

```text
[ViSta       0.6s]   written: stack_detect_continuum_results.json, stack_detect_continuum_spectrum.txt, stack_detect_continuum_uvamp.txt
```

```text
[ViSta       0.6s] Extraction done: F = 0.0002176 +- 3.98e-05   theta = 0.797" +- 0.270"   chi2r = 0.83
```

```text
[ViSta       0.9s]   plot written: stack_detect_continuum_uvamp.png
```

```text
[ViSta       0.5s] z_ref = 2.6218 (median of the sample)
```

```text

centre and width known   detection S/N 11.0, 10.9 sigma; flux quoted: gauss
                                true  no noise        with noise  (m-t)/err
   line flux [Jy km/s]         0.400     0.400    0.369 ± 0.0437       -0.7
     Gaussian                  0.400     0.400    0.369 ± 0.0437       -0.7
     area (window)             0.400     0.400    0.438 ± 0.0641       +0.6
   peak [mJy]                  1.253     1.253    1.156  ± 0.137       -0.7
   centre v0 [km/s]            0.000     0.000    0.000  (fixed)
   FWHM [km/s]               300.000   300.000  300.000  (fixed)
   size [arcsec]               0.800     0.800    0.771  ± 0.123       -0.2
```

The measurement wrote `stack_detect_results.json` (every number),
`_profile.txt`, `_uvamp.txt`, `_noise.txt` and `_diagnostics.npz` (and its
own figures, `stack_detect_profile.png` ...); `examples/plot_line.py` draws
the figures from those files, with the true signal as a `reference` curve,
and the noiseless measurement is added on top.

* **Profile**: the flux density at zero spacing in every channel (the total
  flux density of the stack), the fit and its area (hatched), the window of
  the area flux, ±1σ noise, the imaginary part (a null test: noise only),
  the continuum left outside the window (flagged if significant) and the
  residuals.
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


d, d0 = pl.load("stack_detect"), pl.load("stack_noiseless_detect")
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
n = det.line.summary["noise"]
print(f"noise / weights (kappa):  measured {n['kappa_scatter_over_weights']:.3f}   true {SIM['kappa']}")
rho = n["channel_correlation"]
print(f"channel correlation:      measured {rho[0]:+.2f} {rho[1]:+.2f} {rho[2]:+.2f}   "
      f"true (Hanning) {'+0.67 +0.17 +0.00' if SIM['hanning'] else '0 0 0'}")
print(f"imaginary rms / noise:    {n['imaginary_rms_over_noise']:.3f}   (1 = noise only)")
print(f"line-free rms / noise:    {n['line_free_rms_over_noise']:.3f}   (1 = no residual continuum)")
st = det.line.summary["size_test"]
rc = det.line.summary["continuum_residual"]
print(f"continuum left outside the window: {rc['offset'] * 1e3:+.4f} ± {rc['offset_error'] * 1e3:.4f} mJy "
      f"({rc['offset_nsigma']:+.1f} sigma){' FLAGGED' if rc['detected'] else ''}")
print(f"size test: chi2(point) - chi2(Gaussian) = {st['delta_chi2_point_minus_gauss']:.1f} "
      f"(correlation factor {st['correlation_factor']:.2f}): adopted {st['adopted']}, "
      f"Gaussian weight {st['weight_gauss']:.2f}")
```

```text
noise / weights (kappa):  measured 1.303   true 1.3
channel correlation:      measured +0.64 +0.10 -0.08   true (Hanning) +0.67 +0.17 +0.00
imaginary rms / noise:    1.038   (1 = noise only)
line-free rms / noise:    0.932   (1 = no residual continuum)
continuum left outside the window: +0.0000 ± 0.0035 mJy (+0.0 sigma)
size test: chi2(point) - chi2(Gaussian) = 23.0 (correlation factor 3.84): adopted gauss, Gaussian weight 1.00
```

Without noise the stack returns the true values: the stacking, the
rescaling, the primary-beam correction and the fit add no bias. With noise
the measurement scatters around them by about its error. The noise from the
scatter recovers the factor by which the weights are wrong, and the channel
correlation that of Hanning smoothing; the imaginary part and the line-free
channels are at the noise level.
