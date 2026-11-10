# Validation

How well the measurement works, on simulations where the true line is known.
The [Examples](../examples/index) show the same on a complete simulated
sample, step by step, against the noiseless twin of every dataset.

## Point and resolved sources

Synthetic statistics built cell by cell as the compression builds them: 20
sources, weights wrong by κ = 1.3, channel correlation 0.3; 50–100
realisations per case. *Scatter* is the rms of the flux over the true flux,
*coverage* the fraction of realisations whose quoted ±1σ contains the truth
(ideal 0.68), *noise > 2σ* the fraction of pure-noise stacks with F/σ > 2
(ideal 0.023).

| Method | noise > 2σ | weak (S/N ≈ 3): bias, scatter, coverage | medium (S/N ≈ 6): scatter, coverage | resolved (θ = 1″): bias |
|---|---|---|---|---|
| window ±600 (Sec. 5.3) | 0.07 | −6%, 0.62, 0.59 | 0.32, 0.55 | −61% |
| template, point | 0.09 | −4%, 0.48, 0.65 | 0.24, 0.58 | −69% |
| template, gauss | 0.03 | +128%, 9.0, 0.90 | 0.27, 0.92 | −4% |
| joint, centre and width free | p < 0.05 in 2% | −11%, 0.53, 0.56 | 0.23, 0.73 | +4% |
| **joint, centre and width fixed** (known) | 0.00 | −6%, **0.31**, 0.70 | **0.15**, 0.72 | +2% |
| joint, area flux ±600 | 0.02 | −8%, 0.55, 0.65 | 0.29, 0.62 | +6% |

The errors of the per-annulus chain (bootstrap only) are too small for a
point model (noise gives F/σ > 2 three to four times too often) and too large
for a free Gaussian, whose flux also runs away at low S/N. The joint fit with
a known width halves the scatter of the template method; a free search is
reliable from about 4σ. On whole Measurement Sets (8 simulated ALMA datasets,
Hanning-smoothed noise, weights wrong by a factor 52, continuum subtracted on
18 line-free channels, 16 noise realisations), the measured scatter of the
flux is 71 against 69 quoted (fixed shape) and 108 against 113 quoted (area);
with only the first lags of the channel correlation, the quoted errors were
50 and 70.

## Extended and non-ideal sources

Same simulations, the signal of every cell averaged over the baselines of its
annulus; bias of the total flux and, in brackets, the coverage of the quoted
±1σ.

| Case (S/N) | template, point | template, gauss | joint, fixed shape | joint, free |
|---|---|---|---|---|
| Gaussian 1″, 40 sources (8) | −69% | −4% (0.66) | +3% (0.68) | +5% (0.68) |
| the same, weak (4) | −64% | +18%, scatter 1.25 (0.77) | +8%, scatter 0.62 (0.83) | +10% (0.65) |
| weak, size prior from elsewhere (±0.15 dex) | | | +4%, scatter 0.33 (0.65) | +1% (0.63) |
| sizes from 0.5″ to 1.5″ (8) | −62% | −7% (0.66) | −7% (0.74) | −5% (0.72) |
| point sources, 0.3″ position errors (8) | −54% | +16% (0.82) | +4% (0.82) | +9% (0.66) |
| rotating disk, velocity gradient over 1″ (8) | −62% | −16% (0.58) | +1% (0.72) | −7% (0.64) |
| exponential disk (8) | −68% | −10% (0.44) | −4% (0.62) | −1% (0.54) |
| double-horn profile, point source (8) | 0% (0.66) | +4% (0.70) | +9% (0.60) | +3% (0.76) |
| Gaussian 1″, bright (25) | −68% | 0% (0.57) | +1% (0.60) | +2% (0.60) |

A point model loses 55–70% of the flux of anything extended — including
point sources stacked with 0.3″ position errors, which add up to an extended
stack. The joint fit recovers the total flux within a few per cent when the
source is resolved at S/N ≳ 8, also for an exponential or a rotating disk,
which the Gaussian model does not describe exactly. For a weak resolved stack
the total flux is uncertain because the size is: the averaged model gives an
honest, large error, and a size prior halves it. With the Gaussian shape fixed,
a double-horn profile comes out 9% high: there the area flux is the one to
quote, and `flux_measure="auto"` quotes it when the difference is significant. On whole Measurement Sets with a 1″ source (8 datasets, 12
noise realisations): F = 211 ± 18 against 213 true, size 0.95″ ± 0.24″, while
the point-source flux is 78.

These tables were made before the size test was corrected for the channel
correlation. With the correlation of these simulations (0.3 at lag 1, a
factor of about 1.6 on the χ² difference) the correction changes them within
their noise — on 50 new realisations each, before → after: point source,
bias +0.8% → −0.1%, scatter 0.21 → 0.20; Gaussian 1″ at S/N 8, +3.5% → +3.2%,
coverage 0.68 → 0.70; a very weak 1″ source (S/N ≈ 2), +15% → +9%, coverage
0.78 → 0.68. It matters with Hanning smoothing and a continuum subtraction,
where the factor is 4–5.

## uv plane or image plane

The same stacks measured both ways: 8 simulated ALMA datasets (43 antennas,
baselines up to ~370 m, beam 1.06″ × 0.76″, Hanning-smoothed noise, weights
wrong by 1.3), a line of 200 km/s FWHM, stacked in one pass; in the image
plane, `tclean` of the stacked visibilities over the line window (±300 km/s,
natural weighting, CLEAN to 2σ in a 4″ mask). Mean of measured / true flux and
its scatter over 6 noise realisations per case (so the scatter is known to
about ±30%):

| Method | point source, S/N ≈ 6 | 1″ source, S/N ≈ 8 | 1″ source, S/N ≈ 5 |
|---|---|---|---|
| uv, joint fit, fixed shape | 1.09 ± 0.16 | **1.05 ± 0.12** | 1.16 ± 0.29 |
| uv, joint fit, free | 1.20 ± 0.28 | 1.04 ± 0.14 | 1.12 ± 0.35 |
| uv, window ±300 km/s | 1.20 ± 0.26 | 0.99 ± 0.17 | 1.09 ± 0.41 |
| uv, point-source fit only | **1.00 ± 0.12** | 0.44 ± 0.05 | 0.48 ± 0.11 |
| image, dirty peak | 1.11 ± 0.17 | 0.42 ± 0.07 | 0.52 ± 0.14 |
| image, aperture 1″ / 2″ / 3″ | 1.14 / 0.77 / 0.82 ± 0.37 / 0.63 / 0.78 | 0.74 / 1.09 / 0.87 ± 0.10 / 0.25 / 0.67 | 0.87 / 0.89 / 0.43 ± 0.35 / 0.63 / 1.41 |
| image, aperture 2″, JvM-corrected | 0.78 ± 0.52 | 0.95 ± 0.22 | 0.76 ± 0.57 |
| image, CLEAN model total | 0.58 ± 0.23 | 0.39 ± 0.40 | −0.03 ± 0.67 |
| image, `imfit` Gaussian | 1.41 ± 1.05 | 1.20 ± 0.25 | 4.2 ± 3.1 (2 of 6 failed) |

- **Point source**: the two planes measure the same thing. With natural
  weighting the dirty image at the source position is the weighted mean of
  the real part of the visibilities, i.e. the uv point-source flux. With
  `model="auto"` the joint fit does not know that the source is unresolved
  and averages in a Gaussian fitted to the noise, which costs some scatter
  (on 50 simulated point sources the bias is 0.0% ± 3%); if the source is
  known to be unresolved, `model="point"`.
- **Resolved source**: the peak and the point-source flux lose 55–60%; in
  the image the total flux needs an aperture, and the result depends on it
  (1″ loses a quarter, 3″ is dominated by noise; the dirty-beam wings make the
  raw aperture sum high, the JvM correction brings it back). CLEAN to 2σ of a
  stack at S/N 5–8 leaves most of the flux in the residual. The joint uv fit
  has no aperture to choose and half the scatter of the best image method.
- **Weak resolved source**: every total flux is uncertain because the size
  is; the uv fit gives the honest error, `imfit` fails or runs away. A size
  from elsewhere (`theta_prior_arcsec`) is the way to tighten it.

This agrees with Lindroos et al. (2015, MNRAS 446, 3502), who find
uv stacking more robust than image stacking, with better size estimates.
[Example 2](../examples/02_one_galaxy_without_and_with) and
[Example 9](../examples/09_the_stack_in_the_image) show the comparison on one
galaxy and on a stack, with and without noise.
