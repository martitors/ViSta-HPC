# 4. The measurement

`v.extract()` asks for the measurement, here with all its defaults;
`v.run(only="measure")` makes it on the stack that is already there. It

1. compresses the stack into sums per channel and per *annulus* of baseline
   length (once: every later measurement reads only these sums and takes
   seconds);
2. brings every galaxy to the same redshift (`z_ref`, the median) and
   corrects it for the primary beam;
3. fits all channels and all annuli at once with a line that is a Gaussian
   in velocity, from a source that is a point or a Gaussian on the sky.

Printing the result gives what was measured **and the choices made**.

```python
v.extract()
r = v.run(only="measure")
print(r)
```

```text
[ViSta      12.6s] z_ref = 2.2867 (median of the sample)
```

```text
Line       345.796 GHz, 12 sources: detected at 11.4 sigma (S/N 11.8, searched over centre and width)
  flux     0.3656 ± 0.054 Jy km/s, the area of the Gaussian fit
           because the area agrees with the Gaussian (-0.0 sigma)
           Gaussian fit 0.3656 ± 0.054; area in the window (-500 to +500 km/s) 0.3649 ± 0.068
  Gaussian centre -26 ± 11 km/s, FWHM 267 ± 32 km/s, peak 1.29 ± 0.19 mJy
  source   Gaussian, FWHM 0.67 ± 0.12 arcsec
           because it fits better than a point source: delta chi2 = 18.1 (at least 9 needed to call it extended)
Continuum  subtracted while stacking and measured on its own: 0.177 ± 0.041 mJy, FWHM 1.16 arcsec
           outside the line window: 0.00044 ± 0.0036 mJy (+0.1 sigma): nothing left
Reference  natural weighting, fluxes at z_ref = 2.2867 (median of the sample)
```

How to read it:

* **Line**: how sure the detection is. The centre and the width were
  searched, so the significance is lower than the S/N: it accounts for the
  search.
* **flux**: the number to quote, and where it comes from. Two fluxes are
  always measured: the **area of the Gaussian fit**, and the **area under
  the spectrum in the window** (a plain sum of the channels). The Gaussian
  one is quoted unless the two disagree, which would mean that the line is
  not a Gaussian (§12).
* **Gaussian**: centre, width and peak of the fitted line.
* **source**: the model of the sky that was chosen and why. A Gaussian
  source is used only if it fits better than a point source by a clear
  margin (Δχ² ≥ 9, i.e. 3σ).
* **Continuum**: what was subtracted (measured on its own) and whether
  anything is left outside the line window.
* **Reference**: the weighting and the redshift the fluxes refer to.

**The spectrum.** `r.plot("profile")` draws the stacked spectrum (black)
with its noise (grey band), the fitted Gaussian (red, its area hatched), the
window where the spectrum is summed (shaded) and the continuum level left
(dashed). The box repeats the two fluxes and says which is which. Below, the
residuals and the imaginary part (crosses) in units of the noise: both must
scatter around zero.

```python
r.plot("profile")
```

```{image} figures/04_01.png
:width: 100%
```

**The source.** `r.plot("uv")` draws the line flux measured in every annulus
of baseline length. A point source would give the same flux on all of them;
here it falls on the long baselines: the galaxies are extended. The model
(the thick curve) follows the points and its value on the left, at zero
baseline, is the total flux.

```python
r.plot("uv")
```

```{image} figures/04_02.png
:width: 100%
```
