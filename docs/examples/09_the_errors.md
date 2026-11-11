# 9. The errors

The noise is not taken from the weights of the files: it is measured from
the scatter of the visibilities in every channel and annulus, and so is the
correlation between adjacent channels. `r.plot("noise")` shows the checks:

* left: the noise per channel from the scatter and from the weights. Here
  the weights are 30 % too optimistic (we simulated them so);
* centre: the stacked values away from the line, divided by their noise:
  they follow the bell curve if the noise is right;
* right: the same search for a line repeated on pure noise (grey), and the
  S/N measured (red line): how unlikely it is to be noise.

```python
r.plot("noise")
```

```{image} figures/09_01.png
:width: 100%
```

Trusting the weights instead (`noise="weights"`) gives the same flux, but an
error that is too small and a significance that is too high:

```python
v.extract(**base, noise="weights")
from_weights = v.run(only="measure", tag="weights")

vista.compare({"noise from the data (default)": r, "noise from the weights": from_weights},
              expected=expected)
```

```text
[ViSta      12.8s] WARNING: measure: the imaginary part scatters 1.20 x the noise of the visibilities: centring or calibration effects, the thermal errors are optimistic there
```

```text
[ViSta      12.8s] WARNING: measure: the real part in the line-free channels scatters 1.32 x the noise of the visibilities: residual continuum or baseline effects, the thermal errors are optimistic there
```

```text
                                      flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
noise from the data (default)       0.3656   0.054   gauss   0.3656   0.3509    -26   267   0.67  11.8   11.4σ
noise from the weights              0.3656   0.051   gauss   0.3656   0.3509    -26   267   0.67  15.8   15.5σ
expected                          0.3932   continuum 0.000197
```

```{image} figures/09_02.png
:width: 100%
```
