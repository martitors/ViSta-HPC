# 8. The errors

The noise of every channel and annulus is measured from the scatter of the
visibilities, and the correlation between adjacent channels from the data.
The figure of the noise: per channel, from the scatter and from the weights
(30 % lower here: the weights are wrong); the standardised noise against
N(0, 1); the search on noise that gives the significance.

```python
r.plot("noise")
```

```{image} figures/08_01.png
:width: 100%
```

Noise from the weights (which misjudge it here), or channels taken as
independent: the same fluxes, but errors too small and a significance far
too high.

```python
vista.compare({
    "default": r,
    "noise from the weights": measure(noise="weights"),
    "channels independent": measure(channel_correlation=False),
}, truth=T)
```

```text
[ViSta       9.6s] WARNING: measure: the imaginary part scatters 1.25 x the noise of the visibilities: centring or calibration effects, the thermal errors are optimistic there
```

```text
[ViSta       9.6s] WARNING: measure: the real part in the line-free channels scatters 1.25 x the noise of the visibilities: residual continuum or baseline effects, the thermal errors are optimistic there
```

```text
                               flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
default                      0.3804    0.06   gauss   0.3804   0.3732    +16   307   0.73  10.2    9.7σ
noise from the weights       0.3805   0.058   gauss   0.3805   0.3733    +16   307   0.73  13.6   13.3σ
channels independent         0.3806   0.056   gauss   0.3806   0.3733    +16   307   0.73  21.3   21.0σ
true                       0.4   continuum 0.0002
```

```{image} figures/08_02.png
:width: 100%
```
