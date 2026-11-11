# 1. The simulator, and one galaxy

`examples/simulate.py` writes Measurement Sets like those of a real
observation, so that ViSta reads them as it would read yours. For every
galaxy it makes:

* **the telescope**: an ALMA-like array of 43 antennas within `rmax` metres
  of its centre, observing for 20 minutes; the baselines rotate with the
  Earth, and the longer they are the finer the details they see;
* **the line**: a Gaussian in velocity, with integrated flux `line_flux`
  [Jy km/s] and width `fwhm_kms`, at the frequency where the redshift `z`
  puts it;
* **the continuum**: `cont_flux` [Jy], the dust emission under the line;
* **the shape**: a round Gaussian `size_arcsec` across (FWHM). An extended
  source is fainter on the long baselines;
* **the position**: `offset_arcsec` from the centre of the field, where the
  antennas are less sensitive (the *primary beam*);
* **the noise**: `noise` [Jy] on every visibility, correlated between
  adjacent channels as in real data. The weights written in the file
  underestimate it by a factor `kappa`, as real weights often do.

Here is one galaxy, written twice: without noise and with it.

```python
LINE = 345.79599                       # rest frequency of CO(3-2), GHz

galaxy = dict(z=2.5, rest_freq_ghz=LINE,
              line_flux=0.4, fwhm_kms=300,       # the line: 0.4 Jy km/s, 300 km/s wide
              cont_flux=2e-4,                    # the continuum: 0.2 mJy
              size_arcsec=0.8,                   # the shape
              offset_arcsec=(6, 3),              # 6" east and 3" north of the field centre
              noise=0.1, rmax=600)               # the noise and the size of the array

clean = simulate.make_ms("one_clean.ms", noiseless=True, **galaxy)     # without noise
noisy = simulate.make_ms("one_noisy.ms", **galaxy)                     # as observed
```

`simulate.plot_galaxy` shows them side by side. **Top**: the image of the
channel at the centre of the line (a *dirty* image: the Fourier transform of
the visibilities as they are; the circle is the resolution). **Bottom**: the
spectrum at the position of the galaxy, and in green the true line on its
continuum.

Without noise the galaxy is there, at the centre. The observed spectrum is
below the true one because the source is extended: the long baselines see
only part of it. With the noise of a real observation neither the image nor
the spectrum show it: this is the galaxy that stacking is for.

```python
simulate.plot_galaxy(clean, noisy)
```

```{image} figures/01_01.png
:width: 100%
```
