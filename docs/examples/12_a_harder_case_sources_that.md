# 12. A harder case: sources that are not Gaussians

A third sample: every galaxy is a **ring** 3″ across. ViSta models the
stacked source as a point or a round Gaussian: what happens when it is
neither?

```python
truth_ring = simulate.make_sample(
    "ring", n_sources=12, z_range=(2.0, 3.0), rest_freq_ghz=LINE,
    line_flux=0.6, fwhm_kms=300, size_arcsec=3.0, shape="ring",     # a ring 3" across
    noise=0.04, rmax=600, seed=5)

g = ViSta(truth_ring["input_list"], output_name="ring")
g.stack(line_freq=LINE, line_window=500, velocity_range=900)
g.extract(window=320)
ring = g.run()
ring.plot("image")
```

```text
[ViSta      17.5s] z_ref = 2.3959 (median of the sample)
```

```{image} figures/12_01.png
:width: 100%
```

The image of the stack shows the ring. In the uv plane a ring has a
signature: its flux goes **negative** on some baselines, which no Gaussian
can do. The fitted Gaussian follows the short baselines, which set the total
flux, and misses the rest. The flux is still right within its error, because
it is decided at short baselines; the size is that of "a Gaussian as wide as
the ring", not a property of the galaxies.

The lesson for real data: look at the uv figure and at the image. Points
that go systematically below the model, or below zero, say that the source
is not what the model assumes.

```python
vista.compare({"Gaussian source model": ring}, expected=simulate.expected(truth_ring))
```

```text
                              flux   ± err  quoted    Gauss     area     v0  FWHM   size   S/N signif.  continuum
Gaussian source model       0.6574   0.072   gauss   0.6574   0.6437     -9   319   3.27  13.0   12.7σ
expected                  0.6
```

```{image} figures/12_02.png
:width: 100%
```
