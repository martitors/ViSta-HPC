# 12. A source that is not a Gaussian

Every galaxy of this sample is a **ring** 3″ across. ViSta models the
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
[ViSta       0.4s] z_ref = 2.3959 (median of the sample)
```

```{image} figures/12_01.png
:width: 100%
```

The image of the stack shows the ring. Against the baseline length a ring
has a signature: its flux goes **negative** on some baselines, which no
Gaussian can do. The fitted Gaussian follows the short baselines, which set
the total flux, and misses the rest. The flux is still about right, because
it is decided at the short baselines; the size is that of "a Gaussian as
wide as the ring", not a property of the galaxies.

For real data: look at this plot. Points that go systematically below the
model, or below zero, say that the source is not what the model assumes.

```python
ring.plot("line_uv")
```

```{image} figures/12_02.png
:width: 100%
```

```python
print("measured:", flux(ring))
print("expected:", f"{simulate.expected(truth_ring)['flux'][0]:.3f} Jy km/s")
```

```text
measured: 0.657 ± 0.072 Jy km/s
expected: 0.600 Jy km/s
```
