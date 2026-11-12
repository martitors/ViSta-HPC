# 13. The continuum alone

Sometimes there is no line at all: a continuum survey, galaxies observed in
the same band whatever their redshift. Ten galaxies between z = 1 and 3,
all observed at 100 GHz, with a continuum of 0.3 mJy (at `z_ref`) and no
line.

```python
truth_cont = simulate.make_sample(
    "cont", n_sources=10, z_range=(1.0, 3.0),
    observed_freq_ghz=100.0,               # the same band for all
    line_flux=0.0, cont_flux=3e-4,         # no line; continuum 0.3 mJy
    size_arcsec=0.6, noise=0.05, rmax=600, nchan=64, seed=13)
```

`v.stack()` **without `line_freq`** makes the stack of the continuum: every
channel of every dataset is kept, and the bands need not overlap in the
rest frame. `v.extract()` then measures the continuum: its flux density and
the size of the source, with the same rules as before.

```python
c = ViSta(truth_cont["input_list"], output_name="cont")
c.stack()                                  # no line: the continuum
c.extract()
cont = c.run()
print(cont)
```

```text
[ViSta       0.9s] z_ref = 2.6663 (median of the sample)
```

```text
Stack      cont.zarr: 361,200 rows, 64 channels, 3 s
Continuum  10 sources, every channel: 0.31 ± 0.019 mJy (S/N 16.1), Gaussian, FWHM 0.60 ± 0.06 arcsec
           because its own Gaussian improves the fit by delta chi2 = 166.3 (at least 9 needed to call it extended)
Reference  natural weighting, fluxes at z_ref = 2.6663 (median of the sample)
```

```python
cont.plot("continuum_uv")
```

```{image} figures/13_01.png
:width: 100%
```

```python
print("measured:", flux(cont, "continuum"))
print("size:    ", cont["continuum"]["size"])
print("expected:", f"{simulate.expected(truth_cont)['continuum'] * 1e3:.3f} mJy")
```

```text
measured: 0.310 ± 0.019 mJy
size:     {'value': 0.6000756525377018, 'unit': 'arcsec', 'error': 0.05968234171282386}
```

```text
expected: 0.300 mJy
```
