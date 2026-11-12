# 7. A few galaxies that count much more

`weighting_scheme="natural"` (the default) weights every visibility by its
noise: the deepest observations count most, and the noise of the stack is
the lowest possible. That is right when all the galaxies are alike. It is
not when a few of them are different **and** observed much deeper.

We add to the twelve galaxies of chapter 2 **two lensed ones**: the same
galaxy magnified 5 times, observed 4 times deeper.

```python
lensed = simulate.make_sample(
    "lensed", n_sources=2, z_range=(2.3, 2.6), rest_freq_ghz=LINE,
    line_flux=0.4 * 5, fwhm_kms=300, cont_flux=1e-4 * 5,       # magnified 5 times
    size_arcsec=0.8, z_ref=truth["z_ref"],
    noise=0.07 / 4, kappa=1.3, rmax=600, seed=7)        # 4 times deeper

# one list with the fourteen, and the magnification of each in the NORM column
all14 = simulate.write_list("all14.txt", truth["sources"] + lensed["sources"],
                            z_ref=truth["z_ref"], norm=[1] * 12 + [5] * 2)
print(open("all14.txt").read())
```

```text
# <ms_path>  <redshift>  <RA>  <Dec>  norm=<NORM>
sample/galaxy_00.ms  2.026588  10:00:00.96719368  +02:00:08.5382956  norm=1
sample/galaxy_01.ms  2.065154  10:01:59.55919839  +02:06:08.4777275  norm=1
sample/galaxy_02.ms  2.286817  10:04:00.41672099  +02:12:01.6349063  norm=1
sample/galaxy_03.ms  2.437248  10:05:59.95542388  +02:18:08.7857364  norm=1
sample/galaxy_04.ms  2.484944  10:07:59.26490427  +02:23:51.7829498  norm=1
sample/galaxy_05.ms  2.603148  10:10:00.86329657  +02:30:07.8837517  norm=1
sample/galaxy_06.ms  2.716075  10:11:59.53445747  +02:35:50.0817844  norm=1
sample/galaxy_07.ms  2.777534  10:14:00.44691319  +02:41:47.2381414  norm=1
sample/galaxy_08.ms  2.860394  10:16:00.23368901  +02:48:02.7757871  norm=1
sample/galaxy_09.ms  2.870249  10:17:59.75412473  +02:53:46.6399170  norm=1
sample/galaxy_10.ms  2.915380  10:20:00.21172955  +03:00:02.1567271  norm=1
sample/galaxy_11.ms  2.918238  10:22:00.13803646  +03:05:55.2607827  norm=1
lensed/galaxy_00.ms  2.487529  10:00:00.13341461  +02:00:01.0000000  norm=5
lensed/galaxy_01.ms  2.569164  10:02:00.13342294  +02:06:01.0000000  norm=5
```

## 7.1 The two lensed galaxies alone

(`z_ref` is given, so that every stack of this chapter refers to the same
redshift.)

```python
l = ViSta(lensed["input_list"], output_name="lensed_stack")
l.stack(line_freq=LINE, line_window=500, velocity_range=900)
l.extract(z_ref=truth["z_ref"])
two = l.run()
two.plot("line")
```

```text
[ViSta       0.9s] z_ref = 2.6596 (given)
```

```{image} figures/07_01.png
:width: 100%
```

## 7.2 The fourteen, natural weighting

The stack looks like the two lensed galaxies: they hold most of the weight.

```python
a = ViSta(all14["input_list"], output_name="all14")
a.stack(line_freq=LINE, line_window=500, velocity_range=900)
a.extract(z_ref=truth["z_ref"])
natural = a.run()
natural.plot("line")
```

```text
[ViSta       1.6s] z_ref = 2.6596 (given)
```

```{image} figures/07_02.png
:width: 100%
```

## 7.3 The fourteen, democratic weighting

`weighting_scheme="democratic"` gives every galaxy the same weight: the
stack is the plain mean of the fourteen, with more noise.

```python
a.extract(z_ref=truth["z_ref"], weighting_scheme="democratic")
democratic = a.run(only="measure", tag="democratic")
democratic.plot("line")
```

```{image} figures/07_03.png
:width: 100%
```

The three spectra together: the natural stack lies on the lensed pair, the
democratic one far below. Neither is wrong: they answer different
questions, and each has its own expected value.

```python
vista.compare({"two lensed alone": two, "fourteen, natural": natural,
               "fourteen, democratic": democratic}, table=False)
```

```{image} figures/07_04.png
:width: 100%
```

```python
print("two lensed alone:     ", flux(two))
print("fourteen, natural:    ", flux(natural),
      f"  expected {simulate.expected(all14)['flux'][0]:.3f}")
print("fourteen, democratic: ", flux(democratic),
      f"  expected {simulate.expected(all14, weighting='democratic')['flux'][0]:.3f}")
```

```text
two lensed alone:      1.999 ± 0.016 Jy km/s
```

```text
fourteen, natural:     1.616 ± 0.411 Jy km/s   expected 1.609
fourteen, democratic:  0.653 ± 0.144 Jy km/s   expected 0.629
```

## 7.4 When the magnification is known

`flux_norm=True` divides every galaxy by its `NORM` (here the
magnification). All fourteen are again the same galaxy, and the natural
weighting can use the deep data at full value: the flux of chapter 2, with
a smaller error.

```python
a.extract(z_ref=truth["z_ref"], flux_norm=True, flux_ref=1)
delensed = a.run(only="measure", tag="delensed")
delensed.plot("line")
```

```text
[ViSta       1.9s] z_ref = 2.6596 (given); flux_ref = 1 (given)
```

```{image} figures/07_05.png
:width: 100%
```

```python
print("twelve unlensed (chapter 2):", flux(r))
print("fourteen, de-lensed:        ", flux(delensed))
print("expected:                   ", f"{expected['flux'][0]:.3f} Jy km/s")
```

```text
twelve unlensed (chapter 2): 0.431 ± 0.052 Jy km/s
fourteen, de-lensed:         0.406 ± 0.022 Jy km/s
expected:                    0.400 Jy km/s
```
