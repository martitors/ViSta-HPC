# 7. A few galaxies that count much more

`weighting_scheme="natural"` (the default) weights every visibility by its
noise: the deepest observations count most, and the noise of the stack is
the lowest possible. That is right when all the galaxies are alike. It is
not when a few of them are different **and** observed much deeper.

We add to the twelve galaxies of chapter 2 **two lensed ones**: 5 times
brighter, observed 4 times deeper, and with the shape of a lensed source,
a ring 2.4″ across. The twelve stay as they were.

```python
lensed = simulate.make_sample(
    "lensed", n_sources=2, z_range=(2.3, 2.6), rest_freq_ghz=LINE,
    line_flux=0.4 * 5, fwhm_kms=300, cont_flux=1e-4 * 5,       # magnified 5 times
    size_arcsec=2.4, shape="ring",                             # an Einstein ring
    z_ref=truth["z_ref"],
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

One of the twelve and one of the two, as these baselines see them, without
noise and with the same colour scale:

```python
simulate.plot_images({
    "One normal galaxy, no noise": simulate.clean_image(all14["sources"][5], fov_arcsec=10),
    "One lensed galaxy, no noise": simulate.clean_image(all14["sources"][12], fov_arcsec=10)})
```

```{image} figures/07_01.png
:width: 100%
```

## 7.1 Natural and democratic weighting

The fourteen are stacked once and measured twice: with the natural
weighting, and with `weighting_scheme="democratic"`, which gives every
galaxy the same weight. (`z_ref` is given, so that every stack of this
chapter refers to the same redshift.)

```python
a = ViSta(all14["input_list"], output_name="all14")
a.stack(line_freq=LINE, line_window=500, velocity_range=900)
a.extract(z_ref=truth["z_ref"])
natural = a.run()

a.extract(z_ref=truth["z_ref"], weighting_scheme="democratic")
democratic = a.run(only="measure", tag="democratic")
```

```text
[ViSta       2.2s] z_ref = 2.6596 (given)
```

## 7.2 In the image

The stack of the twelve of chapter 2, and the fourteen with the two
weightings, with the same colour scale. With the natural weighting **the
stack is a ring**: it shows the two lensed galaxies, not the sample. With
the democratic weighting it looks like the twelve.

```python
window = (-500, 500)
simulate.plot_images({
    "The twelve normal": vista.dirty_image("stack.zarr", LINE, window, fov_arcsec=10),
    "Fourteen, natural": vista.dirty_image("all14.zarr", LINE, window, fov_arcsec=10),
    "Fourteen, democratic": vista.dirty_image("all14.zarr", LINE, window, fov_arcsec=10,
                                              weighting="democratic")})
```

```{image} figures/07_02.png
:width: 100%
```

## 7.3 In the spectrum and against the baseline length

The two stacks (the data, and the fit through them), and what we know of
the sample because it is simulated:

* **dotted**, in the colour of each stack: the true value with the weights
  of that stack;
* **green, dashed**: the mean of the twelve normal galaxies, the reference;
* **orange, dashed**: the two lensed ones.

The natural stack lies almost on the two lensed galaxies; against the
baseline it even goes negative near 100 kλ, the signature of a ring. The
democratic stack is much closer to the mean of the twelve.

```python
simulate.plot_weighting(all14, special=[12, 13],
                        results={"natural": natural, "democratic": democratic},
                        names=("the twelve normal", "the two lensed"))
```

```{image} figures/07_03.png
:width: 100%
```

Who makes the stack, in numbers: the share of the weight, and of the flux
of the stack, that comes from the two lensed galaxies.

```python
shares = simulate.weight_shares(all14, special=[12, 13])
for weighting, s in shares.items():
    print(f"{weighting:10s}: the two lensed have {100 * s['weight']:.0f}% of the weight and bring "
          f"{100 * s['flux']:.0f}% of the flux;  the stack should give {s['expected']:.2f} Jy km/s")
print(f"mean of the twelve normal: {shares['natural']['others']:.2f} Jy km/s")
print()
print("measured, natural:   ", flux(natural))
print("measured, democratic:", flux(democratic))
```

```text
natural   : the two lensed have 76% of the weight and bring 94% of the flux;  the stack should give 1.61 Jy km/s
democratic: the two lensed have 14% of the weight and bring 45% of the flux;  the stack should give 0.63 Jy km/s
mean of the twelve normal: 0.40 Jy km/s

measured, natural:    1.808 ± 0.480 Jy km/s
measured, democratic: 0.648 ± 0.179 Jy km/s
```

* **Natural**: observed 4 times deeper, each lensed galaxy weighs 16 times
  a normal one. The two hold 76% of the weight and bring 94% of the flux:
  the stack is theirs.
* **Democratic**: every galaxy weighs 1/14. The two hold 14% of the weight,
  but, 5 times brighter, they still bring 45% of the flux. The stack
  (0.63 Jy km/s) is much closer to the mean of the twelve (0.40) than the
  natural one (1.61), but it does not get there.

Neither weighting is wrong: they answer different questions. But note two
things. The first: a ring is not a Gaussian, so the Gaussian fit against
the baseline describes the natural stack badly, and its flux comes out
above the true value, with a large error that says so (chapter 12 is about
such sources). The second, and more important:

> **When a few sources are much brighter than the others, no weighting
> makes them "one of many".** It is often better to take them out of the
> stack and treat them separately: measure them on their own, and stack the
> rest.

## 7.4 When the magnification is known

`flux_norm=True` divides every galaxy by its `NORM` (here the
magnification). In flux, all fourteen are again the same galaxy, and the
natural weighting can use the deep data at full value. The shape is not
undone: the two are still rings, so the flux is right only as far as a
Gaussian describes the mix.

```python
a.extract(z_ref=truth["z_ref"], flux_norm=True, flux_ref=1)
delensed = a.run(only="measure", tag="delensed")
delensed.plot("line")
```

```text
[ViSta       2.1s] z_ref = 2.6596 (given); flux_ref = 1 (given)
```

```{image} figures/07_04.png
:width: 100%
```

```python
print("twelve unlensed (chapter 2):", flux(r))
print("fourteen, de-lensed:        ", flux(delensed))
print("expected:                   ", f"{expected['flux'][0]:.3f} Jy km/s")
```

```text
twelve unlensed (chapter 2): 0.431 ± 0.052 Jy km/s
fourteen, de-lensed:         0.442 ± 0.034 Jy km/s
expected:                    0.400 Jy km/s
```
