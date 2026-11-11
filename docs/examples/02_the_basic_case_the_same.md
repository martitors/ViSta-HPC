# 2. The basic case: the same galaxy at twelve redshifts

## 2.1 The sample

`simulate.make_sample` writes one observation per galaxy and the *input
list* that ViSta reads. Here **the twelve galaxies are the same galaxy**,
put at twelve redshifts between 2 and 3:

* the same line (0.4 Jy km/s at the reference redshift, 300 km/s wide) and
  the same continuum (0.2 mJy);
* some are seen as **points** and some as **extended** (0.8″), mixed;
* every observation has its band centred a little differently
  (`band_offset_kms`), so the edges of the stacked spectrum are covered by
  fewer galaxies;
* each sits somewhere within 20″ of the centre of its field.

```python
truth = simulate.make_sample(
    "sample", n_sources=12, z_range=(2.0, 3.0), rest_freq_ghz=LINE,
    line_flux=0.4, fwhm_kms=300,                               # the same line for all
    cont_flux=2e-4,                                            # the same continuum, Jy
    size_arcsec=lambda rng: float(rng.choice([0.0, 0.8])),     # a point, or 0.8" across
    band_offset_kms=lambda rng: rng.uniform(-350, 350),        # bands centred differently
    offset_arcsec=lambda rng: tuple(rng.uniform(-15, 15, 2)),  # position in the field
    noise=0.07, kappa=1.3, rmax=600, seed=9)

print(simulate.describe(truth))
```

```text
         z   nu_obs     line   FWHM   size  continuum   noise  array  offset    PB
                GHz  Jy km/s   km/s arcsec        mJy  Jy/vis      m  arcsec
  0  2.027   114.25    0.639    300   0.00      0.320   0.091    600    10.2  0.91
  1  2.065   112.82    0.619    300   0.80      0.309   0.091    600    16.9  0.77
  2  2.287   105.21    0.518    300   0.80      0.259   0.091    600    17.7  0.78
  3  2.437   100.60    0.464    300   0.80      0.232   0.091    600    16.0  0.83
  4  2.485    99.23    0.449    300   0.80      0.224   0.091    600     9.2  0.94
  5  2.603    95.97    0.415    300   0.00      0.207   0.091    600     9.8  0.94
  6  2.716    93.05    0.386    300   0.00      0.193   0.091    600    12.3  0.91
  7  2.778    91.54    0.372    300   0.80      0.186   0.091    600     8.5  0.96
  8  2.860    89.58    0.354    300   0.00      0.177   0.091    600    15.0  0.88
  9  2.870    89.35    0.352    300   0.00      0.176   0.091    600     9.3  0.95
 10  2.915    88.32    0.343    300   0.00      0.172   0.091    600     5.2  0.98
 11  2.918    88.25    0.343    300   0.80      0.171   0.091    600     9.4  0.95
```

The input list has one line per dataset: the Measurement Set, the redshift
and the position of the galaxy. It is all ViSta needs to know.

```python
print(open(truth["input_list"]).read())
```

```text
# <ms_path>  <redshift>  <RA>  <Dec>
sample/galaxy_00.ms  2.026588  10:00:00.56956667  +01:59:54.4681047
sample/galaxy_01.ms  2.065154  10:02:00.56556167  +02:06:14.6322224
sample/galaxy_02.ms  2.286817  10:04:00.84710080  +02:11:47.6919234
sample/galaxy_03.ms  2.437248  10:05:59.38968111  +02:17:46.8676049
sample/galaxy_04.ms  2.484944  10:08:00.24706000  +02:24:08.4628839
sample/galaxy_05.ms  2.603148  10:09:59.37807210  +02:30:02.8785866
sample/galaxy_06.ms  2.716075  10:12:00.60111751  +02:35:51.5873007
sample/galaxy_07.ms  2.777534  10:14:00.56471225  +02:42:01.2020839
sample/galaxy_08.ms  2.860394  10:15:59.02817282  +02:47:56.3165941
sample/galaxy_09.ms  2.870249  10:18:00.58260802  +02:54:03.1715908
sample/galaxy_10.ms  2.915380  10:20:00.13802363  +02:59:55.2607827
sample/galaxy_11.ms  2.918238  10:22:00.49759661  +03:06:05.7918213
```

**The same galaxy looks different at different redshifts.** The farther it
is, the fainter its line (left). Before combining them, ViSta brings every
galaxy to the flux it would have at one reference redshift, `z_ref` (the
median of the sample unless you give it). After this correction the twelve
lines are **one and the same** (right): the stack is then the average of
twelve measurements of the same thing.

```python
simulate.plot_redshift_correction(truth)
```

```{image} figures/02_01.png
:width: 100%
```

What every observation sees on its own: the spectrum at the position of the
galaxy (black) and its true line (green). In none of them can the line be
measured.

```python
simulate.plot_sample(truth)
```

```{image} figures/02_02.png
:width: 100%
```

**The expected answer.** Because the sample is simulated, we know what a
perfect measurement of the stack would give. With real data there is no
such thing: it is here only to check the measurements.

```python
expected = simulate.expected(truth)
print(f"expected line flux of the stack: {expected['flux'][0]:.3f} Jy km/s")
print(f"expected continuum:              {expected['continuum'] * 1e3:.3f} mJy")
```

```text
expected line flux of the stack: 0.400 Jy km/s
expected continuum:              0.200 mJy
```

## 2.2 The stack

Two calls describe it:

* `ViSta(...)`: the input list and the name of what is written;
* `v.stack(...)`: the rest frequency of the line (`line_freq`, GHz, or a
  string as `"345.796GHz"`), the window that holds the line
  (`line_window=500`: from −500 to +500 km/s) and the band to keep
  (`velocity_range=900`: ±900 km/s).

`v.run()` then, for every dataset: brings the frequencies to the rest frame
of the galaxy, moves the phase centre onto the galaxy, puts the channels on
a common grid, and fits and subtracts the continuum on the channels outside
the line window. The result is one dataset, `stack.zarr`.

```python
v = ViSta(truth["input_list"], output_name="stack")
v.stack(line_freq=LINE, line_window=500, velocity_range=900)
v.run()
```

```text
Stack      stack.zarr: 433,440 rows, 90 channels, 4 s
```

**What stacking does, in an image.** One galaxy of the stack, then all
twelve: the line appears at the centre, where every galaxy was moved.
(These images are a quick look; the measurements are made on the
visibilities.)

```python
one = vista.dirty_image("stack.zarr", line_freq=LINE, velocity=(-150, 150), sources=[5],
                        fov_arcsec=16)
vista.plot_image(one, title="One galaxy");
```

```{image} figures/02_03.png
:width: 100%
```

```python
all12 = vista.dirty_image("stack.zarr", line_freq=LINE, velocity=(-150, 150), fov_arcsec=16)
vista.plot_image(all12, title="The stack of twelve");
```

```{image} figures/02_04.png
:width: 100%
```

## 2.3 Measuring the line

`v.extract()` asks for the measurement, here with all its defaults;
`v.run(only="measure")` makes it on the stack that is already there.
Printing the result says what was measured **and why each choice was made**.

```python
v.extract()
r = v.run(only="measure")
print(r)
```

```text
[ViSta       0.3s] z_ref = 2.6596 (median of the sample)
```

```text
Line       345.796 GHz, 12 sources: detected at 18.5 sigma (S/N 18.8, searched over centre and width)
  flux     0.4035 ± 0.04 Jy km/s, the area of the Gaussian fit
           because the area agrees with the Gaussian (-1.2 sigma)
           Gaussian fit 0.4035 ± 0.04; area in the window (-500 to +500 km/s) 0.3713 ± 0.057
  Gaussian centre +15 ± 10 km/s, FWHM 284 ± 21 km/s, peak 1.34 ± 0.13 mJy
  source   Gaussian, FWHM 0.59 ± 0.07 arcsec
           because it fits better than a point source: delta chi2 = 28.1 (at least 9 needed to call it extended)
Continuum  subtracted while stacking and measured on its own: 0.204 ± 0.032 mJy (S/N 6.4), Gaussian, FWHM 0.59 arcsec
           because the source measured on the line is used: its own Gaussian improves the fit by delta chi2 = 1.0 (at least 9 needed to call it extended)
           outside the line window: 0.0036 ± 0.006 mJy (+0.6 sigma): nothing left
Sizes      each on its own: line 0.59 ± 0.07 arcsec, continuum 0.28 ± 0.15 arcsec: compatible (1.9 sigma apart)
Reference  natural weighting, fluxes at z_ref = 2.6596 (median of the sample)
```

How to read it:

* **Line**: the S/N of the line, and how sure the detection is once the
  search over centre and width is accounted for.
* **flux**: the number to quote. Two fluxes are always measured: the **area
  of the Gaussian fit** and the **area under the spectrum in the window**.
  The Gaussian one is quoted unless the two disagree (chapter 11).
* **source**: the model of the sky. A Gaussian source is used only if it
  fits clearly better than a point source (Δχ² ≥ 9).
* **Continuum**: what was subtracted, measured on its own.
* **Sizes**: the size of the source measured on the line and on the
  continuum, and whether they agree.
* **Reference**: the weighting and the redshift the fluxes refer to.

**The spectrum of the line.** The real part (Re) of the stacked spectrum
(black), its noise (grey band), the fit (red) and the window (orange). On
the left the two fluxes, the quoted one in bold; then centre, width and
peak of the Gaussian, and the S/N. Below, the residuals in units of the
noise, and how many galaxies contribute at every velocity: fewer at the
edges, where the noise grows.

```python
r.plot("line")
```

```{image} figures/02_05.png
:width: 100%
```

**The line against the baseline length.** The line flux measured in rings
(*annuli*) of baseline length. A point source would give the same flux on
all of them; here it falls on the long baselines: the stacked source is
extended. Red: the Gaussian source; green dashed: the point source; in bold
the one chosen. The flux is where the model meets zero baseline. The grey
crosses are the imaginary part (Im), which holds only noise.

```python
r.plot("line_uv")
```

```{image} figures/02_06.png
:width: 100%
```

**All the numbers.** `r.results` is a dictionary with everything measured,
every quantity with its value, error and unit.

```python
res = r.results
res["flux"]
```

```text
{'value': 0.4035386592646733,
 'unit': 'Jy km/s',
 'error': 0.03961888730675575,
 'method': 'fit',
 'reason': 'the area agrees with the Gaussian (-1.2 sigma)'}
```

```python
print("flux from the fit:   ", flux(r, "flux_fit"))
print("flux in the window:  ", flux(r, "flux_window"))
print("centre, width:       ", f"{res['centre']['value']:+.0f}, {res['fwhm']['value']:.0f} km/s")
print("S/N, significance:   ", f"{res['snr']:.1f}, {res['significance']:.1f} sigma")
print("size of the source:  ", res["size"])
print("source model:        ", res["model"])
print("noise per channel:   ", res["noise"])
```

```text
flux from the fit:    0.404 ± 0.040 Jy km/s
flux in the window:   0.371 ± 0.057 Jy km/s
centre, width:        +15, 284 km/s
S/N, significance:    18.8, 18.5 sigma
size of the source:   {'value': 0.5923173435129085, 'unit': 'arcsec', 'error': 0.07277499168641591}
source model:         {'chosen': 'gauss', 'size_from': 'own fit', 'probability': 0.9999978029581333, 'delta_chi2': 28.05679302278438, 'delta_chi2_needed': 9.0}
noise per channel:    {'value': 0.0001236310802813826, 'unit': 'Jy', 'per': 'channel', 'channel_width_kms': 20.000001993459648}
```

## 2.4 The continuum

With the default `continuum="subtract"` the continuum is taken out of every
galaxy before stacking, so the spectrum above holds the line only, and the
continuum is **measured on its own**.

**The spectrum of the continuum**: the channels outside the line window
(black), with the level measured (blue).

```python
r.plot("continuum")
```

```{image} figures/02_07.png
:width: 100%
```

**The continuum against the baseline length**, as for the line. This
continuum is weak: on its own it cannot tell whether the source is extended
(the points are too noisy). But the line says it is, and these are the same
galaxies: the continuum is fitted **with the size of the line** (red, in
bold). Chapter 5 looks at this choice.

```python
r.plot("continuum_uv")
```

```{image} figures/02_08.png
:width: 100%
```

```python
res["continuum"]
```

```text
{'value': 0.00020412827670354805,
 'unit': 'Jy',
 'error': 3.1842314044708916e-05,
 'treatment': 'subtracted, measured on its own',
 'snr': 6.410598061966764,
 'size': {'value': 0.5923173435129085, 'unit': 'arcsec'},
 'model': {'chosen': 'gauss',
  'size_from': 'line',
  'probability': 0.3769594679482915,
  'delta_chi2': 0.995052186900903,
  'delta_chi2_needed': 9.0},
 'left_outside_window': {'value': 3.5518374509913434e-06,
  'unit': 'Jy',
  'error': 5.985767788583937e-06,
  'detected': False}}
```

The two sizes, each measured on its own, and how far apart they are in
units of their errors:

```python
res["sizes"]
```

```text
{'line': {'value': 0.5923173435129085,
  'unit': 'arcsec',
  'error': 0.07277499168641591},
 'continuum': {'value': 0.28287759822425596,
  'unit': 'arcsec',
  'error': 0.14899558898296242},
 'difference_sigma': 1.8661315007962558}
```
