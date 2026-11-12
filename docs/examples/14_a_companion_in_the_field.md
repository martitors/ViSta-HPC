# 14. A companion in the field

A stack is centred on its galaxies. When the field of one of them holds a
**second source** (a companion), that source is not at the centre: its
signal oscillates with the baseline, and what is left of it after the
average is added to the stack.

Twelve point-like galaxies, as in chapter 8. Next to the sixth we put a
companion: 1.4″ away, at the same redshift, with a line five times
stronger (2 Jy km/s) and a continuum of 1 mJy.

```python
truth_comp = simulate.make_sample(
    "comp", n_sources=12, z_range=(2.0, 3.0), rest_freq_ghz=LINE,
    line_flux=0.4, fwhm_kms=300, cont_flux=2e-4,
    size_arcsec=0.0,                           # points
    noise=0.05, rmax=250, seed=9)

galaxy = truth_comp["sources"][5]
companion = simulate.add_companion(
    galaxy, offset_arcsec=(1.2, 0.8),          # east, north of the galaxy
    z=galaxy["z"],                             # same redshift: its line falls on ours
    line_flux=2.0, fwhm_kms=300, cont_flux=1e-3)
```

## 14.1 Not removed

Stacked as it is, the companion does two things: the flux comes out too
high, and **the stack looks extended**, although every galaxy is a point.
In the plot against the baseline length the short baselines, which do not
tell the two sources apart, lie above the long ones: exactly what an
extended source does.

```python
k = ViSta(truth_comp["input_list"], output_name="comp_left")
k.stack(line_freq=LINE, line_window=500, velocity_range=900)
k.extract()
left = k.run()
left.plot("line_uv")
```

```text
[ViSta       0.4s] z_ref = 2.6596 (median of the sample)
```

```{image} figures/14_01.png
:width: 100%
```

```python
print("measured:", flux(left), "  size:", f"{left['size']['value']:.2f} arcsec")
print("expected:", f"{simulate.expected(truth_comp)['flux'][0]:.3f} Jy km/s,   size: 0 (points)")
```

```text
measured: 0.522 ± 0.120 Jy km/s   size: 0.72 arcsec
expected: 0.400 Jy km/s,   size: 0 (points)
```

## 14.2 Removed

The companions go in a file, one per line: the dataset they are in, their
position, and their redshift **if it is known**.

```python
simulate.write_companions("comp/companions.txt", [companion])
print(open("comp/companions.txt").read())
```

```text
# <ms_path>  <RA>  <Dec>  [redshift]
comp/galaxy_05.ms  10:10:00.21353660  +02:30:01.8000000  2.603148
```

`v.stack(companions=...)` fits the galaxy and its companions together, as
point sources at their positions, on the visibilities of that dataset, and
subtracts the companions before the continuum subtraction. The galaxy is
always fitted and never touched.

* A companion **with a redshift**: its continuum on every channel, and its
  line, channel by channel, in the window where the line falls.
* A companion **without a redshift** has no known line: only its continuum
  is removed.
* A continuum compatible with zero (S/N below 3) is not subtracted: it
  would only add noise.
* A companion **too close** to the galaxy for the baselines of that dataset
  cannot be told apart from it: it is reported and left where it is,
  rather than removed badly.

The report says, for every companion, what was found and what was done.

```python
k = ViSta(truth_comp["input_list"], output_name="comp_removed")
k.stack(line_freq=LINE, line_window=500, velocity_range=900,
        companions="comp/companions.txt")
k.extract()
removed = k.run()
print(removed)
```

```text
[ViSta       0.4s] z_ref = 2.6596 (median of the sample)
```

```text
Stack      comp_removed.zarr: 433,440 rows, 90 channels, 4 s
Line       345.796 GHz, 12 sources: detected at 44.7 sigma (S/N 44.8, searched over centre and width)
  flux     0.4003 ± 0.015 Jy km/s, the area of the Gaussian fit
           because the area agrees with the Gaussian (+0.6 sigma)
           Gaussian fit 0.4003 ± 0.015; area in the window (-500 to +500 km/s) 0.4046 ± 0.019
  Gaussian centre -2 ± 3 km/s, FWHM 303 ± 10 km/s, peak 1.24 ± 0.047 mJy
  source   point source
           because a Gaussian source improves the fit by delta chi2 = 0.8 only (at least 9 needed to call it extended)
Continuum  subtracted while stacking and measured on its own: 0.195 ± 0.012 mJy (S/N 16.4), point source
           because its own Gaussian improves the fit by delta chi2 = -0.0 (at least 9 needed to call it extended), and the line is a point too
           outside the line window: -0.00013 ± 0.0012 mJy (-0.1 sigma): nothing left
Sizes      each on its own: line 0.28 arcsec, continuum 0.00 ± 115.14 arcsec
Reference  natural weighting, fluxes at z_ref = 2.6596 (median of the sample)
Companions galaxy_05.ms, 1.4" from the target, z = 2.6031: continuum 0.988 ± 0.047 mJy subtracted; line 1.99 ± 0.042 Jy km/s subtracted (50 channels)
```

The fluxes found for the companion are the ones we put in (2 Jy km/s and
1 mJy, seen through the primary beam). The stack is flat again, and the
flux and the size are the right ones.

```python
removed.plot("line_uv")
```

```{image} figures/14_02.png
:width: 100%
```

```python
print("not removed:", flux(left), "  size:", f"{left['size']['value']:.2f} arcsec")
print("removed:    ", flux(removed), "  size:", f"{removed['size']['value']:.2f} arcsec")
print("expected:   ", f"{simulate.expected(truth_comp)['flux'][0]:.3f} Jy km/s,   size: 0 (points)")
```

```text
not removed: 0.522 ± 0.120 Jy km/s   size: 0.72 arcsec
removed:     0.400 ± 0.015 Jy km/s   size: 0.00 arcsec
```

```text
expected:    0.400 Jy km/s,   size: 0 (points)
```
