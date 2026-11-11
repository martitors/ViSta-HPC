# Quickstart

From a list of observations to a measured line, in five short steps. Every
step says what happens and why. The [Examples](../examples/index) do the
same on simulated data, with the right answer known.

## 1. Say where the galaxies are

ViSta needs one text file, the *input list*: one line per observation, with
the Measurement Set, the redshift of the galaxy and its position
([details](input_list)).

```text
# <ms_path>  <redshift>  <RA>  <Dec>
/data/galaxy_01.ms  2.314  10:00:28.61  +02:12:21.5
/data/galaxy_02.ms  2.871  10:01:02.10  +02:30:44.0
/data/galaxy_03.ms  2.442  09:59:51.33  +01:58:07.9
```

## 2. Describe the stack, and run

Three calls describe the work; `run()` does it.

```python
from vista import ViSta

v = ViSta("input_list.txt",           # the input list
          output_name="co32",         # the name of everything that is written
          verbose=True)               # show the progress
v.stack(line_freq=345.79599,          # the line: its rest frequency, GHz (here CO(3-2))
        line_window=500,              # the line is within -500..+500 km/s
        velocity_range=900)           # keep the channels from -900 to +900 km/s
v.extract()                           # measure the stack (without this: only the stack)
result = v.run()
```

What `run()` does, for every observation, in one pass over the data:

1. moves the phase centre onto the galaxy;
2. brings the frequencies to the rest frame of the galaxy;
3. puts the channels on a grid common to all the observations;
4. fits the continuum on the channels **outside** the line window and
   subtracts it.

The result is one dataset, `co32.zarr`: the stack. With `extract()` the
stack is then compressed into a small file of sums (the *statistics*) and
measured. The [stacking pages](../stacking/index) explain every step.

Every other option has a default. `v.help()` lists them all, with their
value now, what they can be and what they do.

## 3. Read the result

```python
print(result)
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

The report says what was measured **and why each choice was made**:

- **Line**: is there a line, and how sure.
- **flux**: the number to quote. Two fluxes are always measured, the area of
  the Gaussian fit and the area under the spectrum in the window; the report
  says which one is quoted and why.
- **source**: a Gaussian or a point, and why.
- **Continuum**: its flux density, and which source model it got.
- **Sizes**: the size seen by the line and by the continuum, and whether
  they agree.
- **Reference**: the weighting, and the redshift the fluxes refer to.

The same numbers are in a dictionary, each with its value, error and unit:

```python
res = result.results
res["flux"]        # {'value': 0.4035, 'error': 0.0396, 'unit': 'Jy km/s', 'method': 'fit', ...}
res["size"]        # {'value': 0.59, 'error': 0.07, 'unit': 'arcsec'}
res["continuum"]   # flux density, S/N, size, model
```

## 4. Look at the figures

`result.plot()` draws them in a notebook; the run also writes them as PNG
(`co32_profile.png`, `co32_uv.png`, `co32_continuum.png`,
`co32_continuum_uv.png`, `co32_noise.png`).

**The line** (`result.plot("line")`). The stacked spectrum (black), its
noise (grey), the fit (red), the window (orange). On the left the two
fluxes, the quoted one in bold. Below: the residuals, and how many galaxies
contribute at every velocity.

```{image} ../_static/fig_line.png
:alt: The stacked line: spectrum, fit, window, the two fluxes
:width: 70%
:align: center
```

**The line against the baseline length** (`result.plot("line_uv")`). A point
source has the same flux on every baseline; an extended one fades on the
long ones. Red is the Gaussian source, green the point source, in bold the
one that was chosen. The total flux is where the model meets zero baseline.

```{image} ../_static/fig_line_uv.png
:alt: Line flux against baseline length, with the Gaussian and the point source models
:width: 60%
:align: center
```

`result.plot("continuum")` and `"continuum_uv"` show the same for the
continuum; `"noise"` the checks on the noise; `"image"` a quick image of the
stack. [Output files and figures](../extraction/outputs) describes them all.

## 5. Change one thing, and measure again

The statistics stay on disk, so a measurement with other options takes
seconds and does not touch the stack. `v.extract(...)` replaces the options,
`run(only="measure")` repeats the measurement.

```python
v.extract(window=320)                             # sum the spectrum over ±320 km/s only
narrow = v.run(only="measure", tag="narrow")      # tag: its files do not replace the others

v.extract(weighting_scheme="democratic")          # every galaxy the same weight
democratic = v.run(only="measure", tag="democratic")

v.extract(model="point")                          # force a point source
point = v.run(only="measure", tag="point")
```

Things you may want to change, each explained on its page:

| To change | Option | Page |
|---|---|---|
| where the line is searched | `v0_range`, `fwhm_range` | [Measuring the line](../extraction/measure_line) |
| the source model | `model`, `continuum_model` | [Measuring the line](../extraction/measure_line), [The continuum](../extraction/extract_flux) |
| which flux is quoted, and its window | `flux_measure`, `window` | [Measuring the line](../extraction/measure_line) |
| how the galaxies are combined | `weighting_scheme`, `z_norm`, `pb_corr`, `flux_norm` | [Weighting](../extraction/weighting) |
| what is done with the continuum | `continuum` (in `v.stack`) | [Continuum](../stacking/continuum) |

## The same from the command line

```bash
./run_vista.sh --input input_list.txt --output-name co32 --verbose \
               --line-freq 345.79599 --line-window 500 --velocity-range 900 --extract
```

It writes the same files ([all the options](../command_line)).

## Before believing a number

- The residuals of the line scatter around zero, within the grey band.
- The imaginary part (grey crosses) scatters around zero: the galaxies are
  where the list says.
- Nothing is left of the continuum outside the window (the report says so).
- The model follows the points against the baseline length.
- The two sizes are compatible.

[Things to check](../extraction/outputs.md#things-to-check) lists the
numbers behind each of these.
