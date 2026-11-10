# Quickstart

A stack in three steps: write the input list, describe the line, run.

## 1. The input list

One line per dataset, with the redshift and the position of the source
([details](input_list)):

```text
# <ms_path>  <redshift>  <RA>  <Dec>
/data/galaxy_01.ms  2.314  10:00:28.61  +02:12:21.5
/data/galaxy_02.ms  2.871  10:01:02.10  +02:30:44.0
/data/galaxy_03.ms  2.442  09:59:51.33  +01:58:07.9
```

## 2. Three calls, and the run

```python
from vista import ViSta

v = ViSta("input_list.txt",           # input, output, how it runs
          output_name="co32",         # co32.zarr, co32_results.json, co32_profile.png ...
          verbose=True)               # show the progress (default: warnings only)
v.stack(line_freq=345.79599,                    # what makes the stack: the line, GHz (CO(3-2))
        line_window=500,              # km/s: the line is within -500..+500
        velocity_range=900)           # channels from -900 to +900 km/s
v.extract()                           # measure it (without this call: only the stack)
result = v.run()
print(result)
```

Every other option has a default. `v.help()` prints them all, step by step,
with their value now, what they can be and what they do; `v.help("stack")`
shows one step, `help(v.extract)` its full description. For example
`v.help("stack")`, after the call above:

```text
v.stack(line_freq=..., ...)  —  what makes the stack (line_freq is required)
  option           now          can be                       what it does
  line_freq        345.796      GHz | '345.8GHz' | list      rest frequency (a number: GHz; a string: with its
                                                             unit); several lines: the first is the one of the
                                                             stack
  line_window      (-500, 500)  W | (lo, hi) | list          km/s, window of every line (500: -500..500): out
                                                             of the continuum fit; lines whose windows overlap
                                                             are fitted together
  velocity_range   900          km/s | (lo, hi) | None       band of the stack around the first line; None:
                                                             the widest coverage of one dataset
  ...
  continuum        'subtract'   'subtract' | 'fit' | 'none'  subtracted while stacking (then measured on its
                                                             own) | left, fitted with the lines | left as it
                                                             is
```

`run()` does everything, in one pass over the data:

* every dataset is re-centred on its source, brought to the rest frame (its
  frequencies first corrected to LSRK, the frame of the redshifts by
  default: `v.stack(..., frame="BARY")` if yours are barycentric) and regridded
  onto the common grid;
* the continuum is fitted row by row on the channels outside the line window
  and subtracted (`MODEL_DATA`, `CORRECTED_DATA`; `DATA` is kept);
* with `extract()`, the visibilities are summed in annuli of baseline
  length, channel by channel, into two files of a few MB (the statistics),
  and the line is measured on all channels and annuli at once, the continuum
  on its own.

The readers, the writers and the threads of the kernel come from the cores
of the machine (`ViSta(n_readers=..., n_threads=...)` to fix them). The [stacking pages](../stacking/index) describe every step.

## 3. Measure again

The statistics stay on disk, so the measurement can be repeated in seconds,
with other options, without touching the stack:

```python
v.extract(fwhm_range=(200, 500),            # limits on the width, km/s
          v0_range=(-150, 150),             # and on the centre
          weighting_scheme="democratic")
result = v.run(only="measure")
ln = result.line.summary["line"]
print(f"F = {ln['flux']:.3f} ± {ln['flux_error']:.3f} Jy km/s ({ln['flux_method']}), "
      f"FWHM = {ln['fwhm_kms']:.0f} km/s, size = {ln['theta_fwhm_arcsec']:.2f}\", "
      f"{result.line.summary['detection']['significance_sigma']:.1f} sigma")
```

With the width known from elsewhere, `fwhm_range=(300, 300)` and
`v0_range=(0, 0)` fix it: a matched filter, the most sensitive measurement.
Two fluxes are always given, the integral of the fitted Gaussian and the sum
of the spectrum over the window; `flux_measure="auto"` quotes the Gaussian one unless
the line is not Gaussian. See [Measuring the line](../extraction/measure_line).

## The same from the command line

```bash
./run_vista.sh --input input_list.txt --output-name co32 --verbose \
               --line-freq 345.79599 --line-window 500 --velocity-range 900 --extract
```

It writes the same files ([all the options](../command_line)).

## Look at the result

`co32_profile.png`, `co32_uv.png` and `co32_noise.png` show the line with its
two fluxes, the flux per annulus of baseline length and the noise.
`examples/plot_line.py` redraws them from the output files in your own style
([Output files and figures](../extraction/outputs)).

Before believing a number, check that the imaginary part and the line-free
channels are at the noise level (`noise.imaginary_rms_over_noise` and
`noise.line_free_rms_over_noise` ≈ 1 in the results) and that no continuum is
left outside the window (`continuum_residual`): see
[Things to check](../extraction/outputs.md#things-to-check).
