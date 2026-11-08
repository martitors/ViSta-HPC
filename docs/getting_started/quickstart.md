# Quickstart

A stack in three steps: write the input list, stack, measure.

## 1. The input list

One line per dataset, with the redshift and the position of the source
([details](input_list)):

```text
# <ms_path>  <redshift>  <RA>  <Dec>
/data/galaxy_01.ms  2.314  10:00:28.61  +02:12:21.5
/data/galaxy_02.ms  2.871  10:01:02.10  +02:30:44.0
/data/galaxy_03.ms  2.442  09:59:51.33  +01:58:07.9
```

## 2. Stack, subtract the continuum and compress, in one pass

```python
from vista import ViSta
from vista.extract import LineConfig, ContinuumConfig, BinningConfig

line = LineConfig(rest_freq_ghz=345.79599,         # CO(3-2)
                  v_window_kms=(-500, 500))         # where the line can be
ViSta("input_list.txt").run(
    output="stack.zarr",                            # the stack (Zarr store)
    central_freq=line.rest_freq_hz,                 # centre of the output grid, Hz
    velocity_range_kms=900,                         # channels from -900 to +900 km/s
    line=line,
    continuum=ContinuumConfig(order=0),             # continuum fitted outside the window
    stats_line="line_stats.npy",                    # compressed statistics of the line
    stats_continuum="cont_stats.npy",               # ... and of the continuum
    binning=BinningConfig(b_min_klambda=3, b_max_klambda=3000, n_bins=18))
```

Each dataset is re-centred on its source, brought to the rest frame (its
frequencies first corrected to LSRK, the frame of the redshifts by default:
`frame="BARY"` if yours are barycentric) and regridded onto the common grid; the continuum is fitted row by row on the
channels outside the line window and subtracted (`MODEL_DATA`,
`CORRECTED_DATA`; `DATA` is kept); and the visibilities are summed in annuli
of baseline length, channel by channel, into two files of a few MB. The
[stacking pages](../stacking/index) describe every step.

## 3. Measure

```python
from vista.extract import measure_line, WeightingConfig, LineFitConfig
from vista.extract.plots_line import plot_line_result

result = measure_line("line_stats.npy", "input_list.txt", line,
                      WeightingConfig(scheme="natural"),   # redshift rescaling and primary beam: on
                      LineFitConfig(mode="search"),        # centre and width free
                      output_prefix="co32")
plot_line_result(result, line, "co32", unit="Jy")          # co32_profile.png, _uv.png, _noise.png
ln = result.summary["line"]
print(f"F = {ln['flux']:.3f} ± {ln['flux_error']:.3f} Jy km/s, FWHM = {ln['fwhm_kms']:.0f} km/s, "
      f"size = {ln['theta_fwhm_arcsec']:.2f}\", "
      f"{result.summary['detection']['significance_sigma']:.1f} sigma")
```

The line is fitted on all channels and annuli at once; the noise is
measured from the data. With the width known from elsewhere,
`LineFitConfig(mode="detect", v0_kms=0, fwhm_kms=300)` is a matched filter,
the most sensitive measurement. See [Measuring the line](../extraction/measure_line).

The measurement takes seconds and can be repeated at will (another
weighting, a size prior, another mode) without touching the stack.

## The same from the command line

```bash
./run_vista.sh --input input_list.txt --rest-freq 345.79599 \
               --velocity-range 900 --v-window "-500 500" --contsub before \
               --method joint --mode search --unit Jy --out-dir co32
```

It writes the stack, the statistics, the results (`*_results.json`) and the
figures in `co32/` ([all the options](../command_line)).

## Look at the result

`co32_profile.png`, `co32_uv.png` and `co32_noise.png` (`plot_line_result`,
or the command line without `--no-plot`) show the line, the flux per annulus of baseline length
and the noise. `examples/plot_line.py` redraws them from the output files in
your own style ([Output files and figures](../extraction/outputs)).

Before believing a number, check that the imaginary part and the line-free
channels are at the noise level (`noise.imaginary_rms_over_noise` and
`noise.line_free_rms_over_noise` ≈ 1 in the results): see
[Things to check](../extraction/outputs.md#things-to-check).
