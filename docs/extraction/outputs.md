# Results, figures and files

**In one sentence.** A run gives you the same measurement in four forms: a
report in words, a dictionary of numbers, figures, and files on disk.

```python
result = v.run()
print(result)            # the report
result.results           # the numbers
result.plot()            # the figures
result.files             # the files
```

## The report

`print(result)` says what was measured **and why each choice was made**:

- **Line**: the S/N, and how sure the detection is after the search over
  centre and width;
- **flux**: the number to quote, and whether it is the Gaussian fit or the
  area in the window, with the reason;
- **source**: a point or a Gaussian, with the Δχ² that decided;
- **Continuum**: its flux density, its model, and what was done with it;
- **Sizes**: the size from the line and from the continuum, each measured
  on its own, and how far apart they are;
- **Reference**: the weighting and the redshift the fluxes refer to.

## The numbers: `result.results`

One dictionary with everything measured. Every quantity has its `value`,
`error` and `unit`.

```python
res = result.results
res["flux"]          # {'value': 0.404, 'error': 0.040, 'unit': 'Jy km/s', 'method': 'fit', 'reason': ...}
res["flux_fit"]      # area of the Gaussian fit
res["flux_window"]   # area under the spectrum in the window
res["centre"], res["fwhm"], res["peak"]
res["snr"], res["significance"]
res["size"]          # FWHM of the source used for the line (0: a point)
res["model"]         # {'chosen': 'gauss', 'size_from': 'own fit', 'probability': ..., 'delta_chi2': ...}
res["window"], res["noise"]
res["continuum"]     # flux density, S/N, size, model, how it was treated
res["sizes"]         # line and continuum, each on its own, and their difference in sigma
res["lines"]         # the same keys for every line (the ones above are of the first)
print(res["summary"])  # all of it in words: the same as print(result)
```

- **Units.** Fluxes are in data units × km/s for a line (Jy km/s for data
  in Jy) and data units for the continuum. The flux normalisation rescales
  them by `flux_ref / NORM`.
- **`model["probability"]`** is how much more likely the chosen source
  model is than the other one (point or Gaussian). It comes from the Δχ² of
  the two fits, with a penalty for the extra parameter of the Gaussian
  (Akaike weights). The choice itself is made on the fixed threshold
  Δχ² ≥ 9.
- **Everything else** (every error separately, the ranges searched, the
  checks of the noise) is in `result.line.summary` and
  `result.continuum.summary`, the dictionaries written to the JSON files.

## The figures

```python
result.plot()                 # all those of the measurement
result.plot("line")           # one at a time
```

| `plot(...)` | File | What it shows |
|---|---|---|
| `"line"` | `<name>_profile.png` | the stacked spectrum of the line, the fit, the window, the two fluxes |
| `"line_uv"` | `<name>_uv.png` | the line flux against the baseline length, with the two source models |
| `"continuum"` | `<name>_continuum.png` | the channels outside the line windows, and the level measured |
| `"continuum_uv"` | `<name>_continuum_uv.png` | the continuum against the baseline length, with the two source models |
| `"noise"` | `<name>_noise.png` | the checks of the noise and of the detection |
| `"image"` | — | a quick image of the stack over the line window |

The files are drawn by the run unless `v.extract(figures_draw=False)`
(`--no-figures`). `save=` writes what `plot` draws.

**How to read them.** The same conventions hold in every figure:

- the **title** says what is stacked, in bold: `line (contsub)` after the
  continuum subtraction, `line + cont` when the continuum was fitted with
  the line, `cont`;
- the axis says **Re**: the real part of the visibilities, where the signal
  of a centred source is. The imaginary part (**Im**, grey crosses) holds
  only noise;
- **red** is the Gaussian (the fit of the line; the Gaussian source);
  **green dashed** is the point source; **blue** is the continuum;
  **orange** is the window;
- what was **chosen** is in **bold**: the flux quoted, the source model
  used.

```{image} ../_static/fig_line.png
:alt: The stacked line
:width: 49%
```
```{image} ../_static/fig_line_uv.png
:alt: The line against the baseline length
:width: 49%
```

```{image} ../_static/fig_continuum.png
:alt: The stacked continuum
:width: 49%
```
```{image} ../_static/fig_continuum_uv.png
:alt: The continuum against the baseline length
:width: 49%
```

[Measuring the line](measure_line) and [The continuum](extract_flux)
explain every element. The image (`"image"`, {func}`vista.dirty_image`) is
a quick look at where the emission is, not a cleaned image.

**Several measurements together.**
`vista.compare({"label": result, ...}, expected=...)` draws them in one
figure and prints their numbers side by side.

## The files

Every file starts with `output_name` (`stacked` by default). With
`run(only="measure", tag="T")` the files of the measurement start with
`<output_name>_T`, so they do not replace the others.

| File | Content |
|---|---|
| `<name>.zarr` (or `.ms`) | the stack ([The stack on disk](../stacking/combination)) |
| `<name>_line_stats.npy`, `<name>_cont_stats.npy` | the statistics of the line and of the continuum ([Compression](../stacking/compression)) |
| `<name>_results.json` | every number of the lines |
| `<name>_profile.txt` | per channel: `v_kms freq_GHz flux_density err_thermal err_bootstrap imaginary model fitted line_free n_sources` |
| `<name>_uvamp.txt` | per annulus: `b_klambda line_flux err_thermal used n_sources line_flux_imaginary` (the first line) |
| `<name>_noise.txt` | per channel: the noise from the scatter of the visibilities and from the weights |
| `<name>_diagnostics.npz` | the channel correlation matrix, the standardised cells, the null distributions of the S/N |
| `<name>_continuum_results.json`, `_spectrum.txt`, `_uvamp.txt` | the continuum ([The continuum](extract_flux)) |

**In the JSON**, every line has (in `lines`; the first also in `line`):

- `flux`, `flux_error` and `flux_method`: the flux quoted, and whether it
  is `gauss` or `area`;
- `flux_gauss`, `flux_area`: the two fluxes, each with its thermal and
  bootstrap error;
- `shape_check`: their difference, its error, the reason for the choice;
- `v0_kms`, `fwhm_kms`, `peak`, with their errors;
- `detection`: S/N, significance, the ranges searched.

The first line also has the size (`theta_fwhm_arcsec`, `size_model`) and
the χ² of the fit.

## Figures in your own style: `plot_line.py`

`examples/plot_line.py` draws the same three figures from the output files
alone, with only NumPy and Matplotlib (it does not import ViSta, so it can be
copied next to a paper):

```bash
python examples/plot_line.py <prefix> --unit Jy --rebin 2 --vlim -800 800 --format pdf
python examples/plot_line.py <prefix> --style thesis.mplstyle --only profile
```

Colours, line styles, sizes and fonts are in the `STYLE` dictionary at the
top, or in a Matplotlib style file (`--style`, applied on top); each figure is
one function (`plot_profile`, `plot_uv`, `plot_noise`) that returns the
figure, commented on what every element is and how it is computed — e.g. the
rebinned errors use the channel correlation matrix,
`var(mean) = Σᵢⱼ Rᵢⱼ σᵢ σⱼ / n²`, not `σ / √n`. From Python:

```python
import plot_line as pl
pl.STYLE["model"]["color"] = "C2"
pl.apply_style()
d = pl.load("<prefix>")
fig = pl.plot_profile(d, unit="Jy", rebin=2, reference=(v_true, s_true))
fig.axes[0].set_ylim(-3, 6)
fig.savefig("line.pdf")
```

`plot_profile` and `plot_uv` take a `reference=(x, y)` curve to compare with,
in data units: the true signal of a simulation (as in the [Examples](../examples/index)),
another line, a model. Without `_diagnostics.npz` (outputs of older
versions) the noise histograms are replaced by the rms values of the JSON
and the rebinned errors ignore the correlation.

## Things to check

| Key | What it tells you |
|---|---|
| `detection.significance_sigma`, `p_value` | is there a line at all. In a search this, not `F / σ_F`, is the evidence |
| `line.flux_method`, `line.shape_check` | which flux is quoted, and whether the line looks Gaussian (`nsigma`: the area minus the Gaussian flux, in units of the error of their difference) |
| `continuum_residual` | offset and slope of the profile outside the windows; `detected: true` means continuum left (or not subtracted) |
| `noise.imaginary_rms_over_noise`, `line_free_rms_over_noise` | ≈ 1. Above: centring, calibration, residual continuum or an unmasked line; the thermal errors are optimistic there |
| `noise.kappa_scatter_over_weights`, `kappa_per_entry` | how wrong the weights are; one entry far from the others has bad weights (harmless, the scatter is used) or bad data |
| `noise.channel_correlation_method` | whether the whole correlation matrix was measured (≥ 30 source × annulus vectors) |
| `line.bootstrap_over_thermal` | ≈ 1 for a uniform population; > 1 the population scatters (or systematics); < 1 with few sources |
| `detection.snr_imaginary` | the imaginary part through the line filter: ≈ 0 |
| `size_test` | point against Gaussian: Δχ² (corrected for the channel correlation) and both fluxes |
| `line.chi2_reduced` | ≈ 1; much larger: one Gaussian, separable in velocity and baseline, does not describe the data |
| `n_sources`, `n_entries` | how many sources the lines of the input list were merged into |
| `warnings` | e.g. a weak line in a search, whose free-shape flux is unreliable; a line that is not Gaussian; continuum outside the windows |
