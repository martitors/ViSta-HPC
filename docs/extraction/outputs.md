# Output files and figures

## Files

Everything a run writes starts with `output_name` (`stacked` by default; with
`run(only="measure", tag="T")` the measurement files start with
`<output_name>_T`):

| File | Content |
|---|---|
| `<name>.zarr` (or `.ms`) | the stack ([The stack on disk](../stacking/combination)) |
| `<name>_line_stats.npy`, `<name>_cont_stats.npy` | the statistics of the line and of the continuum ([Compression](../stacking/compression)) |
| `<name>_results.json` | every number of the lines: `line` (the first), `lines` (all of them), `detection`, `window`, `size_test`, `continuum_residual`, `continuum_fit` (the continuum fitted with the lines, when asked), `noise`, the record of the continuum subtraction |
| `<name>_profile.txt` | per channel: `v_kms freq_GHz flux_density err_thermal err_bootstrap imaginary model fitted line_free n_sources` |
| `<name>_uvamp.txt` | per annulus: `b_klambda line_flux err_thermal used n_sources` (flux of the first line at the fitted shape) |
| `<name>_noise.txt` | per channel: the noise of the profile from the scatter of the visibilities and from the weights |
| `<name>_diagnostics.npz` | `channel_correlation` (the matrix), `z_imaginary`, `z_line_free` (cells over their noise), `null_snr`, `null_snr_phase` (null distributions of the detection S/N) |
| `<name>_continuum_results.json`, `_spectrum.txt`, `_uvamp.txt` | the continuum: flux at zero spacing, size, flux per annulus ([The continuum measurement](extract_flux)) |

The same numbers are in the `RunResult` of `run()`: `result.line.summary`
(the dictionary of the JSON) and `result.continuum.summary`. Fluxes are in
data units × km/s for a line (Jy km/s for data in Jy), data units for the
continuum; the flux normalisation rescales them by `flux_ref / NORM`.

**Every line** has in `lines` (and the first in `line`): `flux`,
`flux_error` (the larger of the thermal and the bootstrap error) and
`flux_method` (which flux they are: `gauss` or `area`); `flux_gauss` and
`flux_area` with their errors (thermal, bootstrap); `flux_optimal` (1.19
FWHM); `shape_check` (the difference of the two fluxes, its error, the
reason for the choice); `v0_kms`, `fwhm_kms`, `peak` with their errors;
`detection` (S/N, significance, the ranges searched). The first line also
has the size (`theta_fwhm_arcsec`, `size_model`) and the χ² of the fit,
shared by all the lines.

## Figures

The run draws (unless `v.extract(figures_draw=False)`, `--no-figures`):

- `<name>_profile.png`: the profile with the fitted lines, the area of the
  Gaussian (hatched) and the window of the area flux (shaded) of every line,
  both fluxes, the ±1σ band, the imaginary part (a null test), the continuum
  left outside the windows and the residuals;
- `<name>_uv.png`: the line flux per annulus against the baseline, with
  the point and the Gaussian model;
- `<name>_noise.png`: the noise per channel from the scatter and from the
  weights, the standardised cells against N(0, 1), the null distribution of
  the detection S/N;
- `<name>_continuum_uvamp.png`: the continuum per annulus, with its fit.

In a notebook the same figures come from the result itself:
`result.plot()` (profile and uv), `result.plot("noise")`,
`result.plot("continuum")`; `save=` also writes them. `vista.compare({label:
result, ...}, truth=...)` draws several measurements in one figure, the
profiles with their fits, the flux per annulus with the source models and
the fluxes with their errors, and prints their numbers side by side (the
[Examples](../examples/index) use it throughout).

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
