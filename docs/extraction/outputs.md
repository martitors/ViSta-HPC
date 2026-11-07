# Output files and figures

## Files

With `output_prefix` (`--out-dir`/`--tag` on the command line),
`measure_line` writes:

| File | Content |
|---|---|
| `<prefix>_results.json` | every number: `line`, `detection`, `window`, `size_test`, `noise`, the record of the continuum subtraction |
| `<prefix>_profile.txt` | per channel: `v_kms freq_GHz flux_density err_thermal err_bootstrap imaginary model fitted line_free n_sources` |
| `<prefix>_uvamp.txt` | per annulus: `b_klambda line_flux err_thermal used n_sources` (line flux at the fitted line shape) |
| `<prefix>_noise.txt` | per channel: the noise of the profile from the scatter of the visibilities and from the weights |
| `<prefix>_diagnostics.npz` | `channel_correlation` (the matrix), `z_imaginary`, `z_line_free` (cells over their noise), `null_snr`, `null_snr_phase` (null distributions of the detection S/N) |

The results are also in `result.summary` (the same dictionary as the JSON).
Fluxes are in data units × km/s for a line (Jy km/s for data in Jy), data
units for the continuum; the flux normalisation rescales them by
`norm_ref / NORM`.

## Figures

`plot_line_result(result, line, prefix)` (and the command line without
`--no-plot`) draws three figures:

- `<prefix>_profile.png`: the profile with the model, the ±1σ band, the
  imaginary part (a null test), the windows and the residuals;
  `--plot-rebin N` averages N channels, with the correlation in the errors;
- `<prefix>_uv.png`: the line flux per annulus against the baseline, with
  the point and the Gaussian model;
- `<prefix>_noise.png`: the noise per channel from the scatter and from the
  weights, the standardised cells against N(0, 1), the null distribution of
  the detection S/N.

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
| `noise.imaginary_rms_over_noise`, `line_free_rms_over_noise` | ≈ 1. Above: centring, calibration, residual continuum or an unmasked line; the thermal errors are optimistic there |
| `noise.kappa_scatter_over_weights`, `kappa_per_entry` | how wrong the weights are; one entry far from the others has bad weights (harmless, the scatter is used) or bad data |
| `noise.channel_correlation_method` | whether the whole correlation matrix was measured (≥ 30 source × annulus vectors) |
| `line.bootstrap_over_thermal` | ≈ 1 for a uniform population; > 1 the population scatters (or systematics); < 1 with few sources |
| `detection.snr_imaginary` | the imaginary part through the line filter: ≈ 0 |
| `size_test` | point against Gaussian: Δχ² (corrected for the channel correlation) and both fluxes |
| `line.chi2_reduced` | ≈ 1; much larger: one Gaussian, separable in velocity and baseline, does not describe the data |
| `n_sources`, `n_entries` | how many sources the lines of the input list were merged into |
| `warnings` | e.g. a weak line in a search, whose free-shape flux is unreliable |
