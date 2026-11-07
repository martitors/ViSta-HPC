# The per-annulus chain

`extract_flux` (`run_vista.sh --method template|window`) is the method of
Sec. 5.3 of the thesis, kept for comparison and for the continuum: the line
shape is fitted once on the spatially integrated spectrum and held fixed, the
amplitude is fitted annulus by annulus, and a circular Gaussian (or a point
source) fitted to the amplitude against the baseline gives the total flux
and the size. The errors come from the bootstrap over the sources.

```python
from vista.extract import (extract_flux, ProfileConfig, UVFitConfig,
                           WeightingConfig, BootstrapConfig)

result = extract_flux("line_stats.npy", "input_list.txt", line,
                      WeightingConfig(scheme="democratic"),
                      ProfileConfig(method="template"),
                      UVFitConfig(model="gauss"),
                      BootstrapConfig(n_realisations=500),
                      output_prefix="co32_democratic")
print(result.summary["flux_total"], result.summary["theta_fwhm_arcsec"])
```

For a line, [`measure_line`](measure_line.md) is the better measurement: in the
simulations the joint fit with a known width halves the scatter of the
template method, the errors of the chain are too small for a point model and
too large for a free Gaussian, and a free Gaussian runs away at low S/N
([Validation](validation)).

## The flux per annulus

| `ProfileConfig.method` | |
|---|---|
| `template` (default) | the shape (centroid, width) is fitted on the integrated spectrum and frozen; only the amplitude is fitted per annulus, and the flux follows analytically, with no truncation and no cross-contamination between blended lines |
| `window` | the plain sum over the line window |
| `continuum` | the mean of the line-free channels: the continuum, from `cont_stats.npy` |

`joint_continuum=True` fits the line on a constant continuum and measures
both (a stack not continuum-subtracted, `--contsub after`). `fit_span_kms`
is the half-width of the shape fit; `shape_tie` ties a second line
(`free`, `centroid`, `centroid+width`); `fixed_shape_kms=(centroid, sigma)`
imposes the shape.

## The source model

| `UVFitConfig` | |
|---|---|
| `model="gauss"` (default) | circular Gaussian: `F_tot = V(b = 0)` and `θ_FWHM` |
| `model="point"` | an unresolved stack: a free Gaussian cannot then overfit a size the data do not constrain |
| `model="gauss2"` | compact plus extended, only with many well-populated annuli |
| `theta_fixed_arcsec`, `theta_prior`, `theta_max_arcsec` | fix the size, tie it softly (`(θ_ref, σ_dex)`), bound it (check `theta_at_bound`) |
| `b_max_klambda`, `min_sources_per_bin` | leave out the longest baselines, or annuli with few sources (raise it to 3–5 on heterogeneous samples) |
| `error_floor_frac` | floor on the bootstrap errors (0.1 of their median); 0 for noiseless simulations |

## The continuum

With `--contsub before`, the continuum is measured from `cont_stats.npy` (the
fitted continuum, `MODEL_DATA`) with `method="continuum"`, in its own fit and
with its own figures: `<tag>_continuum_results.json`, with its flux density
and size. The continuum size can be fed back as a prior on the line size
(`--theta-prior-from <tag>_continuum_results.json`).

## Outputs

`extract_flux` returns an `ExtractionResult`; with `output_prefix` it
writes `<prefix>_results.json` (every fitted quantity and the configuration),
`<prefix>_spectrum.txt` (the stacked visibility average, not a flux),
`<prefix>_spectrum_zerob.txt` (the same line in flux units, extrapolated to
zero baseline), `<prefix>_uvamp.txt` (flux per annulus, with the imaginary
part), and the files of a second line or of the continuum when fitted.
`plot_all(result, line, prefix)` (`--plot`) draws the figures: line flux per
annulus, spectrum and line profile; with `--contsub after` one figure with
three panels.

Things to check: `null_test_max_imaginary_over_sigma` (centring),
`theta_at_bound` and `bootstrap_fraction_at_bound` (unconstrained size),
`chi2_reduced`, `n_bins_in_fit`, `n_bootstrap_failed`, `flux_point_source`
against `flux_total` (a large ratio: a resolved source).
