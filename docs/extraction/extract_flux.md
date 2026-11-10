# The continuum measurement

The continuum subtracted during the stacking (`MODEL_DATA`) is compressed
into `<name>_cont_stats.npy` and measured by every `run()` with the
per-annulus chain of Sec. 5.3 of the thesis (`vista.extract.extract_flux`):
the mean of the channels outside the line windows in every annulus, and a
circular Gaussian fitted to it against the baseline, which gives the flux
density at zero spacing and the size. The errors come from the bootstrap over
the sources, so the continuum is measured only with at least two sources and
`n_bootstrap >= 2`. It writes `<name>_continuum_results.json`, `_spectrum.txt`,
`_uvamp.txt` and the figure `_continuum_uvamp.png`; `result.continuum` holds
the same.

Earlier versions also offered this chain for the line (`--method template`
and `window`). The joint fit ([Measuring the line](measure_line.md)) replaces
it: in the simulations, with a known width, it halves the scatter of the
template method, and its errors are right where those of the chain were too
small for a point model and too large for a free Gaussian
([Validation](validation)). The window flux of the old chain is the area flux
of the joint fit.

## How it is measured

- **Flux per annulus**: the mean of the continuum over the channels covered by
  every source, outside the windows of the lines, annulus by annulus, with
  the same weighting, redshift rescaling and primary-beam correction as the
  line (the weighting options of `v.extract(...)`).
- **Source model**: a circular Gaussian fitted to the flux against the
  baseline, `F_tot = V(b = 0)` and `θ_FWHM`, up to 15″; annuli with fewer
  than two sources are left out.
- **Errors**: from the bootstrap over the sources, with a floor of 0.1 of
  their median, so that no annulus dominates with few sources.

## Outputs

`result.continuum` is an `ExtractionResult`; the run writes
`<name>_continuum_results.json` (every fitted quantity: `flux_total`,
`theta_fwhm_arcsec` and their errors), `<name>_continuum_spectrum.txt` (the
continuum per channel) and `<name>_continuum_uvamp.txt` (flux per annulus,
with the imaginary part), and draws `<name>_continuum_uvamp.png`. The
continuum size can be used as a prior on the line size:
`v.extract(model_prior=(theta, 0.15))`.

Things to check: `null_test_max_imaginary_over_sigma` (centring),
`theta_at_bound` and `bootstrap_fraction_at_bound` (unconstrained size),
`chi2_reduced`, `n_bins_in_fit`, `n_bootstrap_failed`, `flux_point_source`
against `flux_total` (a large ratio: a resolved source).
