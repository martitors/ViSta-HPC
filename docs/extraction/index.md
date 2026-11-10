# Measuring the stack

The stack is measured on its [compressed statistics](../stacking/compression),
not on the visibilities: a file of a few MB, read in a second. Everything
that depends on a choice — which sources, how they are weighted, whether they
are transported to a common redshift, the source model, the line model — is
applied here, so the stack keeps its native amplitudes and the same stack
serves every measurement.

```python
from vista import ViSta

v = ViSta("input_list.txt", output_name="co32")   # where the stack and its statistics are
v.stack(line_freq=345.79599, line_window=500)               # the lines (and how the stack was made)
v.extract(weighting_scheme="natural", z_norm=True, pb_corr=True,
          fwhm_range=(100, 600), n_bootstrap=500)
result = v.run(only="measure")          # from the statistics of an earlier run()
```

| Page | |
|---|---|
| [Measuring the line](measure_line.md) | the joint fit of all channels and annuli, the noise, the search within limits and its significance, several lines, the two fluxes, size, errors, residual continuum |
| [Weighting and normalisation](weighting.md) | natural or democratic; redshift rescaling; primary beam; flux normalisation; repeated observations |
| [Output files and figures](outputs) | what is written, the figures, `plot_line.py`, what to check |
| [The continuum measurement](extract_flux.md) | the per-annulus chain of Sec. 5.3, which measures the continuum |
| [Validation](validation) | how well it works on simulations; uv plane against image plane |

## Why not image the stack?

The stack is a standard interferometric dataset, so it can be imaged
(`export_ms`, then `tclean`) or fitted with a general tool (UVMultiFit). On a
stack of heterogeneous datasets both are awkward: the image inherits a hybrid
beam, made of very different resolutions, and needs an aperture whose size
decides the flux of a resolved source; a general fitter holds all the
visibilities in memory. The measurement in the uv plane fits the visibility
function directly, at every baseline the sample covers, and returns the flux
at zero baseline — the total flux — with no aperture and no deconvolution.
The image is still the place to look at the source: where the emission is,
whether it is symmetric, which the uv fit assumes (`vista.export_ms` writes
the stack as a Measurement Set, for `tclean`).
