# Measuring the stack

The stack is measured on its [compressed statistics](../stacking/compression),
not on the visibilities: a file of a few MB, read in a second. Everything
that depends on a choice — which sources, how they are weighted, whether they
are transported to a common redshift, the source model, the line model — is
applied here, so the stack keeps its native amplitudes and the same stack
serves every measurement.

```python
from vista.extract import (LineConfig, WeightingConfig, LineFitConfig, BootstrapConfig,
                           measure_line)

line = LineConfig(rest_freq_ghz=345.79599, v_window_kms=(-500, 500))
result = measure_line("line_stats.npy", "input_list.txt", line,
                      WeightingConfig(scheme="natural", redshift_rescaling=True, primary_beam=True),
                      LineFitConfig(mode="search"),
                      BootstrapConfig(n_realisations=500),
                      output_prefix="co32")
```

| Page | |
|---|---|
| [Measuring the line](measure_line.md) | `measure_line`: the joint fit of all channels and annuli, the noise, detection and significance, size, errors |
| [Weighting and normalisation](weighting) | natural or democratic; redshift rescaling; primary beam; flux normalisation; repeated observations |
| [Output files and figures](outputs) | what is written, the figures, `plot_line.py`, what to check |
| [The per-annulus chain](extract_flux.md) | `extract_flux`, the method of Sec. 5.3 (template and window), and the continuum measurement |
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
whether it is symmetric, which the uv fit assumes ([Example 9](../examples/09_the_stack_in_the_image)).
