# Measuring the stack

**What is measured.** For every line: its flux, centre, width and peak, how
sure the detection is, and the size of the source. For the continuum: its
flux density and its size.

**On what.** Not on the visibilities, but on their
[compressed statistics](../stacking/compression): a file of a few MB with
the sums of the visibilities per channel and per ring (*annulus*) of
baseline length. It is read in a second, so a measurement can be repeated
at will.

**Why it matters.** Everything that is a choice (which galaxies, how they
are weighted, whether they are brought to the same redshift, the source
model, the limits of the search) is applied here and never to the data on
disk. The same stack serves every measurement.

```python
from vista import ViSta

v = ViSta("input_list.txt", output_name="co32")   # where the stack and its statistics are
v.stack(line_freq=345.79599, line_window=500)     # the lines of the stack
v.extract()                                       # the options of the measurement
result = v.run(only="measure")                    # on the stack that is already there
print(result)
```

## The pages

| Page | What you learn |
|---|---|
| [Measuring the line](measure_line.md) | how the line is fitted, the two fluxes, the source model, the limits of the search, several lines, the errors |
| [The continuum](extract_flux.md) | how the continuum is measured, which size it gets, the continuum alone |
| [Weighting and normalisation](weighting.md) | natural or democratic; the redshift and primary-beam corrections; the flux normalisation |
| [Output files and figures](outputs) | the `results` dictionary, the figures, the files, what to check |
| [Validation](validation) | how well it works on simulations: bias, scatter, and whether the errors hold |

## Why not image the stack?

The stack is a standard interferometric dataset, so it can be imaged
(`vista.export_ms`, then `tclean`). On a stack of different observations
that is awkward: the image has a beam made of very different resolutions,
and the flux of an extended source depends on the aperture you choose.

The measurement in the uv plane fits the visibilities directly, on every
baseline the sample covers. The flux it returns is the one at zero baseline,
which is the total flux: no aperture, no deconvolution.

The image is still the place to **look** at the source: where the emission
is, and whether it is round, which the fit assumes. `result.plot("image")`
gives a quick one.
