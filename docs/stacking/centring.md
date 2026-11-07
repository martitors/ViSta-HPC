# Centring

Every dataset is re-centred on its source: after the phase shift the source
sits at the phase centre, `(l, m) = (0, 0)`, and the visibilities of
different sources add up coherently.

## What is done

For every row, in the kernel, in double precision:

1. the baselines `(u, v, w)` are rotated to the frame of the new phase
   centre (the source position of the input list);
2. the visibilities are multiplied by the phase of the shift,

   ```{math}
   V' = V \exp\left[-2\pi i \,\frac{\nu}{c}\,(u\,l + v\,m + w\,(n - 1))\right],
   \qquad n = \sqrt{1 - l^2 - m^2},
   ```

   with `(l, m)` the direction cosines of the source seen from the old phase
   centre. The `w (n − 1)` term is kept: the shift is exact, not the
   small-field approximation.

The input is single precision (complex64); the phase and the products are
computed in float64 and rounded once, when the stack is written.

## Which data are read

- **The field**: the `FIELD_ID` of the input list, or every field with the
  phase centre of field 0. Give it whenever the MS holds several fields.
- **The scans**: only the scans with the `OBSERVE_TARGET` intent, when the
  MS has intents (calibrators are left out).
- **The column**: `data_column="auto"` reads `CORRECTED_DATA` when present,
  otherwise `DATA`. Name it explicitly when `CORRECTED_DATA` holds something
  else (in a simulation, a corrupted copy).
- **The channels**: only those that fall in the output window
  ([Spectral grid](spectral_grid)).

The log gives, for each dataset, the rows used and those removed by the
selection, and the distance of the source from the phase centre (`offset`).

## The primary beam

A source away from the pointing centre is attenuated by the primary beam.
The re-centring does not change that; the correction is a factor on the
amplitudes, applied when the stack is measured (`WeightingConfig(primary_beam=True)`,
the default): amplitudes divided by the gain `A` and weights multiplied by
`A²`, so an attenuated source is corrected and also counts less. `A` comes
from a Gaussian beam of FWHM `1.2 c / (ν D)`, at the observed frequency of the
line and with the dish diameter `D` of the original MS (read during the
compression). A source beyond half the primary beam gives a warning: the
Gaussian approximation is no longer reliable there.

## Positions matter

The position of the input list is where the phase centre is put. A position
error of a fraction of the beam does not lose the flux of a single source,
but it spreads the stack: sources stacked with random offsets add up to an
*extended* stack. In the simulations, point sources stacked with 0.3″
position errors lost 55% of their flux with a point-source model, while the
joint fit with a free size recovered it within 4% ([Validation](../extraction/validation)).
The imaginary part of the stack is the check: for centred, symmetric sources
it is pure noise (`noise.imaginary_rms_over_noise` ≈ 1).
