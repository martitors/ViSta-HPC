# Centring

**In one sentence.** Every dataset is shifted so that its galaxy sits at
the phase centre; only then do the visibilities of different galaxies add
up.

**Why it matters.** A galaxy away from the phase centre has visibilities
whose phase turns with the baseline. Averaged with those of another galaxy,
somewhere else in its own field, they cancel. At the phase centre the phase
is zero on every baseline, and the signals add.

## What is done

For every row, in the kernel:

1. the baselines `(u, v, w)` are rotated to the frame of the new phase
   centre, the position of the galaxy in the input list;
2. the visibilities are multiplied by the phase of the shift,

   ```{math}
   V' = V \exp\left[-2\pi i \,\frac{\nu}{c}\,(u\,l + v\,m + w\,(n - 1))\right],
   \qquad n = \sqrt{1 - l^2 - m^2},
   ```

   with `(l, m)` the direction of the galaxy seen from the old phase
   centre.

The `w (n − 1)` term is kept: the shift is exact, not the small-field
approximation. The input is single precision; the phase and the products
are computed in double precision and rounded once, when the stack is
written.

## Which data are read

| | What is read | When to say it |
|---|---|---|
| **Field** | the `FIELD_ID` of the input list; without it, every field, with the phase centre of field 0 | whenever the MS holds several fields |
| **Scans** | only those with the `OBSERVE_TARGET` intent, when the MS has intents | — (calibrators are left out) |
| **Column** | `data_column="auto"`: `CORRECTED_DATA` when present, otherwise `DATA` | when `CORRECTED_DATA` holds something else |
| **Channels** | only those in the output window ([Spectral grid](spectral_grid)) | — |

The log gives, for each dataset, the rows used, those removed by the
selection, and the distance of the galaxy from the phase centre (`offset`).

## The primary beam

A galaxy away from the pointing centre is seen fainter: the antennas are
less sensitive there. The re-centring does not change that. The correction
is a factor on the amplitudes, applied when the stack is measured
(`v.extract(pb_corr=True)`, the default; see
[Weighting](../extraction/weighting.md#the-primary-beam)).

A galaxy beyond half the primary beam gives a warning: the Gaussian beam
used for the correction is no longer reliable there.

## Positions matter

**What a wrong position does.** A position error of a fraction of the beam
does not lose the flux of one galaxy, but it smears the stack: galaxies
stacked with random offsets add up to an *extended* source.

**How much.** In the simulations, point sources stacked with 0.3″ position
errors lost 55 % of their flux when measured as a point source. The fit
with a free size recovered it within 4 % ([Validation](../extraction/validation)).

**How to check.** The imaginary part of the stack: for centred, symmetric
sources it is pure noise (`noise.imaginary_rms_over_noise` ≈ 1; the grey
crosses of the figures against the baseline length).
