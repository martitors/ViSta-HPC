# The input list

**In one sentence.** One plain text file, one dataset per line, says which
observations to stack, the redshift of each galaxy and where it is; the
stacking and the measurement both read it.

```text
# <ms_path>  <redshift>  <RA>  <Dec>  [FIELD_ID]  [SPW_IDS]  [NORM]
/data/obs1.ms  2.310  12:34:56.7  +12:34:56.7
/data/obs2.ms  1.987  12:34:56.7  +12:34:56.7  0  1
/data/obs3.ms  2.105  12:34:56.7  +12:34:56.7  3  0,2
```

| Column | Meaning |
|---|---|
| `ms_path` | the Measurement Set |
| `redshift` | the redshift of the source, in the frame of the `frame` option of the stacking (see [Rest frame](../stacking/restframing)) |
| `RA`, `Dec` | the position of the **source**, not the phase centre: the visibilities are re-centred on it. `hh:mm:ss.s ±dd:mm:ss.s` |
| `FIELD_ID` | optional. Without it every field is read and the phase centre is that of field 0. Give it when the MS holds several fields (a mosaic, calibrators) |
| `SPW_IDS` | optional, comma-separated spectral windows; without it all are used. Needs `FIELD_ID` before it |
| `NORM` | optional normalisation of the flux scale, used only by the measurement with `v.extract(flux_norm=True)` (a luminosity, a continuum flux, any proxy). Recognised because it is neither an integer nor a list of integers; `norm=2.91e13` also works |

Text after `#` is a comment.

## One stack entry per line

**What.** Every line of the list becomes one entry of the stack (one
spectral window, in the order of the list).

**Why.** The measurement finds each dataset again from the same file: there
is no second file to keep in sync.

**What to watch.** An entry that gives no data keeps its place, with no
rows. To remove a line by hand after the stacking, replace it with
`# EXCLUDED`, so that the numbering does not shift.

## The same galaxy, observed more than once

Lines with the same coordinates (within `match_tol_arcsec`, 2″ by default)
are different observations of one galaxy:

```text
/data/obs2.ms  2.561  00:32:07.60  -30:37:35.2  11  25,27  6.05e12
/data/obs3.ms  2.561  00:32:07.60  -30:37:35.2  12  23,25  6.05e12
```

The measurement adds their statistics before averaging the galaxies. So
the repeated observations are combined with their own weights, and in the
democratic weighting each galaxy counts once. Nothing else identifies a
galaxy: no name column, no separate coordinate file.
