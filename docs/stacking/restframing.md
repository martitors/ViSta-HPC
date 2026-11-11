# Rest frame

**In one sentence.** The line of every galaxy is observed at a different
frequency, `ν_obs = ν_rest / (1 + z)`; bringing each dataset to the rest
frame of its galaxy puts all the lines at the same frequency.

**Why it matters.** Only then do the channels of different datasets mean
the same thing, and the lines add up instead of being spread over the band.

## What is done

For every dataset with redshift `z`:

| Quantity | Rest frame | Why |
|---|---|---|
| channel frequencies | `ν × (1 + z)` | the line at its rest frequency |
| channel widths | `Δν × (1 + z)` | same velocity width |
| baselines `(u, v, w)` in metres | `/ (1 + z)` | the baselines in wavelengths, `u ν / c`, are unchanged |
| antenna positions, dish diameters | `/ (1 + z)` | same geometry, same primary beam at the rest frequency |

**Nothing physical changes.** The baselines in metres are divided by the
same factor the frequencies are multiplied by. So the baselines in
wavelengths, which is what the interferometer measures, stay the same: an
image of a dataset before and after is identical, only the frequency axis
has new labels. In the stack, the baseline length in kλ of every visibility
is the observed one.

## The frame of the redshifts

**The catch.** `ν × (1 + z)` is right only if the frequencies of the data
are in the frame in which `z` is defined.

**What ViSta does.** It always brings the input frequencies to the frame of
the redshifts: `frame="LSRK"` by default.

**What you have to say.** Give `frame="BARY"` (`--frame BARY`) when your
redshifts are barycentric or heliocentric, as optical spectroscopic
redshifts usually are. The two frames differ by up to ~20 km/s, depending
on the direction of the source.

**How to check the frame of an MS.** `listobs` (column *Frame*), or
`MEAS_FREQ_REF` in its `SPECTRAL_WINDOW` table (1 LSRK, 3 BARY, 5 TOPO).

| Input | Without the correction | What ViSta does |
|---|---|---|
| same frame as `z` | correct | nothing: left as it is |
| TOPO (ALMA, VLA calibrated data) | the Earth's motion moves the line with the date of each observation, by up to ±30 km/s | corrected row by row |
| another frame (BARY with `z` in LSRK, or the reverse) | a constant offset per direction, up to ~20 km/s | corrected |
| NOEMA | already in LSR(K): NOEMA tracks the Doppler shift at the telescope | check that the MS is labelled LSRK; if it says TOPO, fix the label, or it is corrected twice |
| REST | already rest frequencies | not accepted |

## How the correction is made

**The Doppler factor.** For every row that needs it, `k = ν_frame / ν_input`
is computed from the time of the row, the position of the array and the
direction of the source (python-casacore measures). The kernel then uses
the frequencies `ν k` and the redshift `(1 + z) k − 1`.

**It is exact.** The phase shift keeps the observed frequency, and the
channels are rebinned once, with no extra resampling (unlike converting the
inputs with `mstransform` first).

**What it buys.** On a simulated stack of four ALMA observations three
months apart, the line comes back to its true width (60 km/s) and position.
Without the correction it is 73 km/s wide and shifted by 13 km/s.

**Details.**

- `k` is computed before the readers start, at knots every 60 s around the
  data, and interpolated for every row: the error is below 0.2 mm/s. It is
  the same bit for bit whatever the readers, the chunk size or the device.
- The conversion needs the measures data of casacore. Without them, a run
  on inputs in another frame stops with an error that says so, rather than
  stacking lines displaced by the motion of the Earth.
- The stack is labelled LSRK, so that `tclean` leaves its frequencies as
  they are. With a TOPO label it would convert every row with the time of
  its observation and the direction of the stack, and broaden the line.
