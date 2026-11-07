# Rest frame

The line of every source falls at a different observed frequency,
`ν_obs = ν_rest / (1 + z)`. Bringing each dataset to the rest frame of its
source puts all the lines at the same frequency, and all the channels on a
common axis.

## What is done

For every dataset with redshift `z`:

| Quantity | Rest frame | Why |
|---|---|---|
| channel frequencies | `ν × (1 + z)` | the line at its rest frequency |
| channel widths | `Δν × (1 + z)` | same velocity width |
| baselines `(u, v, w)` in metres | `/ (1 + z)` | the baselines in wavelengths, `u ν / c`, are unchanged |
| antenna positions, dish diameters | `/ (1 + z)` | same geometry, same primary beam at the rest frequency |

Since the baselines in metres are divided by the factor by which the
frequencies are multiplied, the uv coordinates in wavelengths — what the
interferometer actually measures — do not change: an image of a dataset
before and after the rest-framing is the same, only the frequency axis is
relabelled. In the stack, the baseline length in kλ of every visibility is
the observed one.

## The frame of the redshifts

`ν × (1 + z)` is right only if the frequencies of the input are in the frame
in which `z` is defined. Check the frame of an MS with `listobs` (column
*Frame*) or `MEAS_FREQ_REF` in its `SPECTRAL_WINDOW` table (1 LSRK, 3 BARY,
5 TOPO):

| Input | What happens | What to do |
|---|---|---|
| same frame as `z` | correct | nothing |
| TOPO (ALMA, VLA calibrated data) | the Earth's motion moves the line with the date of each observation, by up to ±30 km/s | `run(frame=...)` with the frame of `z` |
| another frame (BARY with `z` in LSRK, or the reverse) | a constant offset per direction, up to ~20 km/s | `run(frame=...)` |
| NOEMA | already in LSR(K): NOEMA tracks the Doppler shift at the telescope | check that the MS is labelled LSRK; if it says TOPO, fix the label, do not convert |
| REST | already rest frequencies | not accepted |

With `frame="LSRK"` (or `"BARY"`), for every row the Doppler factor
`k = ν_frame / ν_input` is computed from the time of the row, the position of
the array and the direction of the source (python-casacore measures), and the
kernel uses the frequencies `ν k` and the redshift `(1 + z) k − 1`. This is
exact: the phase shift keeps the observed frequency, and the channels are
rebinned once, with no extra resampling (unlike converting the inputs with
`mstransform` first). On a simulated stack of four ALMA observations three
months apart, the line comes back to its true width (60 km/s) and position,
while without the correction it is 73 km/s wide and shifted by 13 km/s.

`k` is computed in the main process before the readers start, at knots every
60 s around the data, and interpolated linearly for every row: the error is
below 0.2 mm/s. Computed in a fixed order, it is the same bit for bit
whatever the readers, the chunk size or the device.

Without `frame`, the frequencies are used as they are, and TOPO inputs give a
warning with the size of the effect.

The stack is labelled LSRK, so that `tclean` leaves its frequencies as they
are (with a TOPO label it would convert every row with the time of its
observation and the direction of the stack, broadening the line).
