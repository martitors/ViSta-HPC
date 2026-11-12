# Companions

**In one sentence.** A second source in the field of one of your galaxies
is not at the centre of the stack, but part of it ends up in the stack all
the same; `companions` in `v.stack` removes it.

```python
v.stack(line_freq=345.79599, line_window=500, companions="companions.txt")
```

## Why it matters

The stack is centred on its galaxies. A companion a few arcseconds away has
visibilities that oscillate with the baseline: on the long baselines they
average out, on the short ones, which do not tell the two sources apart,
they add to the galaxy. The result is a flux that is too high and a stack
that **looks extended**, because the short baselines lie above the long
ones. [Example 14](../examples/14_a_companion_in_the_field) shows it: a
companion 1.4″ from one galaxy out of twelve turns a 0.40 Jy km/s point
source into a 0.52 Jy km/s source 0.7″ across.

| Companion | Flux of the stack | Size found |
|---|---|---|
| 1.4″ away, not removed | 0.52 ± 0.12 Jy km/s | 0.72″ |
| 2.3″ away, not removed | 0.45 Jy km/s | point |
| 3.6″ away, not removed | 0.42 Jy km/s | point |
| removed (any of the three) | 0.40 ± 0.015 Jy km/s | point |
| *true* | *0.40 Jy km/s* | *point* |

(Twelve point sources of 0.4 Jy km/s on baselines up to 500 m; the
companion has a line of 2 Jy km/s. The farther it is, the faster it
oscillates and the less of it is left.)

```{figure} ../_static/fig_companion_left.png
:width: 70%

The companion left in: the short baselines lie above the long ones, as for
an extended source.
```

```{figure} ../_static/fig_companion_removed.png
:width: 70%

The companion removed: flat, a point source with the right flux.
```

A companion at another redshift has no line in your window: only its
continuum is there, and the continuum subtraction, done row by row, already
removes most of it. A companion **at the same redshift** has its line on
yours, and no continuum fit can remove that.

## The file

One companion per line:

```text
# <ms_path>            <RA>           <Dec>          [redshift]
data/galaxy_05.ms      10:02:31.42    +02:30:12.8    2.6031
data/galaxy_09.ms      10:04:11.07    +02:54:03.1
```

- `ms_path`: the dataset the companion is in, written as in the input list.
- `RA`, `Dec`: its position, as in the input list.
- `redshift`: **only if it is known.** Without it the companion has no
  known line, and only its continuum is removed.

A dataset that is in the input list more than once (several targets in one
field) takes the companion for each of its targets.

## What is done

For every dataset that has companions, after the stack is written and
before the continuum subtraction:

1. **A joint fit.** The galaxy and its companions are fitted together, as
   point sources at their positions, by weighted least squares on the
   visibilities of that dataset. The galaxy is always in the fit and is
   never subtracted.
2. **The continuum.** One flux per source, on the channels outside every
   line window (those of the galaxy, and those of the companions with a
   redshift). It is subtracted from all the channels **if it is detected**
   (S/N ≥ 3): subtracting a flux compatible with zero would only add noise.
3. **The line**, for the companions with a redshift. In the window where
   the line of the companion falls (the window of the galaxy, at the
   velocity of the companion), one flux per channel is fitted and
   subtracted.
4. **Too close to tell apart.** A companion closer to the galaxy than the
   resolution of that dataset cannot be separated from it: any split of the
   flux between the two fits the data. It is recognised by the *variance
   inflation*: by how much the variance of the flux of the companion grows
   because the galaxy is fitted with it (1: not at all). Above 2 the
   companion is left out of the fit and nothing of it is subtracted.

After the subtraction, the galaxy is what the joint fit gave for it: the
companion takes with it only what the data attribute to its position.

## What to check

`print(result)` has one line per companion, and `result["companions"]` the
same numbers:

```text
Companions galaxy_05.ms, 1.4" from the target, z = 2.6031: continuum 0.988 ± 0.047 mJy
           subtracted; line 1.99 ± 0.026 Jy km/s subtracted (50 channels)
```

- **The fluxes of the companion** should be those you know from its image.
  They are what the interferometer sees: not corrected for the primary
  beam.
- **"not subtracted, too close to the target"**: that companion is still in
  the stack. On longer baselines it could be separated; here, consider
  leaving that galaxy out of the stack.
- **"its line is outside the band"**: its redshift puts the line outside
  the channels of the stack; only the continuum matters.
- **The plot against the baseline length** (`result.plot("line_uv")`): flat
  for point sources. Short baselines still above the long ones say that
  something is left.

## Limits

- The companions are **point sources**. An extended companion is only
  partly removed.
- The positions are **fixed**: a wrong position leaves part of the
  companion in the data.
- The subtraction is part of the stacking: to change the companions, stack
  again. The stack keeps what was done (`open_stack(...).get_record("companions")`).
- With companions the stack is read twice more (the fit, the subtraction)
  before the continuum subtraction, so the run takes longer.
