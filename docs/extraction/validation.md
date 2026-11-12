# Validation

**In one sentence.** On synthetic stacks where the truth is known, the flux
of the line and of the continuum comes back within a few per cent for point
and for extended sources, with errors that mean what they say from 5 to 50
galaxies; it does not when the sources of one stack have very different
sizes, or when an extended source is measured as a point.

The [Examples](../examples/index) show single cases, one situation at a
time. This page has the statistics over many noise realisations.

## How it is done

`benchmarks/validation.py` makes the numbers of this page:

```bash
python benchmarks/validation.py --realisations 100 --out validation
```

**The stacks.** They are built directly as the compressed statistics that
ViSta measures, so a stack takes milliseconds to make. Every galaxy has its
own array (43 antennas within 600 m), its redshift (2 to 3) and its depth
(0.15 dex of scatter), so the galaxies do not weigh the same. The noise is
1.3 times what the weights say, and correlated between adjacent channels
(0.3). The continuum is subtracted visibility by visibility outside the
line window, as the stacking does, with the noise that this adds.

**The measurement.** That of `v.extract()` with its defaults, with 200
bootstrap realisations.

**What is counted**, over the realisations of a case:

| | Meaning | A good value |
|---|---|---|
| *bias* | mean of the measured flux over the true one, minus one | 0 |
| *scatter* | rms of the measured flux, over the true one | — |
| *quoted* | the median quoted error, over the true flux | equal to the scatter |
| *coverage* | fraction of the realisations whose flux ± 1σ holds the truth | 0.68 |

With 100 realisations a coverage is known to about ± 0.05, with 60 to
± 0.06.

## The errors, from 5 to 50 galaxies

The same total depth for every number of galaxies, so that only the number
changes. 100 realisations per row. The line has 0.4 Jy km/s and 300 km/s;
the continuum 0.2 mJy, fitted as a free Gaussian for the extended sources
and as a point for the points.

| Sample | Line: bias | scatter | quoted | coverage | *bootstrap alone* | Continuum: bias | scatter | quoted | coverage |
|---|---|---|---|---|---|---|---|---|---|
| extended 0.8″, 5 galaxies | −1% | 0.15 | 0.15 | 0.73 | *0.58* | +14% | 0.52 | 0.40 | 0.79 |
| extended 0.8″, 10 galaxies | +1% | 0.14 | 0.14 | 0.70 | *0.64* | +9% | 0.41 | 0.39 | 0.80 |
| extended 0.8″, 20 galaxies | +1% | 0.13 | 0.14 | 0.68 | *0.62* | +5% | 0.33 | 0.35 | 0.86 |
| extended 0.8″, 50 galaxies | +3% | 0.13 | 0.15 | 0.76 | *0.75* | +1% | 0.28 | 0.36 | 0.86 |
| point, 5 galaxies | +1% | 0.05 | 0.06 | 0.72 | *0.57* | 0% | 0.10 | 0.10 | 0.75 |
| point, 10 galaxies | +1% | 0.06 | 0.05 | 0.70 | *0.62* | 0% | 0.09 | 0.10 | 0.74 |
| point, 20 galaxies | +1% | 0.05 | 0.05 | 0.75 | *0.68* | −1% | 0.08 | 0.09 | 0.76 |
| point, 50 galaxies | 0% | 0.05 | 0.05 | 0.76 | *0.72* | +1% | 0.09 | 0.09 | 0.67 |

**What it says.**

- **The quoted errors hold at every number of galaxies**: the quoted error
  equals the scatter, and the coverage is 0.68 to 0.76 for the line, 0.67
  to 0.86 for the continuum.
- **The bootstrap alone would not**: with 5 to 10 galaxies its coverage is
  0.57 to 0.64. This is why the quoted error is the larger of the thermal
  and the bootstrap one.
- **A weak extended continuum is biased high** (+14 % with 5 galaxies, +5 %
  with 20): its S/N here is about 3, and a Gaussian of free size fitted to
  a weak signal prefers large sizes. Its error is large enough to cover it.
  This is the case where the continuum is better given the size of the line
  (`continuum_model="auto"` does it by itself: see
  [The continuum](extract_flux)).

## Point and extended sources

20 galaxies, 60 realisations per row. *Fit* is the area of the Gaussian,
*window* the area under the spectrum in ± 500 km/s; the flux quoted is one
of the two.

| Case | S/N | Flux quoted: bias | scatter | coverage | Fit: bias | coverage | Window: bias | coverage | Gaussian chosen | size |
|---|---|---|---|---|---|---|---|---|---|---|
| point sources, strong | 39 | +1% | 0.04 | 0.77 | +1% | 0.77 | +1% | 0.73 | 0% | 0.00″ |
| point sources, weak | 8 | +9% | 0.22 | 0.73 | +9% | 0.73 | +8% | 0.75 | 2% | 0.00″ |
| point sources, very weak | 4 | +18% | 0.52 | 0.73 | +18% | 0.73 | +23% | 0.78 | 0% | 0.00″ |
| extended 0.8″, strong | 16 | +2% | 0.11 | 0.68 | +2% | 0.68 | +2% | 0.73 | 100% | 0.81″ |
| extended 0.8″, weak | 5 | +7% | 0.38 | 0.72 | +7% | 0.72 | +5% | 0.75 | 75% | 0.79″ |
| extended 0.8″, weak, centre and width fixed | 5 | +2% | 0.32 | 0.75 | +2% | 0.72 | +5% | 0.75 | 75% | 0.80″ |
| extended 0.8″, fluxes scattered by 0.3 dex | 16 | +3% | 0.14 | 0.80 | +3% | 0.80 | +2% | 0.77 | 100% | 0.81″ |
| **extended 0.8″ measured as a point** (`model="point"`) | 11 | **−71%** | 0.04 | **0.00** | −71% | 0.00 | −71% | 0.00 | — | — |
| **half points, half 0.8″** | 26 | **−16%** | 0.07 | **0.15** | −16% | 0.15 | −16% | 0.25 | 87% | 0.30″ |

**What works.**

- **Strong lines**, point or extended: the flux within 1–2 %, the right
  source model every time, the right size (median 0.81″ for 0.8″), errors
  that cover.
- **Galaxies of different luminosity** (0.3 dex): the same, with a larger
  error, as it should be: the bootstrap sees the scatter of the population.

**What to know.**

- **A weak line found by a search is biased high**: +9 % at S/N 8, +18 % at
  S/N 4. The search over centre and width picks the noise that makes the
  line look brighter. The error is large enough to cover the truth, but the
  bias is real. With **centre and width fixed** from another tracer
  (`v0_range=(0, 0)`, `fwhm_range=(300, 300)`) it goes away: +2 % instead
  of +7 %.
- **An extended source measured as a point loses most of its flux** (−71 %
  here), with a small error that never holds the truth. Never force
  `model="point"` on sources that may be extended.
- **A stack of points and extended sources together is biased low**
  (−16 %), and its error holds the truth one time in seven. Such a stack is
  a flat level plus something that falls with the baseline; a single
  Gaussian settles on a small size (0.30″ here) and misses the flux of the
  extended half. This is a limit of the model, not of the noise: it gets
  worse, not better, with more signal. The figure against the baseline
  length shows it (the short baselines lie above the model), and
  [Example 8](../examples/08_points_and_extended_sources) is this case.
  When the compact and the extended galaxies can be told apart, stack them
  separately.
