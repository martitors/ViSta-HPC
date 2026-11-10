# ViSta benchmark

## Time, memory, disk

| Version | wall time [s] | CPU time [s] | cores used | peak memory [GB] | peak disk [GB] | disk at the end [GB] | runs |
|---|---|---|---|---|---|---|---|
| ViSta 1.0 (CASA) | 266 | 333 | 1.2 | 2.45 | 6.07 | 6.07 | 3 |
| ViSta-HPC extract | 211 | 409 | 1.9 | 0.90 | 2.62 | 2.62 | 3 |
| ViSta-HPC extract_opt | 39 | 78 | 2.0 | 0.47 | 1.61 | 1.61 | 3 |

**ViSta 1.0 (CASA)**, per step (median of 3 runs)

| step | time [s] | peak memory [GB] | peak disk [GB] |
|---|---|---|---|
| start | 2.5 | 0.27 | 0.00 |
| restframing | 3.5 | 0.29 | 1.47 |
| centering | 30.2 | 0.32 | 3.37 |
| rebinning | 35.5 | 0.34 | 4.53 |
| averaging | 29.4 | 0.34 | 4.93 |
| stacking | 8.0 | 0.42 | 5.17 |
| unify | 38.9 | 0.35 | 5.67 |
| contsub | 8.5 | 0.35 | 6.02 |
| fit_size | 9.0 | 2.25 | 6.07 |
| fit_channels | 62.1 | 2.11 | 6.07 |
| fit_profile | 1.3 | 0.36 | 6.07 |
| continuum | 39.6 | 2.45 | 6.07 |

**ViSta-HPC extract**, per step (median of 3 runs)

| step | time [s] | peak memory [GB] | peak disk [GB] |
|---|---|---|---|
| start | 0.0 | 0.00 | 0.00 |
| stacking | 16.1 | 0.75 | 0.90 |
| continuum | 138.7 | 0.68 | 2.62 |
| compression | 46.9 | 0.90 | 2.62 |
| extraction | 10.3 | 0.12 | 2.62 |

**ViSta-HPC extract_opt**, per step (median of 3 runs)

| step | time [s] | peak memory [GB] | peak disk [GB] |
|---|---|---|---|
| start | 0.0 | 0.00 | 0.00 |
| stacking | 19.3 | 0.47 | 1.61 |
| continuum | 0.0 | 0.00 | 0.00 |
| compression | 0.0 | 0.00 | 0.00 |
| extraction | 19.2 | 0.19 | 1.61 |

## Without noise

| Version | line flux [Jy km/s] | FWHM [km/s] | centre [km/s] | size [arcsec] | continuum [mJy] |
|---|---|---|---|---|---|
| ViSta 1.0 (CASA) | 0.3463 (+3.7% true, -1.5% ideal) | 291.4 (-4.1% true, +0.3% ideal) | +6.6 (true +8.4, ideal +6.0) | 0.6038 (+24.4% true, -2.8% ideal) | 0.2133 (-7.6% true, +0.3% ideal) |
| ViSta-HPC extract | 0.3265 (-2.1% true, -7.1% ideal) | 297.2 (-2.4% true, +1.9% ideal) | +5.5 (true +8.0, ideal +5.5) | 0.4984 (+2.9% true, -19.6% ideal) | 0.2066 (-10.2% true, -2.0% ideal) |
| ViSta-HPC extract_opt | 0.352 (+5.5% true, +0.2% ideal) | 292.4 (-4.0% true, +0.2% ideal) | +5.6 (true +8.0, ideal +5.5) | 0.6286 (+29.8% true, +1.4% ideal) | 0.2066 (-10.2% true, -2.0% ideal) |
| true (ViSta-HPC weights) | 0.3335 | 305 | +8.0 | 0.484 | 0.2301 |
| ideal one-Gaussian fit (ViSta-HPC weights) | 0.3514 | 292 | +5.5 | 0.620 | 0.2109 |

## Monte Carlo

| Version | quantity | N | true | without noise | mean | bias (vs true) | scatter | mean quoted error | pull mean | pull rms | within ±1σ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ViSta 1.0 (CASA) | line flux | 20 | 0.334 | 0.3463 | 0.3666 | +9.8% ± 3.5% | 0.0521 | 0.029 | +0.50 | 1.82 | 0.40 |
| ViSta 1.0 (CASA) | FWHM | 20 | 303.7 | 291.4 | 303.2 | -0.2% ± 3.0% | 40.9 | 32.5 | +0.19 | 1.20 | 0.50 |
| ViSta 1.0 (CASA) | centre | 20 | 8.433 | 6.625 | 8.821 | +0.4 ± 2.8 | 12.5 | 11.1 | +0.16 | 1.08 | 0.60 |
| ViSta 1.0 (CASA) | size | 20 | 0.4852 | 0.6038 | 0.6084 | +25.4% ± 6.0% | 0.13 | 0.0972 | -0.06 | 1.28 | 0.65 |
| ViSta 1.0 (CASA) | continuum | 20 | 0.231 | 0.2133 | 0.2068 | -10.5% ± 3.9% | 0.0402 | 0.0358 | -0.24 | 1.15 | 0.60 |
| ViSta-HPC extract | line flux | 20 | 0.3335 | 0.3265 | 0.3626 | +8.7% ± 3.9% | 0.0575 | 0.0628 | +0.55 | 1.04 | 0.70 |
| ViSta-HPC extract | FWHM | 20 | 304.7 | 297.2 | 313.3 | +2.8% ± 3.4% | 46.4 | — | — | — | — |
| ViSta-HPC extract | centre | 20 | 7.958 | 5.511 | 7.01 | -0.9 ± 3.3 | 14.7 | — | — | — | — |
| ViSta-HPC extract | size | 20 | 0.4842 | 0.4984 | 0.5935 | +22.6% ± 6.1% | 0.132 | 0.178 | +0.57 | 1.00 | 0.70 |
| ViSta-HPC extract | continuum | 20 | 0.2301 | 0.2066 | 0.2155 | -6.3% ± 3.2% | 0.0329 | 0.047 | +0.10 | 0.73 | 0.80 |
| ViSta-HPC extract_opt | line flux | 20 | 0.3335 | 0.352 | 0.3731 | +11.9% ± 4.3% | 0.0638 | 0.0713 | +0.26 | 0.92 | 0.60 |
| ViSta-HPC extract_opt | FWHM | 20 | 304.7 | 292.4 | 305.1 | +0.1% ± 3.1% | 42.1 | 31 | +0.27 | 1.37 | 0.50 |
| ViSta-HPC extract_opt | centre | 20 | 7.958 | 5.589 | 7.605 | -0.4 ± 2.8 | 12.7 | 11.5 | +0.14 | 1.04 | 0.60 |
| ViSta-HPC extract_opt | size | 20 | 0.4842 | 0.6286 | 0.5641 | +16.5% ± 12.6% | 0.274 | 0.124 | +0.21 | 0.98 | 0.71 (3 at the boundary, no error) |
| ViSta-HPC extract_opt | continuum | 20 | 0.2301 | 0.2066 | 0.2155 | -6.3% ± 3.2% | 0.0329 | 0.047 | +0.10 | 0.73 | 0.80 |

