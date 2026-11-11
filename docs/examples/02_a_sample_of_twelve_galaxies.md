# 2. A sample of twelve galaxies

`simulate.make_sample` writes one such observation per galaxy and the *input
list* that ViSta reads. The sample is heterogeneous on purpose, as a real
one:

* redshifts between 2 and 3, so the line falls at a different frequency in
  every dataset;
* line fluxes that scatter around 0.4 Jy km/s, widths between 250 and
  350 km/s, centres off by ±20 km/s (redshift errors), sizes between 0.4″
  and 1.2″;
* observations of different depth (`depth_spread_dex`) and with arrays of
  different size (`rmax`): different noise and different resolution;
* galaxies anywhere within 20″ of the field centre.

```python
truth = simulate.make_sample(
    "sample", n_sources=12, z_range=(2.0, 3.0), rest_freq_ghz=LINE,
    line_flux=0.4, flux_scatter_dex=0.15,                      # fluxes around 0.4 Jy km/s
    fwhm_kms=lambda rng: rng.uniform(250, 350),                # a different width each
    v_jitter_kms=20,                                           # redshift errors
    size_arcsec=lambda rng: rng.uniform(0.4, 1.2),             # a different size each
    cont_flux=2e-4,                                            # continuum, Jy
    noise=0.1, kappa=1.3, depth_spread_dex=0.3,                # noise; wrong weights; depths
    rmax=lambda rng: float(rng.choice([300, 600, 1200])),      # a different array each
    offset_arcsec=lambda rng: tuple(rng.uniform(-15, 15, 2)),  # position in the field
    seed=2)

print(simulate.describe(truth))
```

```text
         z   nu_obs     line   FWHM   size  continuum   noise  array  offset    PB
                GHz  Jy km/s   km/s arcsec        mJy  Jy/vis      m  arcsec
  0  2.055   113.18    0.430    317   0.74      0.215   0.084    300    15.1  0.81
  1  2.092   111.84    0.625    301   1.11      0.312   0.125    600     9.9  0.91
  2  2.150   109.77    0.253    261   0.48      0.126   0.159   1200    12.7  0.87
  3  2.188   108.47    0.476    302   0.87      0.238   0.152    300    11.0  0.90
  4  2.262   106.02    0.313    300   0.95      0.157   0.069    600     8.5  0.94
  5  2.275   105.59    0.577    296   1.12      0.288   0.148    600    10.6  0.91
  6  2.298   104.83    0.436    291   0.56      0.218   0.067   1200    10.1  0.92
  7  2.562    97.07    0.412    260   0.76      0.206   0.238    300     3.6  0.99
  8  2.600    96.05    0.320    307   0.89      0.160   0.140    300    16.2  0.84
  9  2.657    94.55    0.357    280   1.14      0.179   0.066    600    16.9  0.83
 10  2.729    92.74    0.217    261   0.45      0.108   0.112    600     5.7  0.98
 11  2.814    90.66    0.211    302   0.41      0.105   0.099   1200    13.7  0.89
```

The input list has one line per dataset: the Measurement Set, the redshift
and the position of the galaxy. It is all ViSta needs to know about the
sample.

```python
print(open(truth["input_list"]).read())
```

```text
# <ms_path>  <redshift>  <RA>  <Dec>
sample/galaxy_00.ms  2.055147  10:00:00.93544175  +02:00:05.4919447
sample/galaxy_01.ms  2.091916  10:02:00.55149827  +02:05:54.5443980
sample/galaxy_02.ms  2.150062  10:04:00.76946651  +02:12:05.3943438
sample/galaxy_03.ms  2.187901  10:06:00.72481989  +02:17:58.1455850
sample/galaxy_04.ms  2.261612  10:08:00.04569709  +02:23:51.4867017
sample/galaxy_05.ms  2.274969  10:10:00.67100506  +02:29:56.5528540
sample/galaxy_06.ms  2.298491  10:11:59.36175542  +02:36:03.1141655
sample/galaxy_07.ms  2.562266  10:13:59.97697158  +02:42:03.6081723
sample/galaxy_08.ms  2.600101  10:16:00.93063467  +02:47:51.7987819
sample/galaxy_09.ms  2.657433  10:18:00.56985950  +02:53:45.3849328
sample/galaxy_10.ms  2.728561  10:19:59.89159119  +02:59:54.5517900
sample/galaxy_11.ms  2.814226  10:21:59.29489396  +03:05:51.2965958
```

What every observation sees on its own: the spectrum at the position of the
galaxy (black) and the true line (green). The deepest observations hint at
the line; in most it is lost in the noise, and in none can it be measured
well.

```python
simulate.plot_sample(truth)
```

```{image} figures/02_01.png
:width: 100%
```

**The expected answer.** Because the sample is simulated, we know what a
perfect measurement of the stack would give: `simulate.expected(truth)`
computes it from the true values (the mean of the galaxies, weighted as the
stack weights them). It is drawn in green in the comparisons below, as
`expected`. With real data there is no such thing: it is here only to check
the measurements.

```python
expected = simulate.expected(truth)
print(f"expected line flux of the stack: {expected['flux'][0]:.3f} Jy km/s")
print(f"expected continuum:              {expected['continuum'] * 1e3:.3f} mJy")
```

```text
expected line flux of the stack: 0.393 Jy km/s
expected continuum:              0.197 mJy
```
