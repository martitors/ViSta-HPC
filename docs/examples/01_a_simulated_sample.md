# 1. A simulated sample

`simulate.make_sample` writes one Measurement Set per galaxy, as an ALMA-like
array (43 antennas within 600 m) would observe it in Band 3, and the input
list for ViSta. Every galaxy has a CO(3–2) line of 0.4 Jy km/s (at the median
redshift) and 300 km/s FWHM, a size of 0.8″ (resolved by the longest
baselines) and a dust continuum of 0.2 mJy, and sits up to 20″ from the
phase centre. As in real data, the datasets have different depths, their
weights misjudge the noise by 30 % (`kappa=1.3`) and adjacent channels are
correlated.

`simulate.expected` gives what the stack should measure: the green lines and
numbers of every comparison below.

```python
LINE = 345.79599                       # rest frequency of CO(3-2), GHz

truth = simulate.make_sample(
    "sim", n_sources=12, z_range=(2.0, 3.0), rest_freq_ghz=LINE,
    line_flux=0.4, fwhm_kms=300, size_arcsec=0.8,    # the line of every galaxy
    cont_flux=2e-4,                                   # the continuum, Jy
    noise=0.1, kappa=1.3, depth_spread_dex=0.2,       # noise, wrong weights, different depths
    offset_arcsec=lambda rng: tuple(rng.uniform(-15, 15, 2)),   # position in the field
    rmax=600, seed=1)
T = simulate.expected(truth)

print(f"true stack: line {T['flux'][0]:.3f} Jy km/s, continuum {T['continuum'] * 1e3:.2f} mJy "
      f"(at z_ref = {truth['z_ref']:.3f})\n")
print(open(truth["input_list"]).read())
```

```text
true stack: line 0.400 Jy km/s, continuum 0.20 mJy (at z_ref = 2.525)

# <ms_path>  <redshift>  <RA>  <Dec>
sim/galaxy_00.ms  2.027559  10:00:00.57720903  +01:59:54.0958449
sim/galaxy_01.ms  2.144160  10:01:59.40651189  +02:05:52.8694002
sim/galaxy_02.ms  2.311831  10:04:00.96218361  +02:12:13.8497158
sim/galaxy_03.ms  2.409199  10:05:59.32075682  +02:18:14.0977624
sim/galaxy_04.ms  2.423326  10:08:00.55385205  +02:24:03.3900990
sim/galaxy_05.ms  2.511822  10:09:59.91859429  +02:29:46.8704874
sim/galaxy_06.ms  2.538143  10:11:59.51970046  +02:36:10.1964456
sim/galaxy_07.ms  2.549594  10:13:59.29506150  +02:42:09.5888016
sim/galaxy_08.ms  2.753513  10:16:00.60545115  +02:47:50.7397178
sim/galaxy_09.ms  2.827703  10:18:00.75403985  +02:53:59.1572916
sim/galaxy_10.ms  2.948649  10:20:00.44042235  +03:00:10.0670765
sim/galaxy_11.ms  2.950464  10:22:00.61100377  +03:06:13.9101262
```
