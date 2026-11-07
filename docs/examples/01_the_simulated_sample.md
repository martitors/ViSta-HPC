# 1. The simulated sample

`examples/simulate.py` writes Measurement Sets of an ALMA-like array (43
antennas of 12 m within 600 m, 20 minutes per galaxy) observing a galaxy at
redshift *z* = 2–3, about 2″ from the phase centre. CO(3–2) (rest 345.796
GHz) falls at 86–115 GHz, in Band 3, where the beam is about 1.4″ × 0.8″.
Each galaxy has

* a CO(3–2) line of 0.4 Jy km/s and 300 km/s FWHM (L′ ≈ 10¹⁰ K km/s pc²: a
  massive star-forming galaxy);
* a dust continuum of 0.2 mJy;
* a size of 0.8″ (FWHM of a circular Gaussian): resolved by the longest
  baselines;
* the primary-beam attenuation at its position;
* noise of 0.1 Jy per visibility (30 s, 20 km/s, close to ALMA in Band 3),
  with weights that misjudge it by `kappa`, channels correlated as after
  Hanning smoothing, and a different depth for every dataset
  (`depth_spread_dex`) — as in real data.

All galaxies have the same luminosity, width, size and centre
(`same_luminosity=True`): the stack, transported to the median redshift
`z_ref`, must return `line_flux`. The same sample is written a second time
**without noise** (`noiseless=True`): the same visibilities and weights, only
the signal.

**Change the parameters** to try other cases: a point source
(`size_arcsec=0`), fewer galaxies or more noise, redshift errors
(`v_jitter_kms`), a spread of luminosities (`flux_scatter_dex`), no
continuum.

```python
REST_GHZ = 345.79599          # CO(3-2)
SIM = dict(
    n_sources=12,             # galaxies in the stack
    z_range=(2.0, 3.0),       # redshifts, uniform in this range
    line_flux=0.4,            # Jy km/s at z_ref: what the stack should give
    fwhm_kms=300.0,           # line FWHM
    size_arcsec=0.8,          # source FWHM (0 = point source)
    cont_flux=2e-4,           # continuum at the line frequency, Jy (0 = none)
    noise=0.1,                # Jy per visibility (real or imaginary part), typical weight
    kappa=1.3,                # the weights misjudge the noise by this factor
    hanning=True,             # adjacent channels correlated (Hanning smoothing)
    depth_spread_dex=0.2,     # datasets of different depth: noise x 10^(+-0.2)
    v_jitter_kms=0.0,         # redshift errors: random shift of the line centres
    flux_scatter_dex=0.0,     # spread of the luminosities
    same_luminosity=True,
    rmax=600.0,               # array radius, m (beam ~1.4" x 0.8" at 100 GHz)
    seed=0,
)
t0 = time.time()
truth = simulate.make_sample("sim", rest_freq_ghz=REST_GHZ, **SIM)
truth0 = simulate.make_sample("sim_noiseless", rest_freq_ghz=REST_GHZ, noiseless=True, **SIM)
LIST, LIST0 = truth["input_list"], truth0["input_list"]
print(f"2 x {SIM['n_sources']} Measurement Sets in {time.time() - t0:.0f} s, "
      f"z_ref = {truth['z_ref']:.3f}\n")
print(f"{'dataset':16s} {'z':>6s} {'nu_obs':>7s} {'F_line':>7s} {'peak':>9s} {'noise':>6s}")
for s in truth["sources"]:
    print(f"{os.path.basename(s['path']):16s} {s['z']:6.3f} {s['nu_obs_ghz']:7.2f} "
          f"{s['line_flux']:7.3f} {s['line_peak'] * 1e3:5.2f} mJy {s['noise']:6.3f}")

# the true values of the stack, to compare with everything below
T = {"flux": truth["line_flux_stack"], "peak": truth["line_peak_stack"],
     "fwhm": truth["fwhm_kms_stack"], "v0": truth["v0_kms_stack"], "size": SIM["size_arcsec"],
     "cont": truth["cont_flux_stack"]}
print(f"\ntrue stack: line {T['flux']:.3f} Jy km/s, peak {T['peak'] * 1e3:.3f} mJy, "
      f"FWHM {T['fwhm']:.0f} km/s, v0 {T['v0']:+.0f} km/s, size {T['size']}\", "
      f"continuum {T['cont'] * 1e3:.3f} mJy")
print("\nthe input list:\n" + open(LIST).read())
```

```text
2 x 12 Measurement Sets in 20 s, z_ref = 2.622

dataset               z  nu_obs  F_line      peak  noise
galaxy_00.ms      2.003  115.16   0.637  2.00 mJy  0.124
galaxy_01.ms      2.017  114.63   0.630  1.97 mJy  0.104
galaxy_02.ms      2.041  113.71   0.616  1.93 mJy  0.065
galaxy_03.ms      2.270  105.75   0.512  1.60 mJy  0.115
galaxy_04.ms      2.544   97.58   0.421  1.32 mJy  0.158
galaxy_05.ms      2.607   95.88   0.404  1.26 mJy  0.115
galaxy_06.ms      2.637   95.08   0.396  1.24 mJy  0.071
galaxy_07.ms      2.729   92.72   0.374  1.17 mJy  0.084
galaxy_08.ms      2.813   90.68   0.355  1.11 mJy  0.149
galaxy_09.ms      2.816   90.62   0.355  1.11 mJy  0.085
galaxy_10.ms      2.913   88.38   0.335  1.05 mJy  0.090
galaxy_11.ms      2.935   87.88   0.331  1.04 mJy  0.112

true stack: line 0.400 Jy km/s, peak 1.253 mJy, FWHM 300 km/s, v0 +0 km/s, size 0.8", continuum 0.200 mJy

the input list:
# <ms_path>  <redshift>  <RA>  <Dec>
sim/galaxy_00.ms  2.002739  10:00:00.13341461  +02:00:01.0000000
sim/galaxy_01.ms  2.016528  10:02:00.13342294  +02:06:01.0000000
sim/galaxy_02.ms  2.040974  10:04:00.13343168  +02:12:01.0000000
sim/galaxy_03.ms  2.269787  10:06:00.13344083  +02:18:01.0000000
sim/galaxy_04.ms  2.543625  10:08:00.13345039  +02:24:01.0000000
sim/galaxy_05.ms  2.606636  10:10:00.13346036  +02:30:01.0000000
sim/galaxy_06.ms  2.636962  10:12:00.13347073  +02:36:01.0000000
sim/galaxy_07.ms  2.729497  10:14:00.13348151  +02:42:01.0000000
sim/galaxy_08.ms  2.813270  10:16:00.13349271  +02:48:01.0000000
sim/galaxy_09.ms  2.815854  10:18:00.13350430  +02:54:01.0000000
sim/galaxy_10.ms  2.912756  10:20:00.13351631  +03:00:01.0000000
sim/galaxy_11.ms  2.935072  10:22:00.13352873  +03:06:01.0000000
```

The input list is the plain ViSta one: `<ms_path> <redshift> <RA> <Dec>
[FIELD_ID] [SPW_IDS] [NORM]`, with the position of the galaxy (not of the
phase centre). The galaxies at lower redshift are brighter (they are nearer);
`noise` is the depth of each dataset.
