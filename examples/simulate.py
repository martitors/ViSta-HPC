"""
simulate.py -- synthetic ALMA-like Measurement Sets to try ViSta on
====================================================================

``make_ms(path, z, ...)`` writes one Measurement Set: an ALMA-like array
(43 antennas of 12 m within ``rmax`` m of the centre) observing a galaxy at
redshift ``z``, a few arcsec from the phase centre, with

* an emission line: a Gaussian of integrated flux ``line_flux`` [Jy km/s],
  width ``fwhm_kms`` and centre ``v0_kms`` (radio velocity w.r.t. the rest
  frequency, as in the extraction);
* a continuum of ``cont_flux`` [Jy] at the line frequency, with a spectral
  index ``cont_index``;
* a circular Gaussian shape of FWHM ``size_arcsec`` (0: a point source), the
  same for line and continuum unless ``cont_size_arcsec`` is given; or a
  ring of that diameter (``shape="ring"``), a source that is not a Gaussian;
* the primary-beam attenuation at the source position (Gaussian beam of
  FWHM 1.2 c / (nu D), the model the extraction corrects for);
* Gaussian noise of ``noise`` [Jy] per real / imaginary part of a typical
  visibility (the weights vary by a factor 4 between rows); the WEIGHT
  column is 1 / sigma^2 of a noise ``kappa`` times smaller than the real one
  (weights that misjudge the noise, as often), and with ``hanning=True`` the
  noise is correlated between adjacent channels as after Hanning smoothing.
  ``noiseless=True`` writes the same dataset without noise (same weights):
  the signal as it is, to compare with.

``make_sample(directory, n_sources, ...)`` writes a sample of them at
different redshifts and the ViSta input list, and returns the true values.
The true signal -- of one dataset, of the stack, of the measured profile and
uv curve -- is computed by ``spectrum_at``, ``expected_in_stack``,
``true_profile`` and ``true_uv``.

Needs numpy and python-casacore (and astropy for the distances); not ViSta.
The visibilities follow V = A S(nu) g(b) exp(+2 pi i (u l + v m + w (n-1)) nu / c),
the convention of the ViSta tests: after the stacking the source is real.
Every channel is labelled LSRK, so no Doppler tracking is simulated.
"""

import json
import math
import os
import warnings

import numpy as np

C = 299792458.0
# sizes of the figures: large enough to read in a notebook
FONTS = {"font.size": 14, "axes.labelsize": 15, "axes.titlesize": 14, "legend.fontsize": 12,
         "xtick.labelsize": 13, "ytick.labelsize": 13}
C_KMS = 299792.458
ALMA_SITE = np.array([2225061.0, -5440057.0, -2481681.0])     # ITRF [m]
LAT, LON = math.radians(-23.0229), math.radians(-67.7552)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def antenna_layout(nant=43, rmax=250.0, min_sep=15.0, seed=12345):
    """Centrally condensed array of ``nant`` antennas within ``rmax`` m, at
    least ``min_sep`` m apart (a compact ALMA configuration for rmax = 250 m,
    beam ~1" at 230 GHz).  Returns ITRF positions [m]."""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < nant:
        r = rmax * rng.uniform() ** 1.4
        a = rng.uniform(0, 2 * math.pi)
        p = np.array([r * math.cos(a), r * math.sin(a)])
        if all(np.hypot(*(p - q)) >= min_sep for q in pts):
            pts.append(p)
    e, n = np.array(pts).T
    # local East-North-Up -> ITRF offsets at the ALMA site
    x = -math.sin(LON) * e - math.sin(LAT) * math.cos(LON) * n
    y = math.cos(LON) * e - math.sin(LAT) * math.sin(LON) * n
    z = math.cos(LAT) * n
    return ALMA_SITE + np.stack([x, y, z], axis=1)


def _basis(ra, dec):
    """Rows: the l, m, n directions of a phase centre (radians)."""
    return np.array([[-math.sin(ra), math.cos(ra), 0.0],
                     [-math.cos(ra) * math.sin(dec), -math.sin(ra) * math.sin(dec),
                      math.cos(dec)],
                     [math.cos(ra) * math.cos(dec), math.sin(ra) * math.cos(dec),
                      math.sin(dec)]])


def primary_beam(offset_arcsec, nu_hz, dish_m=12.0):
    """Gaussian primary beam of FWHM 1.2 c / (nu D) at ``offset_arcsec``."""
    x = math.radians(offset_arcsec / 3600.0) * nu_hz * dish_m / (1.2 * C)
    return math.exp(-4.0 * math.log(2.0) * x * x)


# ---------------------------------------------------------------------------
# the sky
# ---------------------------------------------------------------------------
def source_spectrum(freq_hz, z, rest_freq_ghz, line_flux, fwhm_kms, v0_kms, cont_flux,
                    cont_index, extra_lines=()):
    """Line and continuum flux density [Jy] of the galaxy at the observed
    frequencies ``freq_hz`` (total flux: zero spacing, no primary beam).
    The line is a Gaussian in radio velocity w.r.t. the rest frequency.
    ``extra_lines``: more Gaussians, dicts with ``rest_freq_ghz``,
    ``flux_ratio`` (to ``line_flux``), ``fwhm_kms`` and ``v0_kms`` (in their
    own velocity): other lines of the band, or wings of the same line."""
    nu_r = rest_freq_ghz * 1e9
    nu_obs = nu_r / (1.0 + z)
    v = C_KMS * (nu_r - np.asarray(freq_hz) * (1.0 + z)) / nu_r
    sig = fwhm_kms / 2.3548200450309493
    line = line_flux / (sig * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((v - v0_kms) / sig) ** 2)
    for x in extra_lines:
        nu_x = x["rest_freq_ghz"] * 1e9
        v_x = C_KMS * (nu_x - np.asarray(freq_hz) * (1.0 + z)) / nu_x
        s_x = x["fwhm_kms"] / 2.3548200450309493
        line = line + (line_flux * x["flux_ratio"] / (s_x * math.sqrt(2 * math.pi))
                       * np.exp(-0.5 * ((v_x - x.get("v0_kms", 0.0)) / s_x) ** 2))
    cont = cont_flux * (np.asarray(freq_hz) / nu_obs) ** cont_index
    return line, cont


def shape_visibility(q_wavelengths, size_arcsec, shape="gauss"):
    """Visibility of a source of unit flux at uv distance ``q``
    [wavelengths] (1 for a point source, ``size_arcsec`` 0).

    ``shape="gauss"``: a circular Gaussian of FWHM ``size_arcsec``.
    ``shape="ring"``: a ring of diameter ``size_arcsec``, a quarter of it
    thick: not a Gaussian, its visibility oscillates and goes negative."""
    q = np.asarray(q_wavelengths, float)
    if not size_arcsec:
        return np.ones_like(q)
    t = math.radians(size_arcsec / 3600.0)
    if shape == "ring":
        from scipy.special import j0
        return j0(math.pi * t * q) * np.exp(-(math.pi * 0.25 * t * q) ** 2 / (4 * math.log(2)))
    if shape != "gauss":
        raise ValueError(f"shape: 'gauss' or 'ring', not {shape!r}")
    return np.exp(-(math.pi * t * q) ** 2 / (4 * math.log(2)))


def line_profile(rec, v_kms):
    """The true line of one galaxy (a record of make_ms) against radio
    velocity: its total flux density [Jy] (zero spacing), through the
    primary beam, without continuum."""
    sig = rec["fwhm_kms"] / 2.3548200450309493
    v = np.asarray(v_kms, float)
    return rec["primary_beam"] * rec["line_peak"] * np.exp(-0.5 * ((v - rec["v0_kms"]) / sig) ** 2)


def model_visibility(rec, uvw, freq_hz):
    """The noiseless visibilities of a simulated galaxy (a record of
    make_ms) with the phase centre on it: real, (nrow, nchan), for baselines
    ``uvw`` [m] and observed frequencies ``freq_hz``.  Returns the line part
    and the continuum part, primary beam included."""
    line, cont = source_spectrum(freq_hz, rec["z"], rec["rest_freq_ghz"], rec["line_flux"],
                                 rec["fwhm_kms"], rec["v0_kms"], rec["cont_flux"],
                                 rec["cont_index"], rec.get("extra_lines", ()))
    q = np.hypot(uvw[:, 0], uvw[:, 1])[:, None] * np.asarray(freq_hz)[None, :] / C
    g_line = shape_visibility(q, rec["size_arcsec"], rec.get("shape", "gauss"))
    g_cont = shape_visibility(q, rec["cont_size_arcsec"], rec.get("shape", "gauss"))
    return rec["primary_beam"] * line[None, :] * g_line, rec["primary_beam"] * cont[None, :] * g_cont


# ---------------------------------------------------------------------------
# one dataset
# ---------------------------------------------------------------------------
def make_ms(path, z, rest_freq_ghz=345.79599, line_flux=0.25, fwhm_kms=250.0, v0_kms=0.0,
            cont_flux=0.0, cont_index=3.0, size_arcsec=0.0, cont_size_arcsec=None, shape="gauss",
            phase_centre=(150.0, 2.0), offset_arcsec=(2.0, 1.0), noise=0.05, kappa=1.0,
            hanning=True, nchan=96, dv_kms=20.0, nant=43, rmax=250.0, ntime=40, dt=30.0,
            primary_beam_attenuation=True, noiseless=False, seed=1, band_offset_kms=0.0,
            t0=5.06e9, layout_seed=12345, noise_seed=None, extra_lines=()):
    """Write one Measurement Set; returns a dict of what was simulated.

    The spectral window has ``nchan`` channels of ``dv_kms`` centred on the
    observed line frequency ``rest / (1 + z)`` (channel width in frequency
    ``dv_kms / c * nu_obs``), or ``band_offset_kms`` away from it (positive:
    the band centre at higher velocity, the line nearer its upper-frequency
    edge).  ``offset_arcsec`` is the (east, north) offset of the galaxy from
    the phase centre.  ``t0`` (MJD seconds) sets the start of the
    observation, hence the hour angles and the uv tracks; ``layout_seed`` the
    antenna positions.  ``noise_seed`` draws the noise from its own
    generator: the same ``seed`` and another ``noise_seed`` give the same
    weights and a new realisation of the noise.  ``extra_lines``: more
    Gaussian lines (see :func:`source_spectrum`), sharing the size.
    """
    from casacore import tables as ct

    rng = np.random.default_rng(seed)
    nu_r = rest_freq_ghz * 1e9
    nu_obs = nu_r / (1.0 + z)
    df = dv_kms / C_KMS * nu_obs
    nu_centre = nu_obs * (1.0 - band_offset_kms / C_KMS)
    freq = nu_centre + (np.arange(nchan) - (nchan - 1) / 2.0) * df       # [Hz]

    # positions
    ra0, dec0 = (math.radians(x) for x in phase_centre)
    d_ra = offset_arcsec[0] / 3600.0 / math.cos(dec0)
    ra_s = math.radians(phase_centre[0] + d_ra)
    dec_s = math.radians(phase_centre[1] + offset_arcsec[1] / 3600.0)
    basis = _basis(ra0, dec0)
    s_dir = _basis(ra_s, dec_s)[2]
    l, m, n = basis @ s_dir

    # uvw: the baselines rotating with the Earth, projected on the phase centre
    itrf = antenna_layout(nant, rmax, seed=layout_seed)
    a1, a2 = np.triu_indices(nant, 1)
    times = t0 + np.arange(ntime) * dt
    ha = 2 * math.pi * (times % 86164.0905) / 86164.0905
    bl = itrf[a2] - itrf[a1]                                           # (nbl, 3)
    cos_h, sin_h = np.cos(ha), np.sin(ha)
    rot = np.stack([cos_h[:, None] * bl[None, :, 0] - sin_h[:, None] * bl[None, :, 1],
                    sin_h[:, None] * bl[None, :, 0] + cos_h[:, None] * bl[None, :, 1],
                    np.broadcast_to(bl[None, :, 2], (ntime, len(a1)))], axis=-1)
    uvw = (rot @ basis.T).reshape(-1, 3)                               # (nrow, 3), time-major
    nrow = uvw.shape[0]

    # sky: line + continuum, seen through the primary beam and the baselines
    off = math.hypot(*offset_arcsec)
    sig = fwhm_kms / 2.3548200450309493
    rec = {"path": path, "z": z, "rest_freq_ghz": rest_freq_ghz,
           "ra_deg": math.degrees(ra_s), "dec_deg": math.degrees(dec_s),
           "line_flux": line_flux, "line_peak": line_flux / (sig * math.sqrt(2 * math.pi)),
           "fwhm_kms": fwhm_kms, "v0_kms": v0_kms, "cont_flux": cont_flux,
           "extra_lines": [dict(x) for x in extra_lines],
           "cont_index": cont_index, "size_arcsec": size_arcsec, "shape": shape,
           "rmax_m": rmax,
           "cont_size_arcsec": size_arcsec if cont_size_arcsec is None else cont_size_arcsec,
           "primary_beam": primary_beam(off, nu_obs) if primary_beam_attenuation else 1.0,
           "offset_arcsec": off, "nu_obs_ghz": nu_obs / 1e9, "noise": noise, "kappa": kappa}
    v_line, v_cont = model_visibility(rec, uvw, freq)
    phase = 2 * math.pi * (uvw[:, 0:1] * l + uvw[:, 1:2] * m + uvw[:, 2:3] * (n - 1)) \
        * freq[None, :] / C
    vis = (v_line + v_cont) * np.exp(1j * phase)                       # (nrow, nchan)

    # noise: sigma = noise * kappa / sqrt(w); the weights say noise / sqrt(w).
    # noiseless: the same weights, no noise -- the twin of the noisy dataset
    w = rng.uniform(0.5, 2.0, size=nrow)
    sd = noise * kappa / np.sqrt(w)
    weight = w / noise ** 2
    if noiseless:
        data = np.repeat(vis[:, :, None], 2, axis=2).astype(np.complex64)
    else:
        e = (rng if noise_seed is None else np.random.default_rng(noise_seed)) \
            .standard_normal((2, nrow, nchan, 2))
        if hanning:
            pad = np.pad(e, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="edge")
            e = (0.25 * pad[:, :, :-2] + 0.5 * pad[:, :, 1:-1] + 0.25 * pad[:, :, 2:]) \
                / math.sqrt(0.375)
        data = (vis[:, :, None] + (e[0] + 1j * e[1]) * sd[:, None, None]).astype(np.complex64)

    # the Measurement Set
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)
    desc = ct.maketabdesc([ct.makearrcoldesc("DATA", 0j, ndim=2, valuetype="complex")])
    t = ct.default_ms(path, desc)
    t.addrows(nrow)
    t.putcol("DATA", data)
    t.putcol("FLAG", np.zeros(data.shape, bool))
    t.putcol("UVW", uvw)
    tt = np.repeat(times, len(a1))
    t.putcol("TIME", tt)
    t.putcol("TIME_CENTROID", tt)
    t.putcol("ANTENNA1", np.tile(a1, ntime).astype(np.int32))
    t.putcol("ANTENNA2", np.tile(a2, ntime).astype(np.int32))
    t.putcol("DATA_DESC_ID", np.zeros(nrow, np.int32))
    t.putcol("FIELD_ID", np.zeros(nrow, np.int32))
    t.putcol("STATE_ID", np.full(nrow, -1, np.int32))
    t.putcol("SCAN_NUMBER", (1 + np.repeat(np.arange(ntime), len(a1)) // 10).astype(np.int32))
    t.putcol("WEIGHT", np.repeat(weight[:, None], 2, 1).astype(np.float32))
    t.putcol("SIGMA", np.repeat(1.0 / np.sqrt(weight)[:, None], 2, 1).astype(np.float32))
    t.putcol("FLAG_ROW", np.zeros(nrow, bool))
    t.putcol("EXPOSURE", np.full(nrow, dt))
    t.putcol("INTERVAL", np.full(nrow, dt))
    t.close()

    sw = ct.table(path + "/SPECTRAL_WINDOW", readonly=False, ack=False)
    sw.addrows(1)
    sw.putcell("CHAN_FREQ", 0, freq)
    sw.putcell("CHAN_WIDTH", 0, np.full(nchan, df))
    sw.putcell("EFFECTIVE_BW", 0, np.full(nchan, df))
    sw.putcell("RESOLUTION", 0, np.full(nchan, df))
    sw.putcell("NUM_CHAN", 0, nchan)
    sw.putcell("REF_FREQUENCY", 0, freq[0])
    sw.putcell("TOTAL_BANDWIDTH", 0, df * nchan)
    sw.putcell("MEAS_FREQ_REF", 0, 1)                                # LSRK
    sw.putcell("NAME", 0, "SPW0")
    sw.close()
    dd = ct.table(path + "/DATA_DESCRIPTION", readonly=False, ack=False)
    dd.addrows(1)
    dd.putcol("SPECTRAL_WINDOW_ID", np.zeros(1, np.int32))
    dd.putcol("POLARIZATION_ID", np.zeros(1, np.int32))
    dd.close()
    po = ct.table(path + "/POLARIZATION", readonly=False, ack=False)
    po.addrows(1)
    po.putcell("NUM_CORR", 0, 2)
    po.putcell("CORR_TYPE", 0, np.array([9, 12], np.int32))         # XX, YY
    po.putcell("CORR_PRODUCT", 0, np.array([[0, 0], [1, 1]], np.int32))
    po.close()
    fi = ct.table(path + "/FIELD", readonly=False, ack=False)
    fi.addrows(1)
    for col in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"):
        fi.putcell(col, 0, np.array([[ra0, dec0]]))
    fi.putcell("NAME", 0, "TARGET")
    fi.close()
    an = ct.table(path + "/ANTENNA", readonly=False, ack=False)
    an.addrows(nant)
    an.putcol("POSITION", itrf)
    an.putcol("DISH_DIAMETER", np.full(nant, 12.0))
    an.putcol("NAME", [f"DA{40 + i}" for i in range(nant)])
    an.putcol("STATION", [f"A{i:03d}" for i in range(nant)])
    an.putcol("MOUNT", ["ALT-AZ"] * nant)
    an.close()
    fe = ct.table(path + "/FEED", readonly=False, ack=False)       # for imaging tools
    fe.addrows(nant)
    fe.putcol("ANTENNA_ID", np.arange(nant, dtype=np.int32))
    fe.putcol("FEED_ID", np.zeros(nant, np.int32))
    fe.putcol("SPECTRAL_WINDOW_ID", np.full(nant, -1, np.int32))
    fe.putcol("NUM_RECEPTORS", np.full(nant, 2, np.int32))
    fe.putcol("POLARIZATION_TYPE", np.array([["X", "Y"]] * nant))
    fe.putcol("BEAM_OFFSET", np.zeros((nant, 2, 2)))
    fe.putcol("POL_RESPONSE", np.tile(np.eye(2, dtype=np.complex64), (nant, 1, 1)))
    fe.putcol("RECEPTOR_ANGLE", np.zeros((nant, 2)))
    fe.putcol("POSITION", np.zeros((nant, 3)))
    fe.putcol("TIME", np.full(nant, times[0]))
    fe.putcol("INTERVAL", np.zeros(nant))
    fe.close()
    ob = ct.table(path + "/OBSERVATION", readonly=False, ack=False)
    ob.addrows(1)
    ob.putcell("TELESCOPE_NAME", 0, "ALMA")
    ob.putcell("TIME_RANGE", 0, np.array([times.min(), times.max()]))
    for col in ("LOG", "SCHEDULE"):               # defined, so that readers can infer their shape
        ob.putcell(col, 0, np.array(["none"]))
    ob.close()
    # SOURCE (optional in the format, present in ALMA data; CASA-based tools read it)
    src_desc = ct.required_ms_desc("SOURCE")
    full = ct.complete_ms_desc("SOURCE")
    for col in ("REST_FREQUENCY", "SYSVEL", "TRANSITION"):
        src_desc[col] = full[col]
    so = ct.table(path + "/SOURCE", src_desc, nrow=1, ack=False)
    so.putcell("SOURCE_ID", 0, 0)
    so.putcell("TIME", 0, float(times.mean()))
    so.putcell("INTERVAL", 0, float(times.max() - times.min() + dt))
    so.putcell("SPECTRAL_WINDOW_ID", 0, 0)
    so.putcell("NUM_LINES", 0, 1)
    so.putcell("NAME", 0, "TARGET")
    so.putcell("CALIBRATION_GROUP", 0, 0)
    so.putcell("CODE", 0, "")
    so.putcell("DIRECTION", 0, np.array([ra0, dec0]))
    so.putcell("PROPER_MOTION", 0, np.zeros(2))
    so.putcell("REST_FREQUENCY", 0, np.array([nu_r]))
    so.putcell("SYSVEL", 0, np.array([0.0]))
    so.putcell("TRANSITION", 0, np.array([""]))
    so.close()
    with ct.table(path, readonly=False, ack=False) as t:
        t.putkeyword("SOURCE", "Table: " + os.path.abspath(path) + "/SOURCE")

    return rec


def _centred(rec):
    """The visibilities of one simulated dataset with the phase centre moved
    onto the galaxy: ``data`` (nrow, nchan, ncorr), ``uvw`` [m], the weights
    (0 where flagged) and the channel frequencies [Hz]."""
    from casacore import tables as ct
    path = rec["path"]
    with ct.table(path, ack=False) as t:
        data = t.getcol("DATA")
        flag = t.getcol("FLAG")
        uvw = t.getcol("UVW")
        w = t.getcol("WEIGHT")
    with ct.table(path + "/FIELD", ack=False) as t:
        ra0, dec0 = np.asarray(t.getcell("PHASE_DIR", 0), float).ravel()[:2]
    with ct.table(path + "/SPECTRAL_WINDOW", ack=False) as t:
        freq = np.asarray(t.getcell("CHAN_FREQ", 0), float)
    src = _basis(math.radians(rec["ra_deg"]), math.radians(rec["dec_deg"]))[2]
    l, m, n = _basis(ra0, dec0) @ src
    phase = 2 * math.pi * (uvw[:, 0:1] * l + uvw[:, 1:2] * m + uvw[:, 2:3] * (n - 1)) \
        * freq[None, :] / C
    return (data * np.exp(-1j * phase)[:, :, None], uvw, np.where(flag, 0.0, w[:, None, :]),
            freq)


def channel_image(rec, v_kms=0.0, fov_arcsec=None):
    """Dirty image of one simulated dataset, centred on the galaxy, in the
    channel nearest to the velocity ``v_kms`` (0: the line centre)."""
    from vista.quicklook import image_visibilities
    data, uvw, wt, freq = _centred(rec)
    nu_r = rec["rest_freq_ghz"] * 1e9
    k = int(np.argmin(np.abs(C_KMS * (nu_r - freq * (1 + rec["z"])) / nu_r - v_kms)))
    w = wt[:, k, :].sum(axis=1)
    vis = (wt[:, k, :] * data[:, k, :]).sum(axis=1) / np.maximum(w, 1e-300)
    return image_visibilities(uvw[:, 0] * freq[k] / C, uvw[:, 1] * freq[k] / C, vis, w,
                              fov_arcsec)


def clean_image(rec, fov_arcsec=None):
    """One simulated galaxy without noise, on the baselines of its dataset:
    the dirty image of its line at the peak, centred on the galaxy."""
    from vista.quicklook import image_visibilities
    _, uvw, wt, freq = _centred(rec)
    nu = rec["rest_freq_ghz"] * 1e9 / (1 + rec["z"])
    u, v = uvw[:, 0] * nu / C, uvw[:, 1] * nu / C
    sig = rec["fwhm_kms"] / 2.3548200450309493
    peak = rec["primary_beam"] * rec["line_flux"] / (sig * math.sqrt(2 * math.pi))
    vis = peak * shape_visibility(np.hypot(u, v), rec["size_arcsec"], rec.get("shape", "gauss"))
    return image_visibilities(u, v, vis, wt.sum(axis=(1, 2)), fov_arcsec)


def plot_galaxy(clean, noisy, unit_factor=1e3):
    """One simulated galaxy as the telescope would see it without noise
    (``clean``, a record of ``make_ms(..., noiseless=True)``) and with it
    (``noisy``): the image of the channel at the line centre, and below it
    the spectrum at the position of the galaxy (the visibilities averaged
    over the baselines).  In green the true signal: as these baselines see
    it (solid: what the data scatter around) and its total flux (dashed:
    higher when the source is extended, because the long baselines see only
    part of it)."""
    import matplotlib.pyplot as plt
    from vista.quicklook import plot_image
    plt.rcParams.update(FONTS)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), gridspec_kw={"height_ratios": [1.25, 1]})
    images = [channel_image(clean), channel_image(noisy)]
    top = max(float(images[0].data.max()), 4 * images[1].rms)
    for j, (rec, name, img) in enumerate(zip((clean, noisy), ("without noise", "with noise"),
                                             images)):
        if j == 0:
            img.rms = images[1].rms               # the same colours and contours as the noisy one
        plot_image(img, axes[0, j], title=f"One galaxy, {name}", vmax=top)
        if j == 0:                                # no noise: no rms to quote
            axes[0, j].set_title(f"One galaxy, {name}", fontsize=14)
        sp = spectrum_at(rec)
        ax = axes[1, j]
        o = np.argsort(sp["v"])
        ax.step(sp["v"][o], sp["mean"][o] * unit_factor, where="mid", color="k", lw=1.2,
                label="Observed")
        ax.plot(sp["v"][o], sp["expected"][o] * unit_factor, color="#1e8449", lw=2.4,
                label="True, these baselines")
        ax.plot(sp["v"][o], sp["total"][o] * unit_factor, color="#1e8449", lw=1.6, ls="--",
                label="True, total flux")
        ax.axhline(0, color="0.4", lw=0.6)
        ax.set_xlabel("velocity [km/s]")
        ax.set_ylabel("Re flux density [mJy]")
        ax.legend(frameon=False, loc="upper left")
    lo = min(a.get_ylim()[0] for a in axes[1])
    hi = max(a.get_ylim()[1] for a in axes[1]) * 1.45
    axes[1, 0].set_ylim(lo, hi)
    axes[1, 1].set_ylim(lo, hi)
    fig.tight_layout()


def plot_sample(truth, unit_factor=1e3, ncols=4):
    """The spectrum of every galaxy of a sample at its position (black) and
    its true line (green): what each observation sees on its own."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(FONTS)
    recs = truth["sources"]
    nrows = -(-len(recs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows), sharex=True,
                             sharey=True, squeeze=False)
    for ax, rec in zip(axes.ravel(), recs):
        sp = spectrum_at(rec)
        o = np.argsort(sp["v"])
        ax.step(sp["v"][o], sp["mean"][o] * unit_factor, where="mid", color="k", lw=0.9)
        ax.plot(sp["v"][o], sp["expected"][o] * unit_factor, color="#1e8449", lw=2.2)
        ax.axhline(0, color="0.4", lw=0.5)
        ax.set_title(f"z = {rec['z']:.2f},  "
                     + (f"{rec['size_arcsec']:.1f}\"" if rec["size_arcsec"] else "point"))
    for ax in axes.ravel()[len(recs):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("velocity [km/s]")
    for ax in axes[:, 0]:
        ax.set_ylabel("Re [mJy]")
    fig.tight_layout()


def plot_redshift_correction(truth, unit_factor=1e3):
    """The true line of every galaxy of a sample (no noise), coloured by its
    redshift: as observed (left), and brought to ``z_ref`` as the extraction
    does (right).  For a sample of the same galaxy at different redshifts the
    lines on the right are one and the same."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(FONTS)
    recs = truth["sources"]
    zs = np.array([r["z"] for r in recs])
    v = np.linspace(-700, 700, 400)
    cmap, norm = plt.get_cmap("viridis"), plt.Normalize(zs.min(), zs.max())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True, layout="constrained")
    for r in recs:
        sig = r["fwhm_kms"] / 2.3548200450309493
        prof = r["line_flux"] * np.exp(-0.5 * ((v - r["v0_kms"]) / sig) ** 2) \
            / (sig * math.sqrt(2 * math.pi)) * unit_factor
        axes[0].plot(v, prof, color=cmap(norm(r["z"])), lw=2)
        axes[1].plot(v, prof * alpha_factor(r["z"], truth["z_ref"]), color=cmap(norm(r["z"])),
                     lw=2)
    axes[0].set_title("As observed")
    axes[1].set_title(f"Brought to z_ref = {truth['z_ref']:.2f}")
    axes[0].set_ylabel("flux density [mJy]")
    for ax in axes:
        ax.set_xlabel("velocity [km/s]")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="redshift")


def add_companion(rec, offset_arcsec=(6.0, 3.0), z=None, line_flux=0.0, fwhm_kms=250.0,
                  cont_flux=0.0, cont_index=3.0):
    """Add a second source, a point, to a dataset written by :func:`make_ms`
    (``rec``: its record): a companion of the galaxy, ``offset_arcsec``
    (east, north) from it.  ``z``: its redshift, for a companion with the
    line (``line_flux``, ``fwhm_kms``); None: the continuum only
    (``cont_flux``, at the frequency of the line of the galaxy).  Fluxes as
    in :func:`make_ms`, before the primary beam.  Returns the record of the
    companion (``ra_deg``, ``dec_deg``, ``z``, ...), for
    :func:`write_companions`."""
    from casacore import tables as ct

    t = ct.table(rec["path"], readonly=False, ack=False)
    uvw, data = t.getcol("UVW"), t.getcol("DATA")
    freq = ct.table(rec["path"] + "/SPECTRAL_WINDOW", ack=False).getcell("CHAN_FREQ", 0)
    ra0, dec0 = ct.table(rec["path"] + "/FIELD", ack=False).getcell("PHASE_DIR", 0)[0]
    dec_s = math.radians(rec["dec_deg"] + offset_arcsec[1] / 3600.0)
    ra_s = math.radians(rec["ra_deg"] + offset_arcsec[0] / 3600.0 / math.cos(dec_s))
    l, m, n = _basis(ra0, dec0) @ _basis(ra_s, dec_s)[2]
    from_centre = math.degrees(math.hypot(l, m)) * 3600.0
    pb = primary_beam(from_centre, rec["nu_obs_ghz"] * 1e9)
    line, cont = source_spectrum(freq, rec["z"] if z is None else z, rec["rest_freq_ghz"],
                                 line_flux if z is not None else 0.0, fwhm_kms, 0.0,
                                 cont_flux, cont_index)
    phase = 2 * math.pi * (uvw[:, 0:1] * l + uvw[:, 1:2] * m + uvw[:, 2:3] * (n - 1)) \
        * freq[None, :] / C
    vis = pb * (line + cont)[None, :] * np.exp(1j * phase)
    t.putcol("DATA", (data + vis[:, :, None]).astype(data.dtype))
    t.close()
    return {"path": rec["path"], "ra_deg": math.degrees(ra_s), "dec_deg": math.degrees(dec_s),
            "z": z, "line_flux": line_flux if z is not None else 0.0, "fwhm_kms": fwhm_kms,
            "cont_flux": cont_flux, "primary_beam": pb,
            "separation_arcsec": math.hypot(*offset_arcsec)}


def write_companions(path, companions):
    """Write the file of the companions (records of :func:`add_companion`):
    ``<ms_path> <RA> <Dec> [redshift]``, for ``v.stack(companions=path)``."""
    rows = ["# <ms_path>  <RA>  <Dec>  [redshift]"]
    for c in companions:
        rows.append(f"{c['path']}  {fmt_ra(c['ra_deg'])}  {fmt_dec(c['dec_deg'])}"
                    + (f"  {c['z']:.6f}" if c["z"] is not None else ""))
    with open(path, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    return path


def write_list(path, sources, z_ref=None, norm=None):
    """Write a ViSta input list with these galaxies (records of
    :func:`make_sample`, from one sample or several) and return what the
    other functions take as a sample: ``{"input_list", "z_ref", "sources"}``.
    ``z_ref``: the redshift the fluxes refer to (default: the median).
    ``norm``: one value per galaxy for the NORM column (``flux_norm=True``
    divides every galaxy by it), e.g. the magnification of a lens."""
    sources = list(sources)
    rows = ["# <ms_path>  <redshift>  <RA>  <Dec>" + ("  norm=<NORM>" if norm is not None else "")]
    for k, r in enumerate(sources):
        rows.append(f"{r['path']}  {r['z']:.6f}  {fmt_ra(r['ra_deg'])}  {fmt_dec(r['dec_deg'])}"
                    + (f"  norm={norm[k]:g}" if norm is not None else ""))
    with open(path, "w") as fh:
        fh.write("\n".join(rows) + "\n")
    z = float(np.median([r["z"] for r in sources])) if z_ref is None else float(z_ref)
    return {"input_list": path, "z_ref": z, "sources": sources}


def describe(truth):
    """A table of the galaxies of a sample: what was simulated for each."""
    rows = [f"{'':>3s} {'z':>6s} {'nu_obs':>8s} {'line':>8s} {'FWHM':>6s} {'size':>6s} "
            f"{'continuum':>10s} {'noise':>7s} {'array':>6s} {'offset':>7s} {'PB':>5s}",
            f"{'':>3s} {'':>6s} {'GHz':>8s} {'Jy km/s':>8s} {'km/s':>6s} {'arcsec':>6s} "
            f"{'mJy':>10s} {'Jy/vis':>7s} {'m':>6s} {'arcsec':>7s} {'':>5s}"]
    for k, r in enumerate(truth["sources"]):
        rows.append(f"{k:3d} {r['z']:6.3f} {r['nu_obs_ghz']:8.2f} {r['line_flux']:8.3f} "
                    f"{r['fwhm_kms']:6.0f} {r['size_arcsec']:6.2f} {r['cont_flux'] * 1e3:10.3f} "
                    f"{r['noise'] * r['kappa']:7.3f} {r.get('rmax_m', float('nan')):6.0f} "
                    f"{r['offset_arcsec']:7.1f} {r['primary_beam']:5.2f}")
    return "\n".join(rows)


def spectrum_at(rec):
    """Quick look at one simulated dataset (a record of make_ms /
    make_sample): its visibilities phase-shifted to the galaxy and averaged
    over all baselines with their weights (real part, both correlations).

    Returns a dict: ``v`` (radio velocity, km/s), ``mean`` and ``error`` (of
    the data), ``expected`` (the same average of the noiseless visibilities:
    what the data scatter around) and ``total`` (the true flux density times
    the primary beam: what a zero-length baseline would see).  For a source
    larger than the beam ``expected`` is well below ``total``: the long
    baselines resolve it."""
    data, uvw, wt, freq = _centred(rec)
    shifted = data.real
    wsum = wt.sum(axis=(0, 2))
    mean = (wt * shifted).sum(axis=(0, 2)) / wsum
    # error from the scatter (the weights need not be absolute)
    var = (wt * (shifted - mean[None, :, None]) ** 2).sum(axis=(0, 2)) / wsum
    n_eff = wsum ** 2 / (wt ** 2).sum(axis=(0, 2))
    v_line, v_cont = model_visibility(rec, uvw, freq)
    expected = (wt * (v_line + v_cont)[:, :, None]).sum(axis=(0, 2)) / wsum
    line, cont = source_spectrum(freq, rec["z"], rec["rest_freq_ghz"], rec["line_flux"],
                                 rec["fwhm_kms"], rec["v0_kms"], rec["cont_flux"],
                                 rec["cont_index"], rec.get("extra_lines", ()))
    nu_r = rec["rest_freq_ghz"] * 1e9
    return {"v": C_KMS * (nu_r - freq * (1 + rec["z"])) / nu_r, "mean": mean,
            "error": np.sqrt(var / n_eff), "expected": expected,
            "total": rec["primary_beam"] * (line + cont)}


def expected_in_stack(rec, uvw, weight, flag, rest_freq_hz):
    """The noiseless counterpart of one dataset of a stack (phase centre on
    the galaxy, rest-frame channels ``rest_freq_hz``): the weighted sums over
    rows and correlations of the line and continuum model visibilities, and
    of the weights, per channel -- to average several datasets.  The stack
    keeps the baselines divided by (1 + z) (rest-framed): times (1 + z) they
    are the observed ones again."""
    z = rec["z"]
    v_line, v_cont = model_visibility(rec, np.asarray(uvw) * (1 + z),
                                      np.asarray(rest_freq_hz) / (1 + z))
    wt = np.where(flag, 0.0, weight[:, None, :])
    return ((wt * v_line[:, :, None]).sum(axis=(0, 2)),
            (wt * v_cont[:, :, None]).sum(axis=(0, 2)), wt.sum(axis=(0, 2)))


# ---------------------------------------------------------------------------
# the true values of a stack
# ---------------------------------------------------------------------------
def _stack_weights(truth, rescaled=True, weighting="natural"):
    """Factor applied to the flux of every galaxy and its relative weight in
    a naturally weighted stack (approximate: it ignores the differences in uv
    coverage, here the same for all).  ``rescaled``: as the extraction
    measures it, after the primary-beam correction and the redshift
    rescaling (factor alpha, weight the inverse variance of the rescaled
    visibilities).  Otherwise the plain stacked visibilities, as imaged:
    each galaxy attenuated by the primary beam, weighted by its weights."""
    pb = np.array([r["primary_beam"] for r in truth["sources"]])
    noise = np.array([r["noise"] * r["kappa"] for r in truth["sources"]])
    if rescaled:
        a = np.array([alpha_factor(r["z"], truth["z_ref"]) for r in truth["sources"]])
        # democratic: every galaxy the same weight, whatever its noise
        w = (pb / a / noise) ** 2 if weighting == "natural" else np.ones(len(a))
        return a, w / w.sum()
    w = noise ** -2.0
    return pb, w / w.sum()


def true_profile(truth, v_kms, rescaled=True):
    """The true stacked line profile [Jy] at zero spacing against radio
    velocity: the weighted mean of the line profiles of the galaxies, each
    transported to z_ref as the extraction does (``rescaled``), or as in the
    plain stacked visibilities (``rescaled=False``: no rescaling, attenuated
    by the primary beam).  With ``v_jitter_kms`` it is broader than any one
    galaxy."""
    v = np.asarray(v_kms, float)
    a, w = _stack_weights(truth, rescaled)
    out = np.zeros_like(v)
    for r, ai, wi in zip(truth["sources"], a, w):
        sig = r["fwhm_kms"] / 2.3548200450309493
        phi = np.exp(-0.5 * ((v - r["v0_kms"]) / sig) ** 2) / (sig * math.sqrt(2 * math.pi))
        out += wi * ai * r["line_flux"] * phi
    return out


def true_uv(truth, b_klambda, part="line"):
    """The true stacked line flux [Jy km/s] (``part="line"``) or continuum
    flux density [Jy] (``"continuum"``) against baseline [klambda]: the
    weighted mean of F_i g(b; size_i) over the galaxies, after rescaling."""
    b = np.asarray(b_klambda, float) * 1e3
    a, w = _stack_weights(truth)
    out = np.zeros_like(b)
    for r, ai, wi in zip(truth["sources"], a, w):
        if part == "line":
            out += wi * ai * r["line_flux"] * shape_visibility(b, r["size_arcsec"],
                                                               r.get("shape", "gauss"))
        else:
            out += wi * ai * r["cont_flux"] * shape_visibility(b, r["cont_size_arcsec"],
                                                               r.get("shape", "gauss"))
    return out


def expected(truth, v_kms=None, b_klambda=None, weighting="natural"):
    """What the stack of a sample should give, in the form of
    ``vista.compare(..., truth=...)``: ``flux`` (one value per line: the
    lines of the band, by rest frequency, the first the line of the stack;
    the components at the same rest frequency, e.g. wings, add up),
    ``profile`` (velocity of the first line, flux density at zero spacing:
    every component), ``uv`` (baseline, line flux of the first line) and
    ``continuum``.  As the extraction measures it: rescaled to z_ref, the
    primary beam corrected, with the ``weighting`` asked (``"natural"``: the
    deeper observations count more; ``"democratic"``: every galaxy the
    same).  Known only because the sample is simulated."""
    v = np.linspace(-1500, 1500, 1201) if v_kms is None else np.asarray(v_kms, float)
    b = np.geomspace(1, 3000, 200) if b_klambda is None else np.asarray(b_klambda, float)
    a, w = _stack_weights(truth, weighting=weighting)
    rest0 = truth["sources"][0]["rest_freq_ghz"]
    freqs = [rest0]
    flux, prof, uv = {}, np.zeros_like(v), np.zeros_like(b)
    for r, ai, wi in zip(truth["sources"], a, w):
        comps = [dict(rest_freq_ghz=r["rest_freq_ghz"], flux_ratio=1.0, fwhm_kms=r["fwhm_kms"],
                      v0_kms=r["v0_kms"])] + list(r.get("extra_lines", []))
        for c in comps:
            nu = c["rest_freq_ghz"]
            k = next((i for i, f in enumerate(freqs) if abs(f - nu) < 1e-9), None)
            if k is None:
                freqs.append(nu)
                k = len(freqs) - 1
            f_c = wi * ai * r["line_flux"] * c["flux_ratio"]
            flux[k] = flux.get(k, 0.0) + f_c
            # in the velocity of the first line: shifted, and narrower by nu / rest0
            scale = nu / rest0
            v1 = C_KMS * (1 - scale) + c.get("v0_kms", 0.0) * scale
            sig = c["fwhm_kms"] / 2.3548200450309493 * scale
            prof += f_c / (sig * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((v - v1) / sig) ** 2)
            if k == 0:
                uv += f_c * shape_visibility(b * 1e3, r["size_arcsec"], r.get("shape", "gauss"))
    cont = sum(wi * ai * r["cont_flux"] for r, ai, wi in zip(truth["sources"], a, w))
    return {"flux": [flux[k] for k in range(len(freqs))], "profile": (v, prof), "uv": (b, uv),
            "continuum": float(cont), "rest_freq_ghz": freqs}


def weight_shares(truth, special):
    """Who makes the stack: for the natural and the democratic weighting,
    the share of the weight and of the line flux of the stack that comes
    from the galaxies ``special`` (their indices in the sample), the flux
    the stack should give, and the mean flux of the other galaxies."""
    special = np.zeros(len(truth["sources"]), bool) | np.isin(np.arange(len(truth["sources"])),
                                                              special)
    out = {}
    for weighting in ("natural", "democratic"):
        a, w = _stack_weights(truth, weighting=weighting)
        flux = w * a * np.array([r["line_flux"] for r in truth["sources"]])
        out[weighting] = {"weight": float(w[special].sum()),
                          "flux": float(flux[special].sum() / flux.sum()),
                          "expected": float(flux.sum()),
                          "others": float(flux[~special].sum() / w[~special].sum())}
    return out


def plot_weighting(truth, special, results, names=("the others", "the special ones"),
                   unit_factor=1e3):
    """A stack with the natural and with the democratic weighting
    (``results``: ``{"natural": r, "democratic": r}``), against the velocity
    and against the baseline length: the stacked data and their fits, and,
    dotted, the true value with those weights.  Dashed: the true signal of the galaxies
    that are not ``special`` (their mean: the reference) and of the special
    ones."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(FONTS)
    recs = truth["sources"]
    special = np.isin(np.arange(len(recs)), special)
    everyone = np.ones(len(recs), bool)
    v = np.linspace(-900, 900, 1201)
    lines = [results[k].line for k in ("natural", "democratic")]
    b_all = np.concatenate([ln.baseline_klambda[ln.annulus_ok] for ln in lines])
    b = np.geomspace(b_all.min() * 0.7, b_all.max() * 1.3, 300)

    def mean(weighting, keep, per_galaxy):
        a, w = _stack_weights(truth, weighting=weighting)
        return sum(wi * ai * per_galaxy(r) for r, ai, wi, k in zip(recs, a, w, keep) if k) \
            / w[keep].sum()

    def spectrum(r):
        sig = r["fwhm_kms"] / 2.3548200450309493
        return (r["line_flux"] / (sig * math.sqrt(2 * math.pi))
                * np.exp(-0.5 * ((v - r["v0_kms"]) / sig) ** 2))

    def uv(r):
        return r["line_flux"] * shape_visibility(b * 1e3, r["size_arcsec"],
                                                 r.get("shape", "gauss"))

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(17, 6.8))
    for (weighting, c), line in zip((("natural", "C3"), ("democratic", "C0")), lines):
        o = np.argsort(line.velocity_kms)
        ax.step(line.velocity_kms[o], line.profile[o] * unit_factor, where="mid", color=c,
                lw=1.0, alpha=0.8)
        fit = np.zeros_like(v)
        for blk in line.summary["lines"]:
            fit += blk["peak"] * np.exp(-4 * math.log(2)
                                        * ((v - blk["v0_kms"]) / blk["fwhm_kms"]) ** 2)
        ax.plot(v, fit * unit_factor, color=c, lw=2.4, label=f"Stack, {weighting}")
        ok = line.annulus_ok & np.isfinite(line.annulus_flux)
        bx.errorbar(line.baseline_klambda[ok], line.annulus_flux[ok],
                    line.annulus_flux_error[ok], fmt="o", color=c, ms=4, capsize=2, lw=1)
        p = line.summary["line"]
        theta = p.get("theta_fwhm_arcsec") or 0.0
        bx.plot(b, p["flux_gauss"] * shape_visibility(b * 1e3, theta), color=c, lw=2.4,
                label=f"Stack, {weighting}")
        dotted = f"True, {weighting} weights"
        ax.plot(v, mean(weighting, everyone, spectrum) * unit_factor, color=c, lw=2.2,
                ls=":", label=dotted)
        bx.plot(b, mean(weighting, everyone, uv), color=c, lw=2.2, ls=":", label=dotted)
    for keep, c, lw, label in ((special, "C1", 2, names[1].capitalize()),
                               (~special, "#1e8449", 2.6, f"Mean of {names[0]}")):
        ax.plot(v, mean("natural", keep, spectrum) * unit_factor, color=c, lw=lw, ls="--",
                label=label)
        bx.plot(b, mean("natural", keep, uv), color=c, lw=lw, ls="--", label=label)
    for a_, xlabel, ylabel, title in (
            (ax, "velocity [km/s]", "Re flux density [mJy]", "Against the velocity"),
            (bx, r"baseline [k$\lambda$]", "Re line flux [Jy km/s]",
             "Against the baseline length")):
        a_.axhline(0, color="0.4", lw=0.6)
        a_.set_xlabel(xlabel)
        a_.set_ylabel(ylabel)
        a_.set_title(title)
    ax.set_xlim(-900, 900)
    bx.set_xscale("log")
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle("The stacked line, natural and democratic weighting")
    fig.tight_layout()


def plot_images(images, ncols=None, same_scale=True, unit="Jy"):
    """Several dirty images (``{title: image}``, from ``vista.dirty_image``)
    side by side, with the same colour scale."""
    import matplotlib.pyplot as plt
    from vista.quicklook import plot_image
    plt.rcParams.update(FONTS)
    ncols = ncols or len(images)
    nrows = -(-len(images) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 5.2 * nrows), squeeze=False)
    top = max(max(float(np.nanmax(i.data)), 5 * i.rms) for i in images.values())
    for ax, (title, img) in zip(axes.ravel(), images.items()):
        plot_image(img, ax, unit=unit, title=title, vmax=top if same_scale else None,
                   contours=False)
    for ax in axes.ravel()[len(images):]:
        ax.axis("off")
    fig.tight_layout()


# ---------------------------------------------------------------------------
# a sample
# ---------------------------------------------------------------------------
def fmt_ra(deg):
    h = (deg % 360.0) / 15.0
    hh = int(h)
    mm = int((h - hh) * 60)
    ss = ((h - hh) * 60 - mm) * 60
    return f"{hh:02d}:{mm:02d}:{ss:011.8f}"


def fmt_dec(deg):
    sg = "+" if deg >= 0 else "-"
    d = abs(deg)
    dd = int(d)
    mm = int((d - dd) * 60)
    ss = ((d - dd) * 60 - mm) * 60
    return f"{sg}{dd:02d}:{mm:02d}:{ss:010.7f}"


def alpha_factor(z, z_ref, H0=67.4, Om0=0.315):
    """The redshift rescaling of the extraction (Eq. 27):
    (D_L(z) / D_L(z_ref))^2 (1 + z_ref) / (1 + z)."""
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    d = cosmo.luminosity_distance([z, z_ref]).value
    return (d[0] / d[1]) ** 2 * (1 + z_ref) / (1 + z)


def make_sample(directory, n_sources=12, z_range=(0.4, 0.6), line_flux=0.25,
                same_luminosity=True, z_ref=None, v_jitter_kms=0.0, flux_scatter_dex=0.0,
                depth_spread_dex=0.0, seed=0, input_list="input_list.txt",
                observed_freq_ghz=None, **kwargs):
    """Write ``n_sources`` Measurement Sets and the ViSta input list.

    The redshifts are uniform in ``z_range``.  With ``same_luminosity`` every
    galaxy has the luminosity that gives ``line_flux`` (and ``cont_flux``) at
    ``z_ref`` (default: the median redshift, as in the extraction), so the
    stack with the redshift rescaling should return exactly those values;
    otherwise every galaxy has ``line_flux``.  ``flux_scatter_dex`` spreads
    the luminosities (log-normal), ``v_jitter_kms`` the line centres
    (redshift errors), ``depth_spread_dex`` the noise of the datasets
    (observations of different depth: natural and democratic weighting then
    differ).  ``observed_freq_ghz``: every galaxy is observed in the same
    band, centred here, whatever its redshift (a continuum survey: no line
    in common, ``line_flux=0``).  Other keywords go to :func:`make_ms` (``size_arcsec``,
    ``noise``, ``fwhm_kms``, ...); a callable is evaluated for every source
    with a random generator, e.g. ``size_arcsec=lambda rng: rng.uniform(0.5, 1.5)``.

    Returns a dict with the input list path, one record per source and the
    true values of the stack (``line_flux_stack``, ``line_peak_stack``,
    ``fwhm_kms_stack``, ``v0_kms_stack``, ``cont_flux_stack``); the true
    profile and uv curves are :func:`true_profile` and :func:`true_uv`.
    """
    os.makedirs(directory, exist_ok=True)
    rng = np.random.default_rng(seed)
    zs = np.sort(rng.uniform(*z_range, n_sources))
    z_ref = float(np.median(zs)) if z_ref is None else float(z_ref)
    records, lines = [], []
    for k, z in enumerate(zs):
        scale = 1.0 / alpha_factor(z, z_ref) if same_luminosity else 1.0
        scale *= 10 ** (flux_scatter_dex * rng.standard_normal())
        kw = {key: (val(rng) if callable(val) else val) for key, val in kwargs.items()}
        kw["cont_flux"] = kw.get("cont_flux", 0.0) * scale
        kw["v0_kms"] = kw.get("v0_kms", 0.0) + v_jitter_kms * rng.standard_normal()
        kw["noise"] = kw.get("noise", 0.05) * 10 ** (depth_spread_dex * rng.uniform(-1, 1))
        if observed_freq_ghz is not None:
            kw["rest_freq_ghz"] = observed_freq_ghz * (1.0 + float(z))
        centre = (150.0 + 0.5 * k, 2.0 + 0.1 * k)       # a different field for every galaxy
        path = os.path.join(directory, f"galaxy_{k:02d}.ms")
        rec = make_ms(path, float(z), line_flux=line_flux * scale, phase_centre=centre,
                      seed=seed * 1000 + k, **kw)
        rec["scale"] = scale
        records.append(rec)
        lines.append(f"{path}  {z:.6f}  {fmt_ra(rec['ra_deg'])}  {fmt_dec(rec['dec_deg'])}")
    list_path = os.path.join(directory, input_list)
    with open(list_path, "w") as fh:
        fh.write("# <ms_path>  <redshift>  <RA>  <Dec>\n" + "\n".join(lines) + "\n")
    truth = {"input_list": list_path, "z_ref": z_ref, "sources": records}
    # what the stack should measure: the weighted mean of the rescaled values
    # centre, width and peak: of the Gaussian closest to the true profile,
    # as the extraction fits one (the same as each galaxy's without v_jitter)
    from scipy.optimize import curve_fit
    v = np.linspace(-3000, 3000, 6001)
    prof = true_profile(truth, v)
    gauss = lambda x, a, x0, s: a * np.exp(-0.5 * ((x - x0) / s) ** 2)
    with warnings.catch_warnings():            # an exact Gaussian: no covariance, no matter
        warnings.simplefilter("ignore")
        (peak, v0, sig), _ = curve_fit(gauss, v, prof, p0=(prof.max(), v[np.argmax(prof)], 100.0))
    truth.update({"line_flux_stack": float(true_uv(truth, [0.0])[0]),
                  "cont_flux_stack": float(true_uv(truth, [0.0], "continuum")[0]),
                  "line_peak_stack": float(peak), "v0_kms_stack": round(float(v0), 6) + 0.0,
                  "fwhm_kms_stack": float(2.3548200450309493 * abs(sig))})
    with open(os.path.join(directory, "truth.json"), "w") as fh:
        json.dump(truth, fh, indent=1, default=float)
    return truth
