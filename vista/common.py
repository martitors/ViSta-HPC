"""
vista.common -- pieces shared by the stacking modules
=====================================================

    log, detail, warn        the log of the stacking (the package log, see
                             ``vista.set_verbosity``)
    casacore_lock            dask-ms is not thread safe
    parse_ra_dec, wrap_dra,  sky coordinates
    angular_sep
    fmt_size, fmt_time       numbers for the messages

Internal module of :mod:`vista.pipeline`.
"""

import math
import threading

from . import _detail, _say, _warn

RAD2ARCSEC = 180.0 / math.pi * 3600.0


# =============================================================================
# Log
# =============================================================================
#   [ViSta     12.3s] message
#   [ViSta     12.5s] WARNING: message
# The time is counted from the start of ViSta.run().

def log(msg):
    """Progress line (hidden by ``verbose=False``)."""
    _say(msg)


def detail(msg):
    """Detail line (shown only with ``verbose='debug'``)."""
    _detail(msg)


def warn(msg):
    """Warning line (always shown)."""
    _warn(msg)


# =============================================================================
# Thread safety
# =============================================================================
# The metadata and the subtables are read and written with dask-ms, partly
# from thread pools; every such access to casacore goes through this lock.
# The visibilities are read by the reader processes and written by the single
# writer thread with python-casacore, each on its own table.

casacore_lock = threading.Lock()


# =============================================================================
# Sky coordinates
# =============================================================================

def parse_ra_dec(ra_str, dec_str):
    """RA and Dec strings to radians.

    Parameters
    ----------
    ra_str : str
        Right ascension in hours, in any format astropy understands,
        e.g. ``'12:34:56.7'`` or ``'12h34m56.7s'``.
    dec_str : str
        Declination in degrees, e.g. ``'+12:34:56.7'`` or ``'12.582d'``.  The
        ALMA form with dots, ``'-23.01.45.6'``, is accepted as well.

    Returns
    -------
    (float, float)
        RA and Dec in radians.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    if ":" not in dec_str:
        parts = dec_str.split(".")
        if len(parts) >= 3:
            dec_str = parts[0] + ":" + parts[1] + ":" + ".".join(parts[2:])
    c = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    return float(c.ra.rad), float(c.dec.rad)


def wrap_dra(ra_new, ra_old):
    """Bring *ra_new* within pi of *ra_old*.

    Avoids a jump of 2 pi in the phase shift when the two directions sit on
    opposite sides of RA = 0.

    Parameters
    ----------
    ra_new, ra_old : float
        Right ascensions in radians.

    Returns
    -------
    float
        *ra_new* plus a multiple of 2 pi, within [ra_old - pi, ra_old + pi].
    """
    dra = (ra_new - ra_old + math.pi) % (2.0 * math.pi) - math.pi
    return ra_old + dra


def angular_sep(ra0, dec0, ra1, dec1):
    """Angular distance between two directions (radians in, radians out).

    Haversine formula, accurate at any separation.
    """
    h = (math.sin(0.5 * (dec1 - dec0)) ** 2
         + math.cos(dec0) * math.cos(dec1) * math.sin(0.5 * (ra1 - ra0)) ** 2)
    return 2.0 * math.asin(min(1.0, math.sqrt(h)))


# =============================================================================
# Numbers for the messages
# =============================================================================

def fmt_size(nbytes):
    """Byte count as a readable string, e.g. ``'97.5 MB'``."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def fmt_time(seconds):
    """Duration as a readable string: seconds below two minutes, else minutes."""
    if seconds < 120:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"
