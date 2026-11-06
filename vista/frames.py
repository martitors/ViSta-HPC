"""
vista.frames -- the spectral frame of the inputs
================================================

ViSta computes the rest frequency of every channel as  f * (1 + z).  That is
right only if the frequencies f of the input MS are in the frame in which z
is defined.  ALMA and VLA data are TOPO: the frequency of a line changes with
the date (the Earth's orbit, up to +-30 km/s) and, a little, within an
observation (the Earth's rotation, up to 0.5 km/s).  NOEMA tracks the Doppler
shift at the telescope, so its data are already in LSR(K) once their frame is
labelled correctly.

With ``run(frame="LSRK")`` (or "BARY", ...) every row is corrected: from its
time, the position of the array and the direction of the source, the Doppler
factor

    k = f_frame / f_input

is computed, and the row is processed with the frequencies  f * k  and the
redshift  (1 + z) * k - 1.  This is exact: the phase shift still uses the
frequency at which the row was observed, the rebinning puts the line where it
is in the chosen frame, and nothing is resampled twice (the kernel rebins
once, as without the correction).  The kernel itself is unchanged.

Precision and reproducibility
-----------------------------
k is computed with casacore measures at knots every KNOT_S seconds that
bracket every row time, and interpolated linearly in between: the error is
below 0.2 mm/s (k changes by about 1e-10 per second, smoothly).  It is not
rounded: rows with the same time share their k exactly, rows at different
times are processed with their own k.

casacore measures keeps caches whose state depends on the previous
conversions, which changes the last bits of k (1e-16 .. 1e-14).  So every k
is computed here, in the main process, entry after entry in the order of the
input list and knot after knot in time order, before any reader starts: the
same input list gives the same k, bit for bit, whatever the number and kind
of readers, the chunk size or the device.  The readers only interpolate.

    FRAME_CODES       MEAS_FREQ_REF code of every frame
    check_frames      frame of every entry and, with a target frame, its
                      Doppler factor at the knots (main process)
    EntryFrame        what the reader needs: the knots and the range of k
    k_per_row         k of every row, from its time (reader)
    row_segments      runs of consecutive rows with the same k
    velocity_kms      the velocity shift of a factor k

Shared, unchanged, by the main, extract and extract_opt branches, so that
they give the same k for the same input.
"""

import functools
from dataclasses import dataclass

import numpy as np

FRAME_CODES = {"REST": 0, "LSRK": 1, "LSRD": 2, "BARY": 3, "GEO": 4, "TOPO": 5,
               "GALACTO": 6, "LGROUP": 7, "CMB": 8}
FRAME_NAMES = {code: name for name, code in FRAME_CODES.items()}
TARGET_FRAMES = ("LSRK", "LSRD", "BARY", "GALACTO", "LGROUP", "CMB")

C_KMS = 299792.458
KNOT_S = 60.0         # k is computed every KNOT_S seconds around the data, interpolated in between
SAMPLE_S = 600.0      # times sampled for the warning when no frame is chosen
MAX_TIME_SAMPLES = 20000   # rows of TIME read per MS for that estimate
_F_REF = 1.0e11       # any frequency: the factor does not depend on it


@dataclass
class EntryFrame:
    """Frame correction of one entry (line of the input list).

    Attributes
    ----------
    input_frame, target_frame : str
        Frame of its SPWs (e.g. 'TOPO') and the chosen frame (e.g. 'LSRK').
    knot_t, knot_k : ndarray
        Times [MJD s] and Doppler factors k = f_target / f_input at the
        knots; every row time of the MS lies between two knots.
    k_min, k_mid, k_max : float
        Smallest, median and largest k at the knots.  k_mid sets the
        output grid and the weight factor R of the entry; k_min and k_max
        widen the channel selection, so that no channel is lost for any row.
    """
    input_frame: str
    target_frame: str
    knot_t: np.ndarray
    knot_k: np.ndarray
    k_min: float
    k_mid: float
    k_max: float

    @property
    def k_range(self):
        """(k_min, k_mid, k_max)."""
        return self.k_min, self.k_mid, self.k_max


# =============================================================================
# Reading the MS
# =============================================================================

def read_frame_info(ms, spws):
    """Frame of the SPWs, centre of the array and sampled times of one MS.

    Parameters
    ----------
    ms : str
        Path of the MS.
    spws : list of int or None
        SPWs of the entry (``None``: all).

    Returns
    -------
    (list of int, (float, float, float), ndarray)
        The MEAS_FREQ_REF codes of the SPWs, the mean ITRF position of the
        antennas [m], and the times with data, one per SAMPLE_S seconds
        [MJD s] (from the TIME column: the OBSERVATION table is not always
        complete, e.g. after concat or in simulated data).
    """
    refs, position, times = _ms_frame_info(ms)
    selected = spws if spws is not None else range(len(refs))
    codes = sorted({int(refs[s]) for s in selected})
    return codes, position, times


@functools.lru_cache(maxsize=None)
def _ms_frame_info(ms):
    """MEAS_FREQ_REF of every SPW, array position and sampled times of one MS.

    Cached: an MS listed on several lines of the input list is read once.
    """
    from casacore.tables import table

    with table(f"{ms}::SPECTRAL_WINDOW", ack=False) as t:
        refs = t.getcol("MEAS_FREQ_REF")
    with table(f"{ms}::ANTENNA", ack=False) as t:
        position = tuple(float(x) for x in np.mean(t.getcol("POSITION"), axis=0))
    with table(ms, ack=False) as t:
        nrows = t.nrows()
        step = max(1, nrows // MAX_TIME_SAMPLES)
        times = np.append(t.getcol("TIME", rowincr=step),
                          t.getcol("TIME", startrow=nrows - 1, nrow=1))
    times = np.unique(np.round(times / SAMPLE_S)) * SAMPLE_S
    return refs, position, times


@functools.lru_cache(maxsize=None)
def knot_times(ms):
    """Knots of one MS: the multiples of KNOT_S just below and just above
    every row time [MJD s], so that every row lies between two knots at most
    KNOT_S apart (the whole TIME column is read; cached per MS)."""
    from casacore.tables import table

    with table(ms, ack=False) as t:
        times = np.unique(t.getcol("TIME"))
    return np.unique(np.concatenate([np.floor(times / KNOT_S),
                                     np.ceil(times / KNOT_S)])) * KNOT_S


def clear_cache():
    """Forget what was read from the MSs (they may have changed since)."""
    _ms_frame_info.cache_clear()
    knot_times.cache_clear()


# =============================================================================
# The Doppler factor
# =============================================================================

class DopplerFactor:
    """Doppler factor k(t) = f_target / f_input for one direction.

    Parameters
    ----------
    input_frame, target_frame : str
        Frame of the input SPWs (e.g. 'TOPO') and the chosen frame (e.g. 'LSRK').
    position : (float, float, float)
        ITRF position of the array [m].
    ra, dec : float
        Direction of the source [rad, J2000].
    """

    def __init__(self, input_frame, target_frame, position, ra, dec):
        from casacore.measures import measures
        self.input_frame = input_frame
        self.target_frame = target_frame
        self._me = me = measures()
        x, y, z = position
        me.do_frame(me.position("ITRF", *[f"{float(c)!r}m" for c in (x, y, z)]))
        me.do_frame(me.direction("J2000", f"{float(ra)!r}rad", f"{float(dec)!r}rad"))

    def at(self, t):
        """k at the time t [MJD s]."""
        me = self._me
        me.do_frame(me.epoch("UTC", f"{float(t)!r}s"))
        f = me.measure(me.frequency(self.input_frame, f"{_F_REF!r}Hz"), self.target_frame)
        return f["m0"]["value"] / _F_REF

    def many(self, times):
        """k at every time of *times*, computed in the given order."""
        return np.array([self.at(t) for t in times], dtype=np.float64)


def check_frames(ms_list, spw_list, ra_list, dec_list, frame, lock=None):
    """Frame of every entry and, when needed, its Doppler factor.

    Run in the main process, before the readers get their jobs: every k is
    computed here, in a fixed order (see the module docstring).

    Parameters
    ----------
    ms_list, spw_list : list
        MS and SPWs (None: all) of every entry.
    ra_list, dec_list : list of float
        Direction of the source of every entry [rad, J2000].
    frame : str or None
        The chosen frame (one of TARGET_FRAMES), or None: no correction,
        only a warning when the inputs are TOPO or GEO.
    lock : context manager or None
        Held while an MS is read (the dask-ms readers of the caller).

    Returns
    -------
    (list, list of (str, str))
        For every entry an :class:`EntryFrame` or None (no correction), and
        the messages for the log, as ``(level, text)`` with level 'log',
        'warn' or 'detail'.

    Raises
    ------
    ValueError
        SPWs of one entry in different frames, or labelled REST.
    RuntimeError
        The factor cannot be computed while a frame was asked for.
    """
    import contextlib

    lock = lock or contextlib.nullcontext()
    n = len(ms_list)
    frames = [None] * n
    messages = []
    clear_cache()
    target = frame or "LSRK"
    shifts, inputs = [], {}
    for i in range(n):
        with lock:
            codes, position, times = read_frame_info(ms_list[i], spw_list[i])
        name = ms_list[i].rstrip("/").split("/")[-1]
        if len(codes) > 1:
            raise ValueError(f"{name}: the SPWs of this line are in different frames "
                             f"{[FRAME_NAMES.get(c, c) for c in codes]}: put each SPW "
                             f"on its own line of the input list")
        frame_in = FRAME_NAMES.get(codes[0], str(codes[0]))
        inputs[frame_in] = inputs.get(frame_in, 0) + 1
        if frame_in == "REST":
            raise ValueError(f"{name}: the SPWs are labelled REST (already rest "
                             f"frequencies): ViSta would multiply them by (1+z) again")
        if frame_in == target or (frame is None and frame_in not in ("TOPO", "GEO")):
            continue
        try:
            factor = DopplerFactor(frame_in, target, position, ra_list[i], dec_list[i])
            if frame is None:                   # only the size of the effect
                ks = factor.many(times)
                k_min, k_mid, k_max = float(ks.min()), float(np.median(ks)), float(ks.max())
            else:
                with lock:
                    knots = knot_times(ms_list[i])
                ks = factor.many(knots)
                k_min, k_mid, k_max = float(ks.min()), float(np.median(ks)), float(ks.max())
                frames[i] = EntryFrame(frame_in, frame, knots, ks, k_min, k_mid, k_max)
        except Exception as e:
            if frame is not None:
                raise RuntimeError(f"{name}: cannot compute the Doppler factor "
                                   f"{frame_in} -> {target} (casacore measures data "
                                   f"missing?): {e}") from e
            messages.append(("detail", f"  frames not checked: {e}"))
            return [None] * n, messages
        shifts += [velocity_kms(k_min), velocity_kms(k_max)]
        messages.append(("detail", f"  [{i+1}/{n}]  {name}  {frame_in} -> {target}: "
                                   f"k = {k_mid:.12f}, line at {velocity_kms(k_min):+.3f} .. "
                                   f"{velocity_kms(k_max):+.3f} km/s"
                                   + (f", {len(ks)} knots" if frame is not None else "")))

    found = ", ".join(f"{c} {f}" for f, c in sorted(inputs.items()))
    if shifts and max(shifts) - min(shifts) >= 0.05:
        effect = (f"the line moves by {min(shifts):+.1f} .. {max(shifts):+.1f} km/s "
                  f"from one observation to another")
    elif shifts:
        effect = f"the line is shifted by {shifts[0]:+.1f} km/s"
    if frame is None:
        if shifts:
            messages.append(("warn", f"input frames: {found}.  Without run(frame=...) the "
                                     f"motion of the Earth is not removed: {effect}.  Set "
                                     f"frame to the frame of the redshifts ('LSRK' or 'BARY')"))
    elif shifts:
        messages.append(("log", f"Frame: inputs {found} -> {frame}, corrected row by row "
                                f"(without the correction {effect})"))
    else:
        messages.append(("log", f"Frame: inputs {found}, already {frame}"))
    return frames, messages


# =============================================================================
# In the reader
# =============================================================================

def k_per_row(times, entry):
    """k of every row, from its TIME [MJD s], interpolated between the knots
    of the entry (an :class:`EntryFrame`)."""
    times = np.asarray(times, dtype=np.float64)
    if len(entry.knot_t) == 1:
        return np.full(len(times), entry.knot_k[0])
    return np.interp(times, entry.knot_t, entry.knot_k)


def velocity_kms(k):
    """Velocity shift of a line by the factor k (positive: towards lower frequency)."""
    return C_KMS * (1.0 - 1.0 / k)


def row_segments(k_rows):
    """Runs of consecutive rows with the same k.

    Returns
    -------
    list of (int, int, float)
        ``(first, last + 1, k)`` of every run.  An MS in time order gives one
        run per time stamp (all the baselines of an integration).
    """
    k_rows = np.asarray(k_rows, dtype=np.float64)
    cuts = np.flatnonzero(k_rows[1:] != k_rows[:-1]) + 1
    bounds = np.concatenate([[0], cuts, [len(k_rows)]])
    return [(int(a), int(b), float(k_rows[a])) for a, b in zip(bounds[:-1], bounds[1:])]
