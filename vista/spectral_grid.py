"""
vista.spectral_grid -- the common rest-frame channel grid
=========================================================

Every entry of the stack is rebinned onto channels of the same width.  Each
entry gets an output window with the same number of channels, centred on the
line when its band covers it, otherwise slid to the edge of its band; the
channel edges are always the same for every entry, so the channels of
different entries line up exactly.

    compute_grid           the whole grid: width, number of channels, windows
    choose_channel_width   the common channel width
    choose_nchan           the number of output channels
    place_window           the window of one entry
    snap_to_grid           align a slid window with the common grid
    covered_channels       which output channels an input range overlaps

Internal module of :mod:`vista.pipeline`.
"""

import math
from dataclasses import dataclass, field
from typing import List

import numpy as np

from .common import detail

C_LIGHT_KMS = 299792.458            # speed of light [km/s]


@dataclass
class Grid:
    """The output spectral grid of the stack.

    Attributes
    ----------
    df : float
        Common channel width [Hz].
    nchan : int
        Channels of every output window.
    freq : list of ndarray
        Channel centres of each entry [Hz], one array of *nchan* values.
    shift : list of float
        Centre of each window minus the central frequency [Hz]; 0 when the
        window is centred on it.
    reason : list of str
        How each window was placed: ``'ideal window fits'``,
        ``'shifted to left edge'`` or ``'shifted to right edge'``.
    width_note, nchan_note : str
        Where the width and the number of channels come from (for the log).
    """
    df: float
    nchan: int
    freq: List[np.ndarray] = field(default_factory=list)
    shift: List[float] = field(default_factory=list)
    reason: List[str] = field(default_factory=list)
    width_note: str = ""
    nchan_note: str = ""


def compute_grid(central_freq, ranges, df_rest, nchan_in, names, redshifts,
                 nchan_out=None, velocity_range_kms=None, channel_width_hz=None):
    """Output channel grid of every entry.

    1. The common channel width is the widest rest-frame input channel,
       unless *channel_width_hz* is set.
    2. The number of channels follows from *velocity_range_kms*, from
       *nchan_out*, or from the widest coverage of a single entry.
    3. Each entry gets a window of that size centred on *central_freq*;
       where it does not fit, it is slid to the edge of the band of the
       entry and aligned with the common grid.

    Parameters
    ----------
    central_freq : float
        Rest-frame centre of the grid [Hz].
    ranges : list of (float, float)
        Rest-frame frequency range of each entry, lowest and highest channel
        centre [Hz].
    df_rest : list of float
        Rest-frame channel width of each entry [Hz].
    nchan_in : list of int
        Input channels of each entry.
    names, redshifts : list
        Name and redshift of each entry, for the messages.
    nchan_out : int or None, default None
        Number of channels.  ``None`` takes the widest coverage of one entry.
    velocity_range_kms : float, (float, float) or None, default None
        Bandwidth as a velocity range; it takes precedence over *nchan_out*.
    channel_width_hz : float or None, default None
        Channel width.  ``None`` takes the widest channel of the sample.

    Returns
    -------
    Grid
    """
    df_new, width_note = choose_channel_width(df_rest, channel_width_hz, names, redshifts)
    nn_per_ms = [max(1, int(nc * dr / df_new)) for nc, dr in zip(nchan_in, df_rest)]
    nn, nchan_note = choose_nchan(central_freq, df_new, nn_per_ms, nchan_out,
                                  velocity_range_kms)

    bandwidth = nn * df_new
    start_ideal = central_freq - bandwidth / 2.0
    end_ideal = central_freq + bandwidth / 2.0
    detail(f"Target window: [{start_ideal/1e6:.6f}, {end_ideal/1e6:.6f}] MHz")

    grid = Grid(df=df_new, nchan=nn, width_note=width_note, nchan_note=nchan_note)
    n = len(ranges)
    for i, (msmin, msmax) in enumerate(ranges):
        start, reason = place_window(msmin, msmax, start_ideal, end_ideal,
                                     bandwidth, df_new, names[i])
        offset = start + bandwidth / 2.0 - central_freq
        detail(f"  [{i+1}/{n}]  {names[i]}  start={start/1e6:.6f} MHz  "
               f"end={(start + bandwidth)/1e6:.6f} MHz  offset={offset:+.3e} Hz  ({reason})")
        grid.freq.append((start + df_new / 2.0 + np.arange(nn) * df_new).astype(np.float64))
        grid.shift.append(0.0 if reason == "ideal window fits" else offset)
        grid.reason.append(reason)
    return grid


def choose_channel_width(df_rest, channel_width_hz, names, redshifts):
    """Common rest-frame channel width of the output grid.

    Parameters
    ----------
    df_rest : list of float
        Rest-frame channel width of each entry [Hz].
    channel_width_hz : float or None
        Width asked by the user.  ``None`` takes the widest channel of the
        sample, the finest grid every dataset can support.  A finer value is
        refused: the coarsest dataset would be spread over channels it does
        not resolve, and their noise would be correlated.
    names, redshifts : list
        Name and redshift of each entry, only for the error message.

    Returns
    -------
    (float, str)
        The width [Hz] and where it comes from, for the log.
    """
    floor = max(df_rest)
    if channel_width_hz is None:
        detail(f"Common channel width (df_new) = {floor/1e3:.3f} kHz "
               f"(widest rest-framed channel of the sample)")
        return floor, "width: the coarsest rest-frame channel of the sample"
    width = float(channel_width_hz)
    if width < floor:
        worst = int(np.argmax(df_rest))
        raise ValueError(
            f"[ViSta] requested channel width {width/1e3:.3f} kHz is finer than "
            f"the coarsest rest-framed channel of the sample, {floor/1e3:.3f} kHz "
            f"({names[worst]}, z={redshifts[worst]:.4f}).  Use a width >= "
            f"{floor:.6g} Hz, or drop that dataset.")
    detail(f"Common channel width (df_new) = {width/1e3:.3f} kHz "
           f"(user-set; sample floor is {floor/1e3:.3f} kHz)")
    return width, "width: channel_width_hz"


def choose_nchan(central_freq, df_new, nn_per_ms, nchan_out, velocity_range_kms):
    """Number of channels of the output grid.

    Parameters
    ----------
    central_freq : float
        Rest-frame centre of the grid [Hz].
    df_new : float
        Channel width [Hz].
    nn_per_ms : list of int
        Channels each entry could fill on its own.
    nchan_out : int or None
        Number of channels asked by the user.  ``None`` takes the largest
        number any single entry can fill.
    velocity_range_kms : float, (float, float) or None
        Bandwidth as a velocity range, optical convention.  A number ``v``
        means ``(-v, +v)``.  It takes precedence over *nchan_out*.

    Returns
    -------
    (int, str)
        The number of channels, rounded down to an even number, and where it
        comes from, for the log.
    """
    lo, hi = min(nn_per_ms), max(nn_per_ms)
    if velocity_range_kms is not None:
        if isinstance(velocity_range_kms, (tuple, list)):
            v_lo, v_hi = float(velocity_range_kms[0]), float(velocity_range_kms[1])
        else:
            v_hi = abs(float(velocity_range_kms))
            v_lo = -v_hi
        bandwidth = central_freq * (v_hi - v_lo) / C_LIGHT_KMS
        nn = max(2, int(np.ceil(bandwidth / df_new)))
        note = f"channels: velocity range {v_lo:+.0f} to {v_hi:+.0f} km/s"
        detail(f"Velocity range requested: [{v_lo:.0f}, {v_hi:.0f}] km/s "
               f"(dv={v_hi - v_lo:.0f} km/s)")
        detail(f"  -> bandwidth = {bandwidth/1e6:.3f} MHz / "
               f"df_new={df_new/1e3:.3f} kHz => nn = {nn} channels")
    elif nchan_out is None:
        nn = hi
        note = f"channels: automatic, the MS fill {lo}-{hi} of them"
        detail(f"Output channels (auto): {nn}  [per-MS coverage: {lo}-{hi}]")
    else:
        nn = int(nchan_out)
        note = f"channels: nchan, the MS can fill {lo}-{hi} of them"
        detail(f"Output channels (user-set): {nn}  [per-MS available: {lo}-{hi}]")
    if nn % 2:
        nn -= 1
    detail(f"Final output channels: {nn}")
    return nn, note


def place_window(msmin, msmax, start_ideal, end_ideal, bandwidth, df_new, name):
    """Start of the output window of one entry.

    The common window is used when the entry covers it.  Otherwise it is
    slid to the edge of the band of the entry, never cut, and aligned with
    the common grid.

    Parameters
    ----------
    msmin, msmax : float
        Rest-frame frequency range of the entry [Hz].
    start_ideal, end_ideal : float
        The common window [Hz].
    bandwidth : float
        Its width, ``nchan * df_new`` [Hz].
    df_new : float
        Channel width [Hz].
    name : str
        Name of the entry, for the error message.

    Returns
    -------
    (float, str)
        The start [Hz] and how the window was placed.
    """
    if start_ideal >= msmin and end_ideal <= msmax:
        return start_ideal, "ideal window fits"
    if msmin > start_ideal:
        return snap_to_grid(msmin, start_ideal, df_new, "left"), "shifted to left edge"
    if msmax < end_ideal:
        return (snap_to_grid(msmax - bandwidth, start_ideal, df_new, "right"),
                "shifted to right edge")
    raise RuntimeError(f"[ViSta] Could not place output window for MS {name}")


def snap_to_grid(start, start_ideal, df, side):
    """Move the start of a slid window onto the common grid.

    Parameters
    ----------
    start : float
        Start of the slid window [Hz].
    start_ideal : float
        Start of the common window; the grid is ``start_ideal + k * df``.
    df : float
        Channel width [Hz].
    side : {'left', 'right'}
        ``'left'`` rounds up, so the window starts inside the band of the MS;
        ``'right'`` rounds down, so it ends inside it.

    Returns
    -------
    float
        The aligned start: every output window then has the same channel
        edges, and the channels of different datasets line up exactly.
    """
    k = (start - start_ideal) / df
    k = math.ceil(k - 1e-9) if side == "left" else math.floor(k + 1e-9)
    return start_ideal + k * df


def covered_channels(freq_new, df_new, f_lo, f_hi):
    """Output channels that overlap the input range ``[f_lo, f_hi]``.

    Parameters
    ----------
    freq_new : ndarray
        Centres of the output channels [Hz].
    df_new : float
        Output channel width [Hz].
    f_lo, f_hi : float
        Edges of the input channels [Hz].

    Returns
    -------
    ndarray of bool
        One value per output channel.
    """
    return (np.minimum(freq_new + 0.5 * df_new, f_hi)
            > np.maximum(freq_new - 0.5 * df_new, f_lo))
