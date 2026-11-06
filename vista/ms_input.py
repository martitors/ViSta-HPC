"""
vista.ms_input -- reading the input Measurement Sets
====================================================

    read_ms_metadata    SPWs, channel frequencies and phase centre of one MS
    EntryJob            what to read for one line of the input list
    read_entry_chunks   read one entry: first an EntryInfo (what the entry
                        contributes, what the kernel needs), then its rows,
                        one Chunk at a time
    EntryInfo, Chunk    the two kinds of result

The SPWs of one entry are merged row by row (their channels side by side),
which is correct only if they have the same rows: see :func:`merge_spws`.

The reader runs in the reader processes, which import only this module (plus
numpy, python-casacore and astropy): dask-ms, used for the metadata in the
main process, is imported only there.

Internal module of :mod:`vista.pipeline`.
"""

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .common import RAD2ARCSEC, angular_sep, parse_ra_dec, wrap_dra
from .frames import k_per_row, row_segments
from .spectral_grid import covered_channels

C_LIGHT_MS         = 299792458.0    # speed of light [m/s]
TIME_TOL_S         = 1e-3           # TIME tolerance when matching SPW rows [s]
WIDTH_TOL          = 1e-3           # relative tolerance on channel widths
WEIGHT_RATIO_WARN  = 1.25           # warn above this ratio of mean weights
TARGET_CHUNK_BYTES = 512 * 1024**2  # target size of one chunk of visibilities
MIN_CHUNK_ROWS     = 1000           # never read fewer rows than this per chunk

# Columns copied row by row from the reference SPW (WEIGHT and SIGMA rescaled).
ROW_COLUMNS = ["TIME", "EXPOSURE", "INTERVAL", "WEIGHT", "SIGMA",
               "SCAN_NUMBER", "STATE_ID", "FLAG_ROW"]


# =============================================================================
# Metadata
# =============================================================================

@dataclass
class MsMetadata:
    """Spectral setup and phase centre of one entry, as read from its MS.

    Attributes
    ----------
    spws : list of int
        SPWs used: those of the input list, or every SPW of the MS.
    ddids : list of int
        DATA_DESC_ID of each of them.
    freq, width : list of ndarray
        CHAN_FREQ and CHAN_WIDTH of each SPW, observed frame [Hz].
    phase_dir : (float, float)
        RA and Dec of the phase centre of the field [rad].
    """
    spws: List[int]
    ddids: List[int]
    freq: List[np.ndarray]
    width: List[np.ndarray]
    phase_dir: tuple


def read_ms_metadata(ms, field, spws):
    """SPWs, channel frequencies and phase centre of one input MS.

    Called by the main process (from a thread pool), with dask-ms.

    Parameters
    ----------
    ms : str
        Path of the MS.
    field : int or None
        FIELD_ID of the entry.  ``None`` takes the phase centre of field 0.
    spws : list of int or None
        SPWs of the entry.  ``None`` takes every SPW of the MS.

    Returns
    -------
    MsMetadata
    """
    import dask
    from daskms import xds_from_table
    from .common import casacore_lock

    with casacore_lock:
        dd = xds_from_table(f"{ms}::DATA_DESCRIPTION")[0]
        spw_ids = np.asarray(dd.SPECTRAL_WINDOW_ID.data.compute(scheduler="synchronous"))
        spw2dd = {int(s): row for row, s in enumerate(spw_ids)}

        spw_rows = xds_from_table(f"{ms}::SPECTRAL_WINDOW", group_cols="__row__")
        n_spw = len(spw_rows)
        spws_used = list(range(n_spw)) if spws is None else spws
        lazy = []
        for sp in spws_used:
            if sp >= n_spw:
                raise ValueError(f"{os.path.basename(ms)}: SPW {sp} does not exist "
                                 f"(available: 0-{n_spw-1})")
            lazy += [spw_rows[sp].CHAN_FREQ.data, spw_rows[sp].CHAN_WIDTH.data]
        got = dask.compute(*lazy, scheduler="synchronous") if lazy else ()

        field_rows = xds_from_table(f"{ms}::FIELD", group_cols="__row__")
        fid = field if field is not None else 0
        if fid >= len(field_rows):
            raise ValueError(f"{os.path.basename(ms)}: FIELD {fid} does not exist "
                             f"(available: 0-{len(field_rows)-1})")
        phase_dir = np.asarray(field_rows[fid].PHASE_DIR.data.compute(scheduler="synchronous"),
                               dtype=np.float64).ravel()
    return MsMetadata(
        spws=spws_used,
        ddids=[spw2dd.get(s, s) for s in spws_used],
        freq=[np.asarray(got[2*k], dtype=np.float64).ravel() for k in range(len(spws_used))],
        width=[np.asarray(got[2*k+1], dtype=np.float64).ravel() for k in range(len(spws_used))],
        phase_dir=(float(phase_dir[0]), float(phase_dir[1])),
    )


# =============================================================================
# What to read, and what comes back
# =============================================================================

@dataclass
class EntryJob:
    """What to read for one line of the input list.

    Attributes
    ----------
    index : int
        Line of the input list (0-based); it becomes the DATA_DESC_ID of the
        entry in the output MS.
    n_entries : int
        Lines in the input list (for the ``[i/N]`` tag of the messages).
    ms : str
        Path of the MS.
    z : float
        Redshift of the source.
    ra, dec : str
        Position of the source, as written in the input list.
    field : int or None
        FIELD_ID to read; ``None`` reads every field.
    spws : list of int or None
        SPWs as written in the input list; ``None`` means every SPW.
    meta : MsMetadata
        Spectral setup and phase centre of the MS.
    freq_new : ndarray
        Centres of the output channels of this entry [Hz].
    df_new : float
        Output channel width [Hz].
    chunk_rows : int
        Rows per chunk asked by the user (lowered for very wide spectra).
    ant_offset : int
        Index of the first antenna of this MS in the merged ANTENNA table.
    data_column : str
        Visibility column to read (``'auto'``, ``'DATA'``, ...).
    frame : EntryFrame or None
        The frame correction (:class:`vista.frames.EntryFrame`): input and
        chosen frame, Doppler factor k = f_frame / f_input at the knots;
        None when no correction is needed.
    """
    index: int
    n_entries: int
    ms: str
    z: float
    ra: str
    dec: str
    field: Optional[int]
    spws: Optional[List[int]]
    meta: MsMetadata
    freq_new: np.ndarray
    df_new: float
    chunk_rows: int
    ant_offset: int
    data_column: str = "auto"
    frame: Optional[object] = None

    @property
    def doppler(self):
        """(k_min, k_mid, k_max) of the frame correction, or None: k_mid sets
        the grid and the weight factor R, k_min and k_max the channels read."""
        return None if self.frame is None else self.frame.k_range

    @property
    def name(self):
        """File name of the MS, for the messages."""
        return os.path.basename(self.ms.rstrip("/"))

    @property
    def tag(self):
        """``'[i/N]'``, the position of the entry in the input list."""
        return f"[{self.index + 1:>{len(str(self.n_entries))}}/{self.n_entries}]"


@dataclass
class EntryInfo:
    """What the reader found out about one entry, sent before its chunks.

    Attributes
    ----------
    index : int
        Line of the input list.
    ra1, dec1 : float
        Position of the source [rad], the RA brought near the phase centre.
    freq, width : ndarray
        Rest-frame centres and widths of the merged input channels [Hz].
    weight_scale, sigma_scale : float
        R = df_new / df_old (rest frame) and 1 / sqrt(R): WEIGHT is
        multiplied by R and SIGMA by 1 / sqrt(R), as CASA mstransform does.
    nrows : int
        Rows used.
    rows_all : int or None
        Rows of the reference SPW before the field / target-scan selection
        (None when there is no selection).
    filters : list of str
        The selections applied, e.g. ``['field 4', 'target scans']``.
    data_column : str
        Visibility column read.
    n_spw, nchan, ncorr : int
        Merged SPWs, their input channels, correlations.
    n_covered : int
        Output channels that receive data from this entry.
    offset_arcsec : float
        Distance between the phase centre and the source.
    pb_fwhm_arcsec : float or None
        Primary beam FWHM at the observed frequency (None if unknown).
    chunk_rows : int
        Rows per chunk actually used.
    warnings : list of str
        Problems found while opening the entry, for the log.
    """
    index: int
    ra1: float
    dec1: float
    freq: np.ndarray
    width: np.ndarray
    weight_scale: float
    sigma_scale: float
    nrows: int
    rows_all: Optional[int]
    filters: List[str]
    data_column: str
    n_spw: int
    nchan: int
    ncorr: int
    n_covered: int
    offset_arcsec: float
    pb_fwhm_arcsec: Optional[float]
    chunk_rows: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class Chunk:
    """A block of rows of one entry, as read (before the kernel).

    Attributes
    ----------
    vis, flag : ndarray
        Visibilities (complex64) and flags, shape (rows, channels, corr),
        merged SPWs side by side.  From a reader process they arrive as
        ``(shared-memory name, shape, dtype)`` instead (see vista.readers).
    uvw, ant1, ant2 : ndarray
        UVW and antennas of the rows (reference SPW).
    row_columns : dict of str -> ndarray
        The other row columns of the reference SPW (TIME, WEIGHT, ...).
    last : bool
        True for the last chunk of the entry.
    row0 : int
        Index of the first row of the chunk among the rows of the entry
        (where the chunk goes in a Zarr store).
    segments : list of (int, int, float) or None
        With a frame correction: the runs of rows ``(first, last + 1, k)``
        that share the Doppler factor k (see vista.frames); None otherwise.
    """
    vis: object
    flag: object
    uvw: np.ndarray
    ant1: np.ndarray
    ant2: np.ndarray
    row_columns: dict
    last: bool
    row0: int = 0
    segments: Optional[list] = None


# =============================================================================
# Helpers of the reader
# =============================================================================

def _select_rows(ms, field):
    """TaQL selection of the rows to stack.

    Returns
    -------
    (str, list of str)
        The condition (the requested field and, when the MS has a STATE table
        with ``OBSERVE_TARGET`` intents, only the target scans; an empty
        string keeps every row) and a description of it, for the log.
    """
    from casacore.tables import table

    conditions, filters = [], []
    if field is not None:
        conditions.append(f"FIELD_ID=={int(field)}")
        filters.append(f"field {int(field)}")
    try:
        with table(f"{ms}::STATE", readonly=True, ack=False) as st:
            if st.nrows() > 0:
                modes = st.getcol("OBS_MODE")
                ids = [i for i, m in enumerate(modes) if "OBSERVE_TARGET" in str(m)]
                if ids:
                    conditions.append(f"STATE_ID IN [{','.join(map(str, ids))}]")
                    filters.append("target scans")
    except Exception:
        pass
    return " AND ".join(conditions), filters


def _pick_data_column(colnames, wanted, warnings):
    """Visibility column to read.

    ``'auto'`` prefers CORRECTED_DATA, where a calibrated ALMA MS keeps the
    calibrated visibilities, then DATA.  A missing named column falls back to
    DATA with a warning.  Raises RuntimeError when there is no usable column.
    """
    wanted = wanted or "auto"
    if wanted != "auto":
        if wanted in colnames:
            return wanted
        column = "DATA" if "DATA" in colnames else None
        warnings.append(f"no {wanted} column, {column or 'no DATA either'} used instead")
    elif "CORRECTED_DATA" in colnames:
        column = "CORRECTED_DATA"
    elif "DATA" in colnames:
        column = "DATA"
    else:
        column = None
    if column is None:
        raise RuntimeError("neither DATA nor CORRECTED_DATA in the MS")
    return column


def match_spw_rows(keys, time_tol=TIME_TOL_S):
    """Which SPWs of one MS have exactly the same rows.

    Parameters
    ----------
    keys : list of (ndarray, ndarray, ndarray)
        TIME, ANTENNA1 and ANTENNA2 of each SPW.
    time_tol : float, default 1e-3
        Tolerance on TIME [s].

    Returns
    -------
    (int, list of bool)
        Index of the reference, the SPW with the most rows, and one flag per
        SPW: True when it has the same baselines at the same times, in the
        same order, as the reference.
    """
    ref = int(np.argmax([len(k[0]) for k in keys]))
    t0, a0, b0 = keys[ref]
    same = [i == ref or (len(t) == len(t0)
                         and np.array_equal(a, a0) and np.array_equal(b, b0)
                         and np.allclose(t, t0, rtol=0.0, atol=time_tol))
            for i, (t, a, b) in enumerate(keys)]
    return ref, same


def merge_spws(spws, warnings):
    """Keep the SPWs that can be merged row by row with the reference SPW.

    The channels of the SPWs of one entry are put side by side, row by row,
    with the UVW, TIME and WEIGHT of a single reference SPW.  That is correct
    only if every SPW has the same rows.  The others are left out with a
    warning; listing each SPW on its own line of the input file keeps them.

    Parameters
    ----------
    spws : list of dict
        The usable SPWs, in frequency order (``table``, ``id``, ``width``, ...).
    warnings : list of str
        Where the warnings are appended.

    Returns
    -------
    (list of dict, int)
        The SPWs kept, in the same order, and the position of the reference
        among them.  With a single SPW nothing is read and ``(spws, 0)`` is
        returned.
    """
    if len(spws) == 1:
        return spws, 0
    keys = [(s["table"].getcol("TIME"), s["table"].getcol("ANTENNA1"),
             s["table"].getcol("ANTENNA2")) for s in spws]
    mean_w = [float(np.mean(s["table"].getcol("WEIGHT"))) for s in spws]
    ref, same = match_spw_rows(keys)

    if not all(same):
        left_out = [s["id"] for s, ok in zip(spws, same) if not ok]
        warnings.append(
            f"SPW(s) {left_out} left out: their rows (baselines/times) differ from "
            f"SPW {spws[ref]['id']} (rows per SPW: {[len(k[0]) for k in keys]}).  To "
            f"keep them, put each SPW on its own line of the input list (same MS, "
            f"RA/Dec, z and FIELD_ID)")
    keep = [i for i, ok in enumerate(same) if ok]

    # merged SPWs share one WEIGHT per row and one width factor R
    if len(keep) > 1:
        ids = [spws[i]["id"] for i in keep]
        widths = [float(np.median(spws[i]["width"])) for i in keep]
        if max(widths) > min(widths) * (1.0 + WIDTH_TOL):
            warnings.append(f"merged SPWs {ids} have different channel widths: their "
                            f"weights are only approximate.  Put each SPW on its own "
                            f"line of the input list for exact weights")
        w = [mean_w[i] for i in keep]
        if min(w) > 0 and max(w) > WEIGHT_RATIO_WARN * min(w):
            warnings.append(f"merged SPWs {ids} have mean weights differing by "
                            f"{max(w) / min(w):.2f}x, but share the weights of SPW "
                            f"{spws[ref]['id']}.  Put each SPW on its own line of the "
                            f"input list for exact weights")
    return [spws[i] for i in keep], keep.index(ref)


def _primary_beam_fwhm(ms, nu_obs):
    """Primary beam FWHM [arcsec] at *nu_obs*, ``1.13 lambda / D``; None if unknown."""
    from casacore.tables import table
    try:
        with table(f"{ms}::ANTENNA", readonly=True, ack=False) as at:
            dish = at.getcol("DISH_DIAMETER")
        dish = float(np.median(dish[dish > 0])) if np.any(dish > 0) else 0.0
        if dish > 0:
            return 1.13 * (C_LIGHT_MS / nu_obs) / dish * RAD2ARCSEC
    except Exception:
        pass
    return None


# =============================================================================
# The reader
# =============================================================================

def read_entry_chunks(job, alloc):
    """Read one entry with python-casacore.

    Yields ``(kind, index, obj, seconds)``: first ``("info", index, EntryInfo,
    s)``, then ``("chunk", index, Chunk, s)`` for every block of rows, where
    ``s`` is the time spent reading since the previous message.

    The SPWs of the entry that overlap its output window are taken in
    rest-frame frequency order and, if they have the same rows, merged into
    one spectrum per row (see :func:`merge_spws`).  Only the rows of the
    requested field and of the target scans, and only the channels inside
    the window plus one channel of margin, are read.

    Parameters
    ----------
    job : EntryJob
        The entry.
    alloc : callable
        ``alloc(shape, dtype)`` returns the array VIS / FLAG are read into
        (shared memory in a reader process, plain NumPy in a thread).

    Raises
    ------
    RuntimeError
        When the entry gives no rows or no channels in its window (the
        entry is then skipped).
    """
    from casacore.tables import table

    t0 = time.perf_counter()
    z, meta, df_new, freq_new = job.z, job.meta, job.df_new, job.freq_new
    ra0, dec0 = meta.phase_dir
    ra1, dec1 = parse_ra_dec(job.ra, job.dec)
    ra1 = wrap_dra(ra1, ra0)
    warnings = []

    # weight factor R = df_new / df_old (as CASA mstransform); recomputed
    # below if the reference SPW is not the first one requested
    df_old_first = float(abs(meta.width[0][0])) * (1.0 + z)
    if job.doppler is None:
        weight_scale = df_new / df_old_first
    else:                       # channel widths in the chosen frame
        k_min, k_mid, k_max = job.doppler
        weight_scale = df_new / (df_old_first * k_mid)
    sigma_scale = 1.0 / np.sqrt(weight_scale) if weight_scale > 0.0 else 1.0

    tab = table(job.ms, readonly=True, ack=False)
    opened = []
    try:
        colnames = set(tab.colnames())
        where, filters = _select_rows(job.ms, job.field)
        column = _pick_data_column(colnames, job.data_column, warnings)

        # the SPWs that overlap the window, in rest-frame frequency order
        centre = lambda p: float(meta.freq[p][len(meta.freq[p]) // 2]) * (1.0 + z)
        win_lo, win_hi = freq_new[0] - df_new, freq_new[-1] + df_new
        spws = []
        for pos in sorted(range(len(meta.ddids)), key=centre):
            ddid = int(meta.ddids[pos])
            sub = tab.query(f"DATA_DESC_ID=={ddid}" + (f" AND ({where})" if where else ""))
            opened.append(sub)
            if sub.nrows() == 0:
                continue
            f_rf = meta.freq[pos] * (1.0 + z)
            if job.doppler is None:
                useful = np.where((f_rf >= win_lo) & (f_rf <= win_hi))[0]
            else:               # a channel is kept if any row brings it in the window
                useful = np.where((f_rf * k_max >= win_lo) & (f_rf * k_min <= win_hi))[0]
            if len(useful) == 0:
                continue
            lo, hi = int(useful[0]), int(useful[-1]) + 1
            half = 0.5 * float(abs(meta.width[pos][0])) * (1.0 + z)
            spws.append({
                "table": sub,
                "ddid":  ddid,
                "lo":    lo,
                "hi":    hi,
                "freq":  f_rf[lo:hi],
                "width": np.abs(np.asarray(meta.width[pos], dtype=np.float64)[lo:hi]) * (1.0 + z),
                "edges": (float(f_rf[lo:hi].min()) - half, float(f_rf[lo:hi].max()) + half),
                "id":    job.spws[pos] if job.spws is not None else pos,
            })
        if not spws:
            raise RuntimeError(
                f"no rows or channels in the output window (field={job.field}, "
                f"SPWs {list(meta.ddids)}): check the field, the SPWs and the "
                f"frequency coverage")

        spws, ref_pos = merge_spws(spws, warnings)
        ref = spws[ref_pos]["table"]
        nrows = ref.nrows()
        rows_all = None
        if where:
            with tab.query(f"DATA_DESC_ID=={spws[ref_pos]['ddid']}") as t_all:
                rows_all = t_all.nrows()
        covered = np.zeros(len(freq_new), dtype=bool)
        for s in spws:
            edges = s["edges"] if job.doppler is None else \
                (s["edges"][0] * k_mid, s["edges"][1] * k_mid)
            covered |= covered_channels(freq_new, df_new, *edges)

        # R must use the channel width of the SPW whose weights are kept
        w_ref = float(spws[ref_pos]["width"][0])
        if w_ref > 0 and abs(w_ref - df_old_first) > 1e-6 * w_ref:
            weight_scale = df_new / w_ref if job.doppler is None else df_new / (w_ref * k_mid)
            sigma_scale = 1.0 / np.sqrt(weight_scale)

        freq = np.concatenate([s["freq"] for s in spws])
        width = np.concatenate([s["width"] for s in spws])
        nchan = int(sum(s["hi"] - s["lo"] for s in spws))
        ncorr = int(np.asarray(ref.getcell(column, 0)).shape[-1])
        chunk_rows = min(job.chunk_rows,
                         max(MIN_CHUNK_ROWS, TARGET_CHUNK_BYTES // max(1, nchan * 2 * 8)))

        # row columns of the reference SPW: small, read in one go
        uvw = ref.getcol("UVW").astype(np.float64)
        ant1 = ref.getcol("ANTENNA1")
        ant2 = ref.getcol("ANTENNA2")
        row_columns = {}
        for c in ROW_COLUMNS:
            if c in colnames:
                try:
                    row_columns[c] = ref.getcol(c)
                except Exception:
                    pass

        # Doppler factor of every row, from its time (frame correction),
        # interpolated between the knots computed by the main process
        k_rows = None
        if job.frame is not None:
            times = row_columns["TIME"] if "TIME" in row_columns else ref.getcol("TIME")
            k_rows = k_per_row(times, job.frame)

        # distance of the source from the phase centre, and the primary beam
        # at the observed frequency (sanity check of RA/Dec)
        offset = angular_sep(ra0, dec0, ra1, dec1) * RAD2ARCSEC
        nu_obs = float(np.median(freq)) / (1.0 + z)
        fwhm = _primary_beam_fwhm(job.ms, nu_obs)
        if fwhm is not None and offset > 0.5 * fwhm:
            warnings.append(f"the stacking position is {offset:.1f}\" from the phase "
                            f"centre, beyond half the primary beam (FWHM ~ {fwhm:.0f}\" "
                            f"at {nu_obs / 1e9:.1f} GHz): check RA/Dec in the input list")

        info = EntryInfo(
            index=job.index, ra1=ra1, dec1=dec1, freq=freq, width=width,
            weight_scale=weight_scale, sigma_scale=sigma_scale,
            nrows=nrows, rows_all=rows_all, filters=filters, data_column=column,
            n_spw=len(spws), nchan=nchan, ncorr=ncorr, n_covered=int(covered.sum()),
            offset_arcsec=offset, pb_fwhm_arcsec=fwhm, chunk_rows=chunk_rows,
            warnings=warnings)
        yield ("info", job.index, info, time.perf_counter() - t0)

        for row0 in range(0, nrows, chunk_rows):
            t0 = time.perf_counter()
            n = min(chunk_rows, nrows - row0)
            vis = alloc((n, nchan, ncorr), np.complex64)
            flag = alloc((n, nchan, ncorr), np.bool_)
            _read_block(spws, column, row0, n, ncorr, vis, flag)
            rows = slice(row0, row0 + n)
            chunk = Chunk(vis=vis, flag=flag, uvw=uvw[rows], ant1=ant1[rows], ant2=ant2[rows],
                          row_columns={c: a[rows] for c, a in row_columns.items()},
                          last=row0 + n >= nrows, row0=row0,
                          segments=None if k_rows is None else row_segments(k_rows[rows]))
            yield ("chunk", job.index, chunk, time.perf_counter() - t0)
    finally:
        for sub in opened:
            try:
                sub.close()
            except Exception:
                pass
        tab.close()


def _read_block(spws, column, row0, n, ncorr, vis, flag):
    """Read *n* rows from *row0* of every merged SPW into *vis* and *flag*.

    The channels of the SPWs go side by side.  With a single SPW the data go
    straight into the output arrays, without a copy.
    """
    c0 = 0
    for s in spws:
        w = s["hi"] - s["lo"]
        blc, trc = [s["lo"], 0], [s["hi"] - 1, ncorr - 1]
        if len(spws) == 1:
            tv, tf = vis, flag
        else:
            tv = np.empty((n, w, ncorr), np.complex64)
            tf = np.empty((n, w, ncorr), np.bool_)
        try:
            s["table"].getcolslicenp(column, tv, blc, trc, startrow=row0, nrow=n)
        except Exception:
            tv[...] = s["table"].getcolslice(column, blc, trc, startrow=row0, nrow=n)
        s["table"].getcolslicenp("FLAG", tf, blc, trc, startrow=row0, nrow=n)
        if len(spws) > 1:
            vis[:, c0:c0 + w, :] = tv
            flag[:, c0:c0 + w, :] = tf
        c0 += w
