"""
vista pipeline
pipeline.py — HPC-optimised visibility-domain stacking pipeline
==================================================================

Core processing pipeline for ViSta (Visibility Stacking tool).

The pipeline takes a list of interferometric Measurement Sets (MSs) at
different redshifts and combines them in the visibility (uv) plane by:

  1. Rest-framing each dataset: baseline vectors (UVW) are scaled by
     1/(1+z) and each spectral axis is shifted to the common rest-frame grid.
  2. Centering (phase shift): each dataset is phase-rotated to the same
     sky position (the target stacking position).
  3. Spectral rebinning: visibilities are projected onto a common output
     frequency grid using an overlap-weighted scheme identical to that of
     CASA's mstransform.

How the work is organised
-------------------------
Setup (once): the spectral metadata of every MS are read, the common output
grid is computed and the output subtables are prepared.

Main loop, three stages connected by bounded queues::

    readers (N processes) ──► chunks ──► compute ──► results ──► writer (1 thread)

  * Readers: each reader process takes the next MS and reads it chunk by
    chunk with python-casacore, only the channels that fall in the output
    window.  The visibilities and flags go through shared memory, without
    copies.  Separate processes are needed because python-casacore holds the
    Python GIL while reading: threads would read one at a time.
  * Compute (main thread): the C++ kernel ``ms_ops`` (rest-framing, phase
    shift, rebinning).  On the CPU it runs with OpenMP, one chunk per call; on
    the GPU it takes all the chunks already waiting (up to ``gpu_batch``) in
    one launch, and writes into reused, page-locked output buffers.  The
    kernel releases the GIL, so reading and writing never stop.
  * Writer (one thread): appends every processed chunk to the output MS
    with python-casacore.  A table can be written by one thread only, so the
    writer usually sets the pace of the whole run (the run is I/O bound).

The output holds one spectral window per input MS.  Its rows are in the
order the chunks are processed, not in the input order; the content does not
depend on it.

The C++/OpenMP (and optionally CUDA) kernel is the extension module
``ms_ops``, built from src/ms_ops.cpp and src/ms_ops_cuda.cu.
"""

import os

# ── Threading environment (set before any NumPy / OpenBLAS / MKL import) ──
# A value already set by the user (or by SLURM) is kept.  OpenMP reads it once,
# when ms_ops is imported; run(n_threads=...) changes it afterwards.
_n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 4)
os.environ.setdefault("OMP_NUM_THREADS",        str(_n_threads))
os.environ.setdefault("OPENBLAS_NUM_THREADS",   os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("MKL_NUM_THREADS",        os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("NUMEXPR_NUM_THREADS",    os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", os.environ["OMP_NUM_THREADS"])
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES",    "cores")

import contextlib
import gc
import math
import shutil
import sys
import threading
import time
from collections import Counter
from queue import Queue, Empty, Full

import numpy as np
import dask
import dask.array as da
from daskms import xds_from_table, xds_to_table, Dataset

from . import __version__, _say, _detail, _warn, _reset_clock, set_verbosity

try:
    from . import ms_ops     # installed into vista/ by 'make install'
except ImportError:          # or found on PYTHONPATH
    import ms_ops


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_C_LIGHT_KMS = 299792.458   # speed of light in km/s
_C_LIGHT_MS  = 299792458.0  # speed of light in m/s
_RAD2ARCSEC  = 180.0 / math.pi * 3600.0


def _log(msg):
    """Progress line of the stacking (stdout, through the 'vista' logger)."""
    _say(msg)


# ---------------------------------------------------------------------------
# Thread safety: Casacore Table Data System lock
# ---------------------------------------------------------------------------
# The metadata and the subtables are read and written with dask-ms from
# thread pools; every access to the CTDS goes through this lock.

_casacore_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def _wrap_dra(ra_new, ra_old):
    """Wrap a right ascension difference to [-pi, pi] around *ra_old*.

    Avoids discontinuities at the 0/2*pi boundary when computing phase
    shifts between two pointing directions.
    """
    dra = (ra_new - ra_old + math.pi) % (2.0 * math.pi) - math.pi
    return ra_old + dra


def _parse_ra_dec(ra_str, dec_str):
    """Parse RA / Dec strings (sexagesimal or decimal) and return radians.

    Accepts any format understood by ``astropy.coordinates.SkyCoord``, e.g.::

        ra_str  : '12:34:56.7'   or  '12h34m56.7s'
        dec_str : '+12:34:56.7'  or  '12.582d'

    The Dec string may also use '.' as the d/m/s separator (common in some
    ALMA pipeline products), e.g. ``'-23.01.45.6'`` → ``'-23:01:45.6'``.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    if ":" not in dec_str:
        parts = dec_str.split(".")
        if len(parts) >= 3:
            dec_str = parts[0] + ":" + parts[1] + ":" + ".".join(parts[2:])
    c = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    return float(c.ra.rad), float(c.dec.rad)

# ---------------------------------------------------------------------------
# Array utilities
# ---------------------------------------------------------------------------


def _rename_duplicates(names):
    """Append an index suffix to duplicate strings in a list.

    Used to ensure antenna names are unique when concatenating ANTENNA tables
    from multiple MSs that may share antenna names (e.g. ``'DA41'``).

    Example::

        ['DA41', 'DA41', 'DV01']  →  ['DA41_0', 'DA41_1', 'DV01']
    """
    count = Counter(names)
    seen  = Counter()
    out   = []
    for n in names:
        if count[n] > 1:
            out.append(f"{n}_{seen[n]}")
            seen[n] += 1
        else:
            out.append(n)
    return out


def _to_dask(arr, chunks=None):
    """Wrap a NumPy array in a Dask array with a unique graph name.

    Using a unique name prevents Dask from accidentally sharing computation
    graphs between arrays that happen to have the same shape / dtype.
    """
    import uuid
    name = "array-" + uuid.uuid4().hex[:8]
    if chunks is None:
        return da.from_array(arr, chunks=arr.shape, name=name)
    return da.from_array(arr, chunks=chunks, name=name)


def _fmt_size(nbytes):
    """Format a byte count as a human-readable string (B / KB / MB / GB / TB)."""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"

# ---------------------------------------------------------------------------
# Output subtable builders
# ---------------------------------------------------------------------------
# Each function assembles one or more xarray Datasets that will be written
# to the corresponding subtable of the output Measurement Set via dask-ms.
# All arrays are wrapped in Dask arrays so that the write is lazy and can be
# overlapped with other I/O.
#
# Thread safety: every call that touches CTDS (xds_from_table, dask.compute
# materialising columns read from disk) is wrapped in _casacore_lock.


def _build_antenna_table(ms_list, z_list):
    """Build the ANTENNA subtable for the output MS.

    Antenna positions from all input MSs are concatenated into a single table.
    Each position (and dish diameter) is rescaled by ``1/(1+z)`` to map it to
    the rest-frame spatial frequency plane, exactly as done for the UVW
    coordinates inside the C++ kernel.

    Each input MS is assigned a contiguous block of antenna IDs; the offsets
    are returned so that ANTENNA1 / ANTENNA2 columns can be remapped correctly.

    The reads are parallelised with a ``ThreadPoolExecutor``; all CTDS access
    is serialised via ``_casacore_lock``.

    Returns
    -------
    ds : Dataset
        Combined ANTENNA table as a dask-ms Dataset.
    ant_offsets : list of int
        Starting antenna index for each input MS.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _load_one_ant(args):
        ms, z = args
        inv  = 1.0 / (1.0 + z)
        with _casacore_lock:
            rows = xds_from_table(f"{ms}::ANTENNA", group_cols=[])
            ds   = rows[0] if rows else None
            if ds is None:
                return None
            def get_col(ds, name):
                return ds.data_vars[name].data if name in ds.data_vars else None
            computed = dask.compute(
                get_col(ds,"NAME"), get_col(ds,"POSITION"),
                get_col(ds,"DISH_DIAMETER"), get_col(ds,"MOUNT"),
                get_col(ds,"STATION"), scheduler="synchronous")
        names_arr, pos_arr_raw, diam_arr_raw, mounts_arr, stations_arr = computed
        nant = ds.sizes.get("row", 0)
        names = [str(v) for v in np.asarray(names_arr).ravel()]                 if names_arr is not None else [""]*nant
        if pos_arr_raw is not None:
            pos = np.asarray(pos_arr_raw, dtype=np.float64)
            if pos.shape[0]==3 and pos.ndim==2: pos=pos.T
            positions = list(pos * inv)
        else:
            positions = list(np.zeros((nant,3)))
        diameters = list(np.asarray(diam_arr_raw,dtype=np.float64).ravel()*inv)                     if diam_arr_raw is not None else [12.0*inv]*nant
        mounts   = [str(v) for v in np.asarray(mounts_arr).ravel()]                    if mounts_arr is not None else ["ALT-AZ"]*nant
        stations = [str(v) for v in np.asarray(stations_arr).ravel()]                    if stations_arr is not None else [""]*nant
        return (nant, names, positions, diameters, mounts, stations)

    with ThreadPoolExecutor(max_workers=min(8, len(ms_list))) as pool:
        results = list(pool.map(_load_one_ant, zip(ms_list, z_list)))

    ant_offsets = []
    offset = 0
    all_names, all_positions, all_diameters = [], [], []
    all_mounts, all_stations = [], []
    for r in results:
        ant_offsets.append(offset)
        if r is None:
            continue
        nant, names, positions, diameters, mounts, stations = r
        offset += nant
        all_names.extend(names)
        all_positions.extend(positions)
        all_diameters.extend(diameters)
        all_mounts.extend(mounts)
        all_stations.extend(stations)

    # Ensure antenna names are unique across the combined table
    all_names = _rename_duplicates(all_names)
    nant_tot  = len(all_names)
    pos_arr   = np.array(all_positions, dtype=np.float64).reshape(nant_tot, 3)
    diam_arr  = np.array(all_diameters, dtype=np.float64)
    ds = Dataset({
        "NAME":          (("row",),       _to_dask(np.array(all_names,    dtype=object))),
        "POSITION":      (("row","xyz"),  _to_dask(pos_arr)),
        "DISH_DIAMETER": (("row",),       _to_dask(diam_arr)),
        "MOUNT":         (("row",),       _to_dask(np.array(all_mounts,   dtype=object))),
        "STATION":       (("row",),       _to_dask(np.array(all_stations, dtype=object))),
        "FLAG_ROW":      (("row",),       _to_dask(np.zeros(nant_tot, dtype=bool))),
        "OFFSET":        (("row","xyz"),  _to_dask(np.zeros((nant_tot,3), dtype=np.float64))),
        "TYPE":          (("row",),       _to_dask(np.array(["GROUND-BASED"]*nant_tot, dtype=object))),
    })
    return ds, ant_offsets


def _build_spw_table(freq_list, df_new, ms_list):
    """Build the SPECTRAL_WINDOW subtable for the output MS.

    One spectral window is created per input MS, each with its own output
    frequency grid (which may have a different start but always the same
    channel width *df_new*).  The windows are kept separate rather than
    merged so that each input MS can be identified by its DATA_DESC_ID in
    the output.

    Parameters
    ----------
    freq_list : list of ndarray
        Per-MS output channel centre frequencies (Hz), one array per MS.
    df_new : float
        Common output channel width (Hz).
    ms_list : list of str
        Input MS paths (used only to generate SPW names).
    """
    num_ms = len(ms_list)
    assert len(freq_list) == num_ms

    nchan_per_spw = [len(f) for f in freq_list]
    nchan_max     = max(nchan_per_spw)

    # Build 2-D arrays padded to nchan_max; NUM_CHAN records the true length
    chan_freq_2d = np.zeros((num_ms, nchan_max), dtype=np.float64)
    chan_width_2d = np.zeros((num_ms, nchan_max), dtype=np.float64)
    for i, f in enumerate(freq_list):
        nc = len(f)
        chan_freq_2d[i, :nc]  = f
        chan_width_2d[i, :nc] = df_new

    ref_freqs = np.array([f[len(f)//2] for f in freq_list], dtype=np.float64)
    tot_bws   = np.array([df_new * len(f) for f in freq_list], dtype=np.float64)
    names     = [f"spw_{i}_{os.path.basename(ms)}" for i, ms in enumerate(ms_list)]

    ds = Dataset({
        "CHAN_FREQ":       (("row","chan"), _to_dask(chan_freq_2d)),
        "CHAN_WIDTH":      (("row","chan"), _to_dask(chan_width_2d)),
        "EFFECTIVE_BW":   (("row","chan"), _to_dask(chan_width_2d)),
        "RESOLUTION":     (("row","chan"), _to_dask(chan_width_2d)),
        "REF_FREQUENCY":  (("row",),      _to_dask(ref_freqs)),
        "TOTAL_BANDWIDTH":(("row",),      _to_dask(tot_bws)),
        "NUM_CHAN":        (("row",),      _to_dask(np.array(nchan_per_spw, dtype=np.int32))),
        "NAME":           (("row",),      _to_dask(np.array(names, dtype=object))),
        "FLAG_ROW":       (("row",),      _to_dask(np.zeros(num_ms, dtype=bool))),
        "MEAS_FREQ_REF":  (("row",),      _to_dask(np.full(num_ms, 5, dtype=np.int32))),
    })
    return ds


def _build_field_table(first_ms):
    """Build a minimal FIELD subtable for the output MS.

    The output FIELD table contains a single row with the phase centre set to
    (0, 0) in J2000 radians.  The field name is inherited from the first input
    MS.
    """
    with _casacore_lock:
        rows = xds_from_table(f"{first_ms}::FIELD", group_cols="__row__")
        name = "stacked"
        if rows:
            try: name = str(np.asarray(rows[0].NAME.data).item())
            except Exception: pass
    zero_dir = np.zeros((1,1,2), dtype=np.float64)
    ds = Dataset({
        "NAME":          (("row",),             _to_dask(np.array([name], dtype=object))),
        "CODE":          (("row",),             _to_dask(np.array([""],   dtype=object))),
        "TIME":          (("row",),             _to_dask(np.zeros(1, dtype=np.float64))),
        "NUM_POLY":      (("row",),             _to_dask(np.zeros(1, dtype=np.int32))),
        "PHASE_DIR":     (("row","d0","ra_dec"),_to_dask(zero_dir)),
        "DELAY_DIR":     (("row","d0","ra_dec"),_to_dask(zero_dir)),
        "REFERENCE_DIR": (("row","d0","ra_dec"),_to_dask(zero_dir)),
        "FLAG_ROW":      (("row",),             _to_dask(np.zeros(1, dtype=bool))),
        "SOURCE_ID":     (("row",),             _to_dask(np.zeros(1, dtype=np.int32))),
    })
    return ds


def _build_dd_table(first_ms, num_ms):
    """Build the DATA_DESCRIPTION subtable for the output MS.

    One row is created per input MS, each mapping to a distinct spectral
    window (SPW index == MS index).  The polarisation ID is inherited from the
    first input MS.
    """
    with _casacore_lock:
        rows = xds_from_table(f"{first_ms}::DATA_DESCRIPTION", group_cols="__row__")
        pol_id = 0
        if rows:
            try: pol_id = int(np.asarray(rows[0].POLARIZATION_ID.data).item())
            except Exception: pass
    ds = Dataset({
        "SPECTRAL_WINDOW_ID": (("row",), _to_dask(np.arange(num_ms, dtype=np.int32))),
        "POLARIZATION_ID":    (("row",), _to_dask(np.full(num_ms, pol_id, dtype=np.int32))),
        "FLAG_ROW":           (("row",), _to_dask(np.zeros(num_ms, dtype=bool))),
    })
    return ds


def _copy_subtable_first_row(ms_path, name):
    """Copy the first row of a subtable from an input MS.

    Used to propagate POLARIZATION and similar single-row subtables to the
    output MS without modification.

    Returns None if the subtable does not exist or is empty.
    """
    with _casacore_lock:
        try: rows = xds_from_table(f"{ms_path}::{name}", group_cols="__row__")
        except Exception: return None
        if not rows: return None
        r = rows[0]
        var_names = list(r.data_vars)
        computed  = dask.compute(*[r[v].data for v in var_names])
    data_vars = {}
    for var, arr in zip(var_names, computed):
        arr = np.asarray(arr)
        if arr.ndim >= 1 and r[var].dims[0] == "row":
            arr = arr[:1]
        data_vars[var] = (r[var].dims, _to_dask(arr))
    return Dataset(data_vars)


def _build_observation_table(ms_list):
    """Read OBSERVATION subtables from all input MSs and return them as a list.

    The subtables are read in parallel using a ``ThreadPoolExecutor`` since
    each read is small and dominated by filesystem latency.  All CTDS access
    is serialised via ``_casacore_lock``.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _load_obs(ms):
        with _casacore_lock:
            try:
                rows = xds_from_table(f"{ms}::OBSERVATION", group_cols="__row__")
                return rows if rows else []
            except Exception:
                return []

    with ThreadPoolExecutor(max_workers=min(8, len(ms_list))) as pool:
        all_rows_per_ms = list(pool.map(_load_obs, ms_list))

    flat_rows  = []
    lazy_all   = []
    row_slices = []
    for rows in all_rows_per_ms:
        for r in rows:
            vnames = list(r.data_vars)
            start  = len(lazy_all)
            lazy_all.extend(r[v].data for v in vnames)
            row_slices.append((start, len(lazy_all)))
            flat_rows.append((r, vnames))

    if not lazy_all:
        return []

    with _casacore_lock:
        computed = dask.compute(*lazy_all)

    all_ds = []
    for (r, vnames), (s, e) in zip(flat_rows, row_slices):
        data_vars = {}
        for v, arr in zip(vnames, computed[s:e]):
            data_vars[v] = (r[v].dims, _to_dask(np.asarray(arr)))
        all_ds.append(Dataset(data_vars))
    return all_ds


def _build_source_table(first_ms, central_freq_hz):
    """Build a minimal SOURCE subtable for the output MS.

    A single placeholder row is created with ``central_freq_hz`` as the
    rest frequency.  This is sufficient for imaging tools to recognise the
    dataset as a spectral-line MS.

    Parameters
    ----------
    first_ms : str
        Path to the first input MS (unused, kept for API symmetry).
    central_freq_hz : float
        Rest frequency written into the SOURCE table (Hz).
    """
    ds = Dataset({
        "SOURCE_ID":          (("row",),        _to_dask(np.zeros(1, dtype=np.int32))),
        "TIME":               (("row",),        _to_dask(np.zeros(1, dtype=np.float64))),
        "INTERVAL":           (("row",),        _to_dask(np.zeros(1, dtype=np.float64))),
        "SPECTRAL_WINDOW_ID": (("row",),        _to_dask(np.array([-1], dtype=np.int32))),
        "NUM_LINES":          (("row",),        _to_dask(np.ones(1,  dtype=np.int32))),
        "NAME":               (("row",),        _to_dask(np.array(["J0000+0000"], dtype=object))),
        "CALIBRATION_GROUP":  (("row",),        _to_dask(np.zeros(1, dtype=np.int32))),
        "CODE":               (("row",),        _to_dask(np.array([""], dtype=object))),
        "DIRECTION":          (("row","radec"), _to_dask(np.zeros((1,2), dtype=np.float64))),
        "PROPER_MOTION":      (("row","pm"),    _to_dask(np.zeros((1,2), dtype=np.float64))),
        "REST_FREQUENCY":     (("row","lines"), _to_dask(np.array([[central_freq_hz]], dtype=np.float64))),
        "SYSVEL":             (("row","lines"), _to_dask(np.zeros((1,1),  dtype=np.float64))),
    })
    return [ds]


# ===========================================================================
# Readers: one MS -> context + chunks
# ===========================================================================
# The same generator is used by the reader processes (arrays in shared memory)
# and by the reader threads of reader_mode="thread" (plain NumPy arrays).
# Messages to the main process: (kind, reader_id, ms_idx, obj, read_seconds)
#   kind = "ctx"   once per MS: what the kernel needs + what the log reports
#          "chunk" vis, flag, uvw, antennas and scalar columns of a chunk
#          "skip"  the MS gives no usable data (obj = the reason)
#          "error" unexpected exception in the reader (obj = traceback)
#          "done"  the reader has no more MSs
_SCALAR_COLS = ["TIME", "EXPOSURE", "INTERVAL", "WEIGHT", "SIGMA",
                "SCAN_NUMBER", "STATE_ID", "FLAG_ROW"]


def _angular_sep(ra0, dec0, ra1, dec1):
    """Angular distance between two directions (radians, haversine)."""
    h = (math.sin(0.5 * (dec1 - dec0)) ** 2
         + math.cos(dec0) * math.cos(dec1) * math.sin(0.5 * (ra1 - ra0)) ** 2)
    return 2.0 * math.asin(min(1.0, math.sqrt(h)))


def _iter_ms_chunks(job, alloc):
    """Read one MS: yield ("ctx", ms_idx, ctx, t) once, then ("chunk", ...).

    ``alloc(shape, dtype)`` returns the array VIS/FLAG are read into.
    ``t`` = seconds spent reading since the previous message.
    Raises RuntimeError when the MS gives no usable data (it is skipped).
    """
    from casacore.tables import table

    t0 = time.perf_counter()
    ms_idx = job["ms_idx"]
    ms = job["ms"]
    z = job["z"]
    freq_old_list = job["freq_old_list"]
    width_old_list = job["width_old_list"]
    ddids = job["ddids"]
    req_field = job["field"]
    freq_new_ms = job["freq_new"]
    df_new = job["df_new"]
    ra0_rad, dec0_rad = job["phase_dir"]
    ra1_rad, dec1_rad = _parse_ra_dec(job["ra"], job["dec"])
    ra1_rad = _wrap_dra(ra1_rad, ra0_rad)

    # weight scale R = df_new / df_old_rf (as CASA mstransform)
    df_old_rf_chan0 = float(abs(width_old_list[0][0])) * (1.0 + z)
    weight_scale = df_new / df_old_rf_chan0
    sigma_scale = 1.0 / np.sqrt(weight_scale) if weight_scale > 0.0 else 1.0

    tab = table(ms, readonly=True, ack=False)
    subs = []
    warnings = []
    try:
        colnames = set(tab.colnames())

        # rows: the requested field, target scans only (when the MS has a STATE table)
        conditions, filters = [], []
        if req_field is not None:
            conditions.append(f"FIELD_ID=={int(req_field)}")
            filters.append(f"field {int(req_field)}")
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
        where = " AND ".join(conditions)

        # visibility column: "auto" prefers CORRECTED_DATA (calibrated ALMA MS)
        want = job.get("data_column") or "auto"
        if want != "auto":
            if want in colnames:
                dcol = want
            else:
                dcol = "DATA" if "DATA" in colnames else None
                warnings.append(f"no {want} column, {dcol or 'no DATA either'} used instead")
        elif "CORRECTED_DATA" in colnames:
            dcol = "CORRECTED_DATA"
        elif "DATA" in colnames:
            dcol = "DATA"
        else:
            dcol = None
        if dcol is None:
            raise RuntimeError("neither DATA nor CORRECTED_DATA in the MS")

        def _spw_centre_rf(pos):
            f = freq_old_list[pos]
            return float(f[len(f) // 2]) * (1.0 + z)
        spw_order = sorted(range(len(ddids)), key=_spw_centre_rf)

        f_lo_win = freq_new_ms[0] - df_new
        f_hi_win = freq_new_ms[-1] + df_new
        pieces = []
        ref = None
        n_cov = set()
        rows_all = None
        for pos in spw_order:
            ddid = int(ddids[pos])
            q = f"DATA_DESC_ID=={ddid}" + (f" AND ({where})" if where else "")
            sub = tab.query(q)
            subs.append(sub)
            if sub.nrows() == 0:
                continue
            freq_old_rf = freq_old_list[pos] * (1.0 + z)
            useful = np.where((freq_old_rf >= f_lo_win) & (freq_old_rf <= f_hi_win))[0]
            if len(useful) == 0:
                continue
            ch_lo, ch_hi = int(useful[0]), int(useful[-1]) + 1
            fsl = freq_old_rf[ch_lo:ch_hi]
            half = 0.5 * float(abs(width_old_list[pos][0])) * (1.0 + z)
            in_lo, in_hi = float(fsl.min()) - half, float(fsl.max()) + half
            for j in range(len(freq_new_ms)):
                if min(freq_new_ms[j] + 0.5 * df_new, in_hi) > \
                   max(freq_new_ms[j] - 0.5 * df_new, in_lo):
                    n_cov.add(j)
            pieces.append((sub, dcol, ch_lo, ch_hi, fsl))
            if ref is None:
                ref = sub
                if where:
                    with tab.query(f"DATA_DESC_ID=={ddid}") as t_all:
                        rows_all = t_all.nrows()

        if not pieces or ref is None:
            raise RuntimeError(
                f"no rows or channels in the output window "
                f"(field={req_field}, SPWs {list(ddids)}): check the field, "
                f"the SPWs and the frequency coverage")

        nrows_list = [p[0].nrows() for p in pieces]
        nrows = nrows_list[0]
        if len(set(nrows_list)) > 1:
            keep = [p for p, n in zip(pieces, nrows_list) if n == nrows]
            warnings.append(f"{len(pieces) - len(keep)} SPW(s) dropped: their number of "
                            f"rows differs from the first one ({nrows_list})")
            pieces = keep

        freq_merged = np.concatenate([p[4] for p in pieces])
        nchan_tot = int(sum(p[3] - p[2] for p in pieces))
        ncorr = int(np.asarray(ref.getcell(pieces[0][1], 0)).shape[-1])
        adaptive = max(1000, (512 * 1024 * 1024) // max(1, nchan_tot * 2 * 8))
        eff_chunk = min(job["chunk_rows"], adaptive)

        uvw_all = ref.getcol("UVW").astype(np.float64)
        ant1_all = ref.getcol("ANTENNA1")
        ant2_all = ref.getcol("ANTENNA2")
        scal = {}
        for c in _SCALAR_COLS:
            if c in colnames:
                try:
                    scal[c] = ref.getcol(c)
                except Exception:
                    pass

        # distance of the stacking position from the phase centre, and the
        # primary beam at the observed frequency (sanity check of RA/Dec)
        offset = _angular_sep(ra0_rad, dec0_rad, ra1_rad, dec1_rad) * _RAD2ARCSEC
        nu_obs = float(np.median(freq_merged)) / (1.0 + z)
        fwhm = None
        try:
            with table(f"{ms}::ANTENNA", readonly=True, ack=False) as at:
                dish = at.getcol("DISH_DIAMETER")
            dish = float(np.median(dish[dish > 0])) if np.any(dish > 0) else 0.0
            if dish > 0:
                fwhm = 1.13 * (_C_LIGHT_MS / nu_obs) / dish * _RAD2ARCSEC
        except Exception:
            pass
        if fwhm is not None and offset > 0.5 * fwhm:
            warnings.append(f"the stacking position is {offset:.1f}\" from the phase "
                            f"centre, beyond half the primary beam (FWHM ~ {fwhm:.0f}\" "
                            f"at {nu_obs / 1e9:.1f} GHz): check RA/Dec in the input list")

        ctx = {
            "z": z, "ra0": ra0_rad, "dec0": dec0_rad, "ra1": ra1_rad, "dec1": dec1_rad,
            "freq_old": freq_merged, "freq_new": freq_new_ms,
            "ant_offset": job["ant_offset"],
            "weight_scale": weight_scale, "sigma_scale": sigma_scale,
            "rows": nrows, "rows_all": rows_all, "filters": filters,
            "data_column": dcol, "n_spw": len(pieces), "nchan_in": nchan_tot,
            "ncorr": ncorr, "coverage": len(n_cov), "nchan_out": len(freq_new_ms),
            "offset_arcsec": offset, "pb_fwhm_arcsec": fwhm, "chunk_rows": eff_chunk,
            "warnings": warnings,
        }
        t = time.perf_counter()
        yield ("ctx", ms_idx, ctx, t - t0)

        for row0 in range(0, nrows, eff_chunk):
            t0 = time.perf_counter()
            n = min(eff_chunk, nrows - row0)
            vis = alloc((n, nchan_tot, ncorr), np.complex64)
            flag = alloc((n, nchan_tot, ncorr), np.bool_)
            c0 = 0
            for (sub, col, lo, hi, _f) in pieces:
                w = hi - lo
                blc, trc = [lo, 0], [hi - 1, ncorr - 1]
                if len(pieces) == 1:
                    tv, tf = vis, flag
                else:
                    tv = np.empty((n, w, ncorr), np.complex64)
                    tf = np.empty((n, w, ncorr), np.bool_)
                try:
                    sub.getcolslicenp(col, tv, blc, trc, startrow=row0, nrow=n)
                except Exception:
                    tv[...] = sub.getcolslice(col, blc, trc, startrow=row0, nrow=n)
                sub.getcolslicenp("FLAG", tf, blc, trc, startrow=row0, nrow=n)
                if len(pieces) > 1:
                    vis[:, c0:c0 + w, :] = tv
                    flag[:, c0:c0 + w, :] = tf
                c0 += w
            payload = {
                "vis": vis, "flag": flag,
                "uvw": uvw_all[row0:row0 + n],
                "ant1": ant1_all[row0:row0 + n], "ant2": ant2_all[row0:row0 + n],
                "scal": {c: a[row0:row0 + n] for c, a in scal.items()},
                "last": row0 + n >= nrows,
            }
            yield ("chunk", ms_idx, payload, time.perf_counter() - t0)
    finally:
        for s in subs:
            try:
                s.close()
            except Exception:
                pass
        tab.close()


# ---------------------------------------------------------------------------
# Reader processes: VIS / FLAG in shared memory
# ---------------------------------------------------------------------------

def _shm_create(nbytes):
    from multiprocessing import shared_memory
    try:
        # Python >= 3.13: no resource tracker, the main process frees the segment
        return shared_memory.SharedMemory(create=True, size=max(1, nbytes), track=False)
    except TypeError:
        shm = shared_memory.SharedMemory(create=True, size=max(1, nbytes))
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
        return shm


def _shm_attach(name):
    """Map a segment created by a reader.  (Python < 3.13: attaching registers
    the segment in the resource tracker and unlink() unregisters it.)"""
    from multiprocessing import shared_memory
    try:
        return shared_memory.SharedMemory(name=name, track=False)
    except TypeError:
        return shared_memory.SharedMemory(name=name)


def _shm_unlink_untracked(shm):
    """Remove a segment that is not in the resource tracker (created by a reader)."""
    try:
        import _posixshmem
        _posixshmem.shm_unlink(shm._name)
    except ImportError:
        shm.unlink()


def _reader_process(worker_id, task_q, out_q, stop_ev):
    """Entry point of the reader processes (module level, for 'spawn').

    For "chunk" messages obj["vis"] and obj["flag"] are (shm name, shape,
    dtype): the main process maps them, uses them and unlinks them.
    """
    import queue as _queue
    import casacore.tables  # noqa: F401  (heavy imports while the parent does the setup)
    import astropy.coordinates  # noqa: F401

    fresh = []          # segments created for the current chunk
    pending = []        # segments already sent, local mapping still to close

    def alloc(shape, dtype):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        shm = _shm_create(nbytes)
        fresh.append(shm)
        return np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    def try_close():
        still = []
        for shm in pending:
            try:
                shm.close()
            except BufferError:          # the generator still holds a view
                still.append(shm)
        pending[:] = still

    def put(msg):
        while not stop_ev.is_set():
            try:
                out_q.put(msg, timeout=0.2)
                break
            except _queue.Full:
                continue

    try:
        while not stop_ev.is_set():
            try:
                job = task_q.get(timeout=0.2)
            except _queue.Empty:
                continue
            if job is None:
                break
            try:
                for kind, ms_idx, obj, t in _iter_ms_chunks(job, alloc):
                    if kind == "chunk":
                        shm_v, shm_f = fresh[0], fresh[1]
                        obj = dict(obj)
                        obj["vis"] = (shm_v.name, obj["vis"].shape, "complex64")
                        obj["flag"] = (shm_f.name, obj["flag"].shape, "bool")
                        pending.extend(fresh)
                        fresh.clear()
                    put((kind, worker_id, ms_idx, obj, t))
                    try_close()
            except RuntimeError as e:
                put(("skip", worker_id, job["ms_idx"], str(e), 0.0))
    except Exception:
        import traceback
        put(("error", worker_id, None, traceback.format_exc(), 0.0))
    finally:
        for shm in fresh:                 # created but never sent: removed here
            try:
                _shm_unlink_untracked(shm)
            except Exception:
                pass
            try:
                shm.close()
            except Exception:
                pass
        try_close()
        put(("done", worker_id, None, None, 0.0))


@contextlib.contextmanager
def _spawn_without_main():
    """Start 'spawn' processes without re-running the user's script in them.

    A spawned child normally imports the parent's __main__ module again: a
    script that calls ViSta.run() without an ``if __name__ == "__main__":``
    guard would start the whole run in every reader.  The readers need only
    this module, so __main__ is hidden while they start.
    """
    main = sys.modules.get("__main__")
    saved = {}
    if main is not None:
        for attr in ("__file__", "__spec__"):
            if hasattr(main, attr):
                saved[attr] = getattr(main, attr)
        if "__file__" in saved:
            del main.__file__
        if "__spec__" in saved:
            main.__spec__ = None
    try:
        yield
    finally:
        for attr, value in saved.items():
            setattr(main, attr, value)


# ===========================================================================
# Output
# ===========================================================================

# columns with the same value in every row, written with each chunk
_CONSTANT_COLS = {"FEED1": np.int32, "FEED2": np.int32, "PROCESSOR_ID": np.int32,
                  "ARRAY_ID": np.int32, "OBSERVATION_ID": np.int32,
                  "TIME_CENTROID": np.float64}
_DONE = object()          # end-of-stream marker


class _Abort(Exception):
    pass


def _output_dminfo(nchan, ncorr):
    """Storage managers of the output main table (the layout dask-ms uses)."""
    ch_tile = min(nchan, 16)
    data_rows = max(1, (1 << 20) // (ncorr * ch_tile * 8))
    return {
        "*1": {"TYPE": "IncrementalStMan", "NAME": "IndexingGroup", "SPEC": {},
               "COLUMNS": ["ARRAY_ID", "DATA_DESC_ID", "FIELD_ID", "OBSERVATION_ID",
                           "PROCESSOR_ID", "SCAN_NUMBER", "STATE_ID"]},
        "*2": {"TYPE": "StandardStMan", "NAME": "StandardStMan", "SPEC": {},
               "COLUMNS": ["ANTENNA1", "ANTENNA2", "EXPOSURE", "FEED1", "FEED2",
                           "FLAG_CATEGORY", "FLAG_ROW", "INTERVAL", "TIME",
                           "TIME_CENTROID"]},
        "*3": {"TYPE": "TiledColumnStMan", "NAME": "UVW_GROUP",
               "SPEC": {"DEFAULTTILESHAPE": np.array([3, 65536], np.int32)},
               "COLUMNS": ["UVW"]},
        "*4": {"TYPE": "TiledColumnStMan", "NAME": "WEIGHT_GROUP",
               "SPEC": {"DEFAULTTILESHAPE": np.array([ncorr, 131072], np.int32)},
               "COLUMNS": ["WEIGHT"]},
        "*5": {"TYPE": "TiledColumnStMan", "NAME": "SIGMA_GROUP",
               "SPEC": {"DEFAULTTILESHAPE": np.array([ncorr, 131072], np.int32)},
               "COLUMNS": ["SIGMA"]},
        "*6": {"TYPE": "TiledColumnStMan", "NAME": "FLAG_GROUP",
               "SPEC": {"DEFAULTTILESHAPE": np.array([ncorr, ch_tile, 32768], np.int32)},
               "COLUMNS": ["FLAG"]},
        "*7": {"TYPE": "TiledColumnStMan", "NAME": "DATA_GROUP",
               "SPEC": {"DEFAULTTILESHAPE": np.array([ncorr, ch_tile, data_rows], np.int32)},
               "COLUMNS": ["DATA"]},
    }


class _OutPool:
    """Reused output buffers of the GPU path.

    A fixed set of slots (vis, flag, uvw) as large as a chunk, allocated once
    and page-locked: the GPU copies the results straight into them (DMA) and
    no new memory pages are created for every chunk.  The compute takes a
    free slot, the writer gives it back once written; with no free slot the
    compute waits for the writer.
    """

    def __init__(self, nslots, rows, nchan, ncorr, pin=True):
        self.rows = rows
        self.slots = []
        self.free = Queue()
        self.pinned = 0
        for k in range(nslots):
            sl = (np.zeros((rows, nchan, ncorr), np.complex64),   # zeros: pages touched
                  np.zeros((rows, nchan, ncorr), np.bool_),
                  np.zeros((rows, 3), np.float64))
            if pin:
                self.pinned += int(all(ms_ops.pin_host(a) for a in sl))
            self.slots.append(sl)
            self.free.put(k)

    def views(self, k, n):
        v, f, u = self.slots[k]
        return v[:n], f[:n], u[:n]

    def close(self):
        if self.pinned:
            for sl in self.slots:
                for a in sl:
                    try:
                        ms_ops.unpin_host(a)
                    except Exception:
                        pass
        self.slots = []


def _fmt_time(seconds):
    if seconds < 120:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"


# ===========================================================================
# ViSta — main pipeline class
# ===========================================================================

class ViSta:
    """HPC-optimised visibility-domain stacking pipeline with optional CUDA
    GPU acceleration.

    Reads a list of interferometric Measurement Sets at different redshifts,
    rest-frames and phase-shifts each dataset, regrids all visibilities onto
    a common spectral grid, and writes the result as a single stacked MS.

    Reader processes, the compute kernel and a single writer run at the same
    time, connected by bounded queues (see the module docstring).  The
    compute kernel (``ms_ops``) is written in C++/OpenMP; with a CUDA build
    it can run on a GPU (``run(device="gpu")``).

    Parameters
    ----------
    input_file : str
        Path to a plain-text file listing the MSs to stack.  Each non-comment
        line must contain (whitespace-separated, in order)::

            <ms_path>  <redshift>  <RA>  <Dec>  [FIELD_ID]  [SPW_IDs]

        RA and Dec are parsed by astropy (sexagesimal or decimal).
        FIELD_ID defaults to 0.  SPW_IDs is a comma-separated list of
        spectral window indices (default: all).  An optional last value is
        the normalisation factor used by :mod:`vista.extract`.
    chunk_rows : int
        Number of baseline rows processed per chunk.  Larger values reduce
        Python overhead but increase peak memory usage.  An adaptive mechanism
        may lower this at runtime if the input channel count is very large.
    verbose : bool or "debug"
        ``True`` (default): one line per step and per MS, with the elapsed
        time, plus the warnings.  ``False``: warnings and errors only.
        ``"debug"``: also the details (metadata, queues, GPU buffers).
    """

    def __init__(self, input_file, chunk_rows=5000, verbose=True):
        self.chunk_rows = chunk_rows
        self.verbose    = verbose
        self.ms_list    = []
        self.z_list     = []
        self.ra_list    = []
        self.dec_list   = []
        self.field_list = []
        self.spw_list   = []
        self.norm_list  = []
        self.input_file = input_file
        self.data_column = "auto"
        self._load_input(input_file)

    def _load_input(self, path):
        """Parse the input list file and populate the per-MS attribute lists.

        Expected format (one MS per line, ``#`` for comments)::

            /path/to/ms1  0.045  12:34:56.7  -23:01:45.6  0  16,18
            /path/to/ms2  0.102  12:35:00.0  -23:02:00.0

        An optional extra value may follow, the normalisation factor used by
        :mod:`vista.extract` to put the sources on a common flux scale (a
        luminosity, a continuum flux density, any proxy of the stacked
        emission).  It is ignored by the stacking itself and is recognised
        because it is neither a bare integer nor a comma-separated list of
        integers, so it cannot be confused with ``FIELD_ID`` or ``SPW_IDS``::

            /path/to/ms3  0.102  12:35:00.0  -23:02:00.0  4  27,29  2.91e13
            /path/to/ms4  0.102  12:35:00.0  -23:02:00.0  norm=2.91e13
        """
        from .extract.sources import split_trailing_columns

        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed input line: {line!r}")
                self.ms_list.append(parts[0])
                self.z_list.append(float(parts[1]))
                self.ra_list.append(parts[2])
                self.dec_list.append(parts[3])
                field, spws, norm = split_trailing_columns(parts[4:])
                self.field_list.append(field)
                self.spw_list.append(spws)
                self.norm_list.append(norm)

    # ------------------------------------------------------------------
    # Spectral grid computation
    # ------------------------------------------------------------------

    def _compute_per_ms_grids(self, central_freq, nchan_out=None,
                              velocity_range_kms=None, channel_width_hz=None):
        """Compute the output spectral grid for each input MS.

        This reproduces the channel-grid logic of the original CASA-based
        pipeline (``vista.py``) exactly:

        1. Read SPECTRAL_WINDOW, FIELD, and DATA_DESCRIPTION subtables for
           every MS using dask-ms (parallelised with a ThreadPoolExecutor).
        2. Compute the rest-framed channel width for each MS::

               df_rf_i = df_obs_i * (1 + z_i)

           The common output channel width is the maximum across all MSs::

               df_new = max_i(df_rf_i)

        3. Compute the number of output channels::

               nn = max_i(floor(nchan_i * df_rf_i / df_new))   [if nchan_out is None]
               nn = nchan_out                                    [otherwise]

           ``nn`` is rounded down to an even number.

        4. For each MS, place the output window of width ``nn * df_new``
           centred on *central_freq*.  If the window does not fit within the
           MS's rest-framed frequency range, it is shifted to the nearest edge.

        The metadata read is cached in ``self._cache_*`` so that the reader
        thread can retrieve per-MS frequency arrays without re-reading.

        Parameters
        ----------
        central_freq : float
            Rest-frame central frequency of the output grid (Hz).
        nchan_out : int or None
            If set, fix the number of output channels.
        channel_width_hz : float or None
            Common rest-frame channel width, in Hz.  ``None`` uses the widest
            rest-framed channel of the sample, which is the finest grid every
            dataset supports; a finer value is refused.
        velocity_range_kms : float, tuple, or None
            If set, determines the output bandwidth from a velocity range
            (optical convention, km/s).  A scalar is interpreted as ±v;
            a 2-tuple as (v_lo, v_hi).

        Returns
        -------
        freq_new_per_ms : list of ndarray
            Per-MS output channel centre frequencies (Hz).
        df_new : float
            Common output channel width (Hz).
        """
        from concurrent.futures import ThreadPoolExecutor

        self._cache_freq_old  = {}
        self._cache_width_old = {}
        self._cache_phase_dir = {}
        self._cache_ddids     = {}
        self._freq_range_hz   = None  # not used in this mode
        self._grid_shift      = {}    # ms_idx -> shift of its window (Hz), 0 = centred
        self._grid_note       = ""

        def _read_meta_daskms(i, ms, req_field, req_spws):
            """Read SPECTRAL_WINDOW, FIELD, and DATA_DESCRIPTION for one MS.

            All CTDS access is wrapped in ``_casacore_lock`` since this
            function is called from a ThreadPoolExecutor.
            """
            result = {}

            with _casacore_lock:
                # --- DATA_DESCRIPTION: map spw_id -> dd_row ---
                dd_ds  = xds_from_table(f"{ms}::DATA_DESCRIPTION")[0]
                spwids = np.asarray(
                    dd_ds.SPECTRAL_WINDOW_ID.data.compute(scheduler="synchronous"))
                spw2dd = {int(spwid): dd_row for dd_row, spwid in enumerate(spwids)}
                result["spw2dd"] = spw2dd

                # --- SPECTRAL_WINDOW: one dataset per row (one per SPW) ---
                spw_rows    = xds_from_table(f"{ms}::SPECTRAL_WINDOW", group_cols="__row__")
                n_spw_avail = len(spw_rows)
                spws_to_read = list(range(n_spw_avail)) if req_spws is None else req_spws

                lazy = []
                for sp in spws_to_read:
                    if sp >= n_spw_avail:
                        raise ValueError(f"{os.path.basename(ms)}: SPW {sp} does not exist "
                                         f"(available: 0-{n_spw_avail-1})")
                    lazy.append(spw_rows[sp].CHAN_FREQ.data)
                    lazy.append(spw_rows[sp].CHAN_WIDTH.data)
                computed = dask.compute(*lazy, scheduler="synchronous") if lazy else ()

                freq_per_spw, width_per_spw = [], []
                for k in range(len(spws_to_read)):
                    freq_per_spw.append( np.asarray(computed[2*k],   dtype=np.float64).ravel())
                    width_per_spw.append(np.asarray(computed[2*k+1], dtype=np.float64).ravel())
                result["spws_used"]     = spws_to_read
                result["freq_per_spw"]  = freq_per_spw
                result["width_per_spw"] = width_per_spw

                # --- FIELD: PHASE_DIR of the requested field ---
                field_rows = xds_from_table(f"{ms}::FIELD", group_cols="__row__")
                n_field    = len(field_rows)
                fid = req_field if req_field is not None else 0
                if fid >= n_field:
                    raise ValueError(f"{os.path.basename(ms)}: FIELD {fid} does not exist "
                                     f"(available: 0-{n_field-1})")
                phase_dir = np.asarray(
                    field_rows[fid].PHASE_DIR.data.compute(scheduler="synchronous"),
                    dtype=np.float64).ravel()
                result["phase_dir"] = (float(phase_dir[0]), float(phase_dir[1]))
            return result

        n_ms = len(self.ms_list)
        _detail(f"Reading metadata from {n_ms} MSs (SPECTRAL_WINDOW / FIELD / DATA_DESCRIPTION)...")

        def _load_one(arg):
            i, ms = arg
            req_field = self.field_list[i]
            req_spws  = self.spw_list[i]
            _detail(f"  [{i+1}/{n_ms}]  {os.path.basename(ms)}")
            r = _read_meta_daskms(i, ms, req_field, req_spws)
            _detail(f"  [{i+1}/{n_ms}]  done  spw={r['spws_used']}  "
                 f"{sum(len(f) for f in r['freq_per_spw'])} channels")
            return r

        with ThreadPoolExecutor(max_workers=min(4, n_ms)) as pool:
            meta_list = list(pool.map(_load_one, enumerate(self.ms_list)))
        _detail("Metadata read. Computing per-MS output grids...")

        # ------------------------------------------------------------------
        # Step 1: collect df_max and nchan per MS
        # ------------------------------------------------------------------
        spectral_res_list = []   # rest-framed channel width per MS
        nchan_list        = []   # total input channels per MS
        freq_ranges       = []   # (min_rf, max_rf) per MS

        for ms_idx, (z, meta) in enumerate(zip(self.z_list, meta_list)):
            freq_per_spw  = meta["freq_per_spw"]
            width_per_spw = meta["width_per_spw"]
            spw2dd        = meta["spw2dd"]
            spws_used     = meta["spws_used"]

            self._cache_freq_old[ms_idx]  = freq_per_spw
            self._cache_width_old[ms_idx] = width_per_spw
            self._cache_phase_dir[ms_idx] = meta["phase_dir"]
            self._cache_ddids[ms_idx]     = [spw2dd.get(sp, sp) for sp in spws_used]

            freq_all  = np.concatenate(freq_per_spw)  if freq_per_spw  else np.array([0.0])
            width_all = np.concatenate(width_per_spw) if width_per_spw else np.array([1.0])

            df_obs = float(abs(width_all[0]))
            df_rf  = df_obs * (1.0 + z)
            spectral_res_list.append(df_rf)

            nchan_list.append(len(freq_all))

            freq_rf = freq_all * (1.0 + z)
            freq_ranges.append((float(freq_rf.min()), float(freq_rf.max())))

            _detail(f"  [{ms_idx+1}/{n_ms}]  {os.path.basename(self.ms_list[ms_idx])}  "
                 f"z={z:.5f}  nchan={len(freq_all)}  df_rf={df_rf/1e3:.3f} kHz  "
                 f"range=[{freq_rf.min()/1e6:.3f}, {freq_rf.max()/1e6:.3f}] MHz")

        # Common channel width: by default the widest rest-framed channel of
        # the sample, which is the finest grid every dataset can actually
        # support.  A user-set width must not be finer than that, or the
        # coarsest dataset would be interpolated onto channels it does not
        # resolve, correlating the noise between them.
        df_floor = max(spectral_res_list)
        if channel_width_hz is None:
            df_new = df_floor
            self._width_note = "width: the coarsest rest-frame channel of the sample"
            _detail(f"Common channel width (df_new) = {df_new/1e3:.3f} kHz "
                 f"(widest rest-framed channel of the sample)")
        else:
            df_new = float(channel_width_hz)
            self._width_note = "width: channel_width_hz"
            if df_new < df_floor:
                worst = int(np.argmax(spectral_res_list))
                raise ValueError(
                    f"[ViSta] requested channel width {df_new/1e3:.3f} kHz is "
                    f"finer than the coarsest rest-framed channel of the "
                    f"sample, {df_floor/1e3:.3f} kHz "
                    f"({os.path.basename(self.ms_list[worst])}, "
                    f"z={self.z_list[worst]:.4f}).  Use a width >= "
                    f"{df_floor:.6g} Hz, or drop that dataset."
                )
            _detail(f"Common channel width (df_new) = {df_new/1e3:.3f} kHz "
                 f"(user-set; sample floor is {df_floor/1e3:.3f} kHz)")

        # Maximum output channels each MS can contribute
        nn_per_ms = [max(1, int(nc * dr / df_new))
                     for nc, dr in zip(nchan_list, spectral_res_list)]

        C_KMS = 299792.458
        if velocity_range_kms is not None:
            # Determine output bandwidth from a velocity range
            if isinstance(velocity_range_kms, (tuple, list)):
                v_lo, v_hi = float(velocity_range_kms[0]), float(velocity_range_kms[1])
            else:
                v = abs(float(velocity_range_kms))
                v_lo, v_hi = -v, +v
            dv_total = v_hi - v_lo
            bw_vel   = central_freq * dv_total / C_KMS
            nn = max(2, int(np.ceil(bw_vel / df_new)))
            self._grid_note = f"channels: velocity range {v_lo:+.0f} to {v_hi:+.0f} km/s"
            _detail(f"Velocity range requested: [{v_lo:.0f}, {v_hi:.0f}] km/s "
                 f"(dv={dv_total:.0f} km/s)")
            _detail(f"  -> bandwidth = {bw_vel/1e6:.3f} MHz / "
                 f"df_new={df_new/1e3:.3f} kHz => nn = {nn} channels")
        elif nchan_out is None:
            nn = max(nn_per_ms)
            self._grid_note = (f"channels: automatic, the MS fill "
                               f"{min(nn_per_ms)}-{max(nn_per_ms)} of them")
            _detail(f"Output channels (auto): {nn}  "
                 f"[per-MS coverage: {min(nn_per_ms)}-{max(nn_per_ms)}]")
        else:
            nn = int(nchan_out)
            self._grid_note = (f"channels: nchan_out, the MS can fill "
                               f"{min(nn_per_ms)}-{max(nn_per_ms)} of them")
            _detail(f"Output channels (user-set): {nn}  "
                 f"[per-MS available: {min(nn_per_ms)}-{max(nn_per_ms)}]")

        if nn % 2 != 0:
            nn -= 1
        _detail(f"Final output channels: {nn}")

        bandwidth    = nn * df_new
        start_ideal  = central_freq - bandwidth / 2.0
        end_ideal    = central_freq + bandwidth / 2.0
        _detail(f"Target window: [{start_ideal/1e6:.6f}, {end_ideal/1e6:.6f}] MHz")

        # ------------------------------------------------------------------
        # Step 2: place output window for each MS (shift to edge if needed)
        # ------------------------------------------------------------------
        freq_new_per_ms = []
        for ms_idx, (ms, (msmin, msmax)) in enumerate(zip(self.ms_list, freq_ranges)):
            if (start_ideal >= msmin) and (end_ideal <= msmax):
                start  = start_ideal
                reason = "ideal window fits"
            elif msmin > start_ideal:
                start  = msmin
                reason = "shifted to left edge"
            elif msmax < end_ideal:
                start  = msmax - bandwidth
                reason = "shifted to right edge"
            else:
                raise RuntimeError(
                    f"[ViSta] Could not place output window for MS {ms}"
                )

            end    = start + bandwidth
            center = 0.5 * (start + end)
            offset = center - central_freq
            self._grid_shift[ms_idx] = 0.0 if reason == "ideal window fits" else offset
            _detail(f"  [{ms_idx+1}/{n_ms}]  {os.path.basename(ms)}  "
                 f"start={start/1e6:.6f} MHz  end={end/1e6:.6f} MHz  "
                 f"offset={offset:+.3e} Hz  ({reason})")

            freq_ms = start + df_new / 2.0 + np.arange(nn) * df_new
            freq_new_per_ms.append(freq_ms.astype(np.float64))

        return freq_new_per_ms, df_new

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, ms_out, central_freq, scratch_dir=None, nchan_out=None,
            velocity_range_kms=None, channel_width_hz=None,
            data_column="auto", device="cpu", n_readers=2, n_threads=None,
            gpu_batch=8, timing=False, log_file=None,
            reader_mode="process", chunk_queue=None, out_queue=3):
        """Run the stacking pipeline and write the output Measurement Set.

        Parameters
        ----------
        ms_out : str
            Path to the output Measurement Set.  Any existing file/directory
            at this path will be deleted before writing.
        central_freq : float
            Rest-frame central frequency of the output spectral grid (Hz).
            Identical to the ``central_freq`` parameter of ``ViSta.rebinning()``.
        scratch_dir : str or None
            If set, the MS is first written to this directory (e.g. a local
            NVMe, ``$TMPDIR``, or ``/scratch``) and then moved to *ms_out*
            at the end.  Useful on HPC systems where the final destination is
            on a slow network filesystem.
        nchan_out : int or None
            Number of output channels per spectral window.

            ``None`` (default) — automatic mode: ``nn = max`` channels any
            single MS can contribute.  MSs with narrower bandwidth will have
            partial coverage.

            ``int`` — fix the number of output channels explicitly.  Can be
            larger or smaller than the automatic value.
        velocity_range_kms : float, tuple, or None
            If set, determines the output bandwidth from a velocity range
            (optical convention, km/s).  A scalar is interpreted as ±v;
            a 2-tuple as ``(v_lo, v_hi)``.  Overrides *nchan_out*.
        channel_width_hz : float or None
            Common rest-frame channel width, in Hz.  ``None`` uses the widest
            rest-framed channel of the sample, which is the finest grid every
            dataset supports; a finer value is refused.
        data_column : str
            Visibility column to read: ``"auto"`` (default) prefers
            ``CORRECTED_DATA`` and falls back to ``DATA``; any other name is
            read when present, otherwise ``DATA`` is used (with a warning).
        device : "cpu" or "gpu"
            Where the kernel runs.  ``"gpu"`` needs a CUDA build
            (``-DWITH_CUDA=ON``) and a visible GPU.  The run is usually
            limited by the writing of the output MS, so the GPU pays off only
            when the computation dominates (many channels, many rows).
        n_readers : int
            Reader processes reading different MSs at the same time
            (default 2; more readers help only on fast filesystems).
        n_threads : int or None
            OpenMP threads of the CPU kernel.  ``None`` keeps
            ``OMP_NUM_THREADS``, else ``SLURM_CPUS_PER_TASK``, else all cores.
        gpu_batch : int
            GPU only: maximum number of chunks per GPU launch (default 8).
            The compute takes the chunks already read, without waiting for
            more; larger values use more host memory.
        timing : bool
            At the end, also log how long setup, reading, computing and
            writing took, and which stage limited the run.
        log_file : str or None
            Also append the log to this file.
        reader_mode : "process" or "thread"
            ``"process"`` (default) — parallel reads, data passed through
            shared memory (``/dev/shm``).  ``"thread"`` — readers as threads:
            no shared memory needed (e.g. containers with a small
            ``/dev/shm``), but the reads take turns.
        chunk_queue, out_queue : int or None
            Advanced: chunks waiting for the compute (default
            ``n_readers + 1``, at least ``gpu_batch`` on the GPU) and for the
            writer (default 3).  They bound the memory in use.

        Returns
        -------
        dict
            ``ms_out``, ``n_rows``, ``n_spw``, ``nchan``, ``df_hz``,
            ``flagged_fraction``, ``skipped`` (input indices) and
            ``seconds``.
        """
        set_verbosity(self.verbose, log_file)
        _reset_clock()
        t_start = time.perf_counter()
        dask.config.set(scheduler="synchronous")

        if device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")
        if reader_mode not in ("process", "thread"):
            raise ValueError("reader_mode must be 'process' or 'thread'")
        n_readers = max(1, int(n_readers))
        gpu_batch = max(1, int(gpu_batch))
        chunk_queue = int(chunk_queue) if chunk_queue else n_readers + 1
        if n_threads:
            ms_ops.omp_set_threads(int(n_threads))
        n_omp = ms_ops.omp_max_threads()
        use_gpu = device == "gpu"
        if use_gpu:
            if not ms_ops.built_with_cuda():
                raise RuntimeError("device='gpu': the ms_ops kernel was built without "
                                   "CUDA; rebuild it with cmake -DWITH_CUDA=ON")
            if not ms_ops.gpu_available():
                raise RuntimeError("device='gpu': no GPU visible to CUDA "
                                   "(check CUDA_VISIBLE_DEVICES and the node)")
            # the chunk queue must be able to hold a whole batch
            chunk_queue = max(chunk_queue, gpu_batch)

        n_ms = len(self.ms_list)
        ms_final = ms_out
        if use_gpu:
            gfree, gtot = ms_ops.gpu_memory_info()
            where = f"GPU ({gtot / 2**30:.0f} GB, up to {gpu_batch} chunks per launch)"
        else:
            where = f"CPU ({n_omp} thread{'s' if n_omp > 1 else ''})"
        _log(f"ViSta {__version__} · {n_ms} MS from {os.path.basename(self.input_file)} "
             f"→ {ms_final} · {where} · {n_readers} reader{'s' if n_readers > 1 else ''}")
        _detail(f"chunk_rows={self.chunk_rows}  reader_mode={reader_mode}  "
                f"chunk_queue={chunk_queue}  out_queue={out_queue}")

        # The reader processes start now, so that they import numpy/casacore
        # while the setup runs here; their jobs arrive later on task_q.
        abort = threading.Event()
        procs = []
        if reader_mode == "process":
            import multiprocessing as mp
            mpc = mp.get_context("spawn")
            task_q = mpc.Queue()
            read_q = mpc.Queue(maxsize=chunk_queue)
            stop_ev = mpc.Event()
            procs = [mpc.Process(target=_reader_process, args=(k, task_q, read_q, stop_ev),
                                 name=f"vista-reader-{k}", daemon=True)
                     for k in range(n_readers)]
            with _spawn_without_main():
                for pr in procs:
                    pr.start()
        else:
            task_q = Queue()
            read_q = Queue(maxsize=chunk_queue)
            stop_ev = abort

        def _stop_procs():
            stop_ev.set()
            for pr in procs:
                pr.join(timeout=5)
                if pr.is_alive():
                    pr.terminate()

        # ------------------------------------------------------------------
        # Setup: output grid and subtables
        # ------------------------------------------------------------------
        try:
            if scratch_dir is not None:
                os.makedirs(scratch_dir, exist_ok=True)
                ms_out = os.path.join(scratch_dir, os.path.basename(ms_final))
                _log(f"Writing to the scratch directory {scratch_dir}, "
                     f"moved to {ms_final} at the end")
            for _p in {ms_out, ms_final}:
                if os.path.exists(_p):
                    _detail(f"Removing existing output: {_p}")
                    shutil.rmtree(_p)

            freq_new_per_ms, df_new = self._compute_per_ms_grids(
                central_freq, nchan_out=nchan_out,
                velocity_range_kms=velocity_range_kms,
                channel_width_hz=channel_width_hz)
            nchan_new = len(freq_new_per_ms[0])
            dv = df_new / central_freq * _C_LIGHT_KMS
            half_v = 0.5 * nchan_new * dv
            _log(f"Output grid: {nchan_new} channels × {df_new / 1e3:.1f} kHz "
                 f"({dv:.2f} km/s), ±{half_v:.0f} km/s around "
                 f"{central_freq / 1e9:.4f} GHz")
            _log(f"  ({self._width_note}; {self._grid_note})")
            n_shift = sum(1 for s in self._grid_shift.values() if s)
            if n_shift:
                _log(f"  {n_shift} of {n_ms} MS do not cover the whole window around "
                     f"{central_freq / 1e9:.4f} GHz: their window is moved to the edge "
                     f"of their band (shift in the lines below), so their channel "
                     f"frequencies differ from the others")

            _detail("Building output subtables...")
            ant_ds, ant_offsets = _build_antenna_table(self.ms_list, self.z_list)
            first_ms     = self.ms_list[0]
            spw_ds       = _build_spw_table(freq_new_per_ms, df_new, self.ms_list)
            field_ds     = _build_field_table(first_ms)
            dd_ds        = _build_dd_table(first_ms, n_ms)
            pol_ds       = _copy_subtable_first_row(first_ms, "POLARIZATION")
            obs_ds_list  = _build_observation_table(self.ms_list)
            nant_total   = ant_ds.sizes['row']
            # minimal FEED table covering all antennas
            feed_ds_list = [Dataset({
                'ANTENNA_ID':        (('row',),           _to_dask(np.arange(nant_total, dtype=np.int32))),
                'FEED_ID':           (('row',),           _to_dask(np.zeros(nant_total, dtype=np.int32))),
                'SPECTRAL_WINDOW_ID':(('row',),           _to_dask(np.full(nant_total, -1, dtype=np.int32))),
                'TIME':              (('row',),           _to_dask(np.zeros(nant_total, dtype=np.float64))),
                'INTERVAL':          (('row',),           _to_dask(np.zeros(nant_total, dtype=np.float64))),
                'NUM_RECEPTORS':     (('row',),           _to_dask(np.full(nant_total, 2, dtype=np.int32))),
                'POLARIZATION_TYPE': (('row','receptors'),_to_dask(np.array([['X','Y']]*nant_total, dtype=object))),
                'POL_RESPONSE':      (('row','receptors','receptors2'),
                                      _to_dask(np.tile(np.eye(2, dtype=complex), (nant_total,1,1)))),
                'POSITION':          (('row','xyz'),      _to_dask(np.zeros((nant_total,3), dtype=np.float64))),
                'BEAM_OFFSET':       (('row','receptors','radec'),
                                      _to_dask(np.zeros((nant_total,2,2), dtype=np.float64))),
                'RECEPTOR_ANGLE':    (('row','receptors'),_to_dask(np.zeros((nant_total,2), dtype=np.float64))),
                'BEAM_ID':           (('row',),           _to_dask(np.full(nant_total, -1, dtype=np.int32))),
            })]
            src_ds_list = _build_source_table(first_ms, central_freq)
        except BaseException:
            _stop_procs()
            raise

        jobs = [{
            "ms_idx": i, "ms": self.ms_list[i], "z": float(self.z_list[i]),
            "freq_old_list": self._cache_freq_old[i],
            "width_old_list": self._cache_width_old[i],
            "ddids": [int(d) for d in self._cache_ddids[i]],
            "field": self.field_list[i],
            "phase_dir": self._cache_phase_dir[i],
            "ra": self.ra_list[i], "dec": self.dec_list[i],
            "freq_new": freq_new_per_ms[i], "df_new": float(df_new),
            "chunk_rows": int(self.chunk_rows), "ant_offset": int(ant_offsets[i]),
            "data_column": data_column,
        } for i in range(n_ms)]
        for j in jobs:
            task_q.put(j)
        for _ in range(n_readers):
            task_q.put(None)

        t_loop = time.perf_counter()
        _log(f"Setup done ({nant_total} antennas in total); stacking {n_ms} MS...")

        out_q  = Queue(maxsize=out_queue)
        ctx    = {}
        pools  = {}           # GPU: ncorr -> _OutPool
        errors = []
        stats  = {"rows": 0, "bytes": 0, "flagged": 0, "values": 0, "skipped": {},
                  "ms_done": 0, "t_read": 0.0, "t_compute": 0.0, "t_write": 0.0,
                  "gpu_launches": 0, "gpu_chunks": 0, "gpu_error": False,
                  "t_first": None}
        progress_step = max(1, n_ms // 10) if n_ms >= 20 else None
        width = len(str(n_ms))

        def _put(q, item):
            while True:
                if abort.is_set():
                    raise _Abort()
                try:
                    q.put(item, timeout=0.2)
                    return
                except Full:
                    continue

        def _get(q):
            while True:
                if abort.is_set():
                    raise _Abort()
                try:
                    return q.get(timeout=0.2)
                except Empty:
                    continue

        def _fail(e):
            errors.append(e)
            abort.set()

        # ------------------------------------------------------------------
        # Writer thread: the output table is opened once, then addrows + putcol
        # ------------------------------------------------------------------
        def _writer():
            from casacore.tables import default_ms, makearrcoldesc, maketabdesc
            tab = None
            try:
                while True:
                    item = _get(out_q)
                    if item is _DONE:
                        break
                    t0 = time.perf_counter()
                    if tab is None:
                        nchan, ncorr = item["DATA"].shape[1], item["DATA"].shape[2]
                        desc = maketabdesc([
                            makearrcoldesc("DATA", 0j, shape=[nchan, ncorr], valuetype="complex"),
                            makearrcoldesc("FLAG", False, shape=[nchan, ncorr]),
                            makearrcoldesc("WEIGHT", 0.0, shape=[ncorr], valuetype="float"),
                            makearrcoldesc("SIGMA", 0.0, shape=[ncorr], valuetype="float"),
                        ])
                        tab = default_ms(ms_out, desc, _output_dminfo(nchan, ncorr))
                        out_cols = set(tab.colnames())
                    n = item["DATA"].shape[0]
                    start = tab.nrows()
                    tab.addrows(n)
                    for col, arr in item.items():
                        if col in out_cols:
                            tab.putcol(col, arr, startrow=start, nrow=n)
                    for col, dt in _CONSTANT_COLS.items():
                        if col in out_cols:
                            tab.putcol(col, np.zeros(n, dtype=dt), startrow=start, nrow=n)
                    stats["t_write"] += time.perf_counter() - t0
                    slot = item.get("_slot")
                    if slot is not None:
                        pools[slot[0]].free.put(slot[1])    # buffer free again
                    if item["_last"]:
                        stats["ms_done"] += 1
                        k = stats["ms_done"]
                        if progress_step and (k % progress_step == 0) and k < n_ms:
                            el = time.perf_counter() - t_loop
                            _log(f"Stacking: {k}/{n_ms} MS written "
                                 f"({100 * k / n_ms:.0f}%), about "
                                 f"{_fmt_time(el / k * (n_ms - k))} left")
                if tab is not None:
                    t0 = time.perf_counter()
                    tab.flush()
                    tab.close()
                    tab = None
                    stats["t_write"] += time.perf_counter() - t0
            except _Abort:
                pass
            except Exception as e:
                _fail(e)
            finally:
                if tab is not None:
                    try:
                        tab.close()
                    except Exception:
                        pass

        # ------------------------------------------------------------------
        # Reader threads (reader_mode="thread"), same messages as the processes
        # ------------------------------------------------------------------
        def _thread_reader(wid):
            import traceback

            def put(msg):
                while not abort.is_set():
                    try:
                        read_q.put(msg, timeout=0.2)
                        break
                    except Full:
                        continue
            try:
                while not abort.is_set():
                    job = task_q.get()
                    if job is None:
                        break
                    try:
                        for kind, ms_idx, obj, t in _iter_ms_chunks(
                                job, lambda shape, dt: np.empty(shape, dt)):
                            put((kind, wid, ms_idx, obj, t))
                    except RuntimeError as e:
                        put(("skip", wid, job["ms_idx"], str(e), 0.0))
            except Exception:
                put(("error", wid, None, traceback.format_exc(), 0.0))
            finally:
                put(("done", wid, None, None, 0.0))

        readers = []
        if reader_mode == "thread":
            readers = [threading.Thread(target=_thread_reader, args=(k,),
                                        name=f"vista-reader-{k}", daemon=True)
                       for k in range(n_readers)]
            for t in readers:
                t.start()
        writer = threading.Thread(target=_writer, name="vista-writer", daemon=True)
        writer.start()

        def _chunk_arrays(obj):
            """(vis, flag, shared-memory segments to release)."""
            if reader_mode == "thread":
                return obj["vis"], obj["flag"], []
            shms, arrs = [], []
            for key in ("vis", "flag"):
                name, shape, dt = obj[key]
                shm = _shm_attach(name)
                shms.append(shm)
                arrs.append(np.ndarray(shape, dtype=np.dtype(dt), buffer=shm.buf))
            return arrs[0], arrs[1], shms

        def _release(shms):
            for shm in shms:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass

        def _emit(ms_idx, obj, vis_out, flag_out, uvw_out, slot=None):
            """Pack one processed chunk and hand it to the writer."""
            c = ctx[ms_idx]
            n = vis_out.shape[0]
            out = {
                "DATA": vis_out, "FLAG": flag_out, "UVW": uvw_out,
                "ANTENNA1": obj["ant1"] + c["ant_offset"],
                "ANTENNA2": obj["ant2"] + c["ant_offset"],
                "DATA_DESC_ID": np.full(n, ms_idx, dtype=np.int32),
                "FIELD_ID": np.zeros(n, dtype=np.int32),
            }
            for col, arr in obj["scal"].items():
                if col == "WEIGHT":
                    arr = arr * c["weight_scale"]
                elif col == "SIGMA":
                    arr = arr * c["sigma_scale"]
                out[col] = arr
            out["_slot"] = slot
            out["_last"] = obj["last"]
            stats["rows"] += n
            stats["bytes"] += vis_out.nbytes + flag_out.nbytes + uvw_out.nbytes
            stats["flagged"] += int(np.count_nonzero(flag_out))
            stats["values"] += flag_out.size
            _put(out_q, out)

        def _acquire(ncorr, nrows):
            """A free buffer of the pool for this ncorr: (pool key, index), or
            None if the chunk is larger than the buffers."""
            key = int(ncorr)
            pool = pools.get(key)
            if pool is None:
                nslots = out_queue + gpu_batch + 2
                pool = _OutPool(nslots, int(self.chunk_rows), int(nchan_new), key)
                pools[key] = pool
                _detail(f"GPU: {nslots} output buffers of {int(self.chunk_rows)} rows "
                        f"({pool.pinned}/{nslots} page-locked)")
            if nrows > pool.rows:
                return None
            return (key, _get(pool.free))      # waiting here = waiting for the writer

        def _compute_cpu(ms_idx, obj):
            vis, flag, shms = _chunk_arrays(obj)
            c = ctx[ms_idx]
            try:
                t0 = time.perf_counter()
                vo, fo, uo = ms_ops.full_pipeline(
                    vis, flag, obj["uvw"], c["z"], c["freq_old"],
                    c["ra0"], c["dec0"], c["ra1"], c["dec1"], c["freq_new"])
                stats["t_compute"] += time.perf_counter() - t0
            finally:
                del vis, flag
                _release(shms)
            _emit(ms_idx, obj, vo, fo, uo)

        def _compute_gpu(batch):
            """batch = [(ms_idx, obj), ...] -> one GPU launch into pooled buffers."""
            arrs, shms_all, slots = [], [], []
            try:
                for ms_idx, obj in batch:
                    vis, flag, shms = _chunk_arrays(obj)
                    shms_all += shms
                    arrs.append((vis, flag))
                outs = []
                for (ms_idx, obj), (vis, flag) in zip(batch, arrs):
                    n, ncorr = vis.shape[0], vis.shape[2]
                    sl = _acquire(ncorr, n)
                    slots.append(sl)
                    if sl is None:            # chunk larger than the buffers: new arrays
                        outs.append((np.empty((n, nchan_new, ncorr), np.complex64),
                                     np.empty((n, nchan_new, ncorr), np.bool_),
                                     np.empty((n, 3), np.float64)))
                    else:
                        outs.append(pools[sl[0]].views(sl[1], n))
                cs = [ctx[i] for i, _ in batch]
                t0 = time.perf_counter()
                ms_ops.full_pipeline_batch_into(
                    [a[0] for a in arrs], [a[1] for a in arrs],
                    [o["uvw"] for _, o in batch], [c["z"] for c in cs],
                    [c["freq_old"] for c in cs],
                    [c["ra0"] for c in cs], [c["dec0"] for c in cs],
                    [c["ra1"] for c in cs], [c["dec1"] for c in cs],
                    [c["freq_new"] for c in cs],
                    [o[0] for o in outs], [o[1] for o in outs], [o[2] for o in outs])
                stats["t_compute"] += time.perf_counter() - t0
                stats["gpu_launches"] += 1
                stats["gpu_chunks"] += len(batch)
                if not stats["gpu_error"] and not ms_ops.gpu_available():
                    stats["gpu_error"] = True
                    _warn(f"GPU error, the run continues on the CPU: "
                          f"{ms_ops.gpu_last_error()}")
            except BaseException:
                for sl in slots:              # give back the buffers taken
                    if sl is not None:
                        pools[sl[0]].free.put(sl[1])
                raise
            finally:
                del arrs
                _release(shms_all)
            for (ms_idx, obj), (vo, fo, uo), sl in zip(batch, outs, slots):
                _emit(ms_idx, obj, vo, fo, uo, slot=sl)

        def _handle(kind, wid, ms_idx, obj, t):
            """Bookkeeping of a reader message; True if it is a chunk."""
            stats["t_read"] += t
            if stats["t_first"] is None and kind in ("ctx", "chunk"):
                stats["t_first"] = time.perf_counter() - t_loop
            if kind == "chunk":
                return True
            if kind == "done":
                return False
            if kind == "error":
                raise RuntimeError(f"reader {wid} failed:\n{obj}")
            name = os.path.basename(self.ms_list[ms_idx].rstrip("/"))
            if kind == "skip":
                stats["skipped"][ms_idx] = obj
                _warn(f"[{ms_idx + 1:>{width}}/{n_ms}] {name} skipped: {obj}")
            elif kind == "ctx":
                ctx[ms_idx] = obj
                self._log_ms(ms_idx, n_ms, width, name, obj)
            return False

        # ------------------------------------------------------------------
        # Compute (main thread)
        # ------------------------------------------------------------------
        n_done = 0
        try:
            while n_done < n_readers:
                msg = _get(read_q)
                if msg[0] == "done":
                    n_done += 1
                if not _handle(*msg):
                    continue
                if not use_gpu:
                    _compute_cpu(msg[2], msg[3])
                    continue
                # GPU: also take the chunks already waiting, without waiting
                batch = [(msg[2], msg[3])]
                while len(batch) < gpu_batch:
                    try:
                        m2 = read_q.get_nowait()
                    except Empty:
                        break
                    if m2[0] == "done":
                        n_done += 1
                    if _handle(*m2):
                        batch.append((m2[2], m2[3]))
                _compute_gpu(batch)
            _put(out_q, _DONE)
        except _Abort:
            pass
        except Exception as e:
            _fail(e)
        finally:
            if abort.is_set():
                # free the shared-memory segments still in the queue
                stop_ev.set()
                while True:
                    try:
                        m = read_q.get_nowait()
                    except Exception:
                        break
                    if reader_mode == "process" and m and m[0] == "chunk":
                        for key in ("vis", "flag"):
                            try:
                                _release([_shm_attach(m[3][key][0])])
                            except Exception:
                                pass

        writer.join()
        t_loop_end = time.perf_counter()
        for pool in pools.values():
            pool.close()
        pools.clear()
        if reader_mode == "process":
            _stop_procs()
        else:
            for t in readers:
                t.join(timeout=5)
        if errors:
            raise RuntimeError(f"Pipeline failed: {errors[0]!r}") from errors[0]
        if not os.path.isdir(ms_out):
            raise RuntimeError(
                f"Output table '{ms_out}' was not created: no MS gave usable data "
                f"(see the warnings above for the field, SPW and coverage of each MS)")

        # ------------------------------------------------------------------
        # Subtables, then the optional move from the scratch directory
        # ------------------------------------------------------------------
        def _write_sub(ds_or_list, subtable):
            lst = ds_or_list if isinstance(ds_or_list, list) else [ds_or_list]
            if lst:
                with _casacore_lock:
                    dask.compute(*xds_to_table(lst, f"{ms_out}::{subtable}", "ALL"))
                _detail(f"  Written: {subtable}")

        _detail(f"Writing output subtables to {ms_out} ...")
        _write_sub(spw_ds,      "SPECTRAL_WINDOW")
        _write_sub(field_ds,    "FIELD")
        _write_sub(dd_ds,       "DATA_DESCRIPTION")
        _write_sub(ant_ds,      "ANTENNA")
        if pol_ds:       _write_sub(pol_ds,       "POLARIZATION")
        if obs_ds_list:  _write_sub(obs_ds_list,  "OBSERVATION")
        if feed_ds_list: _write_sub(feed_ds_list, "FEED")
        if src_ds_list:  _write_sub(src_ds_list,  "SOURCE")
        gc.collect()          # dask-ms keeps tables cached: close them now
        if scratch_dir is not None and ms_out != ms_final:
            _log(f"Moving the output from the scratch directory to {ms_final}...")
            shutil.move(ms_out, ms_final)

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        t_end = time.perf_counter()
        total = t_end - t_start
        flagged = stats["flagged"] / stats["values"] if stats["values"] else 0.0
        n_spw = n_ms - len(stats["skipped"])
        if stats["skipped"]:
            names = ", ".join(os.path.basename(self.ms_list[i].rstrip("/"))
                              for i in sorted(stats["skipped"]))
            _warn(f"{len(stats['skipped'])} of {n_ms} MS skipped, no data in the "
                  f"output: {names}")
        if timing:
            loop = t_loop_end - t_loop
            busy = {"reading": stats["t_read"] / n_readers,
                    "computing": stats["t_compute"], "writing": stats["t_write"]}
            limit = max(busy, key=busy.get)
            share = busy[limit] / max(loop, 1e-9)
            _log(f"Timing: setup {_fmt_time(t_loop - t_start)} · stacking "
                 f"{_fmt_time(loop)} · subtables and closing "
                 f"{_fmt_time(t_end - t_loop_end)}")
            _log(f"  during the stacking (overlapping): reading "
                 f"{_fmt_time(busy['reading'])} per reader · computing "
                 f"{_fmt_time(busy['computing'])} · writing {_fmt_time(busy['writing'])}"
                 + (f"  → limited by {limit} ({100 * share:.0f}% busy)" if share >= 0.5 else
                    f"  → no stage busy for half of the time: the first data arrived "
                    f"after {_fmt_time(stats['t_first'] or 0.0)} (start of the readers)"))
            if use_gpu and stats["gpu_launches"]:
                _log(f"  GPU: {stats['gpu_launches']} launches, "
                     f"{stats['gpu_chunks'] / stats['gpu_launches']:.1f} chunks per launch "
                     f"on average")
        _log(f"Done in {_fmt_time(total)}: {ms_final} · {n_spw} SPW × {nchan_new} channels · "
             f"{stats['rows']:,} rows · {_fmt_size(stats['bytes'])} · "
             f"{100 * flagged:.1f}% of the output flagged (no data)")
        return {"ms_out": ms_final, "n_rows": stats["rows"], "n_spw": n_spw,
                "nchan": nchan_new, "df_hz": float(df_new),
                "flagged_fraction": flagged, "skipped": sorted(stats["skipped"]),
                "seconds": total}

    def _log_ms(self, ms_idx, n_ms, width, name, c):
        """One line per MS: what it contributes to the stack."""
        rows = f"{c['rows']:,} rows"
        if c["rows_all"] is not None and c["rows_all"] != c["rows"]:
            rows = f"{c['rows']:,}/{c['rows_all']:,} rows ({', '.join(c['filters'])})"
        cov = 100.0 * c["coverage"] / max(1, c["nchan_out"])
        spw = f"{c['n_spw']} SPW" + ("s" if c["n_spw"] > 1 else "")
        line = (f"[{ms_idx + 1:>{width}}/{n_ms}] {name}  z={c['z']:.4f}  {rows}  "
                f"{c['data_column']}  {spw}, {c['nchan_in']} ch  "
                f"covers {cov:.0f}% of the grid  offset {c['offset_arcsec']:.1f}\"")
        shift = self._grid_shift.get(ms_idx, 0.0)
        if shift:
            line += f"  window {shift / 1e6:+.1f} MHz"
        _log(line)
        _detail(f"    weight scale R={c['weight_scale']:.4f}  chunk={c['chunk_rows']} rows  "
                f"ncorr={c['ncorr']}  primary beam FWHM="
                + (f"{c['pb_fwhm_arcsec']:.1f}\"" if c["pb_fwhm_arcsec"] else "n/a"))
        for w in c["warnings"]:
            _warn(f"{name}: {w}")
