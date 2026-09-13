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

The pipeline is organised as a producer-consumer model with three
concurrent threads (reader, compute, writer) communicating through
bounded queues, keeping all stages continuously busy and hiding I/O
latency behind compute.

The C++/OpenMP (and optionally CUDA) kernel is provided by the compiled
extension module ms_ops.so, built from src/ms_ops.cpp and src/ms_ops_cuda.cu.

"""

import os

# ── Threading environment (set before any NumPy / OpenBLAS / MKL import) ──
_n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 4)
os.environ["OMP_NUM_THREADS"]        = str(_n_threads)
os.environ["OPENBLAS_NUM_THREADS"]   = str(_n_threads)
os.environ["MKL_NUM_THREADS"]        = str(_n_threads)
os.environ["NUMEXPR_NUM_THREADS"]    = str(_n_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(_n_threads)
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("OMP_PLACES",    "cores")

import shutil
import math
import time
import threading
from queue import Queue
import numpy as np
import dask
import dask.array as da
from collections import Counter
from daskms import xds_from_ms, xds_from_table, xds_to_table, Dataset

try:
    from . import ms_ops     # installed into vista/ by 'make install'
except ImportError:          # or found on PYTHONPATH
    import ms_ops

# ---------------------------------------------------------------------------
# GPU availability check
# ---------------------------------------------------------------------------
# Try to call cuda_available() exposed by the CUDA-enabled ms_ops module.
# If the module was compiled without CUDA or no device is present, fall back
# to CPU-only mode transparently.
try:
    import ctypes
    _lib = ctypes.CDLL(ms_ops.__file__)
    _gpu_available = bool(_lib.cuda_available())
except Exception:
    _gpu_available = hasattr(ms_ops, 'full_pipeline_batch')


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_C_LIGHT_KMS = 299792.458   # speed of light in km/s

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
# All log messages are buffered and flushed together to avoid interleaved
# output from concurrent threads.

import sys as _sys
import threading as _threading
import time as _time_module

_log_lock   = _threading.Lock()
_log_buffer = []
_t0_global  = _time_module.perf_counter()
_verbose_global = True


def _log(msg, flush=False):
    """Append a timestamped message to the log buffer.

    Messages are only printed when verbose mode is active.  The timestamp
    is relative to the start of the current run (seconds elapsed).  If
    *flush* is True the entire buffer is printed to stdout immediately.
    """
    if not _verbose_global:
        return
    ts = _time_module.perf_counter() - _t0_global
    line = f"[ViSta  {ts:7.2f}s]  {msg}"
    with _log_lock:
        _log_buffer.append(line)
        if flush:
            _flush_log()


def _flush_log():
    """Flush the log buffer to stdout (must be called with _log_lock held)."""
    if _log_buffer:
        _sys.stdout.write("\n".join(_log_buffer) + "\n")
        _sys.stdout.flush()
        _log_buffer.clear()


def _flush_log_unlocked():
    """Flush the log buffer to stdout, acquiring the lock internally."""
    with _log_lock:
        _flush_log()


# ---------------------------------------------------------------------------
# Thread safety: Casacore Table Data System locks
# ---------------------------------------------------------------------------
# See module docstring for rationale.

_casacore_lock = _threading.Lock()
_ms_locks: dict = {}
_ms_locks_mutex = _threading.Lock()


def _get_ms_lock(ms_path: str) -> _threading.Lock:
    """Return a per-file lock: different MS files can be read in parallel,
    but concurrent access to the *same* MS is serialised.
    """
    with _ms_locks_mutex:
        if ms_path not in _ms_locks:
            _ms_locks[ms_path] = _threading.Lock()
        return _ms_locks[ms_path]


# ---------------------------------------------------------------------------
# Coordinate / frequency helpers
# ---------------------------------------------------------------------------

def _vrange_to_freqrange(restfreq_hz, vmin_kms, vmax_kms):
    """Convert an optical velocity range [vmin, vmax] (km/s) to a frequency
    range in Hz using the optical convention:

        v_opt = c * (f_rest - f) / f_rest   =>   f = f_rest * (1 - v/c)

    Returns (freq_lo_hz, freq_hi_hz) with freq_lo < freq_hi.
    """
    f_at_vmax = restfreq_hz * (1.0 - vmax_kms / _C_LIGHT_KMS)
    f_at_vmin = restfreq_hz * (1.0 - vmin_kms / _C_LIGHT_KMS)
    return min(f_at_vmax, f_at_vmin), max(f_at_vmax, f_at_vmin)


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
# Array / chunk utilities
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


def _make_row_chunks(nrow, chunk_size):
    """Return a tuple of chunk sizes that partition *nrow* rows into blocks
    of at most *chunk_size*, with the remainder as the last (smaller) block.
    """
    n_full = nrow // chunk_size
    n_rem  = nrow %  chunk_size
    return (chunk_size,) * n_full + ((n_rem,) if n_rem else ())


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


def _build_history_table(first_ms):
    """Copy the HISTORY subtable from the first input MS."""
    with _casacore_lock:
        try: rows = xds_from_table(f"{first_ms}::HISTORY", group_cols="__row__")
        except Exception: return []
        return [Dataset({v:(r[v].dims,_to_dask(np.asarray(r[v].data))) for v in r.data_vars}) for r in rows]


def _build_feed_table(ms_list, ant_offsets):
    """Build the FEED subtable by concatenating FEED tables from all input MSs.

    Antenna IDs are remapped using *ant_offsets* so they point to the correct
    rows in the combined ANTENNA table.  All CTDS access is serialised via
    ``_casacore_lock``.
    """
    all_ds = []
    for ms, offset in zip(ms_list, ant_offsets):
        with _casacore_lock:
            try: rows = xds_from_table(f"{ms}::FEED", group_cols="__row__")
            except Exception: continue
            for r in rows:
                vnames   = list(r.data_vars)
                computed = dask.compute(*[r[v].data for v in vnames])
                data_vars = {}
                for v, arr in zip(vnames, computed):
                    arr = np.asarray(arr)
                    if v == "ANTENNA_ID": arr = arr + offset
                    data_vars[v] = (r[v].dims, _to_dask(arr))
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
# ViSta — main pipeline class
# ===========================================================================

class ViSta:
    """HPC-optimised visibility-domain stacking pipeline with optional CUDA
    GPU acceleration.

    Reads a list of interferometric Measurement Sets at different redshifts,
    rest-frames and phase-shifts each dataset, regrids all visibilities onto
    a common spectral grid, and writes the result as a single stacked MS.

    The pipeline uses three concurrent threads (reader, compute, writer)
    connected by bounded queues, effectively hiding I/O latency behind
    compute and vice versa.  The compute kernel (``ms_ops.so``) is written
    in C++/OpenMP and optionally dispatched to a CUDA GPU when available.

    Parameters
    ----------
    input_file : str
        Path to a plain-text file listing the MSs to stack.  Each non-comment
        line must contain (whitespace-separated, in order)::

            <ms_path>  <redshift>  <RA>  <Dec>  [FIELD_ID]  [SPW_IDs]

        RA and Dec are parsed by astropy (sexagesimal or decimal).
        FIELD_ID defaults to 0.  SPW_IDs is a comma-separated list of
        spectral window indices (default: all).
    chunk_rows : int
        Number of baseline rows processed per chunk.  Larger values reduce
        Python overhead but increase peak memory usage.  An adaptive mechanism
        may lower this at runtime if the input channel count is very large.
    verbose : bool
        If True, print timestamped progress messages to stdout.
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
        _log(f"Reading metadata from {n_ms} MSs (SPECTRAL_WINDOW / FIELD / DATA_DESCRIPTION)...")

        def _load_one(arg):
            i, ms = arg
            req_field = self.field_list[i]
            req_spws  = self.spw_list[i]
            _log(f"  [{i+1}/{n_ms}]  {os.path.basename(ms)}")
            r = _read_meta_daskms(i, ms, req_field, req_spws)
            _log(f"  [{i+1}/{n_ms}]  done  spw={r['spws_used']}  "
                 f"{sum(len(f) for f in r['freq_per_spw'])} channels")
            return r

        with ThreadPoolExecutor(max_workers=min(4, n_ms)) as pool:
            meta_list = list(pool.map(_load_one, enumerate(self.ms_list)))
        _log("Metadata read. Computing per-MS output grids...")

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

            _log(f"  [{ms_idx+1}/{n_ms}]  {os.path.basename(self.ms_list[ms_idx])}  "
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
            _log(f"Common channel width (df_new) = {df_new/1e3:.3f} kHz "
                 f"(widest rest-framed channel of the sample)")
        else:
            df_new = float(channel_width_hz)
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
            _log(f"Common channel width (df_new) = {df_new/1e3:.3f} kHz "
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
            _log(f"Velocity range requested: [{v_lo:.0f}, {v_hi:.0f}] km/s "
                 f"(dv={dv_total:.0f} km/s)")
            _log(f"  -> bandwidth = {bw_vel/1e6:.3f} MHz / "
                 f"df_new={df_new/1e3:.3f} kHz => nn = {nn} channels")
        elif nchan_out is None:
            nn = max(nn_per_ms)
            _log(f"Output channels (auto): {nn}  "
                 f"[per-MS coverage: {min(nn_per_ms)}-{max(nn_per_ms)}]")
        else:
            nn = int(nchan_out)
            _log(f"Output channels (user-set): {nn}  "
                 f"[per-MS available: {min(nn_per_ms)}-{max(nn_per_ms)}]")

        if nn % 2 != 0:
            nn -= 1
        _log(f"Final output channels: {nn}")

        bandwidth    = nn * df_new
        start_ideal  = central_freq - bandwidth / 2.0
        end_ideal    = central_freq + bandwidth / 2.0
        _log(f"Target window: [{start_ideal/1e6:.6f}, {end_ideal/1e6:.6f}] MHz")

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
            _log(f"  [{ms_idx+1}/{n_ms}]  {os.path.basename(ms)}  "
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
            data_column="auto"):
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
        channel_width_hz : float or None
            Common rest-frame channel width, in Hz.  ``None`` uses the widest
            rest-framed channel of the sample, which is the finest grid every
            dataset supports; a finer value is refused.
        velocity_range_kms : float, tuple, or None
            If set, determines the output bandwidth from a velocity range
            (optical convention, km/s).  A scalar is interpreted as ±v;
            a 2-tuple as ``(v_lo, v_hi)``.  Overrides *nchan_out*.
        """
        global _verbose_global
        _verbose_global = self.verbose

        # --- Threading coherence ------------------------------------------
        import multiprocessing as _mp
        _slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
        if _slurm_cpus > 0:
            _phys_cores = _slurm_cpus
            _n_readers  = 3
        else:
            _phys_cores = max(1, _mp.cpu_count() // 2)
            _n_readers  = 1
        os.environ["OMP_NUM_THREADS"] = str(_phys_cores)
        _log(f"OMP_NUM_THREADS={_phys_cores}  reader_threads={_n_readers}  "
             f"GPU={'yes' if _gpu_available else 'no'}")
        dask.config.set(scheduler="synchronous")

        self.data_column = data_column
        t_start = time.perf_counter()

        # Determine working path (scratch) vs final destination
        ms_final = ms_out
        if scratch_dir is not None:
            os.makedirs(scratch_dir, exist_ok=True)
            ms_out = os.path.join(scratch_dir, os.path.basename(ms_final))
            _log(f"Writing to scratch: {ms_out}")
            _log(f"  Final destination: {ms_final}")

        # Remove any existing output at both locations
        for _p in ({ms_out, ms_final}):
            if os.path.exists(_p):
                _log(f"Removing existing output: {_p}")
                shutil.rmtree(_p)

        _log("Computing per-MS spectral grids...")
        freq_new_per_ms, df_new = self._compute_per_ms_grids(
            central_freq, nchan_out=nchan_out,
            velocity_range_kms=velocity_range_kms,
            channel_width_hz=channel_width_hz)
        nchan_new = len(freq_new_per_ms[0])

        t_setup = time.perf_counter()
        _log("Building ANTENNA table...")
        ant_ds, ant_offsets = _build_antenna_table(self.ms_list, self.z_list)
        _log(f"  {ant_ds.sizes['row']} antennas in combined table")

        _log("Building output subtables...")
        first_ms     = self.ms_list[0]
        spw_ds       = _build_spw_table(freq_new_per_ms, df_new, self.ms_list)
        field_ds     = _build_field_table(first_ms)
        dd_ds        = _build_dd_table(first_ms, len(self.ms_list))
        pol_ds       = _copy_subtable_first_row(first_ms, "POLARIZATION")
        obs_ds_list  = _build_observation_table(self.ms_list)
        hist_ds_list = []
        nant_total   = ant_ds.sizes['row']

        # Build a minimal FEED table covering all antennas
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

        def _write_sub(ds_or_list, subtable):
            """Write a subtable Dataset (or list of Datasets) to ms_out."""
            lst = ds_or_list if isinstance(ds_or_list, list) else [ds_or_list]
            if lst:
                with _casacore_lock:
                    dask.compute(*xds_to_table(lst, f"{ms_out}::{subtable}", "ALL"))
                _log(f"  Written: {subtable}")

        n_ms = len(self.ms_list)
        _log(f"Starting main loop: {n_ms} MSs to process...")
        total_rows  = 0
        total_bytes = 0
        t_loop_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Inner function: read one MS from disk (runs in reader thread)
        # ------------------------------------------------------------------

        def _read_ms(ms_idx, ms, z, ra_str, dec_str):
            """Read visibilities for one MS and return a processing package.

            When multiple SPWs are requested (e.g. ``spw=[16,18]``), their
            channels are concatenated in rest-frame frequency order before
            being passed as a single array to the C++ kernel.  This means
            the kernel sees ``N rows × (nchan_spw1 + nchan_spw2)`` channels
            instead of ``2N rows × nchan_spw1`` channels, which is both more
            correct (one row per baseline, full bandwidth) and twice as fast.

            SPW ordering
                The input list may specify SPWs in any order, and SPW
                frequencies may increase or decrease across the list (e.g.
                SPW 18 can cover lower frequencies than SPW 16).  We always
                sort SPWs by their rest-framed centre frequency before
                concatenating, so the merged frequency axis is monotonically
                increasing.

            Assumes rows are aligned across SPWs (same TIME / ANTENNA1 /
            ANTENNA2 in the same order), which is guaranteed for standard
            ALMA data where all SPWs are observed simultaneously on the same
            baselines.

            Returns
            -------
            dict
                A processing package containing lazy dask arrays for the
                visibilities and flags, plus materialised scalar columns.
                The heavy data is *not* materialised here — that happens
                one chunk at a time inside ``_process_ms`` to cap peak RAM.

            Raises
            ------
            RuntimeError
                If the MS contributes no valid rows.
            """
            _t_read_start = time.perf_counter()
            from daskms import xds_from_ms

            ant_offset     = ant_offsets[ms_idx]
            freq_old_list  = self._cache_freq_old[ms_idx]
            width_old_list = self._cache_width_old[ms_idx]
            ddids          = self._cache_ddids[ms_idx]
            req_field      = self.field_list[ms_idx]
            ra0_rad, dec0_rad = self._cache_phase_dir[ms_idx]
            ra1_rad, dec1_rad = _parse_ra_dec(ra_str, dec_str)
            ra1_rad = _wrap_dra(ra1_rad, ra0_rad)

            freq_new_ms = freq_new_per_ms[ms_idx]

            SCALAR_COLS = ["TIME","EXPOSURE","INTERVAL","TIME_CENTROID",
                           "SCAN_NUMBER","STATE_ID","ARRAY_ID","OBSERVATION_ID",
                           "FEED1","FEED2","PROCESSOR_ID","WEIGHT","SIGMA","FLAG_ROW"]

            n_covered_global = set()

            # Weight / sigma scaling factor: R = df_new / df_old_rf
            # Identical to the original CASA pipeline weight rescaling.
            df_old_rf_chan0 = float(abs(width_old_list[0][0])) * (1.0 + z)
            weight_scale = df_new / df_old_rf_chan0
            sigma_scale  = 1.0 / np.sqrt(weight_scale) if weight_scale > 0.0 else 1.0
            _log(f"  [{ms_idx+1}/{n_ms}] (read)  R = {weight_scale:.4f}  "
                 f"(df_new={df_new:.1f} Hz, df_old_rf={df_old_rf_chan0:.1f} Hz)")

            with _get_ms_lock(ms):
                # ── Build TaQL filter ────────────────────────────────────
                conditions = []
                if req_field is not None:
                    conditions.append(f"FIELD_ID=={int(req_field)}")
                # Automatically filter for OBSERVE_TARGET state if available
                try:
                    state_ds_list = xds_from_table(f"{ms}::STATE", group_cols=[])
                    if state_ds_list:
                        (obs_modes_arr,) = dask.compute(
                            state_ds_list[0].OBS_MODE.data,
                            scheduler="synchronous"
                        )
                        target_state_ids = [
                            i for i, m in enumerate(np.asarray(obs_modes_arr).ravel())
                            if "OBSERVE_TARGET" in str(m)
                        ]
                        if target_state_ids:
                            ids_str = ",".join(str(s) for s in target_state_ids)
                            conditions.append(f"STATE_ID IN [{ids_str}]")
                except Exception:
                    pass

                taql_where = " AND ".join(conditions)

                datasets = xds_from_ms(
                    ms,
                    group_cols=["DATA_DESC_ID"],
                    index_cols=[],
                    chunks={"row": self.chunk_rows},
                    taql_where=taql_where,
                )

                def _grp_val(ds, key):
                    v = ds.attrs.get(key, None) if hasattr(ds, "attrs") else None
                    if v is None:
                        v = getattr(ds, key, None)
                    return int(np.asarray(v).reshape(-1)[0])

                by_ddid = {_grp_val(ds, "DATA_DESC_ID"): ds for ds in datasets}

                # Sort SPWs by rest-framed centre frequency (ascending)
                # so the merged frequency axis is monotonically increasing.
                def _spw_centre_rf(spw_pos):
                    f = freq_old_list[spw_pos]
                    return float(f[len(f) // 2]) * (1.0 + z)

                spw_order = sorted(range(len(ddids)), key=_spw_centre_rf)

                # Accumulate lazy dask arrays (NOT materialised) per SPW.
                # Data is materialised one chunk at a time later to cap RAM.
                vis_da_per_spw  = []   # dask array (nrow, nchan_k, ncorr)
                flag_da_per_spw = []
                freq_per_spw    = []   # numpy (nchan_k,) — small, OK to materialise

                ref_ds          = None
                nrows           = 0

                # Pre-slicing: only read channels that overlap with the
                # output frequency window (± one channel margin).
                _f_lo_win = freq_new_ms[0] - df_new
                _f_hi_win = freq_new_ms[-1] + df_new

                for spw_pos in spw_order:
                    ddid        = ddids[spw_pos]
                    freq_old    = freq_old_list[spw_pos]
                    freq_old_rf = freq_old * (1.0 + z)

                    ds = by_ddid.get(int(ddid))
                    if ds is None:
                        continue
                    n = ds.sizes["row"]
                    if n == 0:
                        continue

                    # Which input channels fall within the output window?
                    useful = np.where((freq_old_rf >= _f_lo_win) & (freq_old_rf <= _f_hi_win))[0]
                    if len(useful) == 0:
                        continue
                    ch_lo, ch_hi = int(useful[0]), int(useful[-1]) + 1
                    freq_old_rf_sliced = freq_old_rf[ch_lo:ch_hi]

                    # Track which output channels this SPW can contribute to
                    _dfo_half = 0.5 * float(abs(width_old_list[spw_pos][0])) * (1.0 + z)
                    _in_lo = float(freq_old_rf_sliced.min()) - _dfo_half
                    _in_hi = float(freq_old_rf_sliced.max()) + _dfo_half
                    for j in range(len(freq_new_ms)):
                        f_lo = freq_new_ms[j] - 0.5 * df_new
                        f_hi = freq_new_ms[j] + 0.5 * df_new
                        if min(f_hi, _in_hi) > max(f_lo, _in_lo):
                            n_covered_global.add(j)

                    # 'auto' prefers CORRECTED_DATA, as a calibrated ALMA MS
                    # keeps the calibrated visibilities there; set
                    # self.data_column to read a named column instead.
                    _wanted = getattr(self, "data_column", "auto")
                    if _wanted != "auto" and _wanted in ds.data_vars:
                        _data_col = ds.data_vars[_wanted].data
                    elif _wanted != "auto":
                        _log(f"  WARNING: ddid={int(ddid)} has no {_wanted}, "
                             f"falling back")
                        _data_col = (ds.data_vars["DATA"].data
                                     if "DATA" in ds.data_vars else None)
                    elif "CORRECTED_DATA" in ds.data_vars:
                        _data_col = ds.data_vars["CORRECTED_DATA"].data
                    elif "DATA" in ds.data_vars:
                        _data_col = ds.data_vars["DATA"].data
                    else:
                        _log(f"  WARNING: ddid={int(ddid)} has neither DATA nor CORRECTED_DATA, skipping")
                        continue
                    vis_da_per_spw.append(
                        _data_col[:, ch_lo:ch_hi].astype(np.complex64))
                    flag_da_per_spw.append(ds.FLAG.data[:, ch_lo:ch_hi])
                    freq_per_spw.append(freq_old_rf_sliced)

                    if ref_ds is None:
                        ref_ds = ds
                        nrows  = n

            if not vis_da_per_spw or ref_ds is None:
                raise RuntimeError(
                    f"MS {os.path.basename(ms)} (idx={ms_idx}): no valid data read. "
                    f"vis_per_spw={len(vis_da_per_spw)}, ref_ds={'OK' if ref_ds else 'None'}. "
                    f"Check that field={req_field} and the requested SPWs exist in the MS."
                )

            # Sanity check: nrows must match across SPWs (before reading data)
            spw_nrows = [v.shape[0] for v in vis_da_per_spw]
            if len(set(spw_nrows)) > 1:
                ref_n     = spw_nrows[0]
                compat    = [(v, f, fq) for v, f, fq, n in
                             zip(vis_da_per_spw, flag_da_per_spw, freq_per_spw, spw_nrows)
                             if n == ref_n]
                n_dropped = len(vis_da_per_spw) - len(compat)
                _log(f"  WARNING: {n_dropped} SPWs dropped (incompatible nrows, "
                     f"ref={ref_n}).  Shapes: {spw_nrows}")
                if not compat:
                    raise RuntimeError(
                        f"MS {os.path.basename(ms)}: no compatible SPWs.  nrows={spw_nrows}"
                    )
                vis_da_per_spw, flag_da_per_spw, freq_per_spw = zip(*compat)
                nrows = ref_n

            freq_merged = np.concatenate(freq_per_spw)
            nchan_tot   = sum(v.shape[1] for v in vis_da_per_spw)
            _log(f"  [{ms_idx+1}/{n_ms}] (read)  SPWs merged: {len(vis_da_per_spw)} x "
                 f"{vis_da_per_spw[0].shape[1]} ch -> {nchan_tot} ch total")

            # Adaptive chunk sizing: target ~512 MB per chunk to avoid OOM
            # on MSs with very wide bandwidth.
            _TARGET_CHUNK_BYTES = 512 * 1024 * 1024
            ncorr_est   = 2
            _adaptive_chunk = max(
                1000,
                _TARGET_CHUNK_BYTES // max(1, nchan_tot * ncorr_est * 8)
            )
            _effective_chunk = min(self.chunk_rows, _adaptive_chunk)
            if _effective_chunk < self.chunk_rows:
                _log(f"  chunk_rows adapted: {self.chunk_rows} -> {_effective_chunk} "
                     f"(nchan_tot={nchan_tot}, target ~512 MB/chunk)")

            # Read scalar columns in one shot (small: 1D/3D vectors per nrow).
            # Split into "essential" (vary per row, written during processing)
            # and "constant" (written once at the end via casacore putcol).
            SCALAR_COLS_essential = ["TIME", "EXPOSURE", "INTERVAL",
                                     "WEIGHT", "SIGMA", "SCAN_NUMBER",
                                     "STATE_ID", "FLAG_ROW"]
            SCALAR_COLS_constant  = ["FEED1", "FEED2", "PROCESSOR_ID",
                                     "ARRAY_ID", "OBSERVATION_ID",
                                     "TIME_CENTROID"]
            with _get_ms_lock(ms):
                scal_da   = {c: ref_ds[c].data for c in SCALAR_COLS_essential
                             if c in ref_ds.data_vars}
                scal_keys = list(scal_da.keys())
                computed_scal = dask.compute(
                    ref_ds.UVW.data.astype(np.float64),
                    ref_ds.ANTENNA1.data,
                    ref_ds.ANTENNA2.data,
                    *[scal_da[c] for c in scal_keys],
                    scheduler="synchronous"
                )
            uvw_all  = np.asarray(computed_scal[0])
            ant1_all = np.asarray(computed_scal[1])
            ant2_all = np.asarray(computed_scal[2])
            scal_all = [np.asarray(computed_scal[3 + k]) for k in range(len(scal_keys))]

            # NOTE: vis/flag dask arrays are kept LAZY in the package.
            # Materialisation happens one chunk at a time inside _process_ms
            # to cap peak RAM at ~effective_chunk rows × nchan_tot × 2 corr × 8 bytes.

            _t_read_end = time.perf_counter()
            with _timer_read_lock:
                timers["read"] += _t_read_end - _t_read_start

            return {
                "ms_idx": ms_idx, "ms": ms, "z": z,
                "ra0_rad": ra0_rad, "dec0_rad": dec0_rad,
                "ra1_rad": ra1_rad, "dec1_rad": dec1_rad,
                "n_covered": len(n_covered_global),
                # Lazy dask arrays (not materialised)
                "vis_da_per_spw":  vis_da_per_spw,
                "flag_da_per_spw": flag_da_per_spw,
                "freq_merged":     freq_merged,
                "nchan_tot":       nchan_tot,
                "nrows":           nrows,
                "effective_chunk": _effective_chunk,
                # Materialised scalar columns (small: only 1D/3D per nrow)
                "uvw_all":   uvw_all,
                "ant1_all":  ant1_all,
                "ant2_all":  ant2_all,
                "scal_keys": scal_keys,
                "scal_all":  scal_all,
                "ant_offset":    ant_offset,
                "weight_scale":  weight_scale,
                "sigma_scale":   sigma_scale,
            }

        # ------------------------------------------------------------------
        # Inner function: apply C++ kernel to one MS (CPU path)
        # ------------------------------------------------------------------

        def _process_ms(pkg):
            """Apply the C++/OpenMP kernel to one MS package.

            Visibilities are materialised one chunk at a time from the lazy
            dask arrays stored in the package.  Each chunk is processed by
            ``ms_ops.full_pipeline()`` (rest-framing + phase shift + spectral
            rebinning in a single in-memory pass) and immediately wrapped
            in a dask-ms Dataset for writing.

            Peak RAM usage is capped at::

                effective_chunk × nchan_tot × ncorr × 8 bytes

            Returns
            -------
            ms_datasets : list of Dataset
                One Dataset per row chunk, ready for xds_to_table.
            nrows_ms : int
                Total rows processed for this MS.
            bytes_ms : int
                Total bytes of output data produced.
            """
            z         = pkg["z"]
            ra0_rad   = pkg["ra0_rad"];  dec0_rad = pkg["dec0_rad"]
            ra1_rad   = pkg["ra1_rad"];  dec1_rad = pkg["dec1_rad"]

            ms_name  = os.path.basename(pkg["ms"])
            ms_idx   = pkg["ms_idx"]
            freq_new_ms = freq_new_per_ms[ms_idx]
            nchan_ms    = len(freq_new_ms)

            _log(f"[{ms_idx+1}/{n_ms}]  {ms_name}  z={z:.5f}  "
                 f"coverage: {pkg['n_covered']}/{nchan_ms} channels")

            # Recover lazy dask arrays and materialised scalars from package
            vis_da_per_spw  = pkg["vis_da_per_spw"]
            flag_da_per_spw = pkg["flag_da_per_spw"]
            freq_merged     = pkg["freq_merged"]
            nchan_tot       = pkg["nchan_tot"]
            nrows           = pkg["nrows"]
            eff_chunk       = pkg["effective_chunk"]
            uvw_all         = pkg["uvw_all"]
            ant1_all        = pkg["ant1_all"]
            ant2_all        = pkg["ant2_all"]
            scal_keys       = pkg["scal_keys"]
            scal_all        = pkg["scal_all"]
            ant_offset      = pkg["ant_offset"]
            weight_scale    = pkg["weight_scale"]
            sigma_scale     = pkg["sigma_scale"]
            n_spw           = len(vis_da_per_spw)

            freq_merged_use = freq_merged
            if len(freq_merged_use) == 0:
                raise RuntimeError(f"MS {ms_name}: no input channels in output window")

            nrows_ms = 0
            bytes_ms = 0
            ms_datasets = []
            ds_idx = 0

            # Materialise one chunk at a time: peak RAM = 1 chunk × nchan_tot
            for row0 in range(0, nrows, eff_chunk):
                row1 = min(row0 + eff_chunk, nrows)

                with _casacore_lock:
                    chunk_computed = dask.compute(
                        *[v[row0:row1] for v in vis_da_per_spw],
                        *[f[row0:row1] for f in flag_da_per_spw],
                        scheduler="synchronous"
                    )
                vis_chunks  = [np.asarray(chunk_computed[i])       for i in range(n_spw)]
                flag_chunks = [np.asarray(chunk_computed[n_spw+i]) for i in range(n_spw)]

                if n_spw == 1:
                    vis_merged_chunk  = vis_chunks[0]
                    flag_merged_chunk = flag_chunks[0]
                else:
                    vis_merged_chunk  = np.concatenate(vis_chunks,  axis=1)
                    flag_merged_chunk = np.concatenate(flag_chunks, axis=1)

                ant1 = ant1_all[row0:row1] + ant_offset
                ant2 = ant2_all[row0:row1] + ant_offset
                scalars = {}
                for c, arr in zip(scal_keys, scal_all):
                    sl   = arr[row0:row1]
                    dims = ("row", "corr") if sl.ndim == 2 else ("row",)
                    scalars[c] = (dims, sl)

                t_proc = time.perf_counter()

                vis_out, flag_out, uvw_out = ms_ops.full_pipeline(
                    vis_merged_chunk, flag_merged_chunk,
                    uvw_all[row0:row1],
                    z, freq_merged_use,
                    ra0_rad, dec0_rad,
                    ra1_rad, dec1_rad,
                    freq_new_ms,
                )

                # Free input buffers immediately
                del vis_merged_chunk, flag_merged_chunk, vis_chunks, flag_chunks, chunk_computed

                nchan_out_k = vis_out.shape[1]
                _log(f"  [{ms_idx+1}/{n_ms}] (proc) chunk {ds_idx}: "
                     f"{vis_out.shape[0]}r x {nchan_tot}ch  R={weight_scale:.4f}  "
                     f"pipeline [{time.perf_counter()-t_proc:.1f}s] -> "
                     f"{nchan_out_k}/{nchan_ms}ch")

                nrow  = vis_out.shape[0]
                ncorr = vis_out.shape[2]
                nrows_ms += nrow
                bytes_ms += vis_out.nbytes + flag_out.nbytes + uvw_out.nbytes

                row_ch = _make_row_chunks(nrow, eff_chunk)
                ch_3d  = (row_ch, (nchan_ms,), (ncorr,))
                ch_uvw = (row_ch, (3,))
                ch_1d  = (row_ch,)

                data_vars = {
                    "DATA":         (("row","chan","corr"), _to_dask(vis_out,  ch_3d)),
                    "FLAG":         (("row","chan","corr"), _to_dask(flag_out, ch_3d)),
                    "UVW":          (("row","uvw"),         _to_dask(uvw_out,  ch_uvw)),
                    "ANTENNA1":     (("row",),              _to_dask(ant1,     ch_1d)),
                    "ANTENNA2":     (("row",),              _to_dask(ant2,     ch_1d)),
                    "DATA_DESC_ID": (("row",),              _to_dask(np.full(nrow, ms_idx, dtype=np.int32), ch_1d)),
                    "FIELD_ID":     (("row",),              _to_dask(np.zeros(nrow, dtype=np.int32), ch_1d)),
                }

                for col, (dims, arr) in scalars.items():
                    if col == "WEIGHT":
                        arr = arr * weight_scale
                    elif col == "SIGMA":
                        arr = arr * sigma_scale
                    if arr.ndim == 1:   ch = ch_1d
                    elif arr.ndim == 2: ch = (row_ch, (arr.shape[1],))
                    elif arr.ndim == 3: ch = ch_3d
                    else:               ch = None
                    data_vars[col] = (dims, _to_dask(arr, ch))

                ms_datasets.append(Dataset(data_vars))
                ds_idx += 1

            if not ms_datasets:
                raise RuntimeError(
                    f"MS {ms_name} (idx={ms_idx}): no chunks generated.  nrows={nrows}"
                )

            return ms_datasets, nrows_ms, bytes_ms

        # ------------------------------------------------------------------
        # Inner function: GPU batch processing
        # ------------------------------------------------------------------

        def _process_ms_gpu_batch(pkgs):
            """Process a batch of MS packages on the GPU in a single kernel launch.

            Packs all MSs in *pkgs* into the argument lists expected by
            ``ms_ops.full_pipeline_batch()``, which processes them in one
            CUDA kernel launch.  The results are then unpacked and wrapped
            in dask-ms Datasets for writing.

            On CPU (or for single-MS batches), falls through to the
            sequential ``_process_ms`` path.

            Parameters
            ----------
            pkgs : list of dict
                Processing packages as returned by ``_read_ms``.

            Returns
            -------
            list of (ms_datasets, nrows_ms, bytes_ms)
                One tuple per input package.
            """
            results_out = []

            if not pkgs:
                return results_out

            # For GPU batch we need to materialise all data for the batch
            # first, then launch a single kernel.
            t_ms = time.perf_counter()

            # Materialise all lazy dask arrays for every package in the batch
            materialised = []
            for pkg in pkgs:
                vis_da_per_spw  = pkg["vis_da_per_spw"]
                flag_da_per_spw = pkg["flag_da_per_spw"]
                n_spw           = len(vis_da_per_spw)
                nrows           = pkg["nrows"]

                with _casacore_lock:
                    chunk_computed = dask.compute(
                        *[v[:nrows] for v in vis_da_per_spw],
                        *[f[:nrows] for f in flag_da_per_spw],
                        scheduler="synchronous"
                    )
                vis_chunks  = [np.asarray(chunk_computed[i])       for i in range(n_spw)]
                flag_chunks = [np.asarray(chunk_computed[n_spw+i]) for i in range(n_spw)]

                if n_spw == 1:
                    vis_all  = vis_chunks[0]
                    flag_all = flag_chunks[0]
                else:
                    vis_all  = np.concatenate(vis_chunks,  axis=1)
                    flag_all = np.concatenate(flag_chunks, axis=1)

                materialised.append((vis_all, flag_all))

            # Build argument lists for the batch kernel
            vis_list      = [m[0] for m in materialised]
            flag_list     = [m[1] for m in materialised]
            uvw_list      = [pkg["uvw_all"]  for pkg in pkgs]
            z_list        = [pkg["z"]        for pkg in pkgs]
            freq_old_list = [pkg["freq_merged"] for pkg in pkgs]
            ra_old_list   = [pkg["ra0_rad"]  for pkg in pkgs]
            dec_old_list  = [pkg["dec0_rad"] for pkg in pkgs]
            ra_new_list   = [pkg["ra1_rad"]  for pkg in pkgs]
            dec_new_list  = [pkg["dec1_rad"] for pkg in pkgs]
            freq_new_list = [freq_new_per_ms[pkg["ms_idx"]] for pkg in pkgs]

            gpu_results = ms_ops.full_pipeline_batch(
                vis_list, flag_list, uvw_list,
                z_list, freq_old_list,
                ra_old_list, dec_old_list,
                ra_new_list, dec_new_list,
                freq_new_list,
            )
            dt_batch = time.perf_counter() - t_ms
            _log(f"  GPU batch: {len(pkgs)} MSs processed in {dt_batch:.1f}s")

            # Free materialised input buffers
            del materialised, vis_list, flag_list

            # Unpack results and build Datasets
            for pkg, (vis_out, flag_out, uvw_out) in zip(pkgs, gpu_results):
                ms_idx       = pkg["ms_idx"]
                nchan_ms     = len(freq_new_per_ms[ms_idx])
                ant_offset   = pkg["ant_offset"]
                weight_scale = pkg["weight_scale"]
                sigma_scale  = pkg["sigma_scale"]
                ant1_all     = pkg["ant1_all"]
                ant2_all     = pkg["ant2_all"]
                scal_keys    = pkg["scal_keys"]
                scal_all     = pkg["scal_all"]
                nrows        = pkg["nrows"]
                eff_chunk    = pkg["effective_chunk"]

                nrow  = vis_out.shape[0]
                ncorr = vis_out.shape[2]
                nrows_ms = nrow
                bytes_ms = vis_out.nbytes + flag_out.nbytes + uvw_out.nbytes

                ant1 = ant1_all[:nrow] + ant_offset
                ant2 = ant2_all[:nrow] + ant_offset
                scalars = {}
                for c, arr in zip(scal_keys, scal_all):
                    sl   = arr[:nrow]
                    dims = ("row", "corr") if sl.ndim == 2 else ("row",)
                    scalars[c] = (dims, sl)

                row_ch = _make_row_chunks(nrow, eff_chunk)
                ch_3d  = (row_ch, (nchan_ms,), (ncorr,))
                ch_uvw = (row_ch, (3,))
                ch_1d  = (row_ch,)

                data_vars = {
                    "DATA":         (("row","chan","corr"), _to_dask(vis_out,  ch_3d)),
                    "FLAG":         (("row","chan","corr"), _to_dask(flag_out, ch_3d)),
                    "UVW":          (("row","uvw"),         _to_dask(uvw_out,  ch_uvw)),
                    "ANTENNA1":     (("row",),              _to_dask(ant1,     ch_1d)),
                    "ANTENNA2":     (("row",),              _to_dask(ant2,     ch_1d)),
                    "DATA_DESC_ID": (("row",),              _to_dask(np.full(nrow, ms_idx, dtype=np.int32), ch_1d)),
                    "FIELD_ID":     (("row",),              _to_dask(np.zeros(nrow, dtype=np.int32), ch_1d)),
                }
                for col, (dims, arr) in scalars.items():
                    if col == "WEIGHT":
                        arr = arr * weight_scale
                    elif col == "SIGMA":
                        arr = arr * sigma_scale
                    if arr.ndim == 1:   ch = ch_1d
                    elif arr.ndim == 2: ch = (row_ch, (arr.shape[1],))
                    elif arr.ndim == 3: ch = ch_3d
                    else:               ch = None
                    data_vars[col] = (dims, _to_dask(arr, ch))

                ms_datasets = [Dataset(data_vars)]
                results_out.append((ms_datasets, nrows_ms, bytes_ms))

            return results_out

        # ------------------------------------------------------------------
        # Inner function: fill constant columns once after all rows are written
        # ------------------------------------------------------------------

        def _fill_constant_cols(ms_out, total_rows):
            """Write constant scalar columns (FEED1, FEED2, PROCESSOR_ID,
            ARRAY_ID, OBSERVATION_ID, TIME_CENTROID) in a single putcol per
            column — much faster than writing them row-chunk by row-chunk.
            """
            try:
                from casacore.tables import table as _cctable
                with _cctable(ms_out, readonly=False, ack=False) as t:
                    nrows = t.nrows()
                    existing = set(t.colnames())
                    zeros_i32 = np.zeros(nrows, dtype=np.int32)
                    zeros_f64 = np.zeros(nrows, dtype=np.float64)
                    fills = {
                        "FEED1":           zeros_i32,
                        "FEED2":           zeros_i32,
                        "PROCESSOR_ID":    zeros_i32,
                        "ARRAY_ID":        zeros_i32,
                        "OBSERVATION_ID":  zeros_i32,
                        "TIME_CENTROID":   zeros_f64,
                    }
                    for col, arr in fills.items():
                        if col not in existing:
                            continue
                        t.putcol(col, arr)
                _log(f"  Constant columns filled ({total_rows} rows)")
            except Exception as e:
                _log(f"  WARNING: _fill_constant_cols failed: {e}")

        # ------------------------------------------------------------------
        # Inner function: write one MS to disk (runs in writer thread)
        # ------------------------------------------------------------------

        def _write_ms(ms_datasets, nrows_ms, bytes_ms, ms_out):
            """Write the processed datasets for one MS to the output table.

            Uses ``xds_to_table`` (dask-ms) to write all columns, appending
            rows to the existing MAIN table.  CTDS access is serialised via
            ``_casacore_lock``.
            """
            if not ms_datasets:
                return
            t_write = time.perf_counter()
            _log(f"  Writing {nrows_ms:,} rows ({_fmt_size(bytes_ms)}) to disk...")
            with _casacore_lock:
                dask.compute(*xds_to_table(ms_datasets, ms_out, "ALL", descriptor="ms"))
            dt = time.perf_counter() - t_write
            throughput = _fmt_size(bytes_ms / dt) if dt > 0 else "n/a"
            _log(f"  Done in {dt:.1f}s  ({throughput}/s)")

        # ------------------------------------------------------------------
        # Memory-aware queue sizing
        # ------------------------------------------------------------------
        import psutil as _psutil

        _slurm_mem_mb = int(os.environ.get("SLURM_MEM_PER_NODE", 0))
        if _slurm_mem_mb > 0:
            _ram_usable_gb = _slurm_mem_mb / 1024.0 * 0.80
        else:
            _ram_usable_gb = _psutil.virtual_memory().available / 1e9 * 0.80

        # Estimate size of one MS package in RAM using INPUT channel count
        # (not output nchan_new, which is much smaller and would overestimate
        # how many MS packages fit in RAM simultaneously).
        _nrow_est   = 10_000
        _ncorr_est  = 2
        _nchan_input_wc = max(
            sum(len(f) for f in self._cache_freq_old.get(i, [[]]))
            for i in range(n_ms)
        ) if n_ms > 0 else nchan_new
        # 8 bytes/vis (complex64) + 1 byte/flag + ~20% overhead
        _ms_size_gb = max(0.05, _nchan_input_wc * _nrow_est * _ncorr_est * 10 / 1e9)

        _buf_by_ram = max(1, int(_ram_usable_gb / _ms_size_gb / 2))
        READ_BUF    = min(n_ms, max(2, _buf_by_ram))
        # Small WRITE_BUF = backpressure: the compute thread can run ahead by
        # at most 2 MSs while the writer writes, preventing the entire dataset
        # from accumulating in RAM.
        WRITE_BUF   = 2
        _log(f"  Available RAM: {_ram_usable_gb:.1f} GB  |  "
             f"estimated MS size (input worst-case): {_ms_size_gb*1e3:.0f} MB  "
             f"|  nchan_input_wc: {_nchan_input_wc}  "
             f"|  read buffer: {READ_BUF}  |  write buffer: {WRITE_BUF}")

        # ------------------------------------------------------------------
        # Thread timing counters
        # ------------------------------------------------------------------
        timers = {
            "read":       0.0,
            "read_wait":  0.0,
            "compute":    0.0,
            "write_wait": 0.0,
            "write":      0.0,
        }
        _timer_read_lock = threading.Lock()

        read_q  = Queue(maxsize=READ_BUF)
        write_q = Queue(maxsize=WRITE_BUF)
        read_exc  = [None]
        write_exc = [None]

        # Each entry is processed independently — grouping caused aliasing
        # of lazy dask arrays when the same MS had different z values.
        ms_groups = [
            (ms, [(_i, _z, _ra, _dec)])
            for _i, (ms, _z, _ra, _dec) in enumerate(zip(
                self.ms_list, self.z_list, self.ra_list, self.dec_list))
        ]

        def _read_ms_group(ms_path, entries):
            """Read one or more entries from a single MS file.

            Each entry corresponds to one (ms, z, ra, dec) tuple from the
            input list.  Failed entries are logged and skipped; if all entries
            for an MS fail, an empty list is returned.
            """
            pkgs = []
            for (ms_idx, z, ra_str, dec_str) in entries:
                try:
                    pkg = _read_ms(ms_idx, ms_path, z, ra_str, dec_str)
                    pkgs.append(pkg)
                except RuntimeError as e:
                    _log(f"  [{ms_idx+1}/{n_ms}] WARNING: entry skipped: {e}")
            if not pkgs:
                _log(f"  WARNING: MS {os.path.basename(ms_path)} skipped "
                     f"(all {len(entries)} entries failed) — continuing")
                return []
            return pkgs

        def _reader_thread():
            """Read all MSs using a thread pool and push packages onto read_q.

            Multiple reader workers (``_n_readers``) are used on SLURM to
            overlap filesystem latency; on a laptop, a single reader is used.
            """
            from concurrent.futures import ThreadPoolExecutor, as_completed
            try:
                with ThreadPoolExecutor(max_workers=_n_readers) as pool:
                    futures = {}
                    idx = 0
                    while idx < len(ms_groups) and len(futures) < _n_readers:
                        _ms_path, _entries = ms_groups[idx]
                        f = pool.submit(_read_ms_group, _ms_path, _entries)
                        futures[f] = idx; idx += 1
                    while futures:
                        for f in as_completed(futures):
                            del futures[f]
                            exc = f.exception()
                            if exc is not None:
                                read_exc[0] = exc
                                read_q.put(None)
                                return
                            pkgs = f.result()
                            if not pkgs:
                                # All entries failed — submit next MS
                                if idx < len(ms_groups):
                                    _ms_path2, _entries2 = ms_groups[idx]
                                    nf = pool.submit(_read_ms_group, _ms_path2, _entries2)
                                    futures[nf] = idx; idx += 1
                                break
                            for pkg in pkgs:
                                read_q.put(pkg)
                            if idx < len(ms_groups):
                                _ms_path2, _entries2 = ms_groups[idx]
                                nf = pool.submit(_read_ms_group, _ms_path2, _entries2)
                                futures[nf] = idx; idx += 1
                            break
            except Exception as e:
                read_exc[0] = e
            finally:
                read_q.put(None)

        def _writer_thread():
            """Pop processed packages from write_q and write them to disk."""
            try:
                while True:
                    _tw = time.perf_counter()
                    item = write_q.get()
                    timers["write_wait"] += time.perf_counter() - _tw
                    if item is None:
                        break
                    ms_datasets, nrows_ms, bytes_ms = item
                    _t = time.perf_counter()
                    _write_ms(ms_datasets, nrows_ms, bytes_ms, ms_out)
                    timers["write"] += time.perf_counter() - _t
            except Exception as e:
                write_exc[0] = e

        reader = threading.Thread(target=_reader_thread, daemon=True)
        writer = threading.Thread(target=_writer_thread, daemon=True)
        reader.start()
        writer.start()

        # ------------------------------------------------------------------
        # GPU batch configuration
        # ------------------------------------------------------------------
        # On GPU: accumulate up to GPU_BATCH MS packages and process them in
        # a single CUDA kernel launch via ms_ops.full_pipeline_batch().
        # On CPU: GPU_BATCH=1, fall through to the sequential _process_ms path.
        GPU_BATCH  = 20 if _gpu_available else 1
        processed  = 0
        batch_pkgs = []

        def _flush_batch(pkgs):
            """Process a batch of MS packages and push results to write_q.

            On GPU: packs all MSs in the batch into a single kernel launch
            (``ms_ops.full_pipeline_batch``), then unpacks the results.
            On CPU: processes each MS sequentially with ``ms_ops.full_pipeline``.
            """
            nonlocal processed, total_rows, total_bytes

            if not pkgs:
                return

            t_batch = time.perf_counter()

            if GPU_BATCH > 1 and len(pkgs) > 1:
                # ── GPU path: batch dispatch ──
                batch_results = _process_ms_gpu_batch(pkgs)
                timers["compute"] += time.perf_counter() - t_batch

                for (ms_datasets, nrows_ms, bytes_ms) in batch_results:
                    total_rows  += nrows_ms
                    total_bytes += bytes_ms
                    write_q.put((ms_datasets, nrows_ms, bytes_ms))
                    processed += 1
            else:
                # ── CPU path: one MS at a time ──
                for pkg in pkgs:
                    t_one = time.perf_counter()
                    ms_datasets, nrows_ms, bytes_ms = _process_ms(pkg)
                    timers["compute"] += time.perf_counter() - t_one
                    total_rows  += nrows_ms
                    total_bytes += bytes_ms
                    write_q.put((ms_datasets, nrows_ms, bytes_ms))
                    processed += 1

            # Progress report
            elapsed   = time.perf_counter() - t_loop_start
            done_frac = processed / n_ms
            if done_frac > 0:
                eta = elapsed / done_frac * (1 - done_frac)
                _log(f"  Progress: {processed}/{n_ms} MSs  |  ETA: {eta/60:.1f} min")
                _flush_log_unlocked()

        # ------------------------------------------------------------------
        # Main loop: drain read_q and dispatch to _flush_batch
        # ------------------------------------------------------------------
        while True:
            _trw = time.perf_counter()
            pkg = read_q.get()
            timers["read_wait"] += time.perf_counter() - _trw

            if pkg is None:
                # Reader finished — check for errors, flush remaining batch
                if read_exc[0]:
                    raise RuntimeError(
                        f"Reader thread failed: {read_exc[0]}"
                    ) from read_exc[0]
                _flush_batch(batch_pkgs)
                batch_pkgs = []
                break

            if read_exc[0]:
                raise RuntimeError(f"Reader thread failed: {read_exc[0]}") from read_exc[0]

            batch_pkgs.append(pkg)
            if len(batch_pkgs) >= GPU_BATCH:
                _flush_batch(batch_pkgs)
                batch_pkgs = []

        write_q.put(None)   # sentinel to shut down the writer thread
        writer.join()
        if write_exc[0]:
            raise RuntimeError(f"Writer thread failed: {write_exc[0]}") from write_exc[0]

        # Fill constant columns once (much faster than per-chunk)
        _fill_constant_cols(ms_out, total_rows)

        _flush_log_unlocked()
        if not os.path.isdir(ms_out):
            raise RuntimeError(
                f"Output table '{ms_out}' was not created: probably no rows "
                f"were written (all MSs skipped?).  Check the log for spectral "
                f"coverage and field/spw selection errors."
            )

        # ------------------------------------------------------------------
        # Write output subtables
        # ------------------------------------------------------------------
        _log(f"Writing output subtables to {ms_out} ...")
        _write_sub(spw_ds,       "SPECTRAL_WINDOW")
        _write_sub(field_ds,     "FIELD")
        _write_sub(dd_ds,        "DATA_DESCRIPTION")
        _write_sub(ant_ds,       "ANTENNA")
        if pol_ds:       _write_sub(pol_ds,       "POLARIZATION")
        if obs_ds_list:  _write_sub(obs_ds_list,  "OBSERVATION")
        if hist_ds_list: _write_sub(hist_ds_list, "HISTORY")
        if feed_ds_list: _write_sub(feed_ds_list, "FEED")
        if src_ds_list:  _write_sub(src_ds_list,  "SOURCE")

        # ------------------------------------------------------------------
        # Optional staging: move from scratch to final destination
        # ------------------------------------------------------------------
        stage_dt = 0.0
        if scratch_dir is not None and ms_out != ms_final:
            t_stage = time.perf_counter()
            _log(f"Moving completed MS from scratch to final destination...")
            _log(f"  {ms_out}  ->  {ms_final}")
            shutil.move(ms_out, ms_final)
            stage_dt = time.perf_counter() - t_stage
            throughput = _fmt_size(total_bytes / stage_dt) if stage_dt > 0 else "n/a"
            _log(f"  Moved in {stage_dt:.1f}s  ({throughput}/s)")

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        t_total = time.perf_counter() - t_start
        _log("=" * 60)
        _log("Timing profile (cumulative per stage; threads run concurrently,")
        _log("so the sum may exceed the wall-clock time):")
        _log(f"  setup       (main thread)   : {t_loop_start - t_setup:7.1f}s   (antenna + subtables)")
        _log(f"  read        (reader thread) : {timers['read']:7.1f}s")
        _log(f"  read_wait   (main thread)   : {timers['read_wait']:7.1f}s"
             f"   <- near zero means compute never waits for data")
        _log(f"  compute     (main thread)   : {timers['compute']:7.1f}s"
             f"   ({'GPU' if _gpu_available else 'CPU'})")
        _log(f"  write_wait  (writer thread) : {timers['write_wait']:7.1f}s"
             f"   <- near zero means writer is always busy (I/O bound)")
        _eff = total_bytes / timers['write'] if timers['write'] > 0 else 0
        _log(f"  write       (writer thread) : {timers['write']:7.1f}s"
             f"   ({_fmt_size(_eff)}/s)")
        if scratch_dir is not None:
            _eff_s = total_bytes / stage_dt if stage_dt > 0 else 0
            _log(f"  staging     (move)          : {stage_dt:7.1f}s"
                 f"   ({_fmt_size(_eff_s)}/s) scratch -> final")
        _log("=" * 60)
        _log(f"Output MS        : {ms_final}")
        _log(f"Total rows       : {total_rows:,}")
        _log(f"Spectral windows : {n_ms}  (one per input MS, "
             f"{nchan_new} channels, df={df_new/1e3:.3f} kHz)")
        _log(f"Central frequency: {central_freq/1e6:.6f} MHz")
        _log(f"Antennas         : {ant_ds.sizes['row']}")
        _log(f"Data written     : {_fmt_size(total_bytes)}")
        _log(f"Wall-clock time  : {t_total/60:.1f} min")
        _log("=" * 60)
        _flush_log_unlocked()
