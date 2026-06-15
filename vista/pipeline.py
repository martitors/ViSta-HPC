"""
vista/pipeline.py
=================

Core processing pipeline for ViSta (Visibility Stacking tool).

The pipeline takes a list of interferometric Measurement Sets (MSs) at
different redshifts and combines them in the visibility (uv) plane by:

  1. Rest-framing each dataset: baseline vectors are scaled by 1/(1+z)
     and each spectral axis is shifted to the common rest-frame grid.
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
import shutil
import math
import time
import threading
from queue import Queue
from collections import Counter

import numpy as np
import dask
import dask.array as da
from daskms import xds_from_ms, xds_from_table, xds_to_table, Dataset

import ms_ops

# ---------------------------------------------------------------------------
# GPU availability check
# ---------------------------------------------------------------------------
# We try to call cuda_available() exposed by the CUDA-enabled module.
# If the module was compiled without CUDA or no device is present, we fall
# back to CPU-only mode transparently.
try:
    import ctypes
    _lib = ctypes.CDLL(ms_ops.__file__)
    _gpu_available = bool(_lib.cuda_available())
except Exception:
    _gpu_available = hasattr(ms_ops, 'full_pipeline_batch')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_C_LIGHT_KMS = 299792.458  # speed of light in km/s

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
# All log messages are buffered and flushed together to avoid interleaved
# output from concurrent threads.

_log_lock   = threading.Lock()
_log_buffer = []
_t0_global  = time.perf_counter()
_verbose_global = True


def _log(msg, flush=False):
    """Append a timestamped message to the log buffer.

    Messages are only printed if verbose mode is active. The timestamp is
    relative to the start of the current run (seconds elapsed).
    """
    if not _verbose_global:
        return
    ts = time.perf_counter() - _t0_global
    line = f"[ViSta  {ts:7.2f}s]  {msg}"
    with _log_lock:
        _log_buffer.append(line)
        if flush:
            _flush_log()


def _flush_log():
    """Flush the log buffer to stdout (must be called with _log_lock held)."""
    if _log_buffer:
        import sys
        sys.stdout.write("\n".join(_log_buffer) + "\n")
        sys.stdout.flush()
        _log_buffer.clear()


def _flush_log_unlocked():
    """Flush the log buffer to stdout, acquiring the lock internally."""
    with _log_lock:
        _flush_log()


# ---------------------------------------------------------------------------
# Coordinate / frequency helpers
# ---------------------------------------------------------------------------

def _vrange_to_freqrange(restfreq_hz, vmin_kms, vmax_kms):
    """Convert an optical velocity range [vmin, vmax] (km/s) to a frequency
    range in Hz using the optical convention: v = c * (f_rest - f) / f_rest.

    Returns (freq_lo_hz, freq_hi_hz) with freq_lo < freq_hi.
    """
    f_at_vmax = restfreq_hz * (1.0 - vmax_kms / _C_LIGHT_KMS)
    f_at_vmin = restfreq_hz * (1.0 - vmin_kms / _C_LIGHT_KMS)
    return min(f_at_vmax, f_at_vmin), max(f_at_vmax, f_at_vmin)


def _wrap_dra(ra_new, ra_old):
    """Wrap a right ascension difference to the range [-pi, pi] around ra_old.

    This avoids discontinuities at the 0/2*pi boundary when computing phase
    shifts between two pointing directions.
    """
    dra = (ra_new - ra_old + math.pi) % (2.0 * math.pi) - math.pi
    return ra_old + dra


def _parse_ra_dec(ra_str, dec_str):
    """Parse RA/Dec strings (sexagesimal or decimal) and return radians.

    Accepts formats understood by astropy.coordinates.SkyCoord, e.g.:
      ra_str  : '12:34:56.7'  or  '12h34m56.7s'
      dec_str : '+12:34:56.7' or  '12.582d'
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    # Handle Dec strings using '.' as separator instead of ':'
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
    from multiple MSs that may share antenna names (e.g. 'DA41').

    Example: ['DA41', 'DA41', 'DV01'] -> ['DA41_0', 'DA41_1', 'DV01']
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
    """Wrap a NumPy array in a Dask array with a unique name.

    Using a unique name prevents Dask from accidentally sharing computation
    graphs between arrays that happen to have the same shape/dtype.
    """
    import uuid
    name = "array-" + uuid.uuid4().hex[:8]
    if chunks is None:
        return da.from_array(arr, chunks=arr.shape, name=name)
    return da.from_array(arr, chunks=chunks, name=name)


def _make_row_chunks(nrow, chunk_size):
    """Return a tuple of chunk sizes that partition nrow rows into blocks of
    at most chunk_size, with the remainder as the last (smaller) block.
    """
    n_full = nrow // chunk_size
    n_rem  = nrow %  chunk_size
    return (chunk_size,) * n_full + ((n_rem,) if n_rem else ())


def _fmt_size(nbytes):
    """Format a byte count as a human-readable string (B, KB, MB, GB, TB)."""
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

def _build_antenna_table(ms_list, z_list):
    """Build the ANTENNA subtable for the output MS.

    Antenna positions from all input MSs are concatenated into a single table.
    Each position is rescaled by 1/(1+z) to map it to the rest-frame spatial
    frequency plane, exactly as done for the UVW coordinates in the kernel.
    Each input MS is assigned a contiguous block of antenna IDs; the offsets
    are returned so that ANTENNA1/ANTENNA2 columns can be remapped correctly.

    Returns
    -------
    ds : Dataset
        Combined ANTENNA table as a dask-ms Dataset.
    ant_offsets : list of int
        Starting antenna index for each input MS.
    """
    ant_offsets = []
    offset = 0
    all_names, all_positions, all_diameters = [], [], []
    all_mounts, all_stations = [], []

    for ms, z in zip(ms_list, z_list):
        inv = 1.0 / (1.0 + z)
        rows = xds_from_table(f"{ms}::ANTENNA", group_cols=[])
        ds   = rows[0] if rows else None

        if ds is None:
            ant_offsets.append(offset)
            continue

        def get_col(ds, name):
            if name in ds.data_vars:
                return ds.data_vars[name].data
            return None

        computed = dask.compute(
            get_col(ds, "NAME"),
            get_col(ds, "POSITION"),
            get_col(ds, "DISH_DIAMETER"),
            get_col(ds, "MOUNT"),
            get_col(ds, "STATION"),
        )
        names_arr, pos_arr_raw, diam_arr_raw, mounts_arr, stations_arr = computed

        nant = ds.sizes.get("row", 0)
        ant_offsets.append(offset)
        offset += nant

        # Antenna names
        if names_arr is not None:
            for v in np.asarray(names_arr).ravel():
                all_names.append(str(v))
        else:
            all_names.extend([""] * nant)

        # Positions rescaled by 1/(1+z)
        if pos_arr_raw is not None:
            pos = np.asarray(pos_arr_raw, dtype=np.float64)
            if pos.shape[0] == 3 and pos.ndim == 2:
                pos = pos.T
            all_positions.extend(pos * inv)
        else:
            all_positions.extend(np.zeros((nant, 3)))

        # Dish diameters rescaled by 1/(1+z)
        if diam_arr_raw is not None:
            diam = np.asarray(diam_arr_raw, dtype=np.float64).ravel()
            all_diameters.extend(diam * inv)
        else:
            all_diameters.extend([12.0 * inv] * nant)

        if mounts_arr is not None:
            for v in np.asarray(mounts_arr).ravel():
                all_mounts.append(str(v))
        else:
            all_mounts.extend(["ALT-AZ"] * nant)

        if stations_arr is not None:
            for v in np.asarray(stations_arr).ravel():
                all_stations.append(str(v))
        else:
            all_stations.extend([""] * nant)

    # Ensure antenna names are unique across the combined table
    all_names = _rename_duplicates(all_names)
    nant_tot  = len(all_names)
    pos_arr   = np.array(all_positions,  dtype=np.float64).reshape(nant_tot, 3)
    diam_arr  = np.array(all_diameters,  dtype=np.float64)

    ds = Dataset({
        "NAME":          (("row",),       _to_dask(np.array(all_names,    dtype=object))),
        "POSITION":      (("row", "xyz"), _to_dask(pos_arr)),
        "DISH_DIAMETER": (("row",),       _to_dask(diam_arr)),
        "MOUNT":         (("row",),       _to_dask(np.array(all_mounts,   dtype=object))),
        "STATION":       (("row",),       _to_dask(np.array(all_stations, dtype=object))),
        "FLAG_ROW":      (("row",),       _to_dask(np.zeros(nant_tot, dtype=bool))),
        "OFFSET":        (("row", "xyz"), _to_dask(np.zeros((nant_tot, 3), dtype=np.float64))),
        "TYPE":          (("row",),       _to_dask(np.array(["GROUND-BASED"]*nant_tot, dtype=object))),
    })
    return ds, ant_offsets


def _build_spw_table(freq_list, df_new, ms_list):
    """Build the SPECTRAL_WINDOW subtable for the output MS.

    One spectral window is created per input MS, each with its own output
    frequency grid (which may have a different start but always the same
    channel width df_new). The windows are kept separate rather than merged
    so that each input MS can be identified by its DATA_DESC_ID in the output.

    Parameters
    ----------
    freq_list : list of ndarray
        Per-MS output channel centre frequencies (Hz), one array per MS.
    df_new : float
        Common output channel width (Hz).
    ms_list : list of str
        Input MS paths (used only to generate SPW names).
    """
    num_ms    = len(ms_list)
    nchan_per_spw = [len(f) for f in freq_list]
    nchan_max     = max(nchan_per_spw)

    # Build 2-D arrays padded to nchan_max; NUM_CHAN records the true length
    chan_freq_2d  = np.zeros((num_ms, nchan_max), dtype=np.float64)
    chan_width_2d = np.zeros((num_ms, nchan_max), dtype=np.float64)
    for i, f in enumerate(freq_list):
        nc = len(f)
        chan_freq_2d[i, :nc]  = f
        chan_width_2d[i, :nc] = df_new

    ref_freqs = np.array([f[len(f)//2] for f in freq_list], dtype=np.float64)
    tot_bws   = np.array([df_new * len(f) for f in freq_list], dtype=np.float64)
    names     = [f"spw_{i}_{os.path.basename(ms)}" for i, ms in enumerate(ms_list)]

    ds = Dataset({
        "CHAN_FREQ":        (("row", "chan"), _to_dask(chan_freq_2d)),
        "CHAN_WIDTH":       (("row", "chan"), _to_dask(chan_width_2d)),
        "EFFECTIVE_BW":    (("row", "chan"), _to_dask(chan_width_2d)),
        "RESOLUTION":      (("row", "chan"), _to_dask(chan_width_2d)),
        "REF_FREQUENCY":   (("row",),        _to_dask(ref_freqs)),
        "TOTAL_BANDWIDTH": (("row",),        _to_dask(tot_bws)),
        "NUM_CHAN":         (("row",),        _to_dask(np.array(nchan_per_spw, dtype=np.int32))),
        "NAME":            (("row",),        _to_dask(np.array(names, dtype=object))),
        "FLAG_ROW":        (("row",),        _to_dask(np.zeros(num_ms, dtype=bool))),
        "MEAS_FREQ_REF":   (("row",),        _to_dask(np.full(num_ms, 5, dtype=np.int32))),
    })
    return ds


def _build_field_table(first_ms):
    """Build a minimal FIELD subtable for the output MS.

    The output FIELD table contains a single row with the phase centre set to
    (0, 0) in J2000 radians. The field name is inherited from the first input MS.
    """
    rows = xds_from_table(f"{first_ms}::FIELD", group_cols="__row__")
    name = "stacked"
    if rows:
        try:
            name = str(np.asarray(rows[0].NAME.data).item())
        except Exception:
            pass
    zero_dir = np.zeros((1, 1, 2), dtype=np.float64)
    ds = Dataset({
        "NAME":           (("row",),              _to_dask(np.array([name], dtype=object))),
        "CODE":           (("row",),              _to_dask(np.array([""],   dtype=object))),
        "TIME":           (("row",),              _to_dask(np.zeros(1, dtype=np.float64))),
        "NUM_POLY":       (("row",),              _to_dask(np.zeros(1, dtype=np.int32))),
        "PHASE_DIR":      (("row", "d0", "ra_dec"), _to_dask(zero_dir)),
        "DELAY_DIR":      (("row", "d0", "ra_dec"), _to_dask(zero_dir)),
        "REFERENCE_DIR":  (("row", "d0", "ra_dec"), _to_dask(zero_dir)),
        "FLAG_ROW":       (("row",),              _to_dask(np.zeros(1, dtype=bool))),
        "SOURCE_ID":      (("row",),              _to_dask(np.zeros(1, dtype=np.int32))),
    })
    return ds


def _build_dd_table(first_ms, num_ms):
    """Build the DATA_DESCRIPTION subtable for the output MS.

    One row is created per input MS, each mapping to a distinct spectral
    window (SPW index = MS index). The polarisation ID is inherited from the
    first input MS.
    """
    rows = xds_from_table(f"{first_ms}::DATA_DESCRIPTION", group_cols="__row__")
    pol_id = 0
    if rows:
        try:
            pol_id = int(np.asarray(rows[0].POLARIZATION_ID.data).item())
        except Exception:
            pass
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
    try:
        rows = xds_from_table(f"{ms_path}::{name}", group_cols="__row__")
    except Exception:
        return None
    if not rows:
        return None
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

    The subtables are read in parallel using a ThreadPoolExecutor since each
    read is small and dominated by filesystem latency. All rows are returned
    as-is; the output MS will contain one OBSERVATION row per input MS.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _load_obs(ms):
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

    computed = dask.compute(*lazy_all)
    all_ds   = []
    for (r, vnames), (s, e) in zip(flat_rows, row_slices):
        data_vars = {}
        for v, arr in zip(vnames, computed[s:e]):
            data_vars[v] = (r[v].dims, _to_dask(np.asarray(arr)))
        all_ds.append(Dataset(data_vars))
    return all_ds


def _build_feed_table(ms_list, ant_offsets):
    """Build the FEED subtable by concatenating FEED tables from all input MSs.

    Antenna IDs are remapped using ant_offsets so they point to the correct
    rows in the combined ANTENNA table.
    """
    all_ds = []
    for ms, offset in zip(ms_list, ant_offsets):
        try:
            rows = xds_from_table(f"{ms}::FEED", group_cols="__row__")
        except Exception:
            continue
        for r in rows:
            vnames   = list(r.data_vars)
            computed = dask.compute(*[r[v].data for v in vnames])
            data_vars = {}
            for v, arr in zip(vnames, computed):
                arr = np.asarray(arr)
                if v == "ANTENNA_ID":
                    arr = arr + offset
                data_vars[v] = (r[v].dims, _to_dask(arr))
            all_ds.append(Dataset(data_vars))
    return all_ds


def _build_source_table(first_ms):
    """Build a minimal SOURCE subtable for the output MS.

    A single placeholder row is created with the CO(4-3) rest frequency.
    This is sufficient for imaging tools to recognise the dataset as a
    spectral-line MS; the actual source properties are not used by the pipeline.
    """
    CO43_HZ = 461_040_768_000.0
    ds = Dataset({
        "SOURCE_ID":           (("row",),         _to_dask(np.zeros(1, dtype=np.int32))),
        "TIME":                (("row",),         _to_dask(np.zeros(1, dtype=np.float64))),
        "INTERVAL":            (("row",),         _to_dask(np.zeros(1, dtype=np.float64))),
        "SPECTRAL_WINDOW_ID":  (("row",),         _to_dask(np.array([-1], dtype=np.int32))),
        "NUM_LINES":           (("row",),         _to_dask(np.ones(1,  dtype=np.int32))),
        "NAME":                (("row",),         _to_dask(np.array(["J0000+0000"], dtype=object))),
        "CALIBRATION_GROUP":   (("row",),         _to_dask(np.zeros(1, dtype=np.int32))),
        "CODE":                (("row",),         _to_dask(np.array([""], dtype=object))),
        "DIRECTION":           (("row", "radec"), _to_dask(np.zeros((1, 2), dtype=np.float64))),
        "PROPER_MOTION":       (("row", "pm"),    _to_dask(np.zeros((1, 2), dtype=np.float64))),
        "REST_FREQUENCY":      (("row", "lines"), _to_dask(np.array([[CO43_HZ]], dtype=np.float64))),
        "SYSVEL":              (("row", "lines"), _to_dask(np.zeros((1, 1),  dtype=np.float64))),
    })
    return [ds]


# ===========================================================================
# ViSta — main pipeline class
# ===========================================================================

class ViSta:
    """HPC-optimised visibility-domain stacking pipeline.

    Reads a list of interferometric Measurement Sets at different redshifts,
    rest-frames and phase-shifts each dataset, regrids all visibilities onto
    a common spectral grid, and writes the result as a single stacked MS.

    The pipeline uses three concurrent threads (reader, compute, writer)
    connected by bounded queues, effectively hiding I/O latency behind
    compute and vice versa. The compute kernel (ms_ops.so) is written in
    C++/OpenMP and optionally dispatched to a CUDA GPU.

    Parameters
    ----------
    input_file : str
        Path to a plain-text file listing the MSs to stack. Each non-comment
        line must contain (in order):
          <ms_path>  <redshift>  <RA>  <Dec>  [FIELD_ID]  [SPW_ID]
        RA and Dec are parsed by astropy (sexagesimal or decimal).
        FIELD_ID and SPW_ID are optional (default: 0).
    chunk_rows : int
        Number of baseline rows processed per chunk. Larger values reduce
        Python overhead but increase peak memory usage.
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
        self._load_input(input_file)

    def _load_input(self, path):
        """Parse the input list file and populate the per-MS attribute lists."""
        with open(path) as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"Malformed input line: {line!r}")
                self.ms_list.append(parts[0])
                self.z_list.append(float(parts[1]))
                self.ra_list.append(parts[2])
                self.dec_list.append(parts[3])
                self.field_list.append(int(parts[4]) if len(parts) >= 5 else None)
                if len(parts) >= 6:
                    self.spw_list.append([int(s) for s in parts[5].split(",") if s != ""])
                else:
                    self.spw_list.append(None)

    def _compute_per_ms_grids(self, central_freq, nchan_out=None):
        """Compute the output spectral grid for each input MS.

        This reproduces the channel-grid logic of the original CASA-based
        pipeline (vista.py) exactly:

        1. Read SPECTRAL_WINDOW, FIELD, and DATA_DESCRIPTION subtables for
           every MS using dask-ms (parallelised with a ThreadPoolExecutor).
        2. Compute the rest-framed channel width for each MS:
             df_rf_i = df_obs_i * (1 + z_i)
           The common output channel width is the maximum across all MSs:
             df_new = max_i(df_rf_i)
        3. Compute the number of output channels:
             nn = max_i(floor(nchan_i * df_rf_i / df_new))  [if nchan_out is None]
             nn = nchan_out                                   [otherwise]
           nn is rounded down to an even number.
        4. For each MS, place the output window of width nn * df_new centred
           on central_freq. If the window does not fit within the MS's
           rest-framed frequency range, it is shifted to the nearest edge.

        The metadata read is cached in self._cache_* so that the reader
        thread can retrieve per-MS frequency arrays without re-reading.

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

        def _read_meta_daskms(i, ms, req_field, req_spws):
            """Read SPECTRAL_WINDOW, FIELD, and DATA_DESCRIPTION for one MS."""
            result = {}

            # DATA_DESCRIPTION: map spw_id -> DATA_DESC_ID row index
            dd_ds  = xds_from_table(f"{ms}::DATA_DESCRIPTION")[0]
            spwids = np.asarray(
                dd_ds.SPECTRAL_WINDOW_ID.data.compute(scheduler="synchronous"))
            spw2dd = {int(spwid): dd_row for dd_row, spwid in enumerate(spwids)}
            result["spw2dd"] = spw2dd

            # SPECTRAL_WINDOW: one dataset per row (one per SPW)
            spw_rows     = xds_from_table(f"{ms}::SPECTRAL_WINDOW", group_cols="__row__")
            n_spw_avail  = len(spw_rows)
            spws_to_read = list(range(n_spw_avail)) if req_spws is None else req_spws

            lazy = []
            for sp in spws_to_read:
                if sp >= n_spw_avail:
                    raise ValueError(
                        f"{os.path.basename(ms)}: SPW {sp} does not exist "
                        f"(available: 0-{n_spw_avail-1})"
                    )
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

            # FIELD: PHASE_DIR of the requested field
            field_rows = xds_from_table(f"{ms}::FIELD", group_cols="__row__")
            n_field    = len(field_rows)
            fid = req_field if req_field is not None else 0
            if fid >= n_field:
                raise ValueError(
                    f"{os.path.basename(ms)}: FIELD {fid} does not exist "
                    f"(available: 0-{n_field-1})"
                )
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
        spectral_res_list = []
        nchan_list        = []
        freq_ranges       = []

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
                 f"z={z:.5f}  nchan={len(freq_all)}  "
                 f"df_rf={df_rf/1e3:.3f} kHz  "
                 f"range=[{freq_rf.min()/1e6:.3f}, {freq_rf.max()/1e6:.3f}] MHz")

        # Common channel width = widest rest-framed channel across all MSs
        df_new = max(spectral_res_list)
        _log(f"Common channel width (df_new) = {df_new/1e3:.3f} kHz")

        # Maximum number of output channels each MS can contribute
        nn_per_ms = [max(1, int(nc * dr / df_new))
                     for nc, dr in zip(nchan_list, spectral_res_list)]

        if nchan_out is None:
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

        bandwidth   = nn * df_new
        start_ideal = central_freq - bandwidth / 2.0
        end_ideal   = central_freq + bandwidth / 2.0
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

            center = 0.5 * (start + start + bandwidth)
            offset = center - central_freq
            _log(f"  [{ms_idx+1}/{n_ms}]  {os.path.basename(ms)}  "
                 f"start={start/1e6:.6f} MHz  offset={offset:+.3e} Hz  ({reason})")

            freq_ms = start + df_new / 2.0 + np.arange(nn) * df_new
            freq_new_per_ms.append(freq_ms.astype(np.float64))

        return freq_new_per_ms, df_new

    def run(self, ms_out, central_freq, scratch_dir=None, nchan_out=None):
        """Run the stacking pipeline and write the output Measurement Set.

        Parameters
        ----------
        ms_out : str
            Path to the output Measurement Set. Existing files at this path
            will be deleted before writing.
        central_freq : float
            Rest-frame central frequency of the output spectral grid (Hz).
        scratch_dir : str or None
            If set, the MS is first written to this directory (e.g. a local
            NVMe or $TMPDIR) and then moved to ms_out at the end. Useful on
            HPC systems where the final destination is on a slow network
            filesystem.
        nchan_out : int or None
            Number of output channels per spectral window.
            None  -- use the maximum number of channels any single MS can
                     contribute (MSs with narrower bandwidth will have partial
                     coverage).
            int   -- fix the number of output channels explicitly. Can be
                     larger or smaller than the automatic value.
        """
        global _verbose_global
        _verbose_global = self.verbose

        t_start = time.perf_counter()

        # Determine working path (scratch) vs final destination
        ms_final = ms_out
        if scratch_dir is not None:
            os.makedirs(scratch_dir, exist_ok=True)
            ms_out = os.path.join(scratch_dir, os.path.basename(ms_final))
            _log(f"Writing to scratch: {ms_out}")
            _log(f"  Final destination: {ms_final}")

        # Remove any existing output at both locations
        for _p in {ms_out, ms_final}:
            if os.path.exists(_p):
                _log(f"Removing existing output: {_p}")
                shutil.rmtree(_p)

        _log("Computing per-MS spectral grids...")
        freq_new_per_ms, df_new = self._compute_per_ms_grids(
            central_freq, nchan_out=nchan_out
        )
        nchan_new = len(freq_new_per_ms[0])

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
        nant_total   = ant_ds.sizes['row']

        # Build a minimal FEED table covering all antennas
        feed_ds_list = [Dataset({
            'ANTENNA_ID':         (('row',),                _to_dask(np.arange(nant_total, dtype=np.int32))),
            'FEED_ID':            (('row',),                _to_dask(np.zeros(nant_total, dtype=np.int32))),
            'SPECTRAL_WINDOW_ID': (('row',),                _to_dask(np.full(nant_total, -1, dtype=np.int32))),
            'TIME':               (('row',),                _to_dask(np.zeros(nant_total, dtype=np.float64))),
            'INTERVAL':           (('row',),                _to_dask(np.zeros(nant_total, dtype=np.float64))),
            'NUM_RECEPTORS':      (('row',),                _to_dask(np.full(nant_total, 2, dtype=np.int32))),
            'POLARIZATION_TYPE':  (('row', 'receptors'),    _to_dask(np.array([['X','Y']]*nant_total, dtype=object))),
            'POL_RESPONSE':       (('row', 'receptors', 'receptors2'),
                                   _to_dask(np.tile(np.eye(2, dtype=complex), (nant_total, 1, 1)))),
            'POSITION':           (('row', 'xyz'),          _to_dask(np.zeros((nant_total, 3), dtype=np.float64))),
            'BEAM_OFFSET':        (('row', 'receptors', 'radec'),
                                   _to_dask(np.zeros((nant_total, 2, 2), dtype=np.float64))),
            'RECEPTOR_ANGLE':     (('row', 'receptors'),    _to_dask(np.zeros((nant_total, 2), dtype=np.float64))),
            'BEAM_ID':            (('row',),                _to_dask(np.full(nant_total, -1, dtype=np.int32))),
        })]
        src_ds_list = _build_source_table(first_ms)

        def _write_sub(ds_or_list, subtable):
            """Write a subtable Dataset (or list of Datasets) to ms_out."""
            lst = ds_or_list if isinstance(ds_or_list, list) else [ds_or_list]
            if lst:
                dask.compute(*xds_to_table(lst, f"{ms_out}::{subtable}", "ALL"))
                _log(f"  Written: {subtable}")

        n_ms = len(self.ms_list)
        _log(f"Starting main loop: {n_ms} MSs to process...")
        total_rows   = 0
        total_bytes  = 0
        t_loop_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Inner function: read one MS from disk (runs in reader thread)
        # ------------------------------------------------------------------
        def _read_ms(ms_idx, ms, z, ra_str, dec_str):
            """Read visibilities for one MS and return a processing package.

            Visibilities are read from disk using a single dask.compute() call
            per spectral window, materialising all required columns (DATA, FLAG,
            UVW, and a set of scalar columns) into NumPy arrays in one shot.
            This avoids the overhead of repeated lazy-graph construction that
            would result from accessing each column independently.

            Returns None if the MS contributes no valid rows (e.g. the requested
            FIELD_ID or SPW_ID is empty).
            """
            ant_offset     = ant_offsets[ms_idx]
            freq_old_list  = self._cache_freq_old[ms_idx]
            width_old_list = self._cache_width_old[ms_idx]
            ddids          = self._cache_ddids[ms_idx]
            req_field      = self.field_list[ms_idx]
            ra0_rad, dec0_rad = self._cache_phase_dir[ms_idx]
            ra1_rad, dec1_rad = _parse_ra_dec(ra_str, dec_str)
            ra1_rad = _wrap_dra(ra1_rad, ra0_rad)

            freq_new_ms = freq_new_per_ms[ms_idx]

            SCALAR_COLS = ["TIME", "EXPOSURE", "INTERVAL", "TIME_CENTROID",
                           "SCAN_NUMBER", "STATE_ID", "ARRAY_ID", "OBSERVATION_ID",
                           "FEED1", "FEED2", "PROCESSOR_ID", "WEIGHT", "SIGMA", "FLAG_ROW"]

            chunks_data      = []
            n_covered_global = set()

            # Weight and sigma scaling factor: R = df_new / df_old_rf
            # Identical to the original CASA pipeline weight rescaling.
            df_old_rf_chan0 = float(abs(width_old_list[0][0])) * (1.0 + z)
            weight_scale = df_new / df_old_rf_chan0
            sigma_scale  = 1.0 / np.sqrt(weight_scale) if weight_scale > 0.0 else 1.0
            _log(f"  Weight scale R = {weight_scale:.4f}  "
                 f"(df_new={df_new:.1f} Hz, df_old_rf={df_old_rf_chan0:.1f} Hz)")

            taql_where = f"FIELD_ID=={int(req_field)}" if req_field is not None else ""
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

            for spw_pos, ddid in enumerate(ddids):
                freq_old    = freq_old_list[spw_pos]
                freq_old_rf = freq_old * (1.0 + z)
                chan_lo = 0
                chan_hi = len(freq_old) - 1
                freq_old_rf_sl = freq_old_rf[chan_lo:chan_hi+1]

                # Record which output channels this SPW can contribute to
                for j in range(len(freq_new_ms)):
                    f = freq_new_ms[j]
                    if freq_old_rf_sl[0] - df_new <= f <= freq_old_rf_sl[-1] + df_new:
                        n_covered_global.add(j)

                ds = by_ddid.get(int(ddid))
                if ds is None:
                    continue
                nrows = ds.sizes["row"]
                if nrows == 0:
                    continue

                data_da  = ds.DATA.data[:, chan_lo:chan_hi+1, :]
                flag_da  = ds.FLAG.data[:, chan_lo:chan_hi+1, :]
                uvw_da   = ds.UVW.data
                ant1_da  = ds.ANTENNA1.data
                ant2_da  = ds.ANTENNA2.data
                scal_da  = {c: ds[c].data for c in SCALAR_COLS if c in ds.data_vars}
                scal_keys = list(scal_da.keys())

                # Single dask.compute() for the whole SPW: one graph evaluation,
                # one interaction with the TableProxy, no per-chunk overhead.
                out = dask.compute(
                    data_da.astype(np.complex64),
                    flag_da,
                    uvw_da.astype(np.float64),
                    ant1_da,
                    ant2_da,
                    *[scal_da[c] for c in scal_keys],
                    scheduler="synchronous",
                )
                vis_all  = np.asarray(out[0])
                flag_all = np.asarray(out[1])
                uvw_all  = np.asarray(out[2])
                ant1_all = np.asarray(out[3])
                ant2_all = np.asarray(out[4])
                scal_all = [np.asarray(out[5 + k]) for k in range(len(scal_keys))]

                # Split into row chunks for the compute thread
                for row0 in range(0, nrows, self.chunk_rows):
                    row1 = min(row0 + self.chunk_rows, nrows)
                    ant1 = ant1_all[row0:row1] + ant_offset
                    ant2 = ant2_all[row0:row1] + ant_offset
                    scalars = {}
                    for c, arr in zip(scal_keys, scal_all):
                        sl   = arr[row0:row1]
                        dims = ("row", "corr") if sl.ndim == 2 else ("row",)
                        scalars[c] = (dims, sl)
                    chunks_data.append((
                        vis_all[row0:row1], flag_all[row0:row1],
                        uvw_all[row0:row1], scalars, ant1, ant2,
                        freq_old_rf_sl, weight_scale, sigma_scale,
                    ))

            if not chunks_data:
                return None

            return {
                "ms_idx": ms_idx, "ms": ms, "z": z,
                "ra0_rad": ra0_rad, "dec0_rad": dec0_rad,
                "ra1_rad": ra1_rad, "dec1_rad": dec1_rad,
                "n_covered": len(n_covered_global),
                "chunks_data": chunks_data,
            }

        # ------------------------------------------------------------------
        # Inner function: apply C++ kernel (runs in main/compute thread)
        # ------------------------------------------------------------------
        def _process_ms(pkg):
            """Apply the C++/OpenMP kernel to one MS package (CPU path).

            Each chunk within the package is processed independently by calling
            ms_ops.full_pipeline(), which applies rest-framing, phase shift,
            and spectral rebinning in a single in-memory pass.

            Returns a tuple (ms_datasets, nrows_ms, bytes_ms) ready for the
            writer thread.
            """
            z         = pkg["z"]
            ra0_rad   = pkg["ra0_rad"];  dec0_rad = pkg["dec0_rad"]
            ra1_rad   = pkg["ra1_rad"];  dec1_rad = pkg["dec1_rad"]
            ms_name   = os.path.basename(pkg["ms"])
            ms_idx    = pkg["ms_idx"]
            freq_new_ms = freq_new_per_ms[ms_idx]
            nchan_ms    = len(freq_new_ms)

            _log(f"[{ms_idx+1}/{n_ms}]  {ms_name}  z={z:.5f}  "
                 f"coverage: {pkg['n_covered']}/{nchan_ms} channels")

            ms_datasets = []
            nrows_ms = 0
            bytes_ms = 0

            for ds_idx, chunk in enumerate(pkg["chunks_data"]):
                (vis, flag, uvw, scalars, ant1, ant2,
                 freq_old_rf_sl, weight_scale, sigma_scale) = chunk

                t_proc = time.perf_counter()
                vis_out, flag_out, uvw_out = ms_ops.full_pipeline(
                    vis, flag, uvw,
                    z, freq_old_rf_sl,
                    ra0_rad, dec0_rad,
                    ra1_rad, dec1_rad,
                    freq_new_ms,
                )
                dt = time.perf_counter() - t_proc

                nchan_out_actual = vis_out.shape[1]
                nrow  = vis_out.shape[0]
                ncorr = vis_out.shape[2]
                _log(f"  chunk {ds_idx}: {vis.shape[0]} rows x {vis.shape[1]} ch  "
                     f"-> {nchan_out_actual} ch  [{dt:.1f}s]")

                nrows_ms += nrow
                bytes_ms += vis_out.nbytes + flag_out.nbytes + uvw_out.nbytes

                row_ch = _make_row_chunks(nrow, self.chunk_rows)
                ch_3d  = (row_ch, (nchan_ms,), (ncorr,))
                ch_uvw = (row_ch, (3,))
                ch_1d  = (row_ch,)

                data_vars = {
                    "DATA":         (("row", "chan", "corr"), _to_dask(vis_out,  ch_3d)),
                    "FLAG":         (("row", "chan", "corr"), _to_dask(flag_out, ch_3d)),
                    "UVW":          (("row", "uvw"),          _to_dask(uvw_out,  ch_uvw)),
                    "ANTENNA1":     (("row",),                _to_dask(ant1,     ch_1d)),
                    "ANTENNA2":     (("row",),                _to_dask(ant2,     ch_1d)),
                    "DATA_DESC_ID": (("row",),                _to_dask(np.full(nrow, ms_idx, dtype=np.int32), ch_1d)),
                    "FIELD_ID":     (("row",),                _to_dask(np.zeros(nrow, dtype=np.int32), ch_1d)),
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

            return ms_datasets, nrows_ms, bytes_ms

        # ------------------------------------------------------------------
        # Inner function: write one MS to disk (runs in writer thread)
        # ------------------------------------------------------------------
        def _write_ms(ms_datasets, nrows_ms, bytes_ms, ms_out):
            """Write the processed datasets for one MS to the output table.

            Uses xds_to_table (dask-ms) to write all columns chunk by chunk,
            appending rows to the existing MAIN table.
            """
            t_write = time.perf_counter()
            _log(f"  Writing {nrows_ms:,} rows ({_fmt_size(bytes_ms)}) to disk...")
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

        # Estimate size of one MS package in RAM: nchan * nrow * ncorr * 10 bytes
        _nrow_est   = 10_000
        _ncorr_est  = 2
        _ms_size_gb = max(0.05, nchan_new * _nrow_est * _ncorr_est * 10 / 1e9)
        _buf_by_ram = max(2, int(_ram_usable_gb / _ms_size_gb / 2))
        READ_BUF    = min(n_ms, max(4, _buf_by_ram))
        WRITE_BUF   = READ_BUF
        _log(f"  Available RAM: {_ram_usable_gb:.1f} GB  |  "
             f"estimated MS size: {_ms_size_gb*1e3:.0f} MB  |  "
             f"queue depth: {READ_BUF}")

        # ------------------------------------------------------------------
        # Thread timing counters (each key written by exactly one thread)
        # ------------------------------------------------------------------
        timers = {
            "read":       0.0,
            "read_wait":  0.0,
            "compute":    0.0,
            "write_wait": 0.0,
            "write":      0.0,
        }

        read_q    = Queue(maxsize=READ_BUF)
        write_q   = Queue(maxsize=WRITE_BUF)
        read_exc  = [None]
        write_exc = [None]

        ms_args = list(zip(
            range(n_ms),
            self.ms_list, self.z_list, self.ra_list, self.dec_list
        ))

        def _reader_thread():
            """Read all MSs sequentially and push packages onto read_q."""
            try:
                for args in ms_args:
                    _t = time.perf_counter()
                    pkg = _read_ms(*args)
                    timers["read"] += time.perf_counter() - _t
                    if pkg is not None:
                        read_q.put(pkg)
            except Exception as e:
                read_exc[0] = e
            finally:
                read_q.put(None)   # sentinel: no more packages

        def _writer_thread():
            """Pop processed packages from write_q and write them to disk."""
            try:
                while True:
                    _tw = time.perf_counter()
                    item = write_q.get()
                    timers["write_wait"] += time.perf_counter() - _tw
                    if item is None:
                        break   # sentinel received: all MSs written
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

        # GPU batch size: how many MSs to pack into one CUDA kernel launch.
        # On CPU the batch is always 1 (full_pipeline_batch falls back to
        # calling full_pipeline once per MS).
        GPU_BATCH  = 20 if _gpu_available else 1
        processed  = 0
        batch_pkgs = []

        def _flush_batch(pkgs):
            """Process a batch of MS packages and push results to write_q.

            On GPU: packs all MSs in the batch into a single kernel launch
            (ms_ops.full_pipeline_batch), then unpacks the results.
            On CPU: processes each MS sequentially with ms_ops.full_pipeline.
            """
            nonlocal processed, total_rows, total_bytes

            if not pkgs:
                return

            t_ms = time.perf_counter()

            if GPU_BATCH > 1 and len(pkgs) > 1:
                # --- GPU path: batch dispatch ---
                vis_list      = [p["chunks_data"][0][0] for p in pkgs]
                flag_list     = [p["chunks_data"][0][1] for p in pkgs]
                uvw_list      = [p["chunks_data"][0][2] for p in pkgs]
                z_list        = [p["z"]        for p in pkgs]
                freq_old_list = [p["chunks_data"][0][6] for p in pkgs]
                ra_old_list   = [p["ra0_rad"]  for p in pkgs]
                dec_old_list  = [p["dec0_rad"] for p in pkgs]
                ra_new_list   = [p["ra1_rad"]  for p in pkgs]
                dec_new_list  = [p["dec1_rad"] for p in pkgs]
                freq_new_list = [freq_new_per_ms[p["ms_idx"]] for p in pkgs]

                results = ms_ops.full_pipeline_batch(
                    vis_list, flag_list, uvw_list,
                    z_list, freq_old_list,
                    ra_old_list, dec_old_list,
                    ra_new_list, dec_new_list,
                    freq_new_list,
                )
                timers["compute"] += time.perf_counter() - t_ms
                _log(f"  GPU batch: {len(pkgs)} MSs processed in "
                     f"{time.perf_counter()-t_ms:.1f}s")

                for pkg, (vis_out, flag_out, uvw_out) in zip(pkgs, results):
                    chunk        = pkg["chunks_data"][0]
                    scalars      = chunk[3]
                    ant1         = chunk[4]
                    ant2         = chunk[5]
                    weight_scale = chunk[7]
                    sigma_scale  = chunk[8]
                    ms_idx       = pkg["ms_idx"]
                    nchan_ms     = len(freq_new_per_ms[ms_idx])

                    nrow  = vis_out.shape[0]
                    ncorr = vis_out.shape[2]
                    nrows_ms = nrow
                    bytes_ms = vis_out.nbytes + flag_out.nbytes + uvw_out.nbytes

                    row_ch = _make_row_chunks(nrow, self.chunk_rows)
                    ch_3d  = (row_ch, (nchan_ms,), (ncorr,))
                    ch_uvw = (row_ch, (3,))
                    ch_1d  = (row_ch,)

                    data_vars = {
                        "DATA":         (("row", "chan", "corr"), _to_dask(vis_out,  ch_3d)),
                        "FLAG":         (("row", "chan", "corr"), _to_dask(flag_out, ch_3d)),
                        "UVW":          (("row", "uvw"),          _to_dask(uvw_out,  ch_uvw)),
                        "ANTENNA1":     (("row",),                _to_dask(ant1,     ch_1d)),
                        "ANTENNA2":     (("row",),                _to_dask(ant2,     ch_1d)),
                        "DATA_DESC_ID": (("row",),                _to_dask(np.full(nrow, ms_idx, dtype=np.int32), ch_1d)),
                        "FIELD_ID":     (("row",),                _to_dask(np.zeros(nrow, dtype=np.int32), ch_1d)),
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
                    total_rows  += nrows_ms
                    total_bytes += bytes_ms
                    write_q.put((ms_datasets, nrows_ms, bytes_ms))
                    processed += 1

            else:
                # --- CPU path: one MS at a time ---
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
        # Main loop: drain read_q and dispatch batches to _flush_batch
        # ------------------------------------------------------------------
        while True:
            _trw = time.perf_counter()
            pkg = read_q.get()
            timers["read_wait"] += time.perf_counter() - _trw

            if pkg is None:
                # Reader finished: flush any remaining partial batch
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

        _flush_log_unlocked()

        if not os.path.isdir(ms_out):
            raise RuntimeError(
                f"Output table '{ms_out}' was not created. "
                f"Check that at least one MS contributed valid rows "
                f"(verify FIELD_ID / SPW_ID selection and spectral coverage)."
            )

        _log(f"Writing output subtables to {ms_out} ...")
        _write_sub(spw_ds, "SPECTRAL_WINDOW")

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
        _log(f"  read        (reader thread) : {timers['read']:7.1f}s")
        _log(f"  read_wait   (main thread)   : {timers['read_wait']:7.1f}s"
             f"   <- near zero means compute never waits for data")
        _log(f"  compute     (main thread)   : {timers['compute']:7.1f}s")
        _log(f"  write_wait  (writer thread) : {timers['write_wait']:7.1f}s"
             f"   <- near zero means writer is always busy")
        _eff = total_bytes / timers['write'] if timers['write'] > 0 else 0
        _log(f"  write       (writer thread) : {timers['write']:7.1f}s"
             f"   ({_fmt_size(_eff)}/s)")
        if scratch_dir is not None:
            _eff_s = total_bytes / stage_dt if stage_dt > 0 else 0
            _log(f"  staging     (move)         : {stage_dt:7.1f}s"
                 f"   ({_fmt_size(_eff_s)}/s)")
        _log("=" * 60)
        _log(f"Output MS     : {ms_final}")
        _log(f"Total rows    : {total_rows:,}")
        _log(f"Spectral windows : {n_ms}  (one per input MS, "
             f"{nchan_new} channels, df={df_new/1e3:.3f} kHz)")
        _log(f"Central frequency: {central_freq/1e6:.6f} MHz")
        _log(f"Antennas      : {ant_ds.sizes['row']}")
        _log(f"Data written  : {_fmt_size(total_bytes)}")
        _log(f"Wall-clock time  : {t_total/60:.1f} min")
        _log("=" * 60)
        _flush_log_unlocked()
