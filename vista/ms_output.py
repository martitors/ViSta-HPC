"""
vista.ms_output -- writing the stacked Measurement Set
======================================================

    MsWriter          the output main table: created at the first chunk, then
                      filled with addrows + putcol by the single writer thread
    build_subtables   ANTENNA, SPECTRAL_WINDOW, FIELD, ... of the output
    write_subtables   write them, once the main table is closed

The output has one spectral window per line of the input list; its rows are
tagged with DATA_DESC_ID = line, FIELD_ID = 0 (a single field at (0, 0),
since every entry has been re-centred on its source).

Internal module of :mod:`vista.pipeline`.
"""

import os
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import dask
import dask.array as da
from daskms import xds_from_table, xds_to_table, Dataset

from .common import casacore_lock, detail

# Columns with the same value in every row, written with each chunk.
CONSTANT_COLUMNS = {"FEED1": np.int32, "FEED2": np.int32, "PROCESSOR_ID": np.int32,
                    "ARRAY_ID": np.int32, "OBSERVATION_ID": np.int32,
                    "TIME_CENTROID": np.float64}


# =============================================================================
# Helpers
# =============================================================================

def to_dask(arr, chunks=None):
    """Wrap a NumPy array in a dask array with a unique name.

    Parameters
    ----------
    arr : ndarray
        The array.
    chunks : tuple or None, default None
        dask chunks.  ``None`` makes a single chunk.

    Returns
    -------
    dask.array.Array
        The unique name keeps dask from merging the graphs of different
        arrays that happen to have the same content.
    """
    return da.from_array(arr, chunks=arr.shape if chunks is None else chunks,
                         name="array-" + uuid.uuid4().hex[:8])


def rename_duplicates(names):
    """Make a list of names unique by numbering the repeated ones.

    Example: ``['DA41', 'DA41', 'DV01']`` -> ``['DA41_0', 'DA41_1', 'DV01']``.
    Used for the antennas of the merged ANTENNA table.
    """
    count, seen, out = Counter(names), Counter(), []
    for n in names:
        if count[n] > 1:
            out.append(f"{n}_{seen[n]}")
            seen[n] += 1
        else:
            out.append(n)
    return out


# =============================================================================
# Main table
# =============================================================================

def output_dminfo(nchan, ncorr, extra_columns=()):
    """Storage managers of the output main table (the layout dask-ms uses).

    Tiled storage for UVW, WEIGHT, SIGMA, FLAG and DATA, with tiles of about
    1 MB, 16 channels wide at most; *extra_columns* (MODEL_DATA,
    CORRECTED_DATA in an MS exported from a Zarr store) are tiled like DATA.
    """
    ch_tile = min(nchan, 16)
    data_rows = max(1, (1 << 20) // (ncorr * ch_tile * 8))
    dm = {
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
    for i, col in enumerate(extra_columns):
        dm[f"*{8 + i}"] = {"TYPE": "TiledColumnStMan", "NAME": f"{col}_GROUP",
                           "SPEC": {"DEFAULTTILESHAPE": np.array([ncorr, ch_tile, data_rows],
                                                                 np.int32)},
                           "COLUMNS": [col]}
    return dm


class MsWriter:
    """The output main table, written by one thread with python-casacore.

    The table is created at the first chunk, when the number of channels and
    correlations is known; every chunk is then appended with ``addrows`` and
    one ``putcol`` per column.

    Parameters
    ----------
    path : str
        The output MS.
    """

    def __init__(self, path):
        self.path = path
        self._tab = None
        self._columns = set()

    def append(self, columns):
        """Append one chunk.

        Parameters
        ----------
        columns : dict of str -> ndarray
            DATA, FLAG, UVW, ANTENNA1, ... of the rows; columns the table does
            not have are ignored.  The constant columns are added here.
        """
        if self._tab is None:
            extra = [c for c in ("MODEL_DATA", "CORRECTED_DATA") if c in columns]
            self._create(*columns["DATA"].shape[1:], extra=extra)
        n = columns["DATA"].shape[0]
        start = self._tab.nrows()
        self._tab.addrows(n)
        for col, arr in columns.items():
            if col in self._columns:
                self._tab.putcol(col, arr, startrow=start, nrow=n)
        for col, dt in CONSTANT_COLUMNS.items():
            if col in self._columns:
                self._tab.putcol(col, np.zeros(n, dtype=dt), startrow=start, nrow=n)

    def _create(self, nchan, ncorr, extra=()):
        from casacore.tables import default_ms, makearrcoldesc, maketabdesc
        desc = maketabdesc([
            makearrcoldesc("DATA", 0j, shape=[nchan, ncorr], valuetype="complex"),
            makearrcoldesc("FLAG", False, shape=[nchan, ncorr]),
            makearrcoldesc("WEIGHT", 0.0, shape=[ncorr], valuetype="float"),
            makearrcoldesc("SIGMA", 0.0, shape=[ncorr], valuetype="float"),
        ] + [makearrcoldesc(c, 0j, shape=[nchan, ncorr], valuetype="complex")
             for c in extra])
        self._tab = default_ms(self.path, desc, output_dminfo(nchan, ncorr, extra))
        self._columns = set(self._tab.colnames())

    def close(self):
        """Flush and close the table (nothing to do if no chunk arrived)."""
        if self._tab is not None:
            self._tab.flush()
            self._tab.close()
            self._tab = None

    def abandon(self):
        """Close the table after an error, ignoring further errors."""
        if self._tab is not None:
            try:
                self._tab.close()
            except Exception:
                pass
            self._tab = None


# =============================================================================
# Subtables
# =============================================================================
# Each builder returns the dask-ms Dataset(s) of one subtable of the output
# MS.  Everything that reads an input table holds casacore_lock.

def build_subtables(ms_list, z_list, freq_list, df_new, central_freq):
    """Every subtable of the output MS, in the order they are written.

    Parameters
    ----------
    ms_list, z_list : list
        Input MSs and their redshifts, one per entry.
    freq_list : list of ndarray
        Output channel centres of each entry [Hz].
    df_new : float
        Output channel width [Hz].
    central_freq : float
        Rest frequency of the line [Hz] (SOURCE table).

    Returns
    -------
    (dict, list of int)
        Subtable name -> Dataset, list of Datasets or None; and, for each
        entry, the index of its first antenna in the merged ANTENNA table.
    """
    detail("Building ANTENNA table...")
    ant_ds, ant_offsets = build_antenna_table(ms_list, z_list)
    nant = ant_ds.sizes["row"]
    detail(f"  {nant} antennas in combined table")
    detail("Building output subtables...")
    first = ms_list[0]
    subtables = {
        "SPECTRAL_WINDOW":  build_spw_table(freq_list, df_new, ms_list),
        "FIELD":            build_field_table(first),
        "DATA_DESCRIPTION": build_dd_table(first, len(ms_list)),
        "ANTENNA":          ant_ds,
        "POLARIZATION":     copy_subtable_first_row(first, "POLARIZATION"),
        "OBSERVATION":      build_observation_table(ms_list),
        "FEED":             build_feed_table(nant),
        "SOURCE":           build_source_table(central_freq),
    }
    return subtables, ant_offsets


def write_subtables(ms_out, subtables):
    """Write the output subtables, skipping the empty ones.

    Parameters
    ----------
    ms_out : str
        The output MS.
    subtables : dict
        Subtable name -> Dataset, list of Datasets, or None.
    """
    detail(f"Writing output subtables to {ms_out} ...")
    for name, content in subtables.items():
        datasets = content if isinstance(content, list) else [content]
        datasets = [d for d in datasets if d is not None]
        if not datasets:
            continue
        with casacore_lock:
            dask.compute(*xds_to_table(datasets, f"{ms_out}::{name}", "ALL"))
        detail(f"  Written: {name}")


def build_antenna_table(ms_list, z_list):
    """ANTENNA table: the antennas of all the inputs, one after the other.

    Positions and dish diameters are divided by (1+z), like the UVW in the
    kernel, so that they describe the rest-frame array.

    Parameters
    ----------
    ms_list : list of str
        Input MSs.
    z_list : list of float
        Their redshifts.

    Returns
    -------
    (Dataset, list of int)
        The table and, for each input, the index of its first antenna, to be
        added to its ANTENNA1 and ANTENNA2.
    """
    def load(args):
        ms, z = args
        inv = 1.0 / (1.0 + z)
        with casacore_lock:
            rows = xds_from_table(f"{ms}::ANTENNA", group_cols=[])
            if not rows:
                return None
            ds = rows[0]
            col = lambda name: ds.data_vars[name].data if name in ds.data_vars else None
            names, pos, diam, mounts, stations = dask.compute(
                col("NAME"), col("POSITION"), col("DISH_DIAMETER"),
                col("MOUNT"), col("STATION"), scheduler="synchronous")
        nant = ds.sizes.get("row", 0)
        as_str = lambda a, default: ([str(v) for v in np.asarray(a).ravel()]
                                     if a is not None else [default] * nant)
        if pos is not None:
            pos = np.asarray(pos, dtype=np.float64)
            if pos.ndim == 2 and pos.shape[0] == 3:
                pos = pos.T
            positions = list(pos * inv)
        else:
            positions = list(np.zeros((nant, 3)))
        diameters = (list(np.asarray(diam, dtype=np.float64).ravel() * inv)
                     if diam is not None else [12.0 * inv] * nant)
        return (nant, as_str(names, ""), positions, diameters,
                as_str(mounts, "ALT-AZ"), as_str(stations, ""))

    with ThreadPoolExecutor(max_workers=min(8, len(ms_list))) as pool:
        results = list(pool.map(load, zip(ms_list, z_list)))

    offsets, offset = [], 0
    names, positions, diameters, mounts, stations = [], [], [], [], []
    for r in results:
        offsets.append(offset)
        if r is None:
            continue
        offset += r[0]
        names += r[1]; positions += r[2]; diameters += r[3]
        mounts += r[4]; stations += r[5]

    names = rename_duplicates(names)
    n = len(names)
    ds = Dataset({
        "NAME":          (("row",),      to_dask(np.array(names, dtype=object))),
        "POSITION":      (("row", "xyz"), to_dask(np.array(positions, dtype=np.float64).reshape(n, 3))),
        "DISH_DIAMETER": (("row",),      to_dask(np.array(diameters, dtype=np.float64))),
        "MOUNT":         (("row",),      to_dask(np.array(mounts, dtype=object))),
        "STATION":       (("row",),      to_dask(np.array(stations, dtype=object))),
        "FLAG_ROW":      (("row",),      to_dask(np.zeros(n, dtype=bool))),
        "OFFSET":        (("row", "xyz"), to_dask(np.zeros((n, 3), dtype=np.float64))),
        "TYPE":          (("row",),      to_dask(np.array(["GROUND-BASED"] * n, dtype=object))),
    })
    return ds, offsets


def build_spw_table(freq_list, df_new, ms_list):
    """SPECTRAL_WINDOW table: one window per input entry.

    The windows are kept separate, so every entry stays identifiable through
    its DATA_DESC_ID.  They share the channel width but may start at
    different frequencies.

    Parameters
    ----------
    freq_list : list of ndarray
        Output channel centres of each entry [Hz].
    df_new : float
        Common channel width [Hz].
    ms_list : list of str
        Input MSs, for the window names ``spw_<i>_<ms name>``.

    Returns
    -------
    Dataset
        The frequencies are rest-frame frequencies.  MEAS_FREQ_REF is written
        as 1 (LSRK), so that imaging tools leave them as they are: tclean's
        default (specmode='cube', outframe='LSRK') then converts LSRK to LSRK,
        i.e. nothing.  Labelled TOPO, tclean would convert every row with the
        time of its observation and the (0, 0) direction of the stack, which
        broadens the stacked line.
    """
    n = len(ms_list)
    assert len(freq_list) == n
    nchan = [len(f) for f in freq_list]
    chan_freq = np.zeros((n, max(nchan)), dtype=np.float64)
    chan_width = np.zeros_like(chan_freq)
    for i, f in enumerate(freq_list):
        chan_freq[i, :len(f)] = f
        chan_width[i, :len(f)] = df_new
    return Dataset({
        "CHAN_FREQ":       (("row", "chan"), to_dask(chan_freq)),
        "CHAN_WIDTH":      (("row", "chan"), to_dask(chan_width)),
        "EFFECTIVE_BW":    (("row", "chan"), to_dask(chan_width)),
        "RESOLUTION":      (("row", "chan"), to_dask(chan_width)),
        "REF_FREQUENCY":   (("row",), to_dask(np.array([f[len(f)//2] for f in freq_list], dtype=np.float64))),
        "TOTAL_BANDWIDTH": (("row",), to_dask(np.array([df_new * len(f) for f in freq_list], dtype=np.float64))),
        "NUM_CHAN":        (("row",), to_dask(np.array(nchan, dtype=np.int32))),
        "NAME":            (("row",), to_dask(np.array([f"spw_{i}_{os.path.basename(ms)}"
                                                        for i, ms in enumerate(ms_list)], dtype=object))),
        "FLAG_ROW":        (("row",), to_dask(np.zeros(n, dtype=bool))),
        "MEAS_FREQ_REF":   (("row",), to_dask(np.full(n, 1, dtype=np.int32))),   # LSRK
    })


def build_field_table(first_ms):
    """FIELD table: a single field at (0, 0), named after the first input.

    Every entry has been re-centred on its source, so one field is enough.
    """
    name = "stacked"
    with casacore_lock:
        rows = xds_from_table(f"{first_ms}::FIELD", group_cols="__row__")
        if rows:
            try:
                name = str(np.asarray(rows[0].NAME.data).item())
            except Exception:
                pass
    zero_dir = np.zeros((1, 1, 2), dtype=np.float64)
    return Dataset({
        "NAME":          (("row",), to_dask(np.array([name], dtype=object))),
        "CODE":          (("row",), to_dask(np.array([""], dtype=object))),
        "TIME":          (("row",), to_dask(np.zeros(1, dtype=np.float64))),
        "NUM_POLY":      (("row",), to_dask(np.zeros(1, dtype=np.int32))),
        "PHASE_DIR":     (("row", "d0", "ra_dec"), to_dask(zero_dir)),
        "DELAY_DIR":     (("row", "d0", "ra_dec"), to_dask(zero_dir)),
        "REFERENCE_DIR": (("row", "d0", "ra_dec"), to_dask(zero_dir)),
        "FLAG_ROW":      (("row",), to_dask(np.zeros(1, dtype=bool))),
        "SOURCE_ID":     (("row",), to_dask(np.zeros(1, dtype=np.int32))),
    })


def build_dd_table(first_ms, num_ms):
    """DATA_DESCRIPTION table: row i points to spectral window i.

    The polarisation setup is taken from the first input.
    """
    pol_id = 0
    with casacore_lock:
        rows = xds_from_table(f"{first_ms}::DATA_DESCRIPTION", group_cols="__row__")
        if rows:
            try:
                pol_id = int(np.asarray(rows[0].POLARIZATION_ID.data).item())
            except Exception:
                pass
    return Dataset({
        "SPECTRAL_WINDOW_ID": (("row",), to_dask(np.arange(num_ms, dtype=np.int32))),
        "POLARIZATION_ID":    (("row",), to_dask(np.full(num_ms, pol_id, dtype=np.int32))),
        "FLAG_ROW":           (("row",), to_dask(np.zeros(num_ms, dtype=bool))),
    })


def copy_subtable_first_row(ms_path, name):
    """First row of a subtable of an input MS (used for POLARIZATION).

    Returns
    -------
    Dataset or None
        None when the subtable is missing or empty.
    """
    with casacore_lock:
        try:
            rows = xds_from_table(f"{ms_path}::{name}", group_cols="__row__")
        except Exception:
            return None
        if not rows:
            return None
        r = rows[0]
        names = list(r.data_vars)
        got = dask.compute(*[r[v].data for v in names])
    data_vars = {}
    for v, arr in zip(names, got):
        arr = np.asarray(arr)
        if arr.ndim >= 1 and r[v].dims[0] == "row":
            arr = arr[:1]
        data_vars[v] = (r[v].dims, to_dask(arr))
    return Dataset(data_vars)


def build_observation_table(ms_list):
    """OBSERVATION table: the OBSERVATION rows of all the inputs.

    Returns
    -------
    list of Dataset
        One per row; an empty list when no input has the table.
    """
    def load(ms):
        with casacore_lock:
            try:
                return xds_from_table(f"{ms}::OBSERVATION", group_cols="__row__") or []
            except Exception:
                return []

    with ThreadPoolExecutor(max_workers=min(8, len(ms_list))) as pool:
        rows = [r for per_ms in pool.map(load, ms_list) for r in per_ms]
    if not rows:
        return []
    names = [list(r.data_vars) for r in rows]
    lazy = [r[v].data for r, vs in zip(rows, names) for v in vs]
    if not lazy:
        return []
    with casacore_lock:
        got = iter(dask.compute(*lazy))
    return [Dataset({v: (r[v].dims, to_dask(np.asarray(next(got)))) for v in vs})
            for r, vs in zip(rows, names)]


def build_feed_table(nant):
    """FEED table: one dual-linear (X, Y) feed per antenna."""
    return [Dataset({
        "ANTENNA_ID":         (("row",), to_dask(np.arange(nant, dtype=np.int32))),
        "FEED_ID":            (("row",), to_dask(np.zeros(nant, dtype=np.int32))),
        "SPECTRAL_WINDOW_ID": (("row",), to_dask(np.full(nant, -1, dtype=np.int32))),
        "TIME":               (("row",), to_dask(np.zeros(nant, dtype=np.float64))),
        "INTERVAL":           (("row",), to_dask(np.zeros(nant, dtype=np.float64))),
        "NUM_RECEPTORS":      (("row",), to_dask(np.full(nant, 2, dtype=np.int32))),
        "POLARIZATION_TYPE":  (("row", "receptors"),
                               to_dask(np.array([["X", "Y"]] * nant, dtype=object))),
        "POL_RESPONSE":       (("row", "receptors", "receptors2"),
                               to_dask(np.tile(np.eye(2, dtype=complex), (nant, 1, 1)))),
        "POSITION":           (("row", "xyz"), to_dask(np.zeros((nant, 3), dtype=np.float64))),
        "BEAM_OFFSET":        (("row", "receptors", "radec"),
                               to_dask(np.zeros((nant, 2, 2), dtype=np.float64))),
        "RECEPTOR_ANGLE":     (("row", "receptors"), to_dask(np.zeros((nant, 2), dtype=np.float64))),
        "BEAM_ID":            (("row",), to_dask(np.full(nant, -1, dtype=np.int32))),
    })]


def build_source_table(central_freq_hz):
    """SOURCE table: one placeholder source with the line rest frequency.

    Enough for imaging tools to treat the MS as a spectral-line dataset.
    """
    return [Dataset({
        "SOURCE_ID":          (("row",), to_dask(np.zeros(1, dtype=np.int32))),
        "TIME":               (("row",), to_dask(np.zeros(1, dtype=np.float64))),
        "INTERVAL":           (("row",), to_dask(np.zeros(1, dtype=np.float64))),
        "SPECTRAL_WINDOW_ID": (("row",), to_dask(np.array([-1], dtype=np.int32))),
        "NUM_LINES":          (("row",), to_dask(np.ones(1, dtype=np.int32))),
        "NAME":               (("row",), to_dask(np.array(["J0000+0000"], dtype=object))),
        "CALIBRATION_GROUP":  (("row",), to_dask(np.zeros(1, dtype=np.int32))),
        "CODE":               (("row",), to_dask(np.array([""], dtype=object))),
        "DIRECTION":          (("row", "radec"), to_dask(np.zeros((1, 2), dtype=np.float64))),
        "PROPER_MOTION":      (("row", "pm"), to_dask(np.zeros((1, 2), dtype=np.float64))),
        "REST_FREQUENCY":     (("row", "lines"), to_dask(np.array([[central_freq_hz]], dtype=np.float64))),
        "SYSVEL":             (("row", "lines"), to_dask(np.zeros((1, 1), dtype=np.float64))),
    })]
