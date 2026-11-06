"""
vista.stackio — storage of the stacked visibilities
===================================================

Default output of the stacking: a Zarr store with the layout of dask-ms
(``daskms.experimental.zarr``)::

    stack.zarr/
        MAIN/MAIN_<k>/      one group per input dataset, k = its DATA_DESC_ID:
                            DATA, FLAG, UVW, WEIGHT, SIGMA, TIME, ANTENNA1, ...
                            as in an MS main table, chunked along the rows
        SPECTRAL_WINDOW/, DATA_DESCRIPTION/, ANTENNA/, FIELD/, ...  subtables

Every chunk of every column is a file of its own, so several threads write
(and read) different chunks at the same time, without locks; the columns are
compressed (Blosc lz4), which makes flags and empty channels almost free.
The store reads back with ``daskms.experimental.zarr.xds_from_zarr`` (or
plain ``zarr``); :func:`export_ms` turns it into a Measurement Set when one is
needed, e.g. for imaging with tclean.  The rows of each dataset are in their
input order, so the exported MS has the row order of ViSta 2.2.

:func:`open_stack` gives the extraction one interface for both formats, and
:func:`process_pieces` overlaps their I/O with the computation.
"""

import os
import shutil
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import _say, _detail

STORE_VERSION = 1

# Columns of the main table of the store: name -> (dims, dtype)
MAIN_COLUMNS = {
    "DATA":           (("row", "chan", "corr"), np.complex64),
    "FLAG":           (("row", "chan", "corr"), np.bool_),
    "UVW":            (("row", "uvw"), np.float64),
    "WEIGHT":         (("row", "corr"), np.float32),
    "SIGMA":          (("row", "corr"), np.float32),
    "TIME":           (("row",), np.float64),
    "TIME_CENTROID":  (("row",), np.float64),
    "EXPOSURE":       (("row",), np.float64),
    "INTERVAL":       (("row",), np.float64),
    "ANTENNA1":       (("row",), np.int32),
    "ANTENNA2":       (("row",), np.int32),
    "DATA_DESC_ID":   (("row",), np.int32),
    "FIELD_ID":       (("row",), np.int32),
    "SCAN_NUMBER":    (("row",), np.int32),
    "STATE_ID":       (("row",), np.int32),
    "FEED1":          (("row",), np.int32),
    "FEED2":          (("row",), np.int32),
    "PROCESSOR_ID":   (("row",), np.int32),
    "ARRAY_ID":       (("row",), np.int32),
    "OBSERVATION_ID": (("row",), np.int32),
    "FLAG_ROW":       (("row",), np.bool_),
}
# visibility-like columns that the extraction may add (same shape as DATA)
EXTRA_DATA_COLUMNS = ("MODEL_DATA", "CORRECTED_DATA")

_DASKMS_ATTR = "__daskms_zarr_attr__"
_PARTITION_KEY = "__daskms_partition_schema__"


def is_zarr(path):
    return os.path.isfile(os.path.join(path, ".zgroup"))


def is_ms(path):
    return os.path.isfile(os.path.join(path, "table.dat"))


def _compressor(compress):
    if not compress:
        return None
    from numcodecs import Blosc
    return Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)


def _array_attrs(dims):
    """Attributes dask-ms puts on every array of a Zarr store."""
    return {"_ARRAY_DIMENSIONS": list(dims),
            _DASKMS_ATTR: {"array_type": "dask.array.core.Array",
                           "coordinate": False, "dims": list(dims)}}


# ===========================================================================
# Writing the store (stacking)
# ===========================================================================

class ZarrStackWriter:
    """Zarr store being written by the stacking.

    ``add_partition`` is called once per input dataset, before its chunks
    (by a single thread); ``write`` may then be called from many threads at
    once, each with whole chunks of rows (chunk-aligned writes never touch the
    same file).  Columns never written read back as their fill value: 0, or
    the dataset index for DATA_DESC_ID.
    """

    # columns whose value is known without writing it
    _FILLED = {"DATA_DESC_ID", "FIELD_ID", "FEED1", "FEED2", "PROCESSOR_ID",
               "ARRAY_ID", "OBSERVATION_ID", "TIME_CENTROID"}

    def __init__(self, path, compress=True, attrs=None):
        try:
            import zarr
        except ImportError as e:
            raise ImportError("output_format='zarr' needs zarr 2 "
                              "(pip install 'zarr>=2.12,<3'), or use "
                              "output_format='ms'") from e
        self.path = path
        self.compressor = _compressor(compress)
        self.root = zarr.open_group(path, mode="w")
        self.main = self.root.require_group("MAIN")
        self.root.attrs["vista"] = dict(attrs or {}, store_version=STORE_VERSION)
        self.parts = {}

    def add_partition(self, k, nrows, chunk_rows, nchan, ncorr, extra=()):
        """The group of dataset *k*; *extra*: visibility columns written
        besides DATA (MODEL_DATA, CORRECTED_DATA of the continuum subtraction
        during the stacking)."""
        sizes = {"row": int(nrows), "chan": int(nchan), "corr": int(ncorr), "uvw": 3}
        chunk_rows = max(1, min(int(chunk_rows), int(nrows)))
        g = self.main.require_group(f"MAIN_{k}")
        arrays = {}
        columns = dict(MAIN_COLUMNS)
        for name in extra:
            columns[name] = MAIN_COLUMNS["DATA"]
        for name, (dims, dtype) in columns.items():
            shape = tuple(sizes[d] for d in dims)
            chunks = tuple(chunk_rows if d == "row" else sizes[d] for d in dims)
            fill = k if name == "DATA_DESC_ID" else 0
            a = g.create_dataset(name, shape=shape, chunks=chunks, dtype=dtype,
                                 fill_value=fill, compressor=self.compressor,
                                 overwrite=True)
            a.attrs.put(_array_attrs(dims))
            arrays[name] = a
        n_full, rest = divmod(int(nrows), chunk_rows)
        row_chunks = [chunk_rows] * n_full + ([rest] if rest else [])
        g.attrs.put({"DATA_DESC_ID": int(k), "FIELD_ID": 0,
                     _PARTITION_KEY: [["FIELD_ID", "int32"], ["DATA_DESC_ID", "int32"]],
                     _DASKMS_ATTR: {"chunks": {"row": row_chunks, "chan": [int(nchan)],
                                               "corr": [int(ncorr)], "uvw": [3]}}})
        self.parts[k] = arrays

    def write(self, k, row0, item):
        """Write the columns of one chunk (dict name -> array) at row0."""
        arrays = self.parts[k]
        for name, value in item.items():
            a = arrays.get(name)
            if a is None or name in self._FILLED:
                continue
            a[row0:row0 + len(value)] = np.asarray(value, dtype=a.dtype)

    def set_record(self, name, value):
        """Store a JSON-serialisable record in the attributes of the store."""
        attrs = dict(self.root.attrs.get("vista", {}))
        attrs[name] = value
        self.root.attrs["vista"] = attrs

    def write_subtables(self, subtables):
        """subtables: {name: Dataset, list of Datasets or None} (dask-ms Datasets)."""
        import dask
        from daskms.experimental.zarr import xds_to_zarr
        _detail(f"Writing output subtables to {self.path} ...")
        for name, content in subtables.items():
            datasets = content if isinstance(content, list) else [content]
            datasets = [d for d in datasets if d is not None]
            if not datasets:
                continue
            dask.compute(xds_to_zarr(datasets, f"{self.path}::{name}"))
            _detail(f"  Written: {name}")

    def close(self):
        import zarr
        for k in self.parts:
            zarr.consolidate_metadata(
                zarr.DirectoryStore(os.path.join(self.path, "MAIN", f"MAIN_{k}")))
        self.parts = {}


def store_size(path):
    """Bytes on disk of a directory tree."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


# ===========================================================================
# Reading a stack (extraction): Zarr store or MS, one interface
# ===========================================================================

def open_stack(path, writable=False):
    """Open a stacked dataset, Zarr store or Measurement Set."""
    if is_zarr(path):
        return ZarrStack(path, writable)
    if is_ms(path):
        return MSStack(path, writable)
    raise FileNotFoundError(f"{path!r} is neither a ViSta Zarr store nor a "
                            f"Measurement Set (cwd={os.getcwd()})")


class ZarrStack:
    kind = "zarr"
    thread_safe = True          # different chunks can be read/written at once

    def __init__(self, path, writable=False):
        import zarr
        self.path = path
        self.root = zarr.open_group(path, mode="r+" if writable else "r")
        self.groups = {}
        for name, g in self.root["MAIN"].groups():
            self.groups[int(name.rsplit("_", 1)[1])] = g
        self._modified = set()

    def dd_ids(self):
        return sorted(self.groups)

    def _subtable(self, name):
        return self.root[name][f"{name}_0"]

    def dd_frequencies(self):
        spw = self._subtable("SPECTRAL_WINDOW")
        freq = np.asarray(spw["CHAN_FREQ"][:], dtype=np.float64)
        nchan = (np.asarray(spw["NUM_CHAN"][:]) if "NUM_CHAN" in spw
                 else [freq.shape[1]] * freq.shape[0])
        per_spw = [freq[i, :int(nchan[i])] for i in range(freq.shape[0])]
        dd_to_spw = np.asarray(self._subtable("DATA_DESCRIPTION")["SPECTRAL_WINDOW_ID"][:])
        return {dd: per_spw[int(spw_id)] for dd, spw_id in enumerate(dd_to_spw)}

    def dd_to_spw(self):
        return [int(s) for s in self._subtable("DATA_DESCRIPTION")["SPECTRAL_WINDOW_ID"][:]]

    def n_rows(self, dd):
        return int(self.groups[dd]["DATA"].shape[0])

    def columns(self, dd):
        return set(self.groups[dd].array_keys())

    def pieces(self, dd, max_rows=None):
        """Chunk-aligned (row0, n) pieces of a dataset."""
        n = self.n_rows(dd)
        step = int(self.groups[dd]["DATA"].chunks[0])
        return [(r, min(step, n - r)) for r in range(0, n, step)]

    def read(self, dd, row0, n, columns):
        g = self.groups[dd]
        return {c: g[c][row0:row0 + n] for c in columns}

    def read_first(self, dd, n, columns):
        """The first *n* rows of a dataset (all of them if it has fewer)."""
        return self.read(dd, 0, min(int(n), self.n_rows(dd)), columns)

    def write(self, dd, row0, name, value):
        a = self.groups[dd][name]
        a[row0:row0 + len(value)] = np.asarray(value, dtype=a.dtype)
        self._modified.add(dd)

    def get_record(self, name):
        """A record stored with :meth:`set_record`, or None."""
        return dict(self.root.attrs.get("vista", {})).get(name)

    def set_record(self, name, value):
        """Store a JSON-serialisable record in the attributes of the store."""
        attrs = dict(self.root.attrs.get("vista", {}))
        attrs[name] = value
        self.root.attrs["vista"] = attrs

    def add_columns(self, names, like="DATA"):
        """Create visibility columns shaped and chunked like ``like``."""
        for dd, g in self.groups.items():
            ref = g[like]
            for name in names:
                if name in g:
                    continue
                a = g.create_dataset(name, shape=ref.shape, chunks=ref.chunks,
                                     dtype=ref.dtype, fill_value=0,
                                     compressor=ref.compressor)
                a.attrs.put(dict(ref.attrs))
                self._modified.add(dd)

    def close(self):
        import zarr
        for dd in self._modified:
            zarr.consolidate_metadata(
                zarr.DirectoryStore(os.path.join(self.path, "MAIN", f"MAIN_{dd}")))
        self._modified = set()


class MSStack:
    kind = "ms"
    thread_safe = False         # one casacore table: one thread at a time

    def __init__(self, path, writable=False):
        from casacore.tables import table
        self.path = path
        self.tab = table(path, readonly=not writable, ack=False)
        self._dd = np.asarray(self.tab.getcol("DATA_DESC_ID"))

    def _sub(self, name):
        from casacore.tables import table
        return table(f"{self.path}::{name}", ack=False)

    def dd_frequencies(self):
        with self._sub("SPECTRAL_WINDOW") as t:
            n_chan = (t.getcol("NUM_CHAN") if "NUM_CHAN" in t.colnames()
                      else [None] * t.nrows())
            per_spw = []
            for i in range(t.nrows()):
                f = np.asarray(t.getcell("CHAN_FREQ", i), dtype=np.float64)
                if n_chan[i] is not None:           # strip any zero padding
                    f = f[:int(n_chan[i])]
                per_spw.append(f)
        return {dd: per_spw[s] for dd, s in enumerate(self.dd_to_spw())}

    def dd_to_spw(self):
        with self._sub("DATA_DESCRIPTION") as t:
            return [int(s) for s in t.getcol("SPECTRAL_WINDOW_ID")]

    def dd_ids(self):
        return sorted(int(d) for d in np.unique(self._dd))

    def n_rows(self, dd):
        return int(np.count_nonzero(self._dd == dd))

    def columns(self, dd):
        return set(self.tab.colnames())

    def pieces(self, dd, max_rows=None):
        """(row0, n) pieces over runs of consecutive rows of a dataset
        (the rows of a dataset need not be consecutive in the table)."""
        rows = np.flatnonzero(self._dd == dd)
        if rows.size == 0:
            return []
        step = int(max_rows) if max_rows else rows.size
        breaks = np.flatnonzero(np.diff(rows) != 1) + 1
        return [(int(run[0]) + s, min(step, len(run) - s))
                for run in np.split(rows, breaks) for s in range(0, len(run), step)]

    def read(self, dd, row0, n, columns):
        return {c: self.tab.getcol(c, startrow=row0, nrow=n) for c in columns}

    def read_first(self, dd, n, columns):
        """The first *n* rows of a dataset, over its runs of consecutive rows."""
        parts, left = [], int(n)
        for row0, m in self.pieces(dd):
            if left <= 0:
                break
            parts.append(self.read(dd, row0, min(m, left), columns))
            left -= min(m, left)
        if len(parts) == 1:
            return parts[0]
        return {c: np.concatenate([p[c] for p in parts]) for c in columns}

    def write(self, dd, row0, name, value):
        self.tab.putcol(name, value, startrow=row0, nrow=len(value))

    def get_record(self, name):
        """A record stored with :meth:`set_record`, or None."""
        import json
        key = f"VISTA_{name.upper()}"
        if key not in self.tab.getkeywords():
            return None
        try:
            return json.loads(self.tab.getkeyword(key))
        except (TypeError, ValueError):
            return None

    def set_record(self, name, value):
        """Store a record as a JSON keyword of the main table."""
        import json
        self.tab.putkeyword(f"VISTA_{name.upper()}", json.dumps(value))

    def add_columns(self, names, like="DATA"):
        from casacore.tables import makecoldesc, maketabdesc
        missing = [n for n in names if n not in self.tab.colnames()]
        if not missing:
            return
        desc = self.tab.getcoldesc(like)
        shape = self.tab.getcell(like, 0).shape          # (nchan, ncorr)
        nchan, ncorr = int(shape[0]), int(shape[1])
        ch_tile = min(nchan, 16)
        data_rows = max(1, (1 << 20) // (ncorr * ch_tile * 8))
        for name in missing:
            dm = {"TYPE": "TiledColumnStMan", "NAME": f"{name}_GROUP",
                  "SPEC": {"DEFAULTTILESHAPE": np.array([ncorr, ch_tile, data_rows], np.int32)}}
            self.tab.addcols(maketabdesc(makecoldesc(name, desc)), dm)

    def close(self):
        try:
            self.tab.close()
        except Exception:
            pass


# ===========================================================================
# Overlapping I/O and computation
# ===========================================================================

def process_pieces(stack, tasks, read, compute, write=None, n_io=4):
    """Run ``compute`` over ``tasks`` with the I/O in background threads.

    read(task) -> data          in up to n_io threads, a few tasks ahead
    compute(task, data) -> res  here, in task order (the kernels release the GIL)
    write(task, res)            in up to n_io threads (skipped when res is None)

    A Measurement Set is read and written by one thread at a time; a Zarr
    store by n_io threads at once.  Memory: about 2 * n_io + 2 tasks in flight.
    """
    n_io = max(1, int(n_io)) if stack.thread_safe else 1
    lock = threading.Lock() if not stack.thread_safe else None

    def _read(task):
        if lock is None:
            return read(task)
        with lock:
            return read(task)

    def _write(task, res):
        if lock is None:
            return write(task, res)
        with lock:
            return write(task, res)

    it = iter(tasks)
    with ThreadPoolExecutor(n_io) as rpool, ThreadPoolExecutor(n_io) as wpool:
        pending, writes = deque(), deque()
        for _ in range(n_io + 1):
            t = next(it, None)
            if t is None:
                break
            pending.append((t, rpool.submit(_read, t)))
        while pending:
            task, fut = pending.popleft()
            data = fut.result()
            t = next(it, None)
            if t is not None:
                pending.append((t, rpool.submit(_read, t)))
            res = compute(task, data)
            del data
            if write is not None and res is not None:
                writes.append(wpool.submit(_write, task, res))
                while len(writes) > n_io:
                    writes.popleft().result()
        while writes:
            writes.popleft().result()


# ===========================================================================
# Export to a Measurement Set
# ===========================================================================

def export_ms(store, ms_path, overwrite=True, verbose=True, n_io=4):
    """Write a ViSta Zarr store as a Measurement Set.

    The main table gets the columns of the store (with MODEL_DATA and
    CORRECTED_DATA when the continuum subtraction added them), dataset after
    dataset in input order; the subtables are copied with dask-ms.
    """
    import gc
    import time
    import dask
    from casacore.tables import default_ms, makearrcoldesc, maketabdesc
    from daskms import xds_to_table
    from daskms.experimental.zarr import xds_from_zarr
    from . import _reset_clock
    from .ms_output import output_dminfo

    _reset_clock()
    t0 = time.perf_counter()
    st = ZarrStack(store)
    dds = st.dd_ids()
    if not dds:
        raise RuntimeError(f"{store}: no dataset in the store")
    if os.path.exists(ms_path):
        if not overwrite:
            raise FileExistsError(ms_path)
        shutil.rmtree(ms_path)
    first = st.groups[dds[0]]
    nchan, ncorr = int(first["DATA"].shape[1]), int(first["DATA"].shape[2])
    extra = [c for c in EXTRA_DATA_COLUMNS if c in first]
    columns = [c for c in MAIN_COLUMNS if c in first] + extra
    _say(f"Export to MS: {store} → {ms_path}, {len(dds)} datasets, "
         f"{sum(st.n_rows(d) for d in dds):,} rows"
         + (f", with {', '.join(extra)}" if extra else ""), verbose)

    desc = maketabdesc(
        [makearrcoldesc("DATA", 0j, shape=[nchan, ncorr], valuetype="complex"),
         makearrcoldesc("FLAG", False, shape=[nchan, ncorr]),
         makearrcoldesc("WEIGHT", 0.0, shape=[ncorr], valuetype="float"),
         makearrcoldesc("SIGMA", 0.0, shape=[ncorr], valuetype="float")]
        + [makearrcoldesc(c, 0j, shape=[nchan, ncorr], valuetype="complex") for c in extra])
    tab = default_ms(ms_path, desc, output_dminfo(nchan, ncorr, extra))
    out_cols = set(tab.colnames())
    tasks = [(dd, r0, n) for dd in dds for (r0, n) in st.pieces(dd)]

    def read(task):
        dd, r0, n = task
        return st.read(dd, r0, n, [c for c in columns if c in st.groups[dd]])

    def compute(task, data):          # here: append to the MS (one thread)
        n = task[2]
        start = tab.nrows()
        tab.addrows(n)
        for c, a in data.items():
            if c in out_cols:
                tab.putcol(c, a, startrow=start, nrow=n)
        return None

    try:
        process_pieces(st, tasks, read, compute, None, n_io=n_io)
    finally:
        tab.flush()
        tab.close()

    for name in sorted(os.listdir(store)):
        if name == "MAIN" or not os.path.isdir(os.path.join(store, name)):
            continue
        xds = xds_from_zarr(f"{store}::{name}")
        if xds:
            dask.compute(xds_to_table(xds, f"{ms_path}::{name}", "ALL"))
            _detail(f"  Written: {name}")
    gc.collect()
    _say(f"Export done in {time.perf_counter() - t0:.1f} s: {ms_path}", verbose)
    return ms_path


__all__ = ["open_stack", "export_ms", "ZarrStackWriter", "ZarrStack", "MSStack",
           "process_pieces", "is_zarr", "is_ms", "store_size",
           "MAIN_COLUMNS", "EXTRA_DATA_COLUMNS"]
