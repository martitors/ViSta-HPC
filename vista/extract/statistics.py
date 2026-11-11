"""
vista.extract.statistics
========================
Compression of the visibilities into sufficient statistics (Sec. 5.3.1).

For every spectral channel the visibilities are binned radially into
logarithmically spaced annuli of projected rest-frame baseline length, and
five accumulators are stored per annulus:

======  ====================================================
index   accumulator
======  ====================================================
0       ``sum w * Re(V)``
1       ``sum w * Im(V)``
2       ``sum w``
3       ``sum w * b``
4       ``N``, number of visibilities that fell in the bin
5       ``sum w * Re(V)^2``
6       ``sum w * Im(V)^2``
======  ====================================================

The last two (format 4) measure the scatter of the visibilities inside each
cell, hence the noise, independently of the absolute scale of the weights:
see :mod:`vista.extract.noise`.  Files of format 3 have only the first five.

Every subsequent operation, from the spectra to the amplitude profiles, the
model fits and the bootstrap, runs on this compact representation instead of
the full visibility set, which is what makes the repeated resampling cheap.

The stack (Zarr store or MS) is opened in **read-only** mode.  Each line of the input list is
compressed as an independent descriptor, so that different observations of
the same source can be combined later, and the result is checkpointed after
every data descriptor, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .. import _reset_clock, _say, _warn
from .config import BinningConfig, C_MS, Lines
from .sources import Entry, read_input_list

FORMAT_VERSION = 4
N_ACC = 7                     # accumulators per cell (5 in format 3)


# ---------------------------------------------------------------------------
# stack access helpers
# ---------------------------------------------------------------------------
def data_descriptor_frequencies(ms_path: str) -> Dict[int, np.ndarray]:
    """Channel frequencies of every data descriptor of the stack (Zarr or MS)."""
    from ..stackio import open_stack
    stack = open_stack(ms_path)
    try:
        return stack.dd_frequencies()
    finally:
        stack.close()


def _kernel():
    """The compiled kernel (vista/ms_ops), or None: NumPy is used instead."""
    try:
        from .. import ms_ops
    except ImportError:
        try:
            import ms_ops
        except ImportError:
            return None
    return ms_ops if hasattr(ms_ops, "annuli_accumulate") else None


def primary_beam_of_entry(entry: Entry, nu_obs_hz: float):
    """Primary beam response at the source position, for one input line.

    The original phase centre (FIELD ``PHASE_DIR`` of the selected field) and
    the dish diameter (median of ``DISH_DIAMETER``) are read from the
    original MS of the entry.  Returns ``(A, theta_arcsec, dish_m)``, or
    ``(1.0, nan, nan)`` if that MS cannot be read.
    """
    from .sources import primary_beam_gain
    try:
        from casacore.tables import table
        with table(f"{entry.ms}::FIELD", ack=False) as t:
            direction = np.asarray(t.getcell("PHASE_DIR", entry.field or 0),
                                   dtype=float).ravel()
        with table(f"{entry.ms}::ANTENNA", ack=False) as t:
            dish = np.asarray(t.getcol("DISH_DIAMETER"), dtype=float)
    except Exception:
        return 1.0, float("nan"), float("nan")
    ra0, dec0 = float(direction[0]), float(direction[1])
    h = (np.sin(0.5 * (entry.dec - dec0)) ** 2
         + np.cos(dec0) * np.cos(entry.dec) * np.sin(0.5 * (entry.ra - ra0)) ** 2)
    theta = 2.0 * float(np.arcsin(min(1.0, np.sqrt(h))))
    d = float(np.median(dish[dish > 0])) if np.any(dish > 0) else 0.0
    return primary_beam_gain(theta, nu_obs_hz, d), theta * 206264.806, d


def build_global_grid(dd_frequencies: Dict[int, np.ndarray], verbose=True):
    """Common frequency grid onto which all the data descriptors are mapped.

    ViSta gives every output spectral window the same channel width and the
    same number of channels, but a different start frequency, so the grid is
    simply the union of the windows on that common step.
    """
    everything = np.concatenate([f for f in dd_frequencies.values() if len(f)])
    steps = np.concatenate([np.abs(np.diff(f))
                            for f in dd_frequencies.values() if len(f) > 1])
    dnu = float(np.median(steps))
    f0 = float(everything.min())
    n = int(np.rint((everything.max() - f0) / dnu)) + 1
    grid = f0 + np.arange(n) * dnu
    worst = max(np.abs(f - grid[np.clip(np.rint((f - f0) / dnu).astype(int),
                                        0, n - 1)]).max()
                for f in dd_frequencies.values() if len(f))
    _say(f"  common grid: {n} channels of {dnu / 1e6:.3f} MHz "
         f"(largest mapping error {100 * worst / dnu:.1f}% of a channel)", verbose)
    return grid, f0, dnu


def _checkpoint(path: str, payload: dict):
    """Atomic save, so an interrupted write cannot corrupt the file."""
    temporary = path + ".tmp.npy"
    np.save(temporary, payload, allow_pickle=True)
    os.replace(temporary, path)


# ---------------------------------------------------------------------------
# the accumulation kernel
# ---------------------------------------------------------------------------
def accumulate_annuli(accumulator, data, weight, b_lambda, grid_index,
                      grid_ok, edges, n_grid, n_bins):
    """Add one chunk of visibilities to the five accumulators."""
    bin_index = np.searchsorted(edges, b_lambda, side="right") - 1
    in_band = (bin_index >= 0) & (bin_index < n_bins) & grid_ok[:, None]
    bin_index = np.clip(bin_index, 0, n_bins - 1)
    key_2d = np.clip(grid_index, 0, n_grid - 1)[:, None] * n_bins + bin_index
    key = np.broadcast_to(key_2d, weight.shape).ravel()
    # products in float64, as in the kernel (weights and visibilities are
    # float32 on disk: their product in float32 would be rounded)
    w = (np.asarray(weight, dtype=np.float64) * in_band[None, :, :]).ravel()
    real = np.asarray(data.real, dtype=np.float64).ravel()
    imaginary = np.asarray(data.imag, dtype=np.float64).ravel()
    lengths = np.broadcast_to(b_lambda, data.shape).ravel()
    good = w > 0
    if not good.any():
        return
    k = key[good]
    size = n_grid * n_bins
    accumulator[:, 0] += np.bincount(k, weights=w[good] * real[good],
                                     minlength=size)
    accumulator[:, 1] += np.bincount(k, weights=w[good] * imaginary[good],
                                     minlength=size)
    accumulator[:, 2] += np.bincount(k, weights=w[good], minlength=size)
    accumulator[:, 3] += np.bincount(k, weights=w[good] * lengths[good],
                                     minlength=size)
    accumulator[:, 4] += np.bincount(k, minlength=size)
    if accumulator.shape[1] >= 7:
        accumulator[:, 5] += np.bincount(k, weights=w[good] * real[good] ** 2,
                                         minlength=size)
        accumulator[:, 6] += np.bincount(k, weights=w[good] * imaginary[good] ** 2,
                                         minlength=size)


def accumulate_chunk(acc_line, acc_cont, line, cont, flag, flag_row, row_ok,
                     weight, uvw, freq, grid_index, grid_ok, edges,
                     max_amplitude=None, kernel=None):
    """Add one chunk (row-first arrays, as read from the stack) to the
    accumulators; returns the number of corrupt samples rejected.

    line, cont : complex [nrow, nchan, ncorr] (cont may be None)
    flag       : bool [nrow, nchan, ncorr];  flag_row, row_ok : bool [nrow]
    weight     : [nrow, ncorr] (WEIGHT) or [nrow, nchan, ncorr] (WEIGHT_SPECTRUM)
    uvw        : [nrow, 3]
    """
    if kernel is not None:
        return int(kernel.annuli_accumulate(
            acc_line, acc_cont, np.ascontiguousarray(line, np.complex64),
            None if cont is None else np.ascontiguousarray(cont, np.complex64),
            np.ascontiguousarray(flag, bool), np.ascontiguousarray(flag_row, bool),
            np.ascontiguousarray(row_ok, bool), np.ascontiguousarray(weight, np.float32),
            np.ascontiguousarray(uvw, np.float64), np.ascontiguousarray(freq, np.float64),
            np.ascontiguousarray(grid_index, np.int64), np.ascontiguousarray(grid_ok, bool),
            np.ascontiguousarray(edges, np.float64),
            -1.0 if max_amplitude is None else float(max_amplitude)))

    # NumPy: the arrays in the [ncorr, nchan, nrow] layout of the original code
    sel = np.asarray(row_ok, bool)
    n_chan = line.shape[1]
    n_grid, n_bins = acc_line.shape[0] // (len(edges) - 1), len(edges) - 1
    data_line = line[sel].transpose(2, 1, 0)
    data_cont = None if cont is None else cont[sel].transpose(2, 1, 0)
    flg = flag[sel].transpose(2, 1, 0)
    flag_row = np.asarray(flag_row, bool)[sel]
    w = weight[sel]
    w = (w.transpose(2, 1, 0) if w.ndim == 3
         else w.T[:, None, :] * np.ones((1, n_chan, 1)))
    u = uvw[sel].T
    good = ((~flg) & (~flag_row[None, None, :]) & (w > 0)
            & (data_line != 0) & np.isfinite(data_line))
    if max_amplitude is not None:
        good &= np.abs(data_line) <= max_amplitude
    if data_cont is not None:
        corrupt = ~np.isfinite(data_cont)
        if max_amplitude is not None:
            corrupt |= np.abs(data_cont) > max_amplitude
        good &= ~corrupt
    n_rejected = int(np.sum((~good) & (~flg) & (~flag_row[None, None, :])
                            & (w > 0) & (data_line != 0)))
    w = np.where(good, w, 0.0)
    data_line = np.where(good, data_line, 0.0)
    b_lambda = (np.sqrt(u[0] ** 2 + u[1] ** 2)[None, :] * (freq[:, None] / C_MS))
    accumulate_annuli(acc_line, data_line, w, b_lambda, grid_index, grid_ok,
                      edges, n_grid, n_bins)
    if data_cont is not None:
        data_cont = np.where(good, data_cont, 0.0)
        accumulate_annuli(acc_cont, data_cont, w, b_lambda, grid_index, grid_ok,
                          edges, n_grid, n_bins)
    return n_rejected


# ---------------------------------------------------------------------------
# the container
# ---------------------------------------------------------------------------
@dataclass
class SufficientStatistics:
    """Compressed statistics of one stacked dataset."""

    sums: Dict[int, np.ndarray]                 # dd -> (nchan, nbins, 7) (5: format 3)
    freq_grid: np.ndarray                       # (nchan,) Hz
    bin_edges_lambda: np.ndarray                # (nbins+1,)
    ms_path: Optional[str] = None
    meta: Optional[dict] = None

    @classmethod
    def load(cls, path: str) -> "SufficientStatistics":
        """Load a file written by :func:`compress_visibilities`."""
        raw = np.load(path, allow_pickle=True).item()
        if "freq_grid" not in raw:
            raise RuntimeError(
                f"{path}: no frequency grid stored, rerun the compression")
        sums = {int(k): v["sums"] for k, v in raw.items()
                if isinstance(k, (int, np.integer))}
        return cls(sums=sums,
                   freq_grid=np.asarray(raw["freq_grid"]),
                   bin_edges_lambda=np.asarray(raw["bin_edges_lambda"]),
                   ms_path=raw.get("ms_path"),
                   meta={k: v for k, v in raw.items()
                         if not isinstance(k, (int, np.integer))})

    @property
    def n_bins(self) -> int:
        return len(self.bin_edges_lambda) - 1

    @property
    def has_scatter(self) -> bool:
        """True when the cells carry sum w Re^2 and sum w Im^2 (format 4)."""
        return bool(self.sums) and all(np.shape(v)[-1] >= 7 for v in self.sums.values())

    def available(self) -> List[int]:
        return sorted(self.sums)


# ---------------------------------------------------------------------------
# the compression itself
# ---------------------------------------------------------------------------
def compress_visibilities(ms_path: str,
                          input_list: Union[str, Sequence[Optional[Entry]]],
                          line: Optional[Lines] = None,
                          binning: BinningConfig = BinningConfig(),
                          out_line: str = "stack_line_stats.npy",
                          out_continuum: Optional[str] = None,
                          line_column: Optional[str] = None,
                          continuum_column: str = "MODEL_DATA",
                          max_amplitude: Optional[float] = None,
                          chunk_mb: float = 512.0,
                          resume: bool = True,
                          verbose: bool = True,
                          n_io: int = 4,
                          one_channel: bool = False) -> Dict[str, str]:
    """Compress a stack into sufficient statistics, descriptor by
    descriptor.

    Parameters
    ----------
    ms_path
        The stack: Zarr store written by the stacking, or Measurement Set.
        Opened read-only.
    input_list
        Path to the ViSta input list, or the already parsed entries.  Its
        order defines the data descriptor numbering.
    line
        Optional; kept only so that the spectral setup can be recorded in the
        output file.  The compression itself is line-agnostic: every channel
        is accumulated, and the line windows are applied at extraction time.
    binning
        Radial binning of the uv plane.  Fixed here: changing it later means
        recompressing.
    out_line
        Output file with the statistics of the line column.
    out_continuum
        Output file with the statistics of the continuum column.  ``None``
        skips the continuum pass and only the line column is read.
    line_column
        Column holding the line visibilities.  ``None`` picks
        ``CORRECTED_DATA`` when present, otherwise ``DATA``: after a
        continuum subtraction that is the line-only column, and on a stack
        that was never subtracted it is the full signal.
    continuum_column
        Column holding the continuum, normally the ``MODEL_DATA`` written by
        :func:`~vista.extract.contsub.subtract_continuum`.
    max_amplitude
        Discard visibilities whose amplitude exceeds this value.  NaN and
        infinite samples are always discarded.  A stacked MS can carry a
        handful of corrupt cells, and a single one of them would poison the
        accumulators of a whole annulus.
    chunk_mb
        Target size, in MB, of one read chunk: it sets the number of rows read
        at a time and therefore the memory footprint.
    resume
        Reuse the data descriptors already present in the output files.
    one_channel
        Sum every data descriptor over its channels: the statistics of a
        continuum stack, whose datasets need not share any rest frequency.
    n_io
        Threads reading a Zarr store at the same time (a Measurement Set is
        read by one thread).  The accumulation runs in the compiled kernel
        (OpenMP) while the next chunks are read.

    Returns
    -------
    dict
        Paths of the files written, keyed by ``'line'`` and ``'continuum'``.
    """
    from ..stackio import open_stack, process_pieces
    import time

    _reset_clock()
    t_start = time.perf_counter()
    kernel = _kernel()
    entries = (read_input_list(input_list) if isinstance(input_list, str)
               else list(input_list))
    stack = open_stack(ms_path)
    try:
        return _compress(stack, ms_path, entries, line, binning, out_line,
                         out_continuum, line_column, continuum_column,
                         max_amplitude, chunk_mb, resume, verbose, n_io,
                         kernel, t_start, process_pieces, one_channel)
    finally:
        stack.close()


def _compress(stack, ms_path, entries, line, binning, out_line, out_continuum,
              line_column, continuum_column, max_amplitude, chunk_mb, resume,
              verbose, n_io, kernel, t_start, process_pieces, one_channel=False):
    import time

    dd_frequencies = stack.dd_frequencies()
    n_dd = len(dd_frequencies)
    edges = binning.edges_lambda()
    n_bins = len(edges) - 1
    do_continuum = out_continuum is not None

    outputs = {"line": out_line}
    if do_continuum:
        outputs["continuum"] = out_continuum

    # ---- resume -----------------------------------------------------------
    stored: Dict[str, dict] = {}
    for tag, path in outputs.items():
        stored[tag] = {}
        if resume and os.path.exists(path):
            try:
                stored[tag] = np.load(path, allow_pickle=True).item()
                if stored[tag].get("format_version") != FORMAT_VERSION:
                    _warn(f"compression: {path} has format "
                          f"{stored[tag].get('format_version')}, not "
                          f"{FORMAT_VERSION}: not resumed, compressed again")
                    stored[tag] = {}
                done = sum(isinstance(k, (int, np.integer))
                           for k in stored[tag])
                if done:
                    _say(f"Compression: resuming {path}, {done} data "
                         f"descriptors already done", verbose)
            except Exception:
                stored[tag] = {}
        stored[tag].setdefault("bin_edges_lambda", edges)
        stored[tag].setdefault("ms_path", ms_path)
        stored[tag]["format_version"] = FORMAT_VERSION
        if line is not None:
            stored[tag]["rest_freq_ghz"] = line.rest_freq_ghz

    present = stack.dd_ids()
    columns = stack.columns(present[0]) if present else set()
    if line_column is None:
        line_column = ("CORRECTED_DATA" if "CORRECTED_DATA" in columns
                       else "DATA")
    if line_column not in columns:
        raise RuntimeError(f"column {line_column!r} is missing from {ms_path}")
    contsub = stack.get_record("contsub")
    for tag in stored:
        stored[tag]["line_column"] = line_column
        if contsub and contsub.get("line_column") == line_column:
            stored[tag]["contsub"] = contsub
    if do_continuum and continuum_column not in columns:
        raise RuntimeError(f"continuum column {continuum_column!r} is missing "
                           f"from {ms_path}: run the continuum subtraction "
                           f"first, or drop out_continuum")
    _say(f"Compression: {os.path.basename(ms_path.rstrip('/'))} (read-only, "
         f"{'Zarr' if stack.kind == 'zarr' else 'MS'}), {n_dd} data "
         f"descriptors, {n_bins} uv annuli; line from {line_column}"
         + (f", continuum from {continuum_column}" if do_continuum else "")
         + ("" if kernel else "  [NumPy: compiled kernel not found]"), verbose)

    if one_channel:
        every = np.concatenate([np.asarray(f, float) for f in dd_frequencies.values()
                                if len(f)] or [np.zeros(1)])
        grid, f0, dnu = np.array([every.mean()]), 0.0, 1.0
    elif "freq_grid" in stored["line"]:
        grid = np.asarray(stored["line"]["freq_grid"])
        f0, dnu = float(grid[0]), float(grid[1] - grid[0])
    else:
        grid, f0, dnu = build_global_grid(dd_frequencies, verbose)
    n_grid = len(grid)
    for tag in stored:
        stored[tag]["freq_grid"] = grid

    has_weight_spectrum = False
    if stack.kind == "ms" and "WEIGHT_SPECTRUM" in columns and present:
        try:
            p0 = stack.pieces(present[0])[0]
            stack.read(present[0], p0[0], 1, ["WEIGHT_SPECTRUM"])
            has_weight_spectrum = True
        except Exception:
            has_weight_spectrum = False
    weight_column = "WEIGHT_SPECTRUM" if has_weight_spectrum else "WEIGHT"
    read_columns = ["UVW", "FLAG_ROW", "FLAG", weight_column, "ANTENNA1", "ANTENNA2",
                    line_column] + ([continuum_column] if do_continuum else [])

    def _store_empty(dd, freq):
        for tag in stored:
            stored[tag][dd] = {"sums": np.zeros((n_grid, n_bins, N_ACC)), "freq": freq}
            _checkpoint(outputs[tag], stored[tag])

    plan, tasks = {}, []
    width = len(str(n_dd))
    for dd in range(n_dd):
        if all(dd in stored[tag] for tag in stored):
            continue
        entry = entries[dd] if dd < len(entries) else None
        freq = dd_frequencies[dd]
        n_chan = len(freq)
        pieces = []
        if entry is not None and n_chan and dd in present:
            if stack.kind == "ms":
                n_read = 1 + int(do_continuum)
                n_corr = int(stack.read(dd, stack.pieces(dd)[0][0], 1, ["WEIGHT"])["WEIGHT"].shape[-1])
                bytes_per_row = ((16 * n_read + 1 + (8 if has_weight_spectrum else 0))
                                 * n_corr * n_chan + 96)
                pieces = stack.pieces(dd, max(1, int(chunk_mb * 1e6 / bytes_per_row)))
            else:
                pieces = stack.pieces(dd)
        if not pieces:
            _store_empty(dd, freq)
            continue
        grid_index = (np.zeros(n_chan, np.int64) if one_channel
                      else np.rint((freq - f0) / dnu).astype(np.int64))
        plan[dd] = {"entry": entry, "freq": freq, "grid_index": grid_index,
                    "grid_ok": (grid_index >= 0) & (grid_index < n_grid),
                    "left": len(pieces), "rows": 0, "rejected": 0,
                    "acc": {tag: np.zeros((n_grid * n_bins, N_ACC)) for tag in outputs}}
        tasks += [(dd, r0, n) for r0, n in pieces]

    totals = {"rejected": 0}

    def read(task):
        dd, r0, n = task
        return stack.read(dd, r0, n, read_columns)

    def compute(task, d):
        dd = task[0]
        p = plan[dd]
        row_ok = np.asarray(d["ANTENNA1"]) != np.asarray(d["ANTENNA2"])
        p["rows"] += int(np.count_nonzero(row_ok))
        p["rejected"] += accumulate_chunk(
            p["acc"]["line"], p["acc"].get("continuum"), d[line_column],
            d.get(continuum_column) if do_continuum else None,
            d["FLAG"], d["FLAG_ROW"], row_ok, d[weight_column], d["UVW"],
            p["freq"], p["grid_index"], p["grid_ok"], edges, max_amplitude, kernel)
        p["left"] -= 1
        if p["left"] == 0:                       # this data descriptor is complete
            entry = p["entry"]
            nu_obs = ((line.rest_freq_hz if line is not None
                       else float(np.median(p["freq"]))) / (1.0 + entry.z))
            pb_gain, pb_offset, pb_dish = primary_beam_of_entry(entry, nu_obs)
            for tag in stored:
                stored[tag].setdefault("primary_beam", {})[dd] = pb_gain
            if np.isfinite(pb_offset):
                _say(f"  dd {dd}  primary beam A={pb_gain:.3f}  "
                     f"offset={pb_offset:.1f}\"  D={pb_dish:.1f} m", verbose)
            else:
                _warn(f"compression: cannot read {entry.ms} (dd {dd}), "
                      f"primary beam not corrected (A=1)")
            meta = {"z": entry.z, "ms": entry.ms,
                    "ra_deg": entry.coordinates_deg[0],
                    "dec_deg": entry.coordinates_deg[1],
                    "n_rows": int(p["rows"]), "n_rejected": int(p["rejected"])}
            for tag, acc in p["acc"].items():
                stored[tag][dd] = {"sums": acc.reshape(n_grid, n_bins, N_ACC),
                                   "freq": p["freq"], "meta": meta}
                _checkpoint(outputs[tag], stored[tag])
            totals["rejected"] += p["rejected"]
            note = (f", {p['rejected']:,} samples rejected" if p["rejected"] else "")
            _say(f"  [{dd + 1:>{width}}/{n_dd}] dd {dd}  z={entry.z:.4f}  "
                 f"{p['rows']:,} rows{note}  (saved)", verbose)
            del plan[dd]
        return None

    process_pieces(stack, tasks, read, compute, None, n_io=n_io)

    if totals["rejected"]:
        _warn(f"compression: {totals['rejected']:,} corrupt or out-of-range "
              f"samples discarded")
    _say(f"Compression done in {time.perf_counter() - t_start:.1f} s: "
         + ", ".join(outputs.values()), verbose)
    return outputs


__all__ = ["compress_visibilities", "SufficientStatistics",
           "accumulate_annuli", "accumulate_chunk", "build_global_grid",
           "data_descriptor_frequencies", "FORMAT_VERSION"]
