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
======  ====================================================

Every subsequent operation, from the spectra to the amplitude profiles, the
model fits and the bootstrap, runs on this compact representation instead of
the full visibility set, which is what makes the repeated resampling cheap.

The MS is opened in **read-only** mode.  Each line of the input list is
compressed as an independent descriptor, so that different observations of
the same source can be combined later, and the result is checkpointed after
every data descriptor, so an interrupted run resumes where it stopped.

The pass is organised as the stacking is: a reader thread reads the chunks of
rows with python-casacore, the C++/OpenMP kernel (``ms_ops.compress_chunk``)
adds them to the accumulators, and a bounded queue between the two lets the
reading of a chunk overlap the compression of the previous one.
:func:`accumulate_annuli` is the same accumulation in NumPy, kept as the
reference.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .config import BinningConfig, C_MS, LineConfig
from .sources import Entry, read_input_list

FORMAT_VERSION = 3


# ---------------------------------------------------------------------------
# MS access helpers (python-casacore)
# ---------------------------------------------------------------------------
# casacore tables are not safe to use from several threads at once: every
# access goes through this lock, as in the stacking.
_casacore_lock = threading.Lock()


def _table(path: str):
    from casacore.tables import table                # noqa: local import
    return table(path, readonly=True, ack=False)


def _subtable_path(ms_path: str, name: str) -> str:
    try:
        with _table(ms_path) as tb:
            keyword = tb.getkeyword(name)
        if isinstance(keyword, str) and keyword.startswith("Table:"):
            return keyword.split("Table:", 1)[1].strip()
    except Exception:
        pass
    return f"{ms_path}/{name}"


def data_descriptor_frequencies(ms_path: str) -> Dict[int, np.ndarray]:
    """Channel frequencies of every data descriptor of the MS."""
    with _casacore_lock:
        with _table(_subtable_path(ms_path, "SPECTRAL_WINDOW")) as tb:
            n_chan = (tb.getcol("NUM_CHAN") if "NUM_CHAN" in tb.colnames()
                      else [None] * tb.nrows())
            per_spw = []
            for i in range(tb.nrows()):
                freq = np.asarray(tb.getcell("CHAN_FREQ", i), dtype=np.float64)
                if n_chan[i] is not None:            # strip any zero padding
                    freq = freq[:int(n_chan[i])]
                per_spw.append(freq)
        with _table(_subtable_path(ms_path, "DATA_DESCRIPTION")) as tb:
            dd_to_spw = tb.getcol("SPECTRAL_WINDOW_ID")
    return {dd: per_spw[spw] for dd, spw in enumerate(dd_to_spw)}


def primary_beam_of_entry(entry: Entry, nu_obs_hz: float):
    """Primary beam response at the source position, for one input line.

    The original phase centre (FIELD ``PHASE_DIR`` of the selected field) and
    the dish diameter (median of ``DISH_DIAMETER``) are read from the
    original MS of the entry.  Returns ``(A, theta_arcsec, dish_m)``, or
    ``(1.0, nan, nan)`` if that MS cannot be read.
    """
    from .sources import primary_beam_gain
    try:
        with _casacore_lock:
            with _table(_subtable_path(entry.ms, "FIELD")) as tb:
                direction = np.asarray(tb.getcell("PHASE_DIR", entry.field or 0),
                                       dtype=float).ravel()
            with _table(_subtable_path(entry.ms, "ANTENNA")) as tb:
                dish = np.asarray(tb.getcol("DISH_DIAMETER"), dtype=float)
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
    if verbose:
        print(f"[grid] {n} channels, dnu={dnu / 1e6:.3f} MHz, "
              f"max mapping error={100 * worst / dnu:.1f}% of a channel")
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
    """Add one chunk of visibilities to the five accumulators.

    The NumPy version of ``ms_ops.compress_chunk``, kept as the reference:
    ``data`` and ``weight`` are (ncorr, nchan, nrow), with the rejected
    samples already at weight zero.
    """
    bin_index = np.searchsorted(edges, b_lambda, side="right") - 1
    in_band = (bin_index >= 0) & (bin_index < n_bins) & grid_ok[:, None]
    bin_index = np.clip(bin_index, 0, n_bins - 1)
    key_2d = np.clip(grid_index, 0, n_grid - 1)[:, None] * n_bins + bin_index
    key = np.broadcast_to(key_2d, weight.shape).ravel()
    w = (weight * in_band[None, :, :]).ravel()
    real = data.real.ravel()
    imaginary = data.imag.ravel()
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


# ---------------------------------------------------------------------------
# the container
# ---------------------------------------------------------------------------
@dataclass
class SufficientStatistics:
    """Compressed statistics of one stacked dataset."""

    sums: Dict[int, np.ndarray]                 # dd -> (nchan, nbins, 5)
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

    def available(self) -> List[int]:
        return sorted(self.sums)


# ---------------------------------------------------------------------------
# the compression itself
# ---------------------------------------------------------------------------
def compress_visibilities(ms_path: str,
                          input_list: Union[str, Sequence[Optional[Entry]]],
                          line: Optional[LineConfig] = None,
                          binning: BinningConfig = BinningConfig(),
                          out_line: str = "stack_line_stats.npy",
                          out_continuum: Optional[str] = None,
                          line_column: Optional[str] = None,
                          continuum_column: str = "MODEL_DATA",
                          max_amplitude: Optional[float] = None,
                          chunk_mb: float = 512.0,
                          resume: bool = True,
                          verbose: bool = True) -> Dict[str, str]:
    """Compress a stacked MS into sufficient statistics, descriptor by
    descriptor.

    Parameters
    ----------
    ms_path
        Stacked Measurement Set.  Opened read-only.
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
        Target size, in MB, of the chunks in memory at the same time (the one
        being compressed and the two waiting in the queue): it sets the number
        of rows read at a time and therefore the memory footprint.
    resume
        Reuse the data descriptors already present in the output files.

    Returns
    -------
    dict
        Paths of the files written, keyed by ``'line'`` and ``'continuum'``.
    """
    entries = (read_input_list(input_list) if isinstance(input_list, str)
               else list(input_list))
    dd_frequencies = data_descriptor_frequencies(ms_path)
    n_dd = len(dd_frequencies)
    edges = binning.edges_lambda()
    n_bins = len(edges) - 1
    do_continuum = out_continuum is not None

    outputs = {"line": out_line}
    if do_continuum:
        outputs["continuum"] = out_continuum

    # ---- resume and shared frequency grid --------------------------------
    stored: Dict[str, dict] = {}
    for tag, path in outputs.items():
        stored[tag] = {}
        if resume and os.path.exists(path):
            try:
                stored[tag] = np.load(path, allow_pickle=True).item()
                done = sum(isinstance(k, (int, np.integer))
                           for k in stored[tag])
                if verbose:
                    print(f"[resume] {path}: {done} data descriptors")
            except Exception:
                stored[tag] = {}
        stored[tag].setdefault("bin_edges_lambda", edges)
        stored[tag].setdefault("ms_path", ms_path)
        stored[tag]["format_version"] = FORMAT_VERSION
        if line is not None:
            stored[tag]["rest_freq_ghz"] = line.rest_freq_ghz

    if "freq_grid" in stored["line"]:
        grid = np.asarray(stored["line"]["freq_grid"])
        f0, dnu = float(grid[0]), float(grid[1] - grid[0])
    else:
        grid, f0, dnu = build_global_grid(dd_frequencies, verbose)
    n_grid = len(grid)
    for tag in stored:
        stored[tag]["freq_grid"] = grid

    try:
        from .. import ms_ops            # installed into vista/ by 'cmake --install'
    except ImportError:                  # or found on PYTHONPATH
        import ms_ops

    with _casacore_lock:
        tb = _table(ms_path)                          # read-only
        columns = tb.colnames()
    if line_column is None:
        line_column = ("CORRECTED_DATA" if "CORRECTED_DATA" in columns
                       else "DATA")
    if line_column not in columns:
        tb.close()
        raise RuntimeError(f"column {line_column!r} is missing from {ms_path}")
    if do_continuum and continuum_column not in columns:
        tb.close()
        raise RuntimeError(f"continuum column {continuum_column!r} is missing "
                           f"from {ms_path}: run the continuum subtraction "
                           f"first, or drop out_continuum")
    if verbose:
        print(f"[compress] {os.path.basename(ms_path)}  n_dd={n_dd}  "
              f"line={line_column}"
              + (f"  continuum={continuum_column}" if do_continuum else "")
              + "  (read-only)")

    todo = [dd for dd in range(n_dd)
            if not all(dd in stored[tag] for tag in stored)]
    max_amp = -1.0 if max_amplitude is None else float(max_amplitude)
    read_q = Queue(maxsize=2)
    failure = [None]
    timers = {"read": 0.0, "compress": 0.0, "read_wait": 0.0}

    def reader():
        """Read every data descriptor, chunk by chunk, onto the queue:
        ("empty", dd), or ("start", dd, n_rows), ("chunk", dd, arrays)...,
        ("end", dd)."""
        try:
            for dd in todo:
                entry = entries[dd] if dd < len(entries) else None
                n_chan = len(dd_frequencies[dd])
                if entry is None or n_chan == 0:
                    read_q.put(("empty", dd))
                    continue
                t0 = time.perf_counter()
                with _casacore_lock:
                    selection = tb.query(
                        f"DATA_DESC_ID=={dd} && ANTENNA1!=ANTENNA2")
                    n_rows = selection.nrows()
                    if n_rows:
                        n_corr = selection.getcell("WEIGHT", 0).shape[0]
                        has_weight_spectrum = ("WEIGHT_SPECTRUM"
                                               in selection.colnames())
                        if has_weight_spectrum:
                            try:
                                selection.getcell("WEIGHT_SPECTRUM", 0)
                            except Exception:
                                has_weight_spectrum = False
                timers["read"] += time.perf_counter() - t0
                if n_rows == 0:
                    with _casacore_lock:
                        selection.close()
                    read_q.put(("empty", dd))
                    continue
                n_read = 1 + int(do_continuum)
                bytes_per_row = ((8 * n_read + 1 + (4 if has_weight_spectrum
                                                    else 0))
                                 * n_corr * n_chan + 32)
                chunk = max(1, int(chunk_mb * 1e6 / bytes_per_row / 3))
                read_q.put(("start", dd, n_rows))
                for start in range(0, n_rows, chunk):
                    n = min(chunk, n_rows - start)
                    t0 = time.perf_counter()
                    with _casacore_lock:
                        arrays = {
                            "uvw": selection.getcol("UVW", start, n),
                            "flag_row": selection.getcol("FLAG_ROW", start, n),
                            "flag": selection.getcol("FLAG", start, n),
                            "weight": selection.getcol(
                                "WEIGHT_SPECTRUM" if has_weight_spectrum
                                else "WEIGHT", start, n),
                            "line": selection.getcol(line_column, start, n),
                            "cont": (selection.getcol(continuum_column,
                                                      start, n)
                                     if do_continuum else None)}
                    timers["read"] += time.perf_counter() - t0
                    read_q.put(("chunk", dd, arrays))
                    if failure[0] is not None:
                        return
                with _casacore_lock:
                    selection.close()
                read_q.put(("end", dd))
        except Exception as exc:                     # noqa: BLE001
            failure[0] = exc
        finally:
            read_q.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    n_rejected_total = 0
    state = {}
    try:
        while True:
            t0 = time.perf_counter()
            item = read_q.get()
            timers["read_wait"] += time.perf_counter() - t0
            if item is None:
                break
            if failure[0] is not None:
                continue
            kind, dd = item[0], item[1]
            freq = dd_frequencies[dd]
            if kind == "empty":
                empty = {"sums": np.zeros((n_grid, n_bins, 5)), "freq": freq}
                for tag in stored:
                    stored[tag][dd] = dict(empty)
                    _checkpoint(outputs[tag], stored[tag])
            elif kind == "start":
                grid_index = np.rint((freq - f0) / dnu).astype(np.int64)
                state = {"n_rows": item[2], "n_rejected": 0,
                         "grid_index": grid_index,
                         "acc": {tag: np.zeros((n_grid * n_bins, 5))
                                 for tag in outputs}}
            elif kind == "chunk":
                a = item[2]
                t0 = time.perf_counter()
                sums_line, sums_cont, n_rejected = ms_ops.compress_chunk(
                    a["line"], a["cont"], a["weight"], a["flag"],
                    a["flag_row"], a["uvw"], freq, state["grid_index"], edges,
                    n_grid, max_amp)
                timers["compress"] += time.perf_counter() - t0
                state["acc"]["line"] += sums_line
                if do_continuum:
                    state["acc"]["continuum"] += sums_cont
                state["n_rejected"] += int(n_rejected)
            else:                                             # "end"
                entry = entries[dd]
                n_rows, n_rejected = state["n_rows"], state["n_rejected"]
                n_rejected_total += n_rejected
                nu_obs = ((line.rest_freq_hz if line is not None
                           else float(np.median(freq))) / (1.0 + entry.z))
                pb_gain, pb_offset, pb_dish = primary_beam_of_entry(entry, nu_obs)
                for tag in stored:
                    stored[tag].setdefault("primary_beam", {})[dd] = pb_gain
                if verbose:
                    if np.isfinite(pb_offset):
                        print(f"  dd={dd:>3} primary beam A={pb_gain:.3f}  "
                              f"offset={pb_offset:.1f}\"  D={pb_dish:.1f} m")
                        if pb_gain < 0.5:
                            print(f"  dd={dd:>3} WARNING: the source is beyond half the "
                                  f"primary beam (A={pb_gain:.3f}, offset {pb_offset:.1f}\"): "
                                  f"the correction is large and the Gaussian beam "
                                  f"approximate there")
                    else:
                        print(f"  dd={dd:>3} WARNING: cannot read {entry.ms}, "
                              f"primary beam not corrected (A=1)")
                meta = {"z": entry.z, "ms": entry.ms,
                        "ra_deg": entry.coordinates_deg[0],
                        "dec_deg": entry.coordinates_deg[1],
                        "n_rows": int(n_rows), "n_rejected": int(n_rejected)}
                for tag, accumulator in state["acc"].items():
                    stored[tag][dd] = {"sums": accumulator.reshape(n_grid, n_bins, 5),
                                       "freq": freq, "meta": meta}
                    _checkpoint(outputs[tag], stored[tag])
                if verbose:
                    note = (f"  rejected={n_rejected:,}" if n_rejected else "")
                    print(f"  dd={dd:>3} z={entry.z:<7.4f} rows={n_rows:>9,}{note}"
                          f"  [checkpoint]")
                    sys.stdout.flush()
    except Exception as exc:                                  # noqa: BLE001
        failure[0] = exc
        while read_q.get() is not None:                       # let the reader finish
            pass
    finally:
        thread.join()
        with _casacore_lock:
            tb.close()
    if failure[0] is not None:
        raise failure[0]

    if verbose:
        if n_rejected_total:
            print(f"[compress] {n_rejected_total:,} corrupt or out-of-range "
                  f"samples discarded")
        print(f"[compress] read {timers['read']:.1f} s, compression (C++) "
              f"{timers['compress']:.1f} s (wait for the reader "
              f"{timers['read_wait']:.1f} s)")
        print("[compress] done -> " + ", ".join(outputs.values()))
    return outputs


__all__ = ["compress_visibilities", "SufficientStatistics",
           "accumulate_annuli", "build_global_grid",
           "data_descriptor_frequencies", "FORMAT_VERSION"]
