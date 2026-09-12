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
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .config import BinningConfig, C_MS, LineConfig
from .sources import Entry, read_input_list

FORMAT_VERSION = 3


# ---------------------------------------------------------------------------
# MS access helpers
# ---------------------------------------------------------------------------
def _table():
    from casatools import table                      # noqa: local import
    return table()


def _subtable_path(ms_path: str, name: str) -> str:
    tb = _table()
    try:
        tb.open(ms_path)
        keyword = tb.getkeyword(name)
        tb.close()
        if isinstance(keyword, str) and keyword.startswith("Table:"):
            return keyword.split("Table:", 1)[1].strip()
    except Exception:
        try:
            tb.close()
        except Exception:
            pass
    return f"{ms_path}/{name}"


def data_descriptor_frequencies(ms_path: str) -> Dict[int, np.ndarray]:
    """Channel frequencies of every data descriptor of the MS."""
    tb = _table()
    tb.open(_subtable_path(ms_path, "SPECTRAL_WINDOW"))
    n_chan = (tb.getcol("NUM_CHAN") if "NUM_CHAN" in tb.colnames()
              else [None] * tb.nrows())
    per_spw = []
    for i in range(tb.nrows()):
        freq = np.asarray(tb.getcell("CHAN_FREQ", i), dtype=np.float64)
        if n_chan[i] is not None:                    # strip any zero padding
            freq = freq[:int(n_chan[i])]
        per_spw.append(freq)
    tb.close()
    tb = _table()
    tb.open(_subtable_path(ms_path, "DATA_DESCRIPTION"))
    dd_to_spw = tb.getcol("SPECTRAL_WINDOW_ID")
    tb.close()
    return {dd: per_spw[spw] for dd, spw in enumerate(dd_to_spw)}


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
    """Add one chunk of visibilities to the five accumulators."""
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
        Target size, in MB, of one read chunk: it sets the number of rows read
        at a time and therefore the memory footprint.
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

    tb = _table()
    tb.open(ms_path)                                  # read-only
    columns = tb.colnames()
    if line_column is None:
        line_column = ("CORRECTED_DATA" if "CORRECTED_DATA" in columns
                       else "DATA")
    if line_column not in columns:
        raise RuntimeError(f"column {line_column!r} is missing from {ms_path}")
    if do_continuum and continuum_column not in columns:
        raise RuntimeError(f"continuum column {continuum_column!r} is missing "
                           f"from {ms_path}: run the continuum subtraction "
                           f"first, or drop out_continuum")
    if verbose:
        print(f"[compress] {os.path.basename(ms_path)}  n_dd={n_dd}  "
              f"line={line_column}"
              + (f"  continuum={continuum_column}" if do_continuum else "")
              + "  (read-only)")

    n_rejected_total = 0
    for dd in range(n_dd):
        if all(dd in stored[tag] for tag in stored):
            continue
        entry = entries[dd] if dd < len(entries) else None
        freq = dd_frequencies[dd]
        n_chan = len(freq)
        grid_index = np.rint((freq - f0) / dnu).astype(np.int64)
        grid_ok = (grid_index >= 0) & (grid_index < n_grid)
        empty = {"sums": np.zeros((n_grid, n_bins, 5)), "freq": freq}
        if entry is None or n_chan == 0:
            for tag in stored:
                stored[tag][dd] = dict(empty)
                _checkpoint(outputs[tag], stored[tag])
            continue

        selection = tb.query(f"DATA_DESC_ID=={dd} && ANTENNA1!=ANTENNA2")
        n_rows = selection.nrows()
        if n_rows == 0:
            selection.close()
            for tag in stored:
                stored[tag][dd] = dict(empty)
                _checkpoint(outputs[tag], stored[tag])
            continue

        n_corr = selection.getcell("WEIGHT", 0).shape[0]
        has_weight_spectrum = "WEIGHT_SPECTRUM" in selection.colnames()
        if has_weight_spectrum:
            try:
                selection.getcell("WEIGHT_SPECTRUM", 0)
            except Exception:
                has_weight_spectrum = False
        n_read = 1 + int(do_continuum)
        bytes_per_row = ((16 * n_read + 1 + (8 if has_weight_spectrum else 0))
                         * n_corr * n_chan + 96)
        chunk = max(1, int(chunk_mb * 1e6 / bytes_per_row))

        accumulators = {"line": np.zeros((n_grid * n_bins, 5))}
        if do_continuum:
            accumulators["continuum"] = np.zeros((n_grid * n_bins, 5))
        n_rejected = 0

        for start in range(0, n_rows, chunk):
            n = min(chunk, n_rows - start)
            uvw = selection.getcol("UVW", start, n)
            flag_row = selection.getcol("FLAG_ROW", start, n).astype(bool)
            flag = selection.getcol("FLAG", start, n)
            if has_weight_spectrum:
                weight = selection.getcol("WEIGHT_SPECTRUM", start, n)
            else:
                weight = (selection.getcol("WEIGHT", start, n)[:, None, :]
                          * np.ones((1, n_chan, 1)))
            data_line = selection.getcol(line_column, start, n)
            data_cont = (selection.getcol(continuum_column, start, n)
                         if do_continuum else None)

            good = ((~flag) & (~flag_row[None, None, :]) & (weight > 0)
                    & (data_line != 0) & np.isfinite(data_line))
            if max_amplitude is not None:
                good &= np.abs(data_line) <= max_amplitude
            if data_cont is not None:
                corrupt = ~np.isfinite(data_cont)
                if max_amplitude is not None:
                    corrupt |= np.abs(data_cont) > max_amplitude
                good &= ~corrupt
            n_rejected += int(np.sum((~good) & (~flag)
                                     & (~flag_row[None, None, :])
                                     & (weight > 0) & (data_line != 0)))
            weight = np.where(good, weight, 0.0)
            data_line = np.where(good, data_line, 0.0)
            if data_cont is not None:
                data_cont = np.where(good, data_cont, 0.0)

            b_lambda = (np.sqrt(uvw[0] ** 2 + uvw[1] ** 2)[None, :]
                        * (freq[:, None] / C_MS))
            accumulate_annuli(accumulators["line"], data_line, weight,
                              b_lambda, grid_index, grid_ok, edges,
                              n_grid, n_bins)
            if do_continuum:
                accumulate_annuli(accumulators["continuum"], data_cont,
                                  weight, b_lambda, grid_index, grid_ok,
                                  edges, n_grid, n_bins)
            del uvw, data_line, data_cont, weight, good
        selection.close()

        n_rejected_total += n_rejected
        meta = {"z": entry.z, "ms": entry.ms,
                "ra_deg": entry.coordinates_deg[0],
                "dec_deg": entry.coordinates_deg[1],
                "n_rows": int(n_rows), "n_rejected": int(n_rejected)}
        for tag, accumulator in accumulators.items():
            stored[tag][dd] = {"sums": accumulator.reshape(n_grid, n_bins, 5),
                               "freq": freq, "meta": meta}
            _checkpoint(outputs[tag], stored[tag])
        if verbose:
            note = (f"  rejected={n_rejected:,}" if n_rejected else "")
            print(f"  dd={dd:>3} z={entry.z:<7.4f} rows={n_rows:>9,}{note}"
                  f"  [checkpoint]")
            sys.stdout.flush()

    tb.close()
    if verbose:
        if n_rejected_total:
            print(f"[compress] {n_rejected_total:,} corrupt or out-of-range "
                  f"samples discarded")
        print("[compress] done -> " + ", ".join(outputs.values()))
    return outputs


__all__ = ["compress_visibilities", "SufficientStatistics",
           "accumulate_annuli", "build_global_grid",
           "data_descriptor_frequencies", "FORMAT_VERSION"]
