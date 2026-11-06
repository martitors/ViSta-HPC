"""
vista.extract.inline
====================
Continuum subtraction and compression while the stack is written: one pass
over the data instead of three.

In the separate workflow the stacking writes the stack, the continuum
subtraction reads it back and writes ``MODEL_DATA`` and ``CORRECTED_DATA``,
and the compression reads it once more.  Both steps work row by row on the
output grid, so they can run on each chunk right after the stacking kernel,
before it is written: :class:`InlineExtraction` does that for
:meth:`vista.ViSta.run` (``continuum=``, ``stats_line=``, ...).

The stack on disk is the same as with the separate steps -- ``DATA``,
``MODEL_DATA`` and ``CORRECTED_DATA``, bit for bit -- and so are the
statistics files (equal to the last bits of the sums for a Zarr store; for an
MS the separate compression reads larger pieces, so the sums are added in
another order: equal to about 1e-15).  ``DATA`` is always kept: after a look
at the spectrum, the subtraction can be rerun on the stack with another
window (:func:`vista.extract.subtract_continuum`) and the statistics
recomputed (:func:`vista.extract.compress_visibilities`), without stacking
again.

With ``order=-1`` the order of each entry is chosen on its first
``bic_rows`` rows, as in the separate step: the chunks of an entry wait until
that many rows have arrived (one chunk with the default sizes).
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import _say, _warn
from .config import BinningConfig, ContinuumConfig, LineConfig
from .contsub import (_kernel as _contsub_kernel, fit_plan, fit_rows,
                      min_fit_channels, order_from_sample, record)
from .sources import read_input_list
from .statistics import (FORMAT_VERSION, N_ACC, _checkpoint,
                         _kernel as _stats_kernel, accumulate_chunk,
                         build_global_grid, primary_beam_of_entry)


class InlineExtraction:
    """Continuum subtraction and compression of the chunks of a stacking run.

    Parameters
    ----------
    input_file
        The ViSta input list (its order is the data descriptor numbering).
    freqs
        Output channel frequencies of every entry, as in the stack.
    output
        The stack being written (recorded in the statistics files).
    line
        The line: its window (with the second line and ``exclude_v_kms``)
        sets the channels left out of the continuum fit.
    continuum
        Continuum subtraction options, or None for no subtraction.
    stats_line, stats_continuum
        Statistics files to write (as ``compress_visibilities`` would), or
        None.  The line statistics are accumulated from ``CORRECTED_DATA``
        when the continuum is subtracted, from ``DATA`` otherwise; the
        continuum ones from ``MODEL_DATA``.
    binning, max_amplitude
        As in :func:`vista.extract.compress_visibilities`.
    """

    def __init__(self, input_file: str, freqs: Sequence[np.ndarray], output: str,
                 line: Optional[LineConfig], continuum: Optional[ContinuumConfig] = None,
                 stats_line: Optional[str] = None, stats_continuum: Optional[str] = None,
                 binning: Optional[BinningConfig] = None,
                 max_amplitude: Optional[float] = None, verbose=True):
        if line is None:
            raise ValueError("the continuum subtraction and the statistics during the "
                             "stacking need line=LineConfig(...)")
        if stats_continuum is not None and continuum is None:
            raise ValueError("stats_continuum needs the continuum subtraction "
                             "(continuum=ContinuumConfig(...)): it is computed from "
                             "MODEL_DATA")
        if stats_continuum is not None and stats_line is None:
            raise ValueError("stats_continuum needs stats_line too")
        self.line, self.continuum = line, continuum
        self.output, self.verbose = output, verbose
        self.freqs = [np.asarray(f, dtype=np.float64) for f in freqs]
        self.n = len(self.freqs)
        self.t = 0.0                                     # seconds spent here
        self.t_contsub = 0.0
        self.t_stats = 0.0
        self._entries: Dict[int, dict] = {}

        # ---- continuum subtraction ------------------------------------------
        self.do_contsub = continuum is not None
        if self.do_contsub:
            self.k_contsub = _contsub_kernel()
            self.plans = []
            for i, f in enumerate(self.freqs):
                fit_mask, x_all, n_fit = fit_plan(f, line, continuum)
                ok = n_fit >= min_fit_channels(continuum)
                self.plans.append({"fit_mask": fit_mask, "x_all": x_all, "n_fit": n_fit,
                                   "ok": ok})
        self.extra_columns = ((continuum.model_column, continuum.line_column)
                              if self.do_contsub else ())

        # ---- statistics -----------------------------------------------------
        self.do_stats = stats_line is not None
        self.outputs = {}
        if self.do_stats:
            self.k_stats = _stats_kernel()
            self.entries = read_input_list(input_file)
            self.binning = binning or BinningConfig()
            self.edges = self.binning.edges_lambda()
            self.n_bins = len(self.edges) - 1
            self.max_amplitude = max_amplitude
            self.outputs["line"] = stats_line
            if stats_continuum is not None:
                self.outputs["continuum"] = stats_continuum
            self.line_column = continuum.line_column if self.do_contsub else "DATA"
            self.cont_column = continuum.model_column if stats_continuum else None
            self.grid, self.f0, self.dnu = build_global_grid(
                {i: f for i, f in enumerate(self.freqs)}, verbose=False)
            self.stored = {}
            for tag in self.outputs:
                self.stored[tag] = {"bin_edges_lambda": self.edges,
                                    "ms_path": output,
                                    "format_version": FORMAT_VERSION,
                                    "rest_freq_ghz": line.rest_freq_ghz,
                                    "freq_grid": self.grid,
                                    "line_column": self.line_column}

    # ------------------------------------------------------------------------
    def describe(self) -> str:
        """One line for the log of the stacking."""
        parts = []
        if self.do_contsub:
            c = self.continuum
            order = {-1: f"BIC on the first {c.bic_rows} rows", 0: "0", 1: "1"}[c.order]
            n_bad = sum(not p["ok"] for p in self.plans)
            parts.append(f"continuum subtracted (order {order}) -> {c.model_column}, "
                         f"{c.line_column}" + (f"; {n_bad} entries without enough "
                                               f"line-free channels" if n_bad else ""))
        if self.do_stats:
            parts.append(f"statistics from {self.line_column} -> "
                         + ", ".join(self.outputs.values()))
        return "; ".join(parts)

    def _state(self, index):
        st = self._entries.get(index)
        if st is None:
            st = {"order": None, "pending": [], "n_pending": 0, "rows": 0, "rejected": 0,
                  "acc": None}
            if self.do_contsub and self.continuum.order != -1:
                st["order"] = self.continuum.order
            if self.do_stats:
                freq = self.freqs[index]
                gi = np.rint((freq - self.f0) / self.dnu).astype(np.int64)
                st["grid_index"] = gi
                st["grid_ok"] = (gi >= 0) & (gi < len(self.grid))
                st["acc"] = {tag: np.zeros((len(self.grid) * self.n_bins, N_ACC))
                             for tag in self.outputs}
            self._entries[index] = st
        return st

    # ------------------------------------------------------------------------
    def process(self, index: int, row0: int, columns: dict, last: bool) -> List[tuple]:
        """Subtract and accumulate one processed chunk.

        Returns the chunks ready to be written, ``[(row0, columns, last)]``,
        in row order: none while an entry waits for its BIC sample, several
        once the order is chosen.
        """
        t0 = time.perf_counter()
        st = self._state(index)
        ready = [(row0, columns, last)]
        if self.do_contsub and st["order"] is None and self.plans[index]["ok"]:
            st["pending"].append((row0, columns, last))
            st["n_pending"] += len(columns["DATA"])
            bic_rows = int(self.continuum.bic_rows)
            if st["n_pending"] < bic_rows and not last:
                self.t += time.perf_counter() - t0
                return []
            ready, st["pending"], st["n_pending"] = st["pending"], [], 0
            if len(ready) == 1:
                data, flag = ready[0][1]["DATA"][:bic_rows], ready[0][1]["FLAG"][:bic_rows]
            else:
                data = np.concatenate([c["DATA"] for _, c, _ in ready])[:bic_rows]
                flag = np.concatenate([c["FLAG"] for _, c, _ in ready])[:bic_rows]
            p = self.plans[index]
            st["order"], n_valid = order_from_sample(data, flag, p["fit_mask"], p["x_all"],
                                                     self.continuum)
            del data, flag
        for _, cols, is_last in ready:
            if self.do_contsub:
                self._subtract(index, st, cols)
            if self.do_stats:
                self._accumulate(index, st, cols)
            if is_last and self.do_stats:
                self._finish_stats(index, st)
        self.t += time.perf_counter() - t0
        return ready

    def _subtract(self, index, st, cols):
        t0 = time.perf_counter()
        p = self.plans[index]
        c = self.continuum
        if p["ok"]:
            model, line = fit_rows(cols["DATA"], p["fit_mask"], p["x_all"], st["order"],
                                   self.k_contsub, cols["FLAG"])
        else:                     # as the separate step: the columns stay empty
            model = np.zeros_like(cols["DATA"])
            line = np.zeros_like(cols["DATA"])
        cols[c.model_column] = model
        cols[c.line_column] = line
        self.t_contsub += time.perf_counter() - t0

    def _accumulate(self, index, st, cols):
        t0 = time.perf_counter()
        n = len(cols["DATA"])
        row_ok = np.asarray(cols["ANTENNA1"]) != np.asarray(cols["ANTENNA2"])
        st["rows"] += int(np.count_nonzero(row_ok))
        flag_row = cols.get("FLAG_ROW")
        if flag_row is None:
            flag_row = np.zeros(n, dtype=bool)
        acc = st["acc"]
        st["rejected"] += accumulate_chunk(
            acc["line"], acc.get("continuum"), cols[self.line_column],
            cols[self.cont_column] if self.cont_column else None,
            cols["FLAG"], flag_row, row_ok, cols["WEIGHT"], cols["UVW"],
            self.freqs[index], st["grid_index"], st["grid_ok"], self.edges,
            self.max_amplitude, self.k_stats)
        self.t_stats += time.perf_counter() - t0

    def _finish_stats(self, index, st):
        """An entry is complete: primary beam, metadata, its accumulators."""
        entry = self.entries[index] if index < len(self.entries) else None
        st["done"] = True
        if entry is None:                 # an unusable line of the list: left empty
            st["acc"] = None
            return
        nu_obs = self.line.rest_freq_hz / (1.0 + entry.z)
        pb_gain, pb_offset, pb_dish = primary_beam_of_entry(entry, nu_obs)
        if not np.isfinite(pb_offset):
            _warn(f"statistics: cannot read {entry.ms} (dd {index}), primary beam "
                  f"not corrected (A=1)")
        meta = {"z": entry.z, "ms": entry.ms,
                "ra_deg": entry.coordinates_deg[0], "dec_deg": entry.coordinates_deg[1],
                "n_rows": int(st["rows"]), "n_rejected": int(st["rejected"])}
        n_grid = len(self.grid)
        for tag, acc in st["acc"].items():
            self.stored[tag].setdefault("primary_beam", {})[index] = pb_gain
            self.stored[tag][index] = {"sums": acc.reshape(n_grid, self.n_bins, N_ACC),
                                       "freq": self.freqs[index], "meta": meta}
        st["acc"] = None

    # ------------------------------------------------------------------------
    def orders(self) -> dict:
        """Continuum order used for every entry with data and a fit."""
        return {i: st["order"] for i, st in self._entries.items()
                if st["order"] is not None and self.plans[i]["ok"]}

    def no_fit(self) -> list:
        """Entries with data but too few line-free channels."""
        return sorted(i for i in self._entries if not self.plans[i]["ok"])

    def contsub_record(self) -> Optional[dict]:
        if not self.do_contsub:
            return None
        return record(self.line, self.continuum, self.orders(), self.no_fit(),
                      mode="during the stacking")

    def close(self, output: Optional[str] = None) -> dict:
        """Write the statistics files; returns what was done."""
        if output is not None:
            self.output = output
        summary = {}
        if self.do_contsub:
            orders = self.orders()
            summary["contsub_orders"] = orders
            if self.continuum.order == -1 and orders:
                n1 = sum(1 for o in orders.values() if o == 1)
                _say(f"  continuum orders chosen by BIC: {len(orders) - n1} x 0, {n1} x 1",
                     self.verbose)
        if self.do_stats:
            n_grid = len(self.grid)
            rejected = 0
            rec = self.contsub_record()
            for i in range(self.n):
                st = self._entries.get(i)
                if st is not None and not st.get("done"):
                    raise RuntimeError(f"statistics: entry {i} incomplete")
                for tag in self.outputs:
                    if i not in self.stored[tag]:
                        self.stored[tag][i] = {"sums": np.zeros((n_grid, self.n_bins, N_ACC)),
                                               "freq": self.freqs[i]}
                if st is not None:
                    rejected += st["rejected"]
            for tag, path in self.outputs.items():
                self.stored[tag]["ms_path"] = self.output
                if rec is not None:
                    self.stored[tag]["contsub"] = rec
                folder = os.path.dirname(os.path.abspath(path))
                os.makedirs(folder, exist_ok=True)
                _checkpoint(path, self.stored[tag])
            if rejected:
                _warn(f"statistics: {rejected:,} corrupt or out-of-range samples discarded")
            summary["statistics"] = dict(self.outputs)
        return summary


__all__ = ["InlineExtraction"]
