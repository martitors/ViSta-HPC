"""
vista.extract.contsub
=====================
Continuum subtraction on the stacked MS (Sec. 5.2.3).

A low-order polynomial is fitted to the line-free channels, separately for
the real and the imaginary part of the visibilities, and subtracted from
every channel.  The fit runs spectral window by spectral window, reading the
visibilities in chunks of rows, so that even a large stack never has to be
held in memory at once.

The fitted continuum is written to ``MODEL_DATA`` and the subtracted
visibilities to ``CORRECTED_DATA``: the input column is left untouched, so
line quantities can be measured on the subtracted column, continuum
quantities on the model one, and the step can be rerun without damaging the
stack.

The pass is organised as the stacking is: a reader thread reads the chunks
of rows, the C++/OpenMP kernel (``ms_ops.continuum_subtract``) fits and
subtracts the continuum of every spectrum, a writer thread writes the two
columns; bounded queues connect them, so reading, fitting and writing
overlap.  The table is read and written with python-casacore.
"""

from __future__ import annotations

import os
import threading
import time
from queue import Queue
from typing import Iterable, Optional, Sequence

import numpy as np

from .config import ContinuumConfig, LineConfig
from .profiles import line_free_mask


def _log(*args, verbose=True):
    if verbose:
        print(*args, flush=True)


def _bic(rss: float, n: int, k: int) -> float:
    if n <= k or rss <= 0:
        return np.inf
    return n * np.log(rss / n) + k * np.log(n)


def _fit_complex(x_fit, x_all, y_fit, valid, order):
    """Fit a complex spectrum with a polynomial of the given order."""
    xv = x_fit[valid]
    cr = np.polyfit(xv, y_fit[valid].real, order)
    ci = np.polyfit(xv, y_fit[valid].imag, order)
    return np.polyval(cr, x_all) + 1j * np.polyval(ci, x_all), cr, ci


def select_order(x_fit, x_all, y_fit, valid, delta_bic=2.0,
                 require_positive_slope=True):
    """Choose between order 0 and order 1 through their BIC.

    Order 1 is accepted only if it improves the BIC by at least
    ``delta_bic`` and, optionally, if the fitted slope is positive.
    """
    n = int(np.sum(valid))
    if n < 2:
        return 0
    _, cr0, ci0 = _fit_complex(x_fit, x_all, y_fit, valid, 0)
    res0 = y_fit[valid] - (np.polyval(cr0, x_fit[valid])
                           + 1j * np.polyval(ci0, x_fit[valid]))
    bic0 = _bic(float(np.sum(np.abs(res0) ** 2)), n, 1)

    _, cr1, ci1 = _fit_complex(x_fit, x_all, y_fit, valid, 1)
    res1 = y_fit[valid] - (np.polyval(cr1, x_fit[valid])
                           + 1j * np.polyval(ci1, x_fit[valid]))
    bic1 = _bic(float(np.sum(np.abs(res1) ** 2)), n, 2)

    slope_ok = (float(cr1[0]) >= 0) or not require_positive_slope
    return 1 if ((bic0 - bic1) >= delta_bic and slope_ok) else 0


def subtract_continuum(ms_path: str,
                       line: LineConfig,
                       config: ContinuumConfig = ContinuumConfig(),
                       only_dd: Optional[Iterable[int]] = None,
                       start_dd: int = 0,
                       verbose: bool = True) -> dict:
    """Subtract the continuum from a stacked MS, in place, spw by spw.

    Parameters
    ----------
    ms_path
        Path to the stacked Measurement Set.
    line
        Spectral setup: the excluded channels are those inside the line
        window, widened by the second line window and by any extra
        ``exclude_v_kms`` interval.  ``ContinuumConfig.exclude_kms`` (or the
        explicit lo/hi pair) overrides the symmetric default width.
    config
        Fit options: see :class:`~vista.extract.config.ContinuumConfig`.
    only_dd
        Restrict the processing to these data descriptor ids.
    start_dd
        Skip the data descriptors below this id (to resume an interrupted run).

    Returns
    -------
    dict
        ``{dd: order_used}`` for the processed data descriptors.
    """
    from casacore.tables import table, makecoldesc, maketabdesc
    # imported here, not at the top: the extraction must import without the
    # compiled kernel (see vista/__init__.py)
    try:
        from .. import ms_ops            # installed into vista/ by 'cmake --install'
    except ImportError:                  # or found on PYTHONPATH
        import ms_ops

    if not os.path.isdir(ms_path):
        raise FileNotFoundError(f"{ms_path!r} does not exist "
                                f"(cwd={os.getcwd()})")
    wanted = set(only_dd) if only_dd else set()

    with table(f"{ms_path}/SPECTRAL_WINDOW", ack=False) as tb:
        chan_freq_per_spw = [np.asarray(tb.getcell("CHAN_FREQ", i), dtype=float)
                             for i in range(tb.nrows())]
    with table(f"{ms_path}/DATA_DESCRIPTION", ack=False) as tb:
        dd_to_spw = tb.getcol("SPECTRAL_WINDOW_ID")

    use_bic = config.order == -1
    label = (f"BIC (delta={config.delta_bic:.1f})" if use_bic
             else str(config.order))
    _log(f"[contsub] {ms_path}: {len(dd_to_spw)} data descriptors, "
         f"order={label}", verbose=verbose)

    tb = table(ms_path, readonly=False, ack=False)
    # output columns, created on the fly if missing
    existing = tb.colnames()
    for column in (config.model_column, config.line_column):
        if column not in existing:
            tb.addcols(maketabdesc(makecoldesc(
                column, tb.getcoldesc(config.data_column))))
    dd_column = tb.getcol("DATA_DESC_ID")

    # the work: one entry per data descriptor, its rows in contiguous runs
    tasks = []
    for dd, spw in enumerate(dd_to_spw):
        if dd < start_dd:
            continue
        if wanted and dd not in wanted and spw not in wanted:
            continue
        chan_freq = chan_freq_per_spw[spw]
        nchan = len(chan_freq)

        fit_mask = line_free_mask(chan_freq, line,
                                  half_width_kms=config.exclude_kms,
                                  window_kms=(
                                      (config.exclude_kms_lo,
                                       config.exclude_kms_hi)
                                      if (config.exclude_kms_lo is not None
                                          and config.exclude_kms_hi is not None)
                                      else None))
        n_fit = int(fit_mask.sum())
        min_chan = 2 if use_bic else config.order + 1
        if n_fit < min_chan:
            _log(f"  [dd {dd} spw {spw}] only {n_fit} line-free channels "
                 f"- skipped", verbose=verbose)
            continue

        x0 = chan_freq[fit_mask].mean()
        rows = np.where(dd_column == dd)[0]
        if rows.size == 0:
            continue
        _log(f"  [dd {dd} spw {spw}] fit channels={n_fit}/{nchan}  "
             f"rows={rows.size}", verbose=verbose)
        runs = np.split(rows, np.where(np.diff(rows) != 1)[0] + 1)
        chunks = [(int(run[start]), int(min(config.chunk_rows, run.size - start)))
                  for run in runs
                  for start in range(0, run.size, config.chunk_rows)]
        tasks.append({"dd": dd, "fit_mask": fit_mask,
                      "x_fit": chan_freq[fit_mask] - x0,
                      "x_all": chan_freq - x0, "chunks": chunks,
                      "order": None if use_bic else config.order})

    # reader -> kernel -> writer, as in the stacking
    io_lock = threading.Lock()              # one table, one thread at a time
    read_q = Queue(maxsize=2)
    write_q = Queue(maxsize=2)
    failure = [None]
    timers = {"read": 0.0, "compute": 0.0, "write": 0.0,
              "read_wait": 0.0, "write_wait": 0.0}

    def reader():
        try:
            for task in tasks:
                for first, n in task["chunks"]:
                    t0 = time.perf_counter()
                    with io_lock:
                        data = tb.getcol(config.data_column, startrow=first,
                                         nrow=n)
                    timers["read"] += time.perf_counter() - t0
                    read_q.put((task, first, n, data))
                    if failure[0] is not None:
                        return
        except Exception as exc:                     # noqa: BLE001
            failure[0] = exc
        finally:
            read_q.put(None)

    def writer():
        try:
            while True:
                t0 = time.perf_counter()
                item = write_q.get()
                timers["write_wait"] += time.perf_counter() - t0
                if item is None:
                    return
                first, n, model, corrected = item
                t0 = time.perf_counter()
                with io_lock:
                    tb.putcol(config.model_column, model, startrow=first, nrow=n)
                    tb.putcol(config.line_column, corrected, startrow=first,
                              nrow=n)
                timers["write"] += time.perf_counter() - t0
        except Exception as exc:                     # noqa: BLE001
            failure[0] = exc
            while write_q.get() is not None:         # let the producer finish
                pass

    threads = [threading.Thread(target=reader, daemon=True),
               threading.Thread(target=writer, daemon=True)]
    for th in threads:
        th.start()

    orders_used = {}
    try:
        while True:
            t0 = time.perf_counter()
            item = read_q.get()
            timers["read_wait"] += time.perf_counter() - t0
            if item is None:
                break
            if failure[0] is not None:
                continue
            task, first, n, data = item
            # the order is decided once per spectral window, on the mean
            # spectrum of its first chunk: high S/N, no extra read
            if task["order"] is None:
                nonzero = data != 0
                denom = nonzero.sum(axis=(0, 2))
                numer = data.sum(axis=(0, 2), dtype=np.complex128)
                mean_spec = np.where(denom > 0,
                                     numer / np.maximum(denom, 1), np.nan)
                y = mean_spec[task["fit_mask"]]
                valid = np.isfinite(y)
                task["order"] = select_order(task["x_fit"], task["x_all"], y,
                                             valid, config.delta_bic,
                                             config.require_positive_slope)
                _log(f"    [BIC] dd {task['dd']}: order chosen on the mean "
                     f"spectrum ({int(valid.sum())} channels) = "
                     f"{task['order']}", verbose=verbose)
            orders_used[task["dd"]] = task["order"]
            t0 = time.perf_counter()
            model, corrected = ms_ops.continuum_subtract(
                data, task["x_all"], task["fit_mask"], task["order"])
            timers["compute"] += time.perf_counter() - t0
            write_q.put((first, n, model, corrected))
    except Exception as exc:                         # noqa: BLE001
        failure[0] = exc
        while read_q.get() is not None:              # let the reader finish
            pass
    finally:
        write_q.put(None)
        for th in threads:
            th.join()
        tb.flush()
        tb.close()
    if failure[0] is not None:
        raise failure[0]

    _log(f"[contsub] done.  read {timers['read']:.1f} s, "
         f"fit (C++) {timers['compute']:.1f} s, write {timers['write']:.1f} s "
         f"(waits: read {timers['read_wait']:.1f} s, "
         f"write {timers['write_wait']:.1f} s)", verbose=verbose)
    return orders_used


__all__ = ["subtract_continuum", "select_order"]
