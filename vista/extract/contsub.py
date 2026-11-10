"""
vista.extract.contsub
=====================
Continuum subtraction on the stack, Zarr store or MS (Sec. 5.2.3).

A low-order polynomial is fitted to the line-free channels, separately for
the real and the imaginary part of the visibilities, and subtracted from
every channel.  The fit runs spectral window by spectral window, reading the
visibilities in chunks of rows, so that even a large stack never has to be
held in memory at once.

Only the unflagged samples enter the fit.  In a stack, the channels of the
common grid that an entry does not cover are flagged and hold 0: counted in
the fit, they would pull the continuum towards zero by the uncovered
fraction of the line-free channels.

The fitted continuum is written to ``MODEL_DATA`` and the subtracted
visibilities to ``CORRECTED_DATA``: the input column is left untouched, so
line quantities can be measured on the subtracted column, continuum
quantities on the model one, and the step can be rerun without damaging the
stack, e.g. with a wider exclusion window after a look at the spectrum.

The same subtraction can run while the stack is written (``ViSta.run(...,
continuum=...)``, see :mod:`vista.extract.inline`), with the same functions
and identical results.  Either way the parameters are recorded in the stack
(:func:`record`), so it is always known which window produced the
``CORRECTED_DATA`` on disk.
"""

from __future__ import annotations

import datetime
import os
import time
from typing import Iterable, Optional

import numpy as np

from .. import _reset_clock, _say, _warn
from .config import ContinuumConfig, Lines

# A Measurement Set is read in pieces of at most this many bytes of the data
# column (complex64): about 1 GB in memory with the read-ahead, the fit and
# the pending write.
CHUNK_BYTES = 256 * 2**20


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


def _kernel():
    """The compiled kernel (vista/ms_ops), or None: NumPy is used instead."""
    try:
        from .. import ms_ops
    except ImportError:
        try:
            import ms_ops
        except ImportError:
            return None
    return ms_ops if hasattr(ms_ops, "continuum_subtract") else None


def fit_rows(data, fit_mask, x_all, order, kernel=None, flag=None):
    """Per row and correlation: polynomial continuum fitted on the valid
    values of the fit channels (finite and, with *flag*, not flagged), real
    and imaginary parts separately.

    data : complex [nrow, nchan, ncorr]; flag : bool, same shape, or None.
    Returns (model, data - model).  Rows with fewer than order + 1 valid fit
    values get a zero model.
    """
    if kernel is not None:
        return kernel.continuum_subtract(
            np.ascontiguousarray(data, np.complex64), np.ascontiguousarray(fit_mask, bool),
            np.ascontiguousarray(x_all, np.float64), int(order),
            None if flag is None else np.ascontiguousarray(flag, bool))
    # NumPy: in float64 like the kernel, the model and the difference rounded
    # once to the type of the data
    data64 = np.asarray(data, dtype=np.complex128)
    model = np.zeros_like(data64)
    x_fit = x_all[fit_mask]
    for r in range(data.shape[0]):
        for c in range(data.shape[2]):
            y = data64[r, fit_mask, c]
            valid = np.isfinite(y)
            if flag is not None:
                valid &= ~flag[r, fit_mask, c]
            if valid.sum() < order + 1:
                continue
            cr = np.polyfit(x_fit[valid], y[valid].real, order)
            ci = np.polyfit(x_fit[valid], y[valid].imag, order)
            model[r, :, c] = np.polyval(cr, x_all) + 1j * np.polyval(ci, x_all)
    return model.astype(data.dtype), (data64 - model).astype(data.dtype)


def fit_plan(chan_freq, line: Lines, config: ContinuumConfig):
    """Line-free channels of one data descriptor.

    Returns ``(fit_mask, x_all, n_fit)``: the channels used in the fit, the
    abscissa of every channel (frequency minus the mean frequency of the fit
    channels, which keeps the linear fit well conditioned) and their number.
    """
    chan_freq = np.asarray(chan_freq, dtype=np.float64)
    fit_mask = line.free_mask(chan_freq)
    n_fit = int(fit_mask.sum())
    x0 = chan_freq[fit_mask].mean() if n_fit else 0.0
    return fit_mask, chan_freq - x0, n_fit


def min_fit_channels(config: ContinuumConfig) -> int:
    """Line-free channels needed by the fit (2 to choose the order)."""
    return 2 if config.order == -1 else config.order + 1


def order_from_sample(data, flag, fit_mask, x_all, config: ContinuumConfig):
    """Order chosen by BIC on the mean spectrum of a sample of rows.

    The sample is the first ``config.bic_rows`` rows of the data descriptor,
    whatever the chunking: the separate step and the subtraction during the
    stacking see the same rows and choose the same order.  Unflagged nonzero
    values only.  Returns ``(order, n_channels)``.
    """
    good = (data != 0) & ~flag
    denom = good.sum(axis=(0, 2))
    numer = np.where(good, data, 0).sum(axis=(0, 2), dtype=np.complex128)
    mean_spec = np.where(denom > 0, numer / np.maximum(denom, 1), np.nan)
    y = mean_spec[fit_mask]
    valid = np.isfinite(y)
    order = select_order(x_all[fit_mask], x_all, y, valid, config.delta_bic,
                         config.require_positive_slope)
    return order, int(valid.sum())


def record(line: Lines, config: ContinuumConfig, orders: dict,
           skipped=(), mode: str = "separate") -> dict:
    """Description of a subtraction, stored in the stack (and in the
    statistics) so that the window behind CORRECTED_DATA is never ambiguous."""
    from .. import __version__
    return {
        "vista_version": __version__,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "data_column": config.data_column, "model_column": config.model_column,
        "line_column": config.line_column,
        "rest_freq_ghz": float(line.rest_freq_ghz),
        "v_window_kms": [float(v) for v in line.v_window_kms],
        "lines": line.as_dict(),
        "order": int(config.order), "delta_bic": float(config.delta_bic),
        "require_positive_slope": bool(config.require_positive_slope),
        "bic_rows": int(config.bic_rows),
        "orders": {str(int(k)): int(v) for k, v in sorted(orders.items())},
        "no_fit": sorted(int(d) for d in skipped),
    }


def describe_record(rec: dict) -> str:
    """One line for the log: when and with which window."""
    lines = rec.get("lines")
    if lines:
        parts = [f"{f:.6g} GHz {w[0]:+.0f}..{w[1]:+.0f} km/s"
                 for f, w in zip(lines["rest_freq_ghz"], lines["window_kms"])]
        window = "; ".join(parts)
        extra = ", and between the lines" if lines.get("exclude_between") else ""
    elif rec.get("exclude_kms_lo") is not None and rec.get("exclude_kms_hi") is not None:
        window = f"{rec['exclude_kms_lo']:+.0f}..{rec['exclude_kms_hi']:+.0f} km/s"
        extra = ""
    else:                                   # recorded by a version before 3.0
        window = f"+-{rec.get('exclude_kms', float('nan')):.0f} km/s"
        extra = ""
    order = {-1: "BIC", 0: "0", 1: "1"}.get(rec.get("order"), "?")
    return (f"{rec.get('date', '?')} ({rec.get('mode', '?')}): window {window}{extra}, "
            f"order {order}")


def subtract_continuum(ms_path: str,
                       line: Lines,
                       config: ContinuumConfig = ContinuumConfig(),
                       only_dd: Optional[Iterable[int]] = None,
                       start_dd: int = 0,
                       verbose: bool = True,
                       n_io: int = 4) -> dict:
    """Subtract the continuum from a stack, in place, spw by spw.

    Parameters
    ----------
    ms_path
        Path to the stack: the Zarr store written by the stacking, or a
        Measurement Set.
    line
        The lines (:class:`~vista.extract.config.Lines`): the channels inside
        their windows (and between them with ``exclude_between``) are left
        out of the fit.
    config
        Fit options: see :class:`~vista.extract.config.ContinuumConfig`.
    only_dd
        Restrict the processing to these data descriptor ids.
    start_dd
        Skip the data descriptors below this id (to resume an interrupted run).
    n_io
        Threads reading and writing a Zarr store at the same time (a
        Measurement Set is always read and written by one thread).

    The fit of every row runs in the compiled kernel (OpenMP) when it is
    available, while the next chunks are read and the previous ones written.

    Returns
    -------
    dict
        ``{dd: order_used}`` for the processed data descriptors.
    """
    from ..stackio import open_stack, process_pieces

    if not os.path.isdir(ms_path):
        raise FileNotFoundError(f"{ms_path!r} does not exist "
                                f"(cwd={os.getcwd()})")
    wanted = set(only_dd) if only_dd else set()
    _reset_clock()
    t_start = time.perf_counter()
    kernel = _kernel()
    stack = open_stack(ms_path, writable=True)
    try:
        dd_freq = stack.dd_frequencies()
        dd_to_spw = stack.dd_to_spw()
        present = set(stack.dd_ids())

        use_bic = config.order == -1
        label = (f"BIC (delta={config.delta_bic:.1f})" if use_bic
                 else str(config.order))
        _say(f"Continuum subtraction: {ms_path} ({'Zarr' if stack.kind == 'zarr' else 'MS'}), "
             f"{len(dd_to_spw)} data descriptors, polynomial order {label}; "
             f"{config.data_column} -> continuum in {config.model_column}, "
             f"line in {config.line_column}"
             + ("" if kernel else "  [NumPy: compiled kernel not found]"), verbose)

        previous = stack.get_record("contsub")
        if previous:
            _say(f"  replaces the subtraction of {describe_record(previous)}", verbose)

        # output columns, created on the fly if missing
        stack.add_columns([config.model_column, config.line_column],
                          like=config.data_column)

        plan, tasks, no_fit = {}, [], []
        for dd, spw in enumerate(dd_to_spw):
            if dd < start_dd:
                continue
            if wanted and dd not in wanted and spw not in wanted:
                continue
            if dd not in present:
                continue
            chan_freq = dd_freq[dd]
            nchan = len(chan_freq)
            fit_mask, x_all, n_fit = fit_plan(chan_freq, line, config)
            if n_fit < min_fit_channels(config):
                _warn(f"continuum subtraction: dd {dd} (SPW {spw}) has only {n_fit} "
                      f"line-free channels: no fit, its {config.line_column} stays "
                      f"empty and it drops out of the line statistics")
                no_fit.append(dd)
                continue
            # the rows of a data descriptor need not be consecutive in an MS:
            # the pieces follow the runs of consecutive rows, at most
            # chunk_rows and CHUNK_BYTES each (Zarr: the chunks of the store)
            pieces = stack.pieces(dd)
            if not pieces:
                continue
            if stack.kind == "ms":
                ncorr = stack.read(dd, pieces[0][0], 1,
                                   [config.data_column])[config.data_column].shape[-1]
                max_rows = min(int(config.chunk_rows),
                               max(1, CHUNK_BYTES // (nchan * ncorr * 8)))
                pieces = stack.pieces(dd, max_rows)
            order = None if use_bic else config.order
            n_rows = sum(n for _, n in pieces)
            _say(f"  dd {dd} (SPW {spw}): fit on {n_fit}/{nchan} line-free channels, "
                 f"{n_rows:,} rows", verbose)
            if use_bic:
                # chosen once per data descriptor, on the mean spectrum of
                # its first bic_rows rows: high S/N, a small extra read
                sample = stack.read_first(dd, int(config.bic_rows),
                                          [config.data_column, "FLAG"])
                order, n_valid = order_from_sample(sample[config.data_column],
                                                   sample["FLAG"], fit_mask, x_all,
                                                   config)
                del sample
                _say(f"    order {order}, chosen by BIC on the mean spectrum of the "
                     f"first {min(n_rows, int(config.bic_rows)):,} rows "
                     f"({n_valid} channels)", verbose)
            plan[dd] = {"fit_mask": fit_mask, "x_all": x_all, "order": order}
            tasks += [(dd, r0, n) for r0, n in pieces]

        def read(task):
            dd, r0, n = task
            return stack.read(dd, r0, n, [config.data_column, "FLAG"])

        def compute(task, columns):
            p = plan[task[0]]
            return fit_rows(columns[config.data_column], p["fit_mask"], p["x_all"],
                            p["order"], kernel, columns["FLAG"])

        def write(task, result):
            dd, r0, _ = task
            stack.write(dd, r0, config.model_column, result[0])
            stack.write(dd, r0, config.line_column, result[1])

        process_pieces(stack, tasks, read, compute, write, n_io=n_io)
        orders_used = {dd: p["order"] for dd, p in plan.items()}
        stack.set_record("contsub", record(line, config, orders_used, no_fit,
                                           mode="separate"))
    finally:
        stack.close()

    _say(f"Continuum subtraction done in {time.perf_counter() - t_start:.1f} s: "
         f"{len(orders_used)} data descriptors", verbose)
    return orders_used


__all__ = ["subtract_continuum", "select_order", "fit_rows", "fit_plan",
           "order_from_sample", "min_fit_channels", "record", "describe_record"]
