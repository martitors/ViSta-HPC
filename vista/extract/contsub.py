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
"""

from __future__ import annotations

import os
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
    from casatools import table                      # noqa: local import

    if not os.path.isdir(ms_path):
        raise FileNotFoundError(f"{ms_path!r} does not exist "
                                f"(cwd={os.getcwd()})")
    wanted = set(only_dd) if only_dd else set()

    tb = table()
    tb.open(f"{ms_path}/SPECTRAL_WINDOW")
    chan_freq_per_spw = [np.asarray(tb.getcell("CHAN_FREQ", i), dtype=float)
                         for i in range(tb.nrows())]
    tb.close()
    tb.open(f"{ms_path}/DATA_DESCRIPTION")
    dd_to_spw = tb.getcol("SPECTRAL_WINDOW_ID")
    tb.close()

    use_bic = config.order == -1
    label = (f"BIC (delta={config.delta_bic:.1f})" if use_bic
             else str(config.order))
    _log(f"[contsub] {ms_path}: {len(dd_to_spw)} data descriptors, "
         f"order={label}", verbose=verbose)

    # output columns, created on the fly if missing
    tb.open(ms_path, nomodify=False)
    existing = tb.colnames()
    for column in (config.model_column, config.line_column):
        if column not in existing:
            tb.addcols({column: tb.getcoldesc(config.data_column)})
    tb.close()

    orders_used = {}
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
        x_fit = chan_freq[fit_mask] - x0
        x_all = chan_freq - x0

        tb.open(ms_path)
        dd_column = tb.getcol("DATA_DESC_ID")
        tb.close()
        rows = np.where(dd_column == dd)[0]
        if rows.size == 0:
            continue
        _log(f"  [dd {dd} spw {spw}] fit channels={n_fit}/{nchan}  "
             f"rows={rows.size}", verbose=verbose)

        order = None if use_bic else config.order
        tb.open(ms_path, nomodify=False)
        for start in range(0, rows.size, config.chunk_rows):
            chunk = rows[start:start + config.chunk_rows]
            first, n = int(chunk[0]), len(chunk)
            data = tb.getcol(config.data_column, startrow=first, nrow=n)
            ncorr = data.shape[0]
            model = np.zeros_like(data)

            # the order is decided once per spectral window, on the mean
            # spectrum of the first chunk: high S/N, no extra read
            if use_bic and order is None:
                nonzero = data != 0
                denom = nonzero.sum(axis=(0, 2))
                numer = data.sum(axis=(0, 2))
                mean_spec = np.where(denom > 0,
                                     numer / np.maximum(denom, 1), np.nan)
                y = mean_spec[fit_mask]
                valid = np.isfinite(y)
                order = select_order(x_fit, x_all, y, valid,
                                     config.delta_bic,
                                     config.require_positive_slope)
                _log(f"    [BIC] order chosen on the mean spectrum "
                     f"({int(valid.sum())} channels) = {order}",
                     verbose=verbose)

            need = order + 1
            for c in range(ncorr):
                for r in range(n):
                    y = data[c, :, r][fit_mask]
                    valid = np.isfinite(y)
                    if valid.sum() < need:
                        continue
                    cr = np.polyfit(x_fit[valid], y[valid].real, order)
                    ci = np.polyfit(x_fit[valid], y[valid].imag, order)
                    model[c, :, r] = (np.polyval(cr, x_all)
                                      + 1j * np.polyval(ci, x_all))

            tb.putcol(config.model_column, model, startrow=first, nrow=n)
            tb.putcol(config.line_column, data - model,
                      startrow=first, nrow=n)
        tb.close()
        orders_used[dd] = order

    _log("[contsub] done.", verbose=verbose)
    return orders_used


__all__ = ["subtract_continuum", "select_order"]
