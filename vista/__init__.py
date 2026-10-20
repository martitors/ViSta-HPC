"""
ViSta: Visibility Stacking tool for broadband interferometric data.

Main class:
    ViSta  -- HPC-optimised visibility-domain stacking pipeline
              (C++/OpenMP kernel, optional CUDA GPU)

Subpackage:
    vista.extract -- post-processing (continuum subtraction) and
                     visibility-domain signal extraction of the stack

Example usage::

    from vista import ViSta
    p = ViSta("input_list.txt")
    p.run("stacked_output.ms", central_freq=153.253e9, nchan_out=1000)

Log: every step of the package writes timestamped lines to stdout through the
logger ``"vista"``; ``vista.set_verbosity()`` chooses how much (and an
optional log file).

New in v2.1:
    - Transparent GPU/CUDA acceleration (batch kernel dispatch)
    - Thread-safe Casacore access (global + per-MS locking)
    - Lazy chunk-at-a-time materialisation with adaptive chunk sizing
    - velocity_range_kms parameter for velocity-based output bandwidth
    - Automatic OBSERVE_TARGET state filtering
    - CORRECTED_DATA preference with DATA fallback
    - Full output subtable set (FIELD, DD, ANTENNA, POL, OBS, FEED, SOURCE)

New in v2.2:
    - vista.extract: continuum subtraction, compression into sufficient
      statistics, and uv-domain flux extraction with bootstrap errors
    - Optional normalisation factor as the last column of the input list

New in v2.3:
    - Optimised stacking loop: parallel reader processes (shared memory),
      GIL-free compute kernel, a single direct casacore writer; the output is
      the same as v2.2 (bit-identical on the CPU), only the row order changes
    - New GPU engine: many chunks per launch, persistent device buffers,
      page-locked output buffers, automatic split into passes; chosen with
      ``run(device="gpu")``
    - Clear log for the whole package (stacking and extraction): elapsed
      time on every line, what each input contributes, warnings for
      suspicious inputs; ``verbose="debug"`` and ``run(timing=True)`` for
      the details
"""

import logging as _logging
import sys as _sys
import time as _time

__all__ = ["ViSta", "set_verbosity"]
__version__ = "2.3.0"


# ---------------------------------------------------------------------------
# Log shared by the whole package
# ---------------------------------------------------------------------------
#   [ViSta     12.3s] message
#   [ViSta     12.5s] WARNING: message
# The time is counted from the start of the current step (ViSta.run, the
# continuum subtraction, the compression, the extraction).

_logger = _logging.getLogger("vista")
_clock = [_time.perf_counter()]


class _Formatter(_logging.Formatter):
    def format(self, record):
        elapsed = _time.perf_counter() - _clock[0]
        msg = record.getMessage()
        if record.levelno >= _logging.WARNING:
            msg = f"{record.levelname}: {msg}"
        return f"[ViSta {elapsed:9.1f}s] {msg}"


if not any(getattr(h, "_vista", None) == "stdout" for h in _logger.handlers):
    _handler = _logging.StreamHandler(_sys.stdout)
    _handler.setFormatter(_Formatter())
    _handler._vista = "stdout"
    _logger.addHandler(_handler)
    _logger.setLevel(_logging.INFO)
    _logger.propagate = False


def set_verbosity(verbose=True, log_file=None):
    """Choose how much the package writes, and optionally where else.

    Parameters
    ----------
    verbose : bool or "debug"
        ``True`` (default) — progress, results and warnings;
        ``False`` — warnings and errors only;
        ``"debug"`` — also the details (per-MS metadata, queues, buffers,
        every subtable written), meant for diagnosing problems.
    log_file : str or None
        Also append the log to this file.  ``None`` stops writing to a file
        set by an earlier call.
    """
    if verbose == "debug":
        level = _logging.DEBUG
    elif verbose:
        level = _logging.INFO
    else:
        level = _logging.WARNING
    _logger.setLevel(level)
    for h in list(_logger.handlers):
        if getattr(h, "_vista", None) == "file":
            _logger.removeHandler(h)
            h.close()
    if log_file:
        fh = _logging.FileHandler(log_file, mode="a")
        fh.setFormatter(_Formatter())
        fh._vista = "file"
        _logger.addHandler(fh)


def _reset_clock():
    """Restart the elapsed time shown on the log lines (start of a step)."""
    _clock[0] = _time.perf_counter()


def _say(msg, verbose=True):
    """Progress / result line (hidden by ``verbose=False``)."""
    if verbose:
        _logger.info(msg)


def _detail(msg):
    """Detail line, shown only with ``set_verbosity("debug")``."""
    _logger.debug(msg)


def _warn(msg):
    """Warning: always shown."""
    _logger.warning(msg)


def __getattr__(name):
    """Import the stacking pipeline lazily (PEP 562).

    ``vista.pipeline`` needs the compiled kernel ``ms_ops`` and the dask-ms
    stack, which the extraction subpackage does not: importing it eagerly
    here would make ``import vista.extract`` fail on a machine where the
    kernel was never built.  ``from vista import ViSta`` still works.
    """
    if name == "ViSta":
        from .pipeline import ViSta
        return ViSta
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
