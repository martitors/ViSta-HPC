"""
ViSta: Visibility Stacking tool for broadband interferometric data.

::

    from vista import ViSta

    v = ViSta("input_list.txt", output_name="co32")   # input, output, how it runs
    v.stack(line_freq=345.796, line_window=500)                  # what makes the stack
    v.extract()                                        # what is measured (skip: stack only)
    result = v.run()            # stack, subtract the continuum, compress, measure
    print(result)

``ViSta(...)`` takes the input list, the outputs and how the run uses the
machine; ``v.stack(...)`` the lines, the band and the continuum;
``v.extract(...)`` the measurement (without it, ``run()`` only writes the
stack); ``v.help()`` lists every option with its value; ``v.run()`` runs
them (``v.run(only="measure")`` repeats the measurement).

Log: nothing but warnings by default; ``ViSta(..., verbose=True)`` shows the
progress and the results, ``verbose="debug"`` also the details and the time
of every stage, and ``log_file`` writes the same to a file.

Output: the stack is a Zarr store (written in parallel) or a Measurement
Set; ``vista.export_ms(store, ms)`` turns a store into a Measurement Set.

The history of the versions is in the documentation (Changelog).
"""

import logging as _logging
import sys as _sys
import time as _time

__all__ = ["ViSta", "set_verbosity", "export_ms", "open_stack"]
__version__ = "3.0.0"


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
        if record.levelno >= _logging.WARNING and not getattr(record, "vista_notice", False):
            msg = f"{record.levelname}: {msg}"
        return f"[ViSta {elapsed:9.1f}s] {msg}"


if not any(getattr(h, "_vista", None) == "stdout" for h in _logger.handlers):
    _handler = _logging.StreamHandler(_sys.stdout)
    _handler.setFormatter(_Formatter())
    _handler._vista = "stdout"
    _logger.addHandler(_handler)
    _logger.setLevel(_logging.WARNING)
    _logger.propagate = False


def set_verbosity(verbose=False, log_file=None):
    """Choose how much the package writes, and optionally where else
    (:class:`ViSta` calls it with its own ``verbose`` and ``log_file``).

    Parameters
    ----------
    verbose : bool or "debug"
        ``False`` (default) — warnings and errors only;
        ``True`` — progress, results and warnings;
        ``"debug"`` — also the details (per-MS metadata, queues, buffers,
        every subtable written) and the time of every stage of the
        stacking, meant for diagnosing problems.
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


def _notice(msg):
    """A result always shown, also with ``verbose=False`` (not a warning)."""
    _logger.warning(msg, extra={"vista_notice": True})


def __getattr__(name):
    """Import the stacking pipeline lazily (PEP 562).

    ``vista.pipeline`` needs the compiled kernel ``ms_ops`` and the dask-ms
    stack, which the extraction subpackage does not: importing it eagerly
    here would make ``import vista.extract`` fail on a machine where the
    kernel was never built.  :class:`ViSta` imports it only to stack.
    """
    if name in ("ViSta", "RunResult"):
        from . import api
        return getattr(api, name)
    if name in ("export_ms", "open_stack"):
        from . import stackio
        return getattr(stackio, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
