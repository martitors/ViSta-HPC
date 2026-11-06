"""
vista.polarization -- the correlations of the inputs
====================================================

Every row of a stack has the same number of correlations: a Measurement Set
stores DATA with one shape per column, and the output POLARIZATION table is
the one of the first input.  So the entries must all have the same number of
correlations (e.g. XX, YY); otherwise the run would stop while writing, after
most of the work.  :func:`check_correlations` finds it before anything is
read, and also warns when the correlation types differ (XX, YY next to RR,
LL): the stack then labels them all as the first entry does, which is
harmless for Stokes I, (XX + YY) / 2 = (RR + LL) / 2, but not for the single
correlations.

Shared, unchanged, by the main, extract and extract_opt branches.
"""

import contextlib

import numpy as np

CORR_NAMES = {1: "I", 2: "Q", 3: "U", 4: "V", 5: "RR", 6: "RL", 7: "LR", 8: "LL",
              9: "XX", 10: "XY", 11: "YX", 12: "YY"}


def entry_correlations(ms, spws):
    """Correlations of the SPWs of one entry.

    Returns
    -------
    dict
        SPW id -> tuple of correlation names, e.g. ``('XX', 'YY')``, for the
        requested SPWs (``spws`` None: every SPW with a data description).
    """
    from casacore.tables import table

    with table(f"{ms}::DATA_DESCRIPTION", ack=False) as t:
        dd_spw = t.getcol("SPECTRAL_WINDOW_ID")
        dd_pol = t.getcol("POLARIZATION_ID")
    with table(f"{ms}::POLARIZATION", ack=False) as t:
        types = [tuple(CORR_NAMES.get(int(c), str(int(c))) for c in np.ravel(t.getcell("CORR_TYPE", i)))
                 for i in range(t.nrows())]
    wanted = set(dd_spw.tolist()) if spws is None else set(spws)
    return {int(s): types[int(p)] for s, p in zip(dd_spw, dd_pol) if int(s) in wanted}


def check_correlations(ms_list, spw_list, lock=None):
    """Check that every entry has the same number of correlations.

    Parameters
    ----------
    ms_list, spw_list : list
        MS and SPWs (None: all) of every entry.
    lock : context manager or None
        Held while an MS is read.

    Returns
    -------
    list of (str, str)
        Messages for the log, ``(level, text)``: a warning when the
        correlation types differ.

    Raises
    ------
    ValueError
        The SPWs of an entry, or different entries, have different numbers
        of correlations.
    """
    lock = lock or contextlib.nullcontext()
    per_entry = []
    for i, (ms, spws) in enumerate(zip(ms_list, spw_list)):
        name = ms.rstrip("/").split("/")[-1]
        with lock:
            corr = entry_correlations(ms, spws)
        setups = sorted(set(corr.values()))
        if len({len(s) for s in setups}) > 1:
            raise ValueError(
                f"{name}: its SPWs have different numbers of correlations "
                f"({', '.join(f'SPW {s}: ' + ','.join(c) for s, c in sorted(corr.items()))}); "
                f"their channels cannot be merged row by row.  Put the SPWs with "
                f"different correlations on lines of their own, in different stacks")
        per_entry.append((i, name, setups[0] if setups else ()))
    counts = sorted({len(c) for _, _, c in per_entry if c})
    if len(counts) > 1:
        groups = "; ".join(
            f"{n} correlations: " + ", ".join(f"[{i + 1}] {name}" for i, name, c in per_entry
                                               if len(c) == n)
            for n in counts)
        raise ValueError(
            f"the entries have different numbers of correlations ({groups}).  A stack "
            f"needs the same correlations in every row: stack them separately, or "
            f"bring them to the same correlations first (e.g. with mstransform, "
            f"correlation='XX,YY')")
    messages = []
    kinds = sorted({c for _, _, c in per_entry if c})
    if len(kinds) > 1:
        messages.append(("warn", f"the entries have different correlation types "
                                 f"({' / '.join(','.join(k) for k in kinds)}): the stack "
                                 f"labels them as {','.join(per_entry[0][2])}, like the "
                                 f"first entry.  Stokes I is right, single correlations are not"))
    return messages
