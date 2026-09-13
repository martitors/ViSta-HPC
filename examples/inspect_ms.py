#!/usr/bin/env python
"""
Read one dataset and work out the ranges the stacking and the extraction
should use, instead of guessing them.

    python inspect_ms.py sim_gauss_input.txt --rest-freq 345.7959899
    python inspect_ms.py sim_gauss_input.txt --rest-freq 345.7959899 -n 7
    python inspect_ms.py sim_gauss_input.txt --rest-freq 345.7959899 --all

Prints, for the chosen entry: the spectral coverage in velocity around the
line, the channel width in km/s, the range of projected baselines in the rest
frame, and the settings that follow from them.  With ``--all`` it scans every
entry and reports the values the whole sample can support, which are the ones
that actually matter: the stack can only be as wide as its narrowest member
and only as fine as its coarsest.
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vista.extract.sources import read_input_list

C_KMS = 299792.458
C_MS = 299792458.0


def open_table(path):
    try:
        from casatools import table
        tb = table()
        tb.open(path)
        return tb
    except ImportError:
        from casacore.tables import table as cctable
        return cctable(path, readonly=True, ack=False)


def inspect(entry, rest_freq_ghz, max_rows=200_000, verbose=True):
    """Spectral and uv coverage of one dataset, in the frame of the line."""
    nu_line = rest_freq_ghz * 1e9 / (1.0 + entry.z)        # observed, Hz

    tb = open_table(os.path.join(entry.ms, "SPECTRAL_WINDOW"))
    n_chan = tb.getcol("NUM_CHAN")
    spws = entry.spws if entry.spws else list(range(len(n_chan)))
    freqs, widths = [], []
    for spw in spws:
        freqs.append(np.asarray(tb.getcell("CHAN_FREQ", spw), float)[:int(n_chan[spw])])
        widths.append(abs(float(np.asarray(tb.getcell("CHAN_WIDTH", spw), float)[0])))
    tb.close()
    freq = np.concatenate(freqs)
    df = float(np.median(widths))

    velocity = C_KMS * (nu_line - freq) / nu_line
    dv = C_KMS * df / nu_line

    # projected baselines, in the rest frame of the line
    tb = open_table(entry.ms)
    n_rows = tb.nrows()
    step = max(1, n_rows // max_rows)
    uvw = tb.getcol("UVW")[:, ::step] if hasattr(tb, "getcol") else None
    tb.close()
    b_metres = np.sqrt(uvw[0] ** 2 + uvw[1] ** 2)
    b_lambda = np.outer(freq / C_MS, b_metres)
    b_kl = np.array([b_lambda.min(), np.percentile(b_lambda, 50),
                     b_lambda.max()]) / 1e3

    if verbose:
        print(f"  z = {entry.z}   nu_line(obs) = {nu_line / 1e9:.4f} GHz")
        print(f"  {len(freq)} channels over {len(spws)} spw, "
              f"{freq.min() / 1e9:.4f} - {freq.max() / 1e9:.4f} GHz")
        print(f"  channel width: {df / 1e6:.3f} MHz = {dv:.1f} km/s "
              f"(rest frame: {df * (1 + entry.z) / 1e6:.3f} MHz)")
        print(f"  velocity coverage: {velocity.min():+.0f} to "
              f"{velocity.max():+.0f} km/s   (line at 0)")
        print(f"  baselines: {b_kl[0]:.1f} / {b_kl[1]:.1f} / {b_kl[2]:.1f} "
              f"klambda  (min / median / max)")
        print(f"  rows: {n_rows:,}")
    return dict(z=entry.z, nu_line=nu_line, n_chan=len(freq), df=df, dv=dv,
                v_lo=float(velocity.min()), v_hi=float(velocity.max()),
                df_rest=df * (1 + entry.z),
                b_min=float(b_kl[0]), b_max=float(b_kl[2]), n_rows=n_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_list")
    parser.add_argument("--rest-freq", type=float, required=True, metavar="GHZ")
    parser.add_argument("-n", "--index", type=int, default=0,
                        help="which entry to inspect (default the first)")
    parser.add_argument("--all", action="store_true",
                        help="scan every entry and report what the sample supports")
    args = parser.parse_args()

    entries = [e for e in read_input_list(args.input_list) if e is not None]
    print(f"{args.input_list}: {len(entries)} entries, "
          f"line at {args.rest_freq} GHz\n")

    if not args.all:
        entry = entries[args.index]
        print(f"entry {args.index}: {os.path.basename(entry.ms)}")
        info = inspect(entry, args.rest_freq)
        results = [info]
    else:
        results = []
        for i, entry in enumerate(entries):
            try:
                info = inspect(entry, args.rest_freq, verbose=False)
            except Exception as exc:
                print(f"  [{i}] cannot read {os.path.basename(entry.ms)}: {exc}")
                continue
            results.append(info)
            print(f"  [{i:>3}] z={info['z']:<6} "
                  f"v=[{info['v_lo']:+6.0f},{info['v_hi']:+6.0f}] km/s  "
                  f"dv={info['dv']:5.1f}  "
                  f"b=[{info['b_min']:6.1f},{info['b_max']:7.1f}] kl")
        if not results:
            sys.exit("nothing could be read")

    # ---- what the sample as a whole supports ------------------------------
    v_lo = max(r["v_lo"] for r in results)        # narrowest common coverage
    v_hi = min(r["v_hi"] for r in results)
    dv_max = max(r["dv"] for r in results)
    df_rest_max = max(r["df_rest"] for r in results)
    b_min = min(r["b_min"] for r in results)
    b_max = max(r["b_max"] for r in results)

    # round to a step that suits the band: a 50 km/s grid is meaningless
    # when the whole coverage is 200 km/s wide
    coverage = min(abs(v_lo), abs(v_hi))
    step = 50 if coverage > 500 else (10 if coverage > 100 else 5)
    half = int(math.floor(coverage / step) * step)
    window = int(math.ceil(0.6 * half / step) * step)

    print(f"\n{'=' * 66}")
    print("what the sample supports")
    print(f"{'=' * 66}")
    print(f"  common velocity coverage : {v_lo:+.0f} to {v_hi:+.0f} km/s")
    print(f"  coarsest channel         : {dv_max:.1f} km/s "
          f"({df_rest_max / 1e6:.3f} MHz rest frame)")
    print(f"  channels across the band : ~{int(2 * half / dv_max)}")
    print(f"  baselines                : {b_min:.1f} to {b_max:.1f} klambda")

    print(f"\nsuggested settings")
    print(f"  --velocity-range {half}")
    print(f"      the widest window every dataset covers, rounded down.  More "
          f"than\n      this and ViSta slides the window to the band edge for "
          f"the datasets\n      that fall short, so for them the stack is no "
          f"longer centred on the line.")
    print(f"  --v-window \"-{window} {window}\"")
    print(f"      a starting guess: {window / dv_max:.0f} channels, and it "
          f"leaves\n      {half - window:.0f} km/s of line-free band on each "
          f"side.  Set it from the\n      line profile once you have seen it, "
          f"wide enough to contain the\n      wings and narrow enough to leave "
          f"line-free channels.")
    print(f"  --fit-span-kms {half}")
    print(f"      the shape fit cannot use more than the band, and the default "
          f"of\n      1500 km/s is meant for far wider data than these.")
    print(f"  --bins \"{max(1, math.floor(b_min)):g} "
          f"{math.ceil(b_max / 10) * 10:g} {12 if len(results) < 30 else 16}\"")
    print(f"      brackets the baselines actually present.  The default 3-3000 "
          f"would\n      leave most annuli empty, and they would be dropped.")
    if len(results) > 1 and max(r["dv"] for r in results) / min(
            r["dv"] for r in results) > 1.3:
        print(f"\n  note: the channel width in velocity runs from "
              f"{min(r['dv'] for r in results):.1f} to {dv_max:.1f} km/s "
              f"across the\n  sample, so a line of fixed intrinsic width is "
              f"sampled differently at\n  each redshift.  Expected when the "
              f"simulations share an observed\n  channel width in Hz.")


if __name__ == "__main__":
    main()
