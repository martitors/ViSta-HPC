#!/usr/bin/env python
"""
Recover the rest frequency the data were built with, from the rest-framed
copies that sit next to each dataset.

    python check_rest.py sim_gauss_input.txt
    python check_rest.py sim_gauss_input.txt --suffix .ms.rest.center
    python check_rest.py sim_gauss_input.txt --candidates 345.7959899 345.758964

For each entry it opens the observed MS and its ``.rest`` counterpart, and
reports three things: the redshift implied by the ratio of the two frequency
axes, the frequency at the centre of the rest-framed band, and what velocity
that centre corresponds to under each candidate rest frequency.

If the line sits at the centre of every band, as it does in a simulation, the
candidate that gives a velocity near zero is the one the data were built with.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vista.extract.sources import read_input_list

C_KMS = 299792.458


def open_table(path):
    try:
        from casatools import table
        tb = table()
        tb.open(path)
        return tb
    except ImportError:
        from casacore.tables import table as cctable
        return cctable(path, readonly=True, ack=False)


def frequencies(ms_path, spws=None):
    """Concatenated channel frequencies of the selected spectral windows."""
    tb = open_table(os.path.join(ms_path, "SPECTRAL_WINDOW"))
    n_chan = tb.getcol("NUM_CHAN")
    wanted = spws if spws else list(range(len(n_chan)))
    out = []
    for spw in wanted:
        freq = np.asarray(tb.getcell("CHAN_FREQ", spw), float)
        out.append(freq[:int(n_chan[spw])])
    tb.close()
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_list")
    parser.add_argument("--suffix", default=".rest",
                        help="appended to the MS path to find the rest-framed "
                             "copy (default .rest)")
    parser.add_argument("--candidates", nargs="+", type=float,
                        default=[345.7959899, 345.758964], metavar="GHZ",
                        help="rest frequencies to test")
    parser.add_argument("--max", type=int, default=6,
                        help="how many entries to check (default 6)")
    args = parser.parse_args()

    entries = [e for e in read_input_list(args.input_list) if e is not None]
    step = max(1, len(entries) // args.max)
    chosen = entries[::step][:args.max]

    header = f"{'dd':>4} {'z':>6} {'z_implied':>10} {'band centre':>14} "
    header += "".join(f"{c:>14.6f}" for c in args.candidates)
    print(header)
    print(f"{'':>4} {'':>6} {'':>10} {'GHz (rest)':>14} "
          + "".join(f"{'km/s':>14}" for _ in args.candidates))
    print("-" * len(header))

    velocities = {c: [] for c in args.candidates}
    for entry in chosen:
        rest_path = entry.ms + args.suffix
        if not os.path.isdir(rest_path):
            print(f"{entry.dd:>4} {entry.z:>6}  missing: "
                  f"{os.path.basename(rest_path)}")
            continue
        try:
            observed = frequencies(entry.ms, entry.spws)
            rested = frequencies(rest_path)
        except Exception as exc:
            print(f"{entry.dd:>4} {entry.z:>6}  cannot read: {exc}")
            continue

        z_implied = float(np.median(rested) / np.median(observed)) - 1.0
        centre = 0.5 * (rested.min() + rested.max())
        row = f"{entry.dd:>4} {entry.z:>6} {z_implied:>10.5f} " \
              f"{centre / 1e9:>14.6f} "
        for c in args.candidates:
            nu = c * 1e9
            v = C_KMS * (nu - centre) / nu
            velocities[c].append(v)
            row += f"{v:>14.1f}"
        print(row)

    print()
    for c in args.candidates:
        values = velocities[c]
        if not values:
            continue
        mean = float(np.mean(values))
        spread = float(np.std(values))
        verdict = ("  <-- this is the one" if abs(mean) < 5 and spread < 5
                   else "")
        print(f"  {c:.6f} GHz : the band centre sits at "
              f"{mean:+.1f} +- {spread:.1f} km/s{verdict}")
    print("\n  A candidate that puts every band centre at zero, with no "
          "scatter, is\n  the rest frequency the cubes were written with.  A "
          "constant nonzero\n  offset is a different rest frequency; an offset "
          "that varies with the\n  entry is a redshift problem instead.")


if __name__ == "__main__":
    main()
