#!/usr/bin/env python
"""
Image one dataset as a cube, measure the line in it, and print the limits the
stacking and the extraction should use.

    python inspect_line.py sim_gauss_input.txt --rest-freq 345.7959899
    python inspect_line.py sim_gauss_input.txt --rest-freq 345.7959899 -n 25
    python inspect_line.py sim_gauss_input.txt --rest-freq 345.7959899 --keep

Runs tclean with no deconvolution, finds the source in the collapsed cube,
takes the spectrum at that position, and fits a Gaussian to it.  From the
fitted centroid and width, and from the channels the band actually covers, it
works out the velocity window, the fit span and the radial binning.

The absolute scale of a dirty cube is not the source flux, so the amplitudes
printed here are only indicative; the shape is what this is for.
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vista.extract.sources import read_input_list

C_KMS = 299792.458


def make_cube(entry, name, imsize, cell, nchan=-1):
    from casatasks import tclean
    for suffix in (".image", ".psf", ".residual", ".model", ".pb", ".sumwt"):
        if os.path.exists(name + suffix):
            os.system(f"rm -rf {name}{suffix}")
    tclean(vis=entry.ms, imagename=name, specmode="cube", datacolumn="data",
           imsize=[imsize, imsize], cell=f"{cell}arcsec", niter=0,
           gridder="standard", weighting="natural", pblimit=-1,
           nchan=nchan, interpolation="nearest")
    return name + ".image"


def read_cube(path):
    """Cube as (nchan, ny, nx), plus the frequency of each channel."""
    from casatools import image
    ia = image()
    ia.open(path)
    chunk = ia.getchunk()                      # (nx, ny, npol, nchan)
    csys = ia.coordsys()
    shape = ia.shape()
    n_chan = shape[3] if len(shape) > 3 else 1
    freq = np.array([csys.toworld([0, 0, 0, c], "n")["numeric"][3]
                     for c in range(n_chan)])
    beam = ia.restoringbeam()
    ia.close()
    cube = np.moveaxis(chunk[:, :, 0, :], -1, 0)      # (nchan, nx, ny)
    return cube, freq, beam


def fit_gaussian(velocity, spectrum):
    """Amplitude, centroid, sigma and baseline of the line, all in km/s."""
    from scipy.optimize import curve_fit

    def model(_x, amplitude, centroid, sigma, offset):
        return offset + amplitude * np.exp(
            -0.5 * ((velocity - centroid) / sigma) ** 2)

    dv = float(np.median(np.abs(np.diff(np.sort(velocity)))))
    baseline = float(np.median(spectrum))
    peak = float(np.max(spectrum) - baseline)
    guess = [peak, float(velocity[np.argmax(spectrum)]),
             max(2 * dv, 0.1 * (velocity.max() - velocity.min())), baseline]
    bounds = ([-np.inf, velocity.min(), 0.5 * dv, -np.inf],
              [np.inf, velocity.max(), velocity.max() - velocity.min(), np.inf])
    p, _ = curve_fit(model, np.arange(len(velocity)), spectrum, p0=guess,
                     bounds=bounds, maxfev=20000)
    return p, model(None, *p)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_list")
    parser.add_argument("--rest-freq", type=float, required=True, metavar="GHZ")
    parser.add_argument("-n", "--index", type=int, default=0)
    parser.add_argument("--imsize", type=int, default=256)
    parser.add_argument("--cell", type=float, default=0.05, metavar="ARCSEC")
    parser.add_argument("--box", type=int, default=3,
                        help="half-size, in pixels, of the box summed around "
                             "the source (default 3)")
    parser.add_argument("--keep", action="store_true",
                        help="do not delete the cube afterwards")
    parser.add_argument("--plot", default=None, metavar="PNG",
                        help="save the spectrum and the fit to this file")
    args = parser.parse_args()

    entries = [e for e in read_input_list(args.input_list) if e is not None]
    entry = entries[args.index]
    nu_line = args.rest_freq * 1e9 / (1.0 + entry.z)
    print(f"entry {args.index}: {os.path.basename(entry.ms)}")
    print(f"  z = {entry.z}   line observed at {nu_line / 1e9:.4f} GHz\n")

    name = f"inspect_line_{args.index}"
    cube_path = make_cube(entry, name, args.imsize, args.cell)
    cube, freq, beam = read_cube(cube_path)
    velocity = C_KMS * (nu_line - freq) / nu_line
    dv = float(np.median(np.abs(np.diff(velocity))))

    # where the source is: brightest pixel of the collapsed cube
    collapsed = np.nansum(cube, axis=0)
    iy, ix = np.unravel_index(np.nanargmax(collapsed), collapsed.shape)
    half = args.box
    box = cube[:, max(iy - half, 0):iy + half + 1,
               max(ix - half, 0):ix + half + 1]
    spectrum = np.nansum(box, axis=(1, 2))

    print(f"  cube: {len(freq)} channels, {dv:.2f} km/s each")
    print(f"  band: {velocity.min():+.0f} to {velocity.max():+.0f} km/s "
          f"(line at 0)")
    print(f"  source at pixel ({ix}, {iy}), "
          f"{math.hypot(ix - args.imsize // 2, iy - args.imsize // 2) * args.cell:.2f}\" "
          f"from the phase centre")

    try:
        (amplitude, centroid, sigma, offset), fitted = fit_gaussian(
            velocity, spectrum)
        sigma = abs(sigma)
        fwhm = 2.3548200450309493 * sigma
        print(f"\n  line fit on the spectrum:")
        print(f"    centroid = {centroid:+.1f} km/s   "
              f"({centroid / dv:+.1f} channels from the band centre)")
        print(f"    FWHM     = {fwhm:.1f} km/s = {fwhm / dv:.1f} channels")
        print(f"    peak / baseline = {amplitude:.4g} / {offset:.4g}  "
              f"(ratio {amplitude / offset if offset else float('nan'):.2f})")
        if abs(offset) > 0.1 * abs(amplitude):
            print(f"    the baseline is not negligible: there is a continuum, "
                  f"so use\n    --contsub after (or before), not the default")
    except Exception as exc:
        print(f"\n  the line fit failed ({exc}); falling back to the moments")
        weights = np.clip(spectrum - np.median(spectrum), 0, None)
        centroid = float(np.sum(weights * velocity) / np.sum(weights))
        sigma = float(np.sqrt(np.sum(weights * (velocity - centroid) ** 2)
                              / np.sum(weights)))
        fwhm = 2.3548200450309493 * sigma
        offset = float(np.median(spectrum))
        print(f"    centroid = {centroid:+.1f} km/s   FWHM = {fwhm:.1f} km/s")

    # ---- the limits that follow ------------------------------------------
    window = 3.0 * sigma                       # +-3 sigma holds 99.7%
    reach = min(abs(velocity.min()), abs(velocity.max()))
    step = 5 if reach < 100 else (10 if reach < 500 else 50)
    window = int(math.ceil(window / step) * step)
    span = int(math.floor(reach / step) * step)

    print(f"\n{'=' * 62}")
    print("limits for this dataset")
    print(f"{'=' * 62}")
    print(f"  --v-window \"-{window} {window}\"")
    print(f"      +-3 sigma of the fitted line, {2 * window / dv:.0f} channels; "
          f"it leaves\n      {span - window} km/s of line-free band on each "
          f"side for the continuum.")
    print(f"  --fit-span-kms {span}")
    print(f"      the shape fit cannot use more than the band.")
    print(f"  --velocity-range {span}")
    print(f"      ask for more and ViSta slides the window to the band edge "
          f"for the\n      datasets that fall short.")
    if span <= window:
        print(f"\n  WARNING: the window fills the band, no line-free channels "
              f"are left.\n  Narrow --v-window, or accept that the continuum "
              f"cannot be fitted here.")
    print(f"\n  run inspect_ms.py --all to see what the whole sample supports: "
          f"these\n  numbers come from one dataset, and the stack is bound by "
          f"its narrowest.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        ax.step(velocity, spectrum, where="mid", color="0.3", lw=1.0,
                label="spectrum at the source")
        try:
            ax.plot(velocity, fitted, "-", color="C3", lw=1.5,
                    label=f"FWHM = {fwhm:.0f} km/s")
            ax.axhline(offset, color="C0", ls=":", lw=1.0, label="baseline")
        except NameError:
            pass
        ax.axvspan(-window, window, color="C1", alpha=0.10)
        ax.set_xlabel("velocity [km/s]")
        ax.set_ylabel("cube amplitude")
        ax.legend(fontsize=8, frameon=False)
        figure.savefig(args.plot, dpi=150)
        print(f"\n  plot: {args.plot}")

    if not args.keep:
        for suffix in (".image", ".psf", ".residual", ".model", ".pb",
                       ".sumwt"):
            os.system(f"rm -rf {name}{suffix}")
    else:
        print(f"\n  cube kept: {cube_path}")


if __name__ == "__main__":
    main()
