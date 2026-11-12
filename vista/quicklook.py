"""
vista.quicklook
===============
A quick look at visibilities in the image plane: the **dirty image**, the
Fourier transform of the visibilities as they are (natural weights, no
deconvolution).  It is what the data show before any cleaning: a source
convolved with the response of the array, sidelobes included.  Enough to
see where the emission is and how it compares with the noise; for a proper
image, export the stack (:func:`vista.export_ms`) and clean it.

::

    img = vista.dirty_image("stacked.zarr", line_freq=345.796, velocity=(-300, 300))
    vista.plot_image(img)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

C = 299792458.0
C_KMS = 299792.458
ARCSEC = np.pi / 180.0 / 3600.0


@dataclass
class DirtyImage:
    """A dirty image: ``data`` [data unit / beam], north up and east left
    when drawn with ``extent`` [arcsec]; ``rms`` away from the centre;
    ``beam_arcsec``, the approximate resolution (FWHM)."""

    data: np.ndarray
    extent: tuple
    rms: float
    beam_arcsec: float
    label: str = ""


def image_visibilities(u, v, vis, weight, fov_arcsec=None, pixels_per_beam=5):
    """Dirty image of visibilities ``vis`` at ``(u, v)`` [wavelengths] with
    weights ``weight`` (one value each): they are put on a grid, with their
    complex conjugates at ``(-u, -v)``, and Fourier transformed.  The image
    is normalised by the sum of the weights, so a point source of flux S at
    the centre peaks at S.  The field of view defaults to 30 beams."""
    u, v, vis, weight = (np.asarray(a).ravel() for a in (u, v, vis, weight))
    keep = (weight > 0) & np.isfinite(vis)
    u, v, vis, weight = u[keep], v[keep], vis[keep], weight[keep]
    if not len(u):
        raise ValueError("no visibility to image")
    q = np.hypot(u, v)
    q_max = float(q.max())
    # resolution: about 1 / (2 q_rms), FWHM of the natural-weight beam
    beam = 0.9 / (2.0 * float(np.sqrt(np.average(q ** 2, weights=weight)))) / ARCSEC
    cell = min(beam / pixels_per_beam, 1.0 / (2.2 * q_max) / ARCSEC)   # at least Nyquist
    fov = 30.0 * beam if fov_arcsec is None else float(fov_arcsec)
    n_show = max(16, int(round(fov / cell)) // 2 * 2)
    n = int(2 ** np.ceil(np.log2(2 * n_show)))                   # padded: no aliasing in view
    du = 1.0 / (n * cell * ARCSEC)
    iu, iv = np.rint(u / du).astype(np.int64), np.rint(v / du).astype(np.int64)
    grid = np.zeros(n * n, complex)
    for su, sv, val in ((iu, iv, vis), (-iu, -iv, np.conj(vis))):
        idx = (sv % n) * n + (su % n)
        grid += (np.bincount(idx, weight * val.real, n * n)
                 + 1j * np.bincount(idx, weight * val.imag, n * n))
    # V = S exp(+2 pi i (u l + v m)): the image is the transform with exp(-...)
    full = np.fft.fftshift(np.fft.fft2(grid.reshape(n, n))).real / (2.0 * weight.sum())
    c, h = n // 2, n_show // 2
    data = full[c - h:c + h, c - h:c + h]
    half = h * cell
    yy, xx = np.mgrid[-h:h, -h:h]
    far = np.hypot(xx, yy) * cell > max(3 * beam, 0.25 * half)
    rms = float(np.sqrt(np.mean(data[far] ** 2))) if far.any() else float("nan")
    # column index grows with l (east): east is drawn on the left
    return DirtyImage(data=data, extent=(-half, half, -half, half), rms=rms, beam_arcsec=beam)


def dirty_image(path, line_freq=None, velocity: Optional[Sequence[float]] = None,
                column="auto", sources=None, fov_arcsec=None, weighting="natural"):
    """Dirty image of a ViSta stack (a Zarr store or a Measurement Set).

    The stack is already centred on the sources, so the stacked emission is
    at the centre of the image.

    Parameters
    ----------
    path : str
        The stack.
    line_freq, velocity
        Average the channels between ``velocity = (lo, hi)`` km/s around the
        rest frequency ``line_freq`` (GHz, or a string with its unit): the
        map of a line.  Without them, all the channels.
    column : str
        ``"auto"``: ``CORRECTED_DATA`` (the continuum subtracted) when the
        stack has it, otherwise ``DATA``.
    sources : list of int, optional
        Only these datasets of the stack (0-based, in the order of the input
        list): the image of one galaxy, or of a subsample.
    fov_arcsec : float, optional
        Size of the image; default 30 beams.
    weighting : {'natural', 'democratic'}
        ``'natural'``: every visibility with its weight, the deepest datasets
        count most.  ``'democratic'``: every dataset the same total weight.
    """
    from .extract.config import parse_freq_ghz
    from .stackio import open_stack

    if weighting not in ("natural", "democratic"):
        raise ValueError(f"weighting must be 'natural' or 'democratic', not {weighting!r}")
    stack = open_stack(path)
    freqs = stack.dd_frequencies()
    wanted = stack.dd_ids() if sources is None else [int(s) for s in sources]
    us, vs, vis, wts = [], [], [], []
    col = None
    for dd in wanted:
        freq = np.asarray(freqs[dd], float)
        if velocity is not None:
            if line_freq is None:
                raise ValueError("dirty_image: velocity needs line_freq")
            nu0 = parse_freq_ghz(line_freq) * 1e9
            vel = C_KMS * (nu0 - freq) / nu0
            chans = (vel >= min(velocity)) & (vel <= max(velocity))
        else:
            chans = np.ones(len(freq), bool)
        if not chans.any():
            continue
        if col is None:
            col = column if column != "auto" else (
                "CORRECTED_DATA" if "CORRECTED_DATA" in stack.columns(dd) else "DATA")
        nu = float(freq[chans].mean())
        first = len(wts)
        for row0, nrow in stack.pieces(dd):
            d = stack.read(dd, row0, nrow, ["UVW", "WEIGHT", "FLAG", col])
            w = np.where(d["FLAG"][:, chans, :], 0.0, d["WEIGHT"][:, None, :])
            wsum = w.sum(axis=(1, 2))
            good = wsum > 0
            mean = np.zeros(len(wsum), complex)
            mean[good] = (w * d[col][:, chans, :]).sum(axis=(1, 2))[good] / wsum[good]
            us.append(d["UVW"][:, 0] * nu / C)
            vs.append(d["UVW"][:, 1] * nu / C)
            vis.append(mean)
            wts.append(wsum)
        if weighting == "democratic":
            total = sum(float(w.sum()) for w in wts[first:])
            if total > 0:
                wts[first:] = [w / total for w in wts[first:]]
    if not us:
        raise ValueError(f"dirty_image: no channel of {path} in the velocity range {velocity}")
    img = image_visibilities(np.concatenate(us), np.concatenate(vs), np.concatenate(vis),
                             np.concatenate(wts), fov_arcsec)
    img.label = (f"{velocity[0]:+.0f} to {velocity[1]:+.0f} km/s" if velocity is not None
                 else "all channels") + f", {col}"
    return img


def plot_image(image, ax=None, unit: Optional[str] = "Jy", title=None, vmax=None,
               colorbar=True, contours=True):
    """Draw a :class:`DirtyImage` (north up, east left), with the contours at
    3, 5, 7... times its rms (in the title; ``contours=False``: without) and
    the resolution in the corner.  Returns the axes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    f, ulabel = (1e3, "mJy") if unit and unit.lower() == "jy" else (1.0, unit or "data units")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.4, 5.4))
    lo, hi = image.extent[0], image.extent[1]
    top = vmax * f if vmax is not None else max(np.nanmax(image.data) * f, 5 * image.rms * f)
    im = ax.imshow(image.data * f, origin="lower", extent=(lo, hi, lo, hi), cmap="inferno",
                   vmin=-3 * image.rms * f, vmax=top, interpolation="nearest")
    if contours and np.isfinite(image.rms) and image.rms > 0:
        levels = [k * image.rms * f for k in (3, 5, 7, 10, 15, 20, 30)
                  if k * image.rms < np.nanmax(image.data)]
        if levels:
            ax.contour(image.data * f, levels=levels, colors="w", linewidths=0.8,
                       extent=(lo, hi, lo, hi), origin="lower")
    ax.set_xlim(hi, lo)                                  # east on the left
    ax.add_patch(Circle((hi - 0.12 * (hi - lo), lo + 0.12 * (hi - lo)), image.beam_arcsec / 2,
                        fc="none", ec="w", lw=1.2))
    ax.set_xlabel("east offset [arcsec]", fontsize=14)
    ax.set_ylabel("north offset [arcsec]", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.set_title((title + "\n" if title else "") + f"rms {image.rms * f:.2g} {ulabel}/beam",
                 fontsize=14)
    if colorbar:
        bar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        bar.set_label(f"{ulabel} / beam", fontsize=14)
        bar.ax.tick_params(labelsize=13)
    return ax


__all__ = ["DirtyImage", "dirty_image", "image_visibilities", "plot_image"]
