# Spectral grid

In the rest frame every dataset has its own channels: different widths,
different edges, different bandwidths. They are rebinned onto one grid, built
in three steps.

## The channel width

A channel of observed width `Δν_obs` is `Δν_obs (1 + z)` wide in the rest
frame. The common width is the **largest** of these over the sample: the
finest grid every dataset can fill, since resampling a coarse spectrum onto
finer channels would only correlate the noise between them.
`channel_width` (`--channel-width`) sets it; a value finer than that floor is
refused, with an error naming the dataset that sets it.

## The number of channels

`velocity_range` (`--velocity-range`) gives the bandwidth as a velocity
range around the line — a scalar `v` means `±v`, a pair `(lo, hi)` an
asymmetric range — and the number of channels follows from the width.
`nchan` (`--nchan`) sets it directly. Without either, ViSta takes the
widest coverage any single dataset can provide. The count is rounded down to
an even number.

## The placement

The window is centred on the rest frequency of the line, the same for every
dataset. Where a dataset does not cover the whole window, its window is slid
to the edge of its band rather than cut, and the log gives the shift. Every
output spectral window then has the same width and the same number of
channels, possibly a different start; the channel edges are always on the
same grid, so the channels of different datasets line up exactly. The
channels a dataset does not cover are flagged: the stack records, channel by
channel, which datasets contribute, and the coverage thins out in the wings.

Ask for a range the sample covers: with a window wider than the narrowest
dataset, that dataset's window slides off-centre and its line sits near an
edge.

## The rebinning

Each output channel is the weighted sum of the input channels it overlaps,
each weighted by the fraction of the overlap, with the width of every input
channel from its neighbours (as CASA `mstransform`). Since an output channel
collects the signal of `R = Δν_new / Δν_old` input channels, the weights are
multiplied by `R` and `SIGMA` divided by `√R`: the noise of a wider channel
is lower, and the natural weighting of datasets with different channel widths
stays right. These products are computed in float64.

The rebinning, like any channel averaging, correlates the noise of adjacent
channels when the edges do not match, on top of the correlation already in
the data (Hanning smoothing: +0.67 at lag 1). The measurement measures this
correlation from the data and uses it in every error ([The noise](../extraction/measure_line.md#the-noise)).
