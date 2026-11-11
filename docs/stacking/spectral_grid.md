# Spectral grid

**In one sentence.** In the rest frame every dataset has its own channels
(different widths, edges and bandwidths); they are all rebinned onto one
grid, so that channel 10 of one dataset is channel 10 of every other.

The grid is decided by three things: the width of a channel, how many
channels, and where they are put.

## The channel width

**What.** The common width is the **largest** rest-frame channel of the
sample. A channel of observed width `Δν_obs` is `Δν_obs (1 + z)` wide in
the rest frame.

**Why the largest.** It is the finest grid that every dataset can fill.
Resampling a coarse spectrum onto finer channels adds no information, it
only correlates the noise between them.

**Your choice.** `channel_width` (`--channel-width`) sets a coarser one. A
finer one is refused, with an error naming the dataset that sets the limit.

## The number of channels

| You give | The grid |
|---|---|
| `velocity_range=900` | from −900 to +900 km/s around the line |
| `velocity_range=(-700, 2200)` | that range |
| `nchan=200` | that many channels |
| nothing | the widest coverage any single dataset has |

The number is rounded down to an even one.

## Where the channels are

**Centred on the line.** The window is centred on the rest frequency of the
line, the same for every dataset. The channel edges are always on the same
grid, so the channels of different datasets line up exactly.

**When a dataset does not cover it all.** Its window is slid to the edge of
its band rather than cut, and the log gives the shift. The channels it does
not cover are flagged. The stack records, channel by channel, which
datasets contribute: the coverage thins out towards the edges (the panel
*sources* under the stacked spectrum).

**What to watch.** Ask for a range that the sample covers. With a window
wider than the narrowest dataset, the window of that dataset slides
off-centre, and its line sits near an edge.

## The rebinning

**How.** Each output channel is the weighted sum of the input channels it
overlaps, each weighted by the fraction of the overlap (as CASA
`mstransform`).

**The weights.** An output channel collects the signal of
`R = Δν_new / Δν_old` input channels. The weights are multiplied by `R`
(and `SIGMA` divided by `√R`): the noise of a wider channel is lower. So
datasets with different channel widths keep the right relative weights.

**A side effect.** Rebinning correlates the noise of adjacent channels when
the edges do not match, on top of the correlation already in the data
(Hanning smoothing: +0.67 between neighbours). The measurement measures
this correlation from the data and uses it in every error
([The noise](../extraction/measure_line.md#the-noise)).
