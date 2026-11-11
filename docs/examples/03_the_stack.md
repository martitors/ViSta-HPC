# 3. The stack

Two calls describe it:

* `ViSta(...)`: the input list and the name of what is written;
* `v.stack(...)`: the rest frequency of the line (`line_freq`, GHz, or a
  string as `"345.796GHz"`), the window that holds the line
  (`line_window=500`: from −500 to +500 km/s around it) and the band to keep
  (`velocity_range=900`: ±900 km/s).

`v.run()` then, for every dataset: brings the frequencies to the rest frame
of the galaxy, moves the phase centre onto the galaxy, puts the channels on
a common grid, and fits and subtracts the continuum on the channels outside
the line window. The result is one dataset, `stack.zarr`, with the
visibilities of all the galaxies.

```python
v = ViSta(truth["input_list"], output_name="stack")
v.stack(line_freq=LINE, line_window=500, velocity_range=900)
v.run()
```

```text
Stack      stack.zarr: 433,440 rows, 90 channels, 6 s
```

**What stacking does, in an image.** `vista.dirty_image` makes the image of
the channels between two velocities; `sources=[5]` keeps only one galaxy of
the stack (the sixth, of average depth). On its own it shows at most a hint,
no stronger than the noise peaks around it; the twelve together show the
line at the centre, where every galaxy was moved.

(These images are a quick look. The measurements below are made on the
visibilities, not on images.)

```python
one = vista.dirty_image("stack.zarr", line_freq=LINE, velocity=(-150, 150), sources=[5],
                        fov_arcsec=16)
all12 = vista.dirty_image("stack.zarr", line_freq=LINE, velocity=(-150, 150), fov_arcsec=16)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
vista.plot_image(one, axes[0], title="one galaxy, −150 to +150 km/s")
vista.plot_image(all12, axes[1], title="the stack of twelve, −150 to +150 km/s")
fig.tight_layout()
```

```{image} figures/03_01.png
:width: 100%
```
