# 4. Not subtracting the continuum

`continuum` in `v.stack` says what to do with the continuum:

* `"subtract"` (default): taken out of every galaxy before stacking, and
  measured on its own. Line and continuum can have different sizes. The
  choice for most cases.
* `"fit"`: left in, and fitted **together with the line** as a constant
  level. For when few channels are free of the line (a broad line, a narrow
  band). It uses one size for both.
* `"none"`: nothing is done. Only when the continuum is negligible.

The stack keeps the data before the subtraction, so no new stack is needed:
only the measurement is repeated.

## 4.1 Fitted with the line

The spectrum now shows the line **on top of** the continuum (blue dashed).

```python
v.stack(line_freq=LINE, line_window=500, velocity_range=900, continuum="fit")
v.extract()
fitted = v.run(only="measure", tag="fitted")
fitted.plot("line")
```

```{image} figures/04_01.png
:width: 100%
```

## 4.2 Left in the data

Here there is a continuum, and nothing accounts for it: the fit takes it
for part of the line and the flux is too high. ViSta notices: it measures
the level outside the window (blue dashed) and the report warns.

```python
v.stack(line_freq=LINE, line_window=500, velocity_range=900, continuum="none")
left = v.run(only="measure", tag="left")
left.plot("line")
```

```{image} figures/04_02.png
:width: 100%
```

```python
print("subtracted (default): ", flux(r))
print("fitted with the line: ", flux(fitted))
print("left in the data:     ", flux(left))
print("expected:             ", f"{expected['flux'][0]:.3f} Jy km/s")
```

```text
subtracted (default):  0.431 ± 0.052 Jy km/s
fitted with the line:  0.418 ± 0.044 Jy km/s
left in the data:      0.493 ± 0.042 Jy km/s
expected:              0.400 Jy km/s
```

```python
v.stack(line_freq=LINE, line_window=500, velocity_range=900)     # back to the subtraction
```

```text
ViSta('sample/input_list.txt': 12 datasets, Lines(345.796 GHz [-500, +500] km/s), → stack.zarr, measured)
```
