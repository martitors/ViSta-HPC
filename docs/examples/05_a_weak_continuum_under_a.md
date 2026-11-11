# 5. A weak continuum under a strong line

This is the sample of chapter 2: the line is well detected and shows that
the source is extended; the continuum is detected, but too weakly to
measure a size.

`continuum_model` says which source model the continuum gets, and
`measure="continuum"` repeats **only** the continuum.

## 5.1 The continuum left to itself

`continuum_model="gauss"` fits its own Gaussian. The points are too noisy to
tell a Gaussian from a point (the two curves fit equally well): the size
that comes out is not a measurement, and the flux depends on it.

```python
v.extract(continuum_model="gauss", measure="continuum")
own = v.run(only="measure", tag="own")
own.plot("continuum_uv")
```

```{image} figures/05_01.png
:width: 100%
```

## 5.2 As a point

`continuum_model="point"`: a smaller error, but the flux is too low,
because the source is not a point.

```python
v.extract(continuum_model="point", measure="continuum")
as_point = v.run(only="measure", tag="as_point")
as_point.plot("continuum_uv")
```

```{image} figures/05_02.png
:width: 100%
```

## 5.3 With the size of the line

The default (`continuum_model="auto"`) does this by itself when the
continuum cannot measure its size and the line can; `continuum_model="line"`
asks for it. It assumes that gas and dust have the same extent: the report
says so, and gives the two sizes side by side. The three choices give three
different fluxes:

```python
print("its own Gaussian:        ", flux(own, "continuum"))
print("a point:                 ", flux(as_point, "continuum"))
print("the size of the line:    ", flux(r, "continuum"))
print("expected:                ", f"{expected['continuum'] * 1e3:.3f} mJy")
```

```text
its own Gaussian:         0.180 ± 0.029 mJy
a point:                  0.166 ± 0.025 mJy
the size of the line:     0.204 ± 0.032 mJy
expected:                 0.200 mJy
```
