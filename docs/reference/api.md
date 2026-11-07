# Python API

The public interface: the stacking (`vista`), its configuration and the
measurement (`vista.extract`). Every function is documented in its
docstring; this page collects them.

## Stacking

```{eval-rst}
.. autoclass:: vista.ViSta
   :members: run

.. autofunction:: vista.export_ms

.. autofunction:: vista.open_stack

.. autofunction:: vista.set_verbosity
```

## Configuration

```{eval-rst}
.. autoclass:: vista.extract.LineConfig
   :members:

.. autoclass:: vista.extract.ContinuumConfig

.. autoclass:: vista.extract.BinningConfig

.. autoclass:: vista.extract.WeightingConfig

.. autoclass:: vista.extract.LineFitConfig

.. autoclass:: vista.extract.BootstrapConfig

.. autoclass:: vista.extract.ProfileConfig

.. autoclass:: vista.extract.UVFitConfig
```

## Continuum subtraction and compression

```{eval-rst}
.. autofunction:: vista.extract.subtract_continuum

.. autofunction:: vista.extract.compress_visibilities

.. autoclass:: vista.extract.SufficientStatistics
   :members: load
```

## Measurement

```{eval-rst}
.. autofunction:: vista.extract.measure_line

.. autoclass:: vista.extract.LineResult
   :members: save

.. autofunction:: vista.extract.extract_flux

.. autoclass:: vista.extract.ExtractionResult

.. autofunction:: vista.extract.alpha_factor

.. autofunction:: vista.extract.read_input_list
```

## Figures

```{eval-rst}
.. autofunction:: vista.extract.plots_line.plot_line_result

.. autofunction:: vista.extract.plots.plot_all
```
