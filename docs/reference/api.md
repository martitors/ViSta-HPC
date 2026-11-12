# Python API

The whole method is driven by one object, {class}`vista.ViSta`: its
constructor takes the input, the output and how the run uses the machine;
`stack()` says what makes the stack (the lines, the band, the continuum);
`extract()` what is measured on it (without it, `run()` only writes the
stack); `help()` lists every option with its value, and `run()` runs them.
The options are listed in [Parameters](parameters).

```{eval-rst}
.. autoclass:: vista.ViSta
   :members: stack, extract, run, help, files

.. autoclass:: vista.RunResult
   :members: plot, report, as_dict, keys
   :special-members: __getitem__

.. autofunction:: vista.compare

.. autofunction:: vista.dirty_image

.. autofunction:: vista.plot_image
```

## Other functions

```{eval-rst}
.. autofunction:: vista.export_ms

.. autofunction:: vista.open_stack

.. autofunction:: vista.set_verbosity
```

## The results

```{eval-rst}
.. autoclass:: vista.extract.LineResult
   :members: save

.. autoclass:: vista.extract.ContinuumResult
   :members: save
```

## Figures

```{eval-rst}
.. autofunction:: vista.extract.plots_line.plot_profile

.. autofunction:: vista.extract.plots_line.plot_uv

.. autofunction:: vista.extract.plots_line.plot_line_result
```

## The steps underneath

`ViSta.run()` calls these; they are documented in their modules for whoever
needs a step on its own: {mod}`vista.pipeline` (the stacking),
{mod}`vista.companions` (the companions),
{mod}`vista.extract.contsub` (the continuum subtraction),
{mod}`vista.extract.statistics` (the compression), {mod}`vista.extract.measure`
(the lines), {mod}`vista.extract.stack` (the continuum measurement) and
{mod}`vista.resources` (how the cores and the memory are shared).
