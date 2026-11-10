# 13. On your own data

1. Write the input list, one line per dataset:
   `<ms_path> <redshift> <RA> <Dec> [FIELD_ID] [SPW_IDS] [NORM]`, with the
   position of the galaxy (hh:mm:ss.s ±dd:mm:ss.s).
2. `python examples/inspect_ms.py input_list.txt --line-freq 345.796GHz --all`
   reads the bands of the datasets and suggests `velocity_range`,
   `line_window` and `annuli`.
3. `ViSta(...)`, `v.stack(line_freq=..., line_window=...)` with a window wide
   enough for the whole line (one per line if the band holds more than one),
   `v.extract()`, `r = v.run()`, `r.plot()`.
4. Check `r.plot("noise")`, the continuum outside the window (on the
   profile), the significance, the size test and the two fluxes.
5. Change the options as above, with `run(only="measure")`, and compare the
   results with `vista.compare`.

The documentation, <https://vista-hpc.readthedocs.io>, explains every option.
