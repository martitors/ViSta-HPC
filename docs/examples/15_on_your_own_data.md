# 15. On your own data

1. Write the input list, one line per dataset:
   `<ms_path> <redshift> <RA> <Dec> [FIELD_ID] [SPW_IDS] [NORM]`, with the
   position of the galaxy (hh:mm:ss.s ±dd:mm:ss.s).
2. `python examples/inspect_ms.py input_list.txt --line-freq 345.796GHz --all`
   reads the bands of the datasets and suggests `velocity_range`,
   `line_window` and `annuli`.
3. Stack and measure as in §3 and §4, with a window wide enough for the
   whole line.
4. Read the report (`print(r)`) and look at `r.plot("profile")`,
   `r.plot("uv")`, `r.plot("noise")`, `r.plot("image")`: the residuals and
   the imaginary part around zero, nothing left of the continuum, the model
   on the points.
5. Refine as in §5–§10, one option at a time, with `run(only="measure")`.
   `v.help()` lists every option with its value and what it can be.

The documentation, <https://vista-hpc.readthedocs.io>, explains every option.
