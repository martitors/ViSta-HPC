# 16. On your own data

1. Write the input list, one line per dataset:
   `<ms_path> <redshift> <RA> <Dec> [FIELD_ID] [SPW_IDS] [NORM]`, with the
   position of the galaxy (hh:mm:ss.s ±dd:mm:ss.s).
2. `python examples/inspect_ms.py input_list.txt --line-freq 345.796GHz --all`
   reads the bands of the datasets and suggests `velocity_range`,
   `line_window` and `annuli`.
3. Stack and measure as in chapter 2, with a window wide enough for the
   whole line.
4. Read the report (`print(r)`) and look at the plots (`r.plot()` draws the
   line and the continuum, spectrum and baselines; `r.plot("noise")`,
   `r.plot("image")`): residuals around zero, nothing left of the
   continuum, the model on the points, the two sizes in agreement.
5. Other sources in the fields of your galaxies: list them in a file and
   give it to `v.stack(companions=...)`, as in chapter 14.
6. Change one option at a time, with `run(only="measure")`, as in chapters
   3 to 8. `v.help()` lists every option with its value and what it can be.

The documentation, <https://vista-hpc.readthedocs.io>, explains every option.
