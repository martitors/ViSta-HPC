# Examples

A worked example: a simulated sample of ALMA Band 3 observations of galaxies
at z = 2–3, whose CO(3–2) line no single galaxy detects, stacked and
measured with ViSta; then **one option at a time** changed, to show what
each one does, against the true values of the simulation. Every comparison
is one call, `vista.compare({...}, truth=...)`: the profiles at zero
spacing with their fits, the line flux per annulus with the source model,
the fluxes with their errors, and a table of the numbers.

These pages are the notebook `examples/ViSta_tutorial.ipynb`, run once, with
its results. To run it yourself:

- **On Google Colab**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martitors/ViSta-HPC/blob/extract_opt/examples/ViSta_tutorial.ipynb)
  and *Runtime → Run all* (about 10 minutes on the CPU). It clones the
  repository, builds the kernel and installs what it needs.
- **Locally**: Jupyter, with `VISTA_REPO` set to a built checkout.

Every parameter is in the code cells: change it and run again from that
cell.

| | |
|---|---|
| [0. Setup](00_setup) | clone, build, imports |
| [1. A simulated sample](01_a_simulated_sample) | twelve galaxies at z = 2–3 and what the stack should give |
| [2. Stack and measure](02_stack_and_measure) | the three calls, `run()`, and the figures of the measurement |
| [3. One option at a time](03_one_option_at_a_time) | the measurement again in seconds; every option of `extract()` |
| [4. Centre and width](04_centre_and_width_searched_or) | searched, known, or assumed wrong |
| [5. The source](05_the_source_point_or_extended) | point, Gaussian, a fixed size, a range of sizes |
| [6. The window of the area flux](06_the_window_of_the_area_flux) | the Gaussian and the area flux |
| [7. Weighting, redshift and primary beam](07_weighting_redshift_and) | natural or democratic, `z_norm`, `pb_corr` |
| [8. The errors](08_the_errors) | the noise from the data or from the weights, the channel correlation |
| [9. The continuum](09_the_continuum_subtracted) | subtracted, fitted with the line, or left in the data |
| [10. A line that is not Gaussian](10_a_line_that_is_not_gaussian) | broad wings: the Gaussian, the area, and `auto` |
| [11. Two lines in the band](11_two_lines_in_the_band) | one listed or both: blended lines fitted together |
| [12. The command line](12_the_command_line) | the same with `python -m vista` |
| [13. On your own data](13_on_your_own_data) | the steps |

The pages are generated from an executed copy of the notebook with
`docs/_tools/notebook_to_pages.py`.
