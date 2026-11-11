# Examples

A worked example, **one step at a time**: simulated ALMA observations of
twelve different galaxies at z = 2–3, stacked and measured with ViSta. Every
step changes one thing, says why, and shows what happens, in the spectrum,
in the uv plane and in images, against the answer that the simulation knows.

These pages are the notebook `examples/ViSta_tutorial.ipynb`, run once, with
its results. To run it yourself:

- **On Google Colab**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martitors/ViSta-HPC/blob/extract_opt/examples/ViSta_tutorial.ipynb)
  and *Runtime → Run all* (about 10 minutes on the CPU). It clones the
  repository, builds the kernel and installs what it needs.
- **Locally**: Jupyter, with `VISTA_REPO` set to a built checkout.

| | |
|---|---|
| [0. Setup](00_setup) | clone, build, imports |
| [1. The simulator, and one galaxy](01_the_simulator_and_one_galaxy) | what is simulated; one galaxy without and with noise, image and spectrum |
| [2. A sample of twelve galaxies](02_a_sample_of_twelve_galaxies) | a heterogeneous sample, the input list, the expected answer |
| [3. The stack](03_the_stack) | `ViSta()`, `stack()`, `run()`; one galaxy and the stack in an image |
| [4. The measurement](04_the_measurement) | `extract()`; how to read the report, the spectrum and the uv figure |
| [5. Refining: the window](05_refining_the_window) | the area flux and its window |
| [6. The source model](06_the_source_model_the_wrong) | a point source where it is wrong; a known size |
| [7. The normalisations](07_the_normalisations) | the redshift and the primary beam, on and off |
| [8. The weighting](08_the_weighting) | natural and democratic |
| [9. The errors](09_the_errors) | the noise from the data and from the weights |
| [10. The continuum](10_the_continuum) | subtracted, left in the data, fitted with the line |
| [11. A line that is not a Gaussian](11_a_harder_case_a_line_that_is) | broad wings: the Gaussian and the area flux |
| [12. Sources that are not Gaussians](12_a_harder_case_sources_that) | rings: what the image and the uv plane show |
| [13. Two lines in the band](13_a_harder_case_two_lines_in) | one listed, then both, fitted together |
| [14. The command line](14_the_command_line) | the same with `python -m vista` |
| [15. On your own data](15_on_your_own_data) | the steps |

The pages are generated from an executed copy of the notebook with
`docs/_tools/notebook_to_pages.py`.
