# Examples

Worked examples, **one situation at a time**: simulated ALMA observations
stacked and measured with ViSta. Every chapter is one situation, on a
simulation made for it, so that the right answer is known; every step does
one thing, and the text says what and why.

These pages are the notebook `examples/ViSta_tutorial.ipynb`, run once, with
its results. To run it yourself:

- **On Google Colab**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martitors/ViSta-HPC/blob/extract_opt/examples/ViSta_tutorial.ipynb)
  and *Runtime → Run all* (about 20 minutes on the CPU). It clones the
  repository, builds the kernel and installs what it needs.
- **Locally**: Jupyter, with `VISTA_REPO` set to a built checkout.

| | |
|---|---|
| [0. Setup](00_setup) | clone, build, imports |
| [1. One galaxy](01_one_galaxy) | what is simulated; one galaxy without and with noise, image and spectrum |
| [2. The basic case](02_the_basic_case_the_same) | the same galaxy at twelve redshifts: the stack, the line, the continuum, the numbers |
| [3. The window, the normalisations, the noise](03_the_window_the) | the flux in the window, a narrower window, the redshift and primary-beam corrections, the noise |
| [4. Not subtracting the continuum](04_not_subtracting_the) | fitted with the line, or left in the data |
| [5. A weak continuum under a strong line](05_a_weak_continuum_under_a) | the continuum takes the size of the line |
| [6. A weak line on a strong continuum](06_a_weak_line_on_a_strong) | the line takes the size of the continuum |
| [7. A few galaxies that count much more](07_a_few_galaxies_that_count) | lensed galaxies: natural and democratic weighting, the flux normalisation |
| [8. Points and extended sources](08_points_and_extended_sources) | which source model, what the wrong one does, and what a mixed sample does |
| [9. Two lines far apart](09_two_lines_far_apart) | measured each on its own |
| [10. Two lines that blend](10_two_lines_that_blend) | fitted together |
| [11. A line that is not a Gaussian](11_a_line_that_is_not_a) | broad wings: the flux in the window |
| [12. A source that is not a Gaussian](12_a_source_that_is_not_a) | rings: what the image and the baselines show |
| [13. The continuum alone](13_the_continuum_alone) | a stack without any line |
| [14. A companion in the field](14_a_companion_in_the_field) | a second source next to a galaxy: what it does, how it is removed |
| [15. The command line](15_the_command_line) | the same with `python -m vista` |
| [16. On your own data](16_on_your_own_data) | the steps |

The pages are generated from an executed copy of the notebook with
`docs/_tools/notebook_to_pages.py`.
