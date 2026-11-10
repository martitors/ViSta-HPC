# Examples

A complete worked example: a simulated sample of ALMA Band 3 observations of
galaxies at z = 2–3, whose CO(3–2) line is too faint to be measured in any of
them, stacked and measured with ViSta. Every dataset is simulated twice,
**with and without noise**, so every step is compared with the true signal
(green, dashed in the figures).

These pages are the notebook `examples/ViSta_tutorial.ipynb`, run once, with
its results. To run it yourself:

- **On Google Colab**: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martitors/ViSta-HPC/blob/extract_opt/examples/ViSta_tutorial.ipynb)
  and *Runtime → Run all* (about 15 minutes on the CPU). It clones the
  repository, builds the kernel and installs what it needs.
- **Locally**: Jupyter, with `VISTA_REPO` set to a built checkout, and
  `pip install casatasks casatools` for the images (section 2 and 9 only).

Every parameter is in the code cells: change it and run again from that
cell.

| | |
|---|---|
| [0. Setup](00_setup) | clone, build, imports |
| [1. The simulated sample](01_the_simulated_sample) | twelve galaxies at z = 2–3, with and without noise, and the true values of the stack |
| [2. One galaxy, without and with noise](02_one_galaxy_without_and_with) | the uv plane (mean over the baselines, flux per annulus) and the image (CLEANed cube: peak, aperture), against the truth |
| [3. Stacking](03_stacking) | stacking, continuum subtraction and compression in one pass; a first look |
| [4. Measuring the line at a known width](04_measuring_the_stacked_line) | centre and width fixed: profile, the two fluxes, flux per annulus, noise, against the truth |
| [5. Searching for the line](05_searching_for_the_line) | centre and width free within limits, significance |
| [6. What each option changes](06_what_each_option_changes) | size model, size prior, noise, weighting, rescaling, primary beam, assumed width |
| [7. The continuum](07_the_continuum) | its flux and size |
| [8. Redoing the continuum subtraction](08_redoing_the_continuum) | another window and order, without stacking again |
| [9. The stack in the image](09_the_stack_in_the_image) | the stack imaged with `tclean`, without and with noise |
| [10. Command line and your own data](10_from_the_command_line_and_on) | `run_vista.sh`, and the steps on real data |

The pages are generated from an executed copy of the notebook with
`docs/_tools/notebook_to_pages.py`.
