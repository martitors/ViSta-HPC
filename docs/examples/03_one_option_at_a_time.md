# 3. One option at a time

The stack and its statistics are written: every measurement below reads only
the statistics and takes seconds. `measure(...)` sets the options of
`extract()` and measures again; everything else stays as in §2. `v.help()`
lists every option, with its value now and what it can be.

```python
def measure(**options):
    """The measurement again with these options of extract(); the stack is not touched."""
    v.extract(**options)
    return v.run(only="measure", tag="variant")


v.help("extract")
```

```text
v.extract(...)  —  what is measured; without this call run() only writes the stack
  option                   now            can be                               what it does
  annuli                   (3, 3000, 18)  (kλ min, kλ max, n)                  annuli of baseline length,
                                                                               logarithmic
  weighting_scheme         'natural'      'natural' | 'democratic'             weights of the data (deepest
                                                                               count most) | every source the
                                                                               same
  z_norm                   True           True | False                         bring every source to the
                                                                               redshift z_ref
  z_ref                    None           float | None                         that redshift; None: the median
                                                                               of the sample (reported)
  pb_corr                  True           True | False                         correct every source for the
                                                                               primary beam
  flux_norm                False          True | False                         divide every source by the NORM
                                                                               column of the list
  flux_ref                 None           float | None                         reference of that
                                                                               normalisation; None: the median
                                                                               (reported)
  v0_range                 None           (lo, hi) | list | None               km/s, allowed centre of every
                                                                               line; lo = hi fixes it; None:
                                                                               in the window
  fwhm_range               None           (lo, hi) | list | None               km/s, allowed FWHM; lo = hi
                                                                               fixes it; None: two channels to
                                                                               the window
  model                    'auto'         'auto' | 'point' | 'gauss' | arcsec  the source: Gaussian if the
                                                                               data prefer it | point |
                                                                               Gaussian | Gaussian of this
                                                                               FWHM
  model_prior              None           (arcsec, dex) | None                 soft tie of the size of the
                                                                               Gaussian
  theta_range              (0, 15)        (min, max)                           arcsec, allowed size of the
                                                                               Gaussian (lines and continuum);
                                                                               min > 0 leaves the point source
                                                                               out
  flux_measure             'auto'         'auto' | 'gauss' | 'area'            flux quoted (both are given):
                                                                               Gaussian unless not Gaussian |
                                                                               Gaussian | sum over the window
  shape_nsigma             2              float                                threshold of that test: area
                                                                               and Gaussian differ by more
                                                                               than this many sigma
  window                   None           W | (lo, hi) | list | None           window of the area flux; None:
                                                                               the line windows
  fit_span                 1500           km/s | None                          channels farther than this from
                                                                               every line are not fitted;
                                                                               None: every channel
  min_sources_per_annulus  2              int                                  annuli with fewer sources are
                                                                               left out
  n_bootstrap              500            int                                  resamplings of the sources, for
                                                                               the errors (0: none)
  n_null                   1000           int                                  searches on noise, for the
                                                                               significance (0: Bonferroni
                                                                               bound)
  seed                     42             int                                  seed of the random numbers of
                                                                               bootstrap and null searches:
                                                                               the same seed gives the same
                                                                               numbers
  figures_draw             True           True | False                         draw the figures (profile, uv,
                                                                               noise, continuum)
  figures_unit             'Jy'           str                                  unit of the data, for the
                                                                               labels
  (advanced)                              keywords                             noise, channel_correlation,
                                                                               n_null_phase, residual_nsigma,
                                                                               H0, Om0

help(ViSta), help(v.stack), help(v.extract): the full description.
```
