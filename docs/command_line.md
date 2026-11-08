# Command line

`run_vista.sh` runs the whole workflow, from the input list to the measured
line, with the same steps as the Python API:

1. **stack** the datasets (Zarr store, or Measurement Set);
2. **subtract the continuum**, dataset by dataset (optional);
3. **compress** the stack into statistics;
4. **measure** the line (and the continuum) on the statistics.

When the stack is built in the same run, steps 2 and 3 are done while it is
written, in one pass. Every step can be run on its own, so a refined step
does not force the others to be repeated: steps 1–3 touch the stack, step 4
works on a few MB and is cheap to repeat. The package must be importable
(run it from the repository, or set `PYTHONPATH`).

```bash
# everything, the continuum subtracted first, the line searched for
./run_vista.sh --input input_list.txt --rest-freq 345.79599 \
               --velocity-range 900 --v-window "-500 500" --contsub before \
               --method joint --mode search --unit Jy --out-dir co32

# the width known: a matched filter
./run_vista.sh --input input_list.txt --rest-freq 345.79599 --out-dir co32 \
               --only fit --contsub before --method joint --mode detect --fwhm 300

# measure again, another weighting, nothing else
./run_vista.sh --input input_list.txt --rest-freq 345.79599 --out-dir co32 \
               --only fit --contsub before --weighting democratic

# redo the continuum subtraction with another window, without stacking again
./run_vista.sh --input input_list.txt --rest-freq 345.79599 --out-dir co32 \
               --skip-stack --contsub before --exclude-v "400 700"
```

`./run_vista.sh --help` prints all the options. They are grouped below; the
meaning of each is on the page of its step.

## Inputs and outputs

| Option | |
|---|---|
| `--input FILE` | the [input list](getting_started/input_list) (required) |
| `--rest-freq GHZ` | rest frequency of the line (required) |
| `--tag NAME` | name of the run (default: from the rest frequency, e.g. `line345796`) |
| `--out-dir DIR` | where everything is written (default `.`) |

## Which steps

| Option | |
|---|---|
| `--contsub {before,after}` | the [continuum](stacking/continuum): subtract it, then fit line and continuum separately (`before`), or fit it with the line (`after`). Without it nothing is subtracted and no continuum is fitted |
| `--skip-stack`, `--skip-contsub`, `--skip-compress`, `--skip-fit` | skip a step whose result exists |
| `--only {stack,contsub,compress,fit}` | run one step |
| `--two-pass` | stack first, then subtract and compress reading the stack back (default: during the stacking) |

## Stacking

| Option | Default | |
|---|---|---|
| `--velocity-range KMS` | 2000 | half-width of the output band ([Spectral grid](stacking/spectral_grid)) |
| `--nchan N` | | number of output channels instead |
| `--width HZ` | coarsest rest-frame channel | common channel width |
| `--frame {LSRK,BARY}` | `LSRK` | frame of the redshifts: the inputs are always brought to it ([Rest frame](stacking/restframing)) |
| `--in-column NAME` | `auto` | column read: `auto` prefers `CORRECTED_DATA` |
| `--format {zarr,ms}` | `zarr` | format of the stack ([The stack on disk](stacking/combination)) |
| `--export-ms` | | also write the stack as a Measurement Set |
| `--chunk-rows N` | 5000 | rows per chunk |
| `--readers N`, `--threads N`, `--writers N`, `--io-threads N` | 2, all cores, 4, 4 | parallelism |
| `--device {cpu,gpu}`, `--gpu-batch N` | `cpu`, 8 | the GPU |
| `--scratch DIR` | `$TMPDIR` | build the stack there, then move it |
| `--timing`, `--debug` | | timing of the stages; detailed log |

## Continuum subtraction and compression

| Option | Default | |
|---|---|---|
| `--cont-order {0,1,-1}` | 0 | constant, slope, or BIC |
| `--exclude LO HI` | the line window | asymmetric window left out of the continuum fit |
| `--bins "MIN MAX N"` | `"3 3000 18"` | annuli of baseline length, kλ ([Compression](stacking/compression)) |
| `--max-amplitude X` | | discard visibilities above `\|V\| = X` |
| `--chunk-mb X` | 512 | read chunk size |

## The line

| Option | Default | |
|---|---|---|
| `--v-window "LO HI"` | `"-600 600"` | velocity window of the line |
| `--second-rest-freq GHZ`, `--second-v-window "LO HI"` | | a second line in the band |
| `--exclude-v "LO HI ..."` | | more intervals kept out of the line-free channels |

## Weighting and normalisation

| Option | Default | |
|---|---|---|
| `--weighting {natural,democratic}` | `natural` | ([Weighting](extraction/weighting)) |
| `--norm-z` / `--no-norm-z`, `--z-ref Z` | on, median | redshift rescaling |
| `--norm-flux` / `--no-norm-flux`, `--norm-ref X` | off, median | flux normalisation by the `NORM` column |
| `--already-normalised` | | the data already carry the redshift factor |
| `--match-tol-arcsec X` | 2.0 | same-source tolerance |

## The measurement

| Option | Default | |
|---|---|---|
| `--method {joint,template,window}` | `joint` | `joint`: [`measure_line`](extraction/measure_line); `template`, `window`: [the per-annulus chain](extraction/extract_flux) |
| `--mode {search,detect}` | `search` | joint: free centre and width, or fixed (`--v0`, `--fwhm`) |
| `--v0 KMS`, `--fwhm KMS` | 0, — | joint: centre and width (`--fwhm` required in `detect`) |
| `--size {auto,point,gauss}` | `auto` | joint: the source size |
| `--n-null N` | 1000 | joint: realisations of the search on noise |
| `--theta-prior-from FILE.json`, `--theta-prior-sigma DEX` | —, 0.15 | a size prior from a previous fit, e.g. the continuum |
| `--theta-fixed ARCSEC`, `--theta-max ARCSEC` | —, 15 | fix or bound the size |
| `--model {gauss,point,gauss2}` | `gauss` | per-annulus chain: source model |
| `--b-max-kl X`, `--min-src-bin N` | —, 2 | annuli left out |
| `--fit-span KMS`, `--shape-tie` | 1500, `free` | per-annulus chain: shape fit |
| `--nboot N`, `--seed N` | 500, 42 | bootstrap |

## Figures

| Option | |
|---|---|
| `--unit NAME` | amplitude unit on the axes, e.g. `Jy` |
| `--v-limits "LO HI"` | velocity range of the spectrum |
| `--plot-rebin N` | joint: channels per bin in the profile |
| `--no-plot` | no figures |

At the end the script prints the main results and the checks worth reading
before believing them (imaginary part, size at its bound, χ²).

## One step at a time

The steps are also available as subcommands of the package:

```bash
python -m vista.extract contsub   stack.zarr     --rest-freq 345.79599
python -m vista.extract compress  stack.zarr     --input input_list.txt \
        --rest-freq 345.79599 --out-line line_stats.npy
python -m vista.extract flux      line_stats.npy --input input_list.txt \
        --rest-freq 345.79599 --weighting democratic --out co32_democratic
python -m vista.extract export-ms stack.zarr     stack.ms
```

`python -m vista.extract <subcommand> --help` lists their options.
