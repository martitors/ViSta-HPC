# Command line

**In one sentence.** `run_vista.sh` (or `python -m vista`) does from a
shell what {class}`vista.ViSta` does from Python: every option of
`ViSta()`, `stack()` and `extract()` is one flag, with the same default.

- Without `--extract` only the stack is written, as in Python.
- The package must be importable: run it from the repository, or set
  `PYTHONPATH`. After `pip install -e .` the same command is also called
  `vista`.
- At the end it prints what it did (the stack, the lines with `z_ref`, the
  continuum) and the files it wrote. `--help` lists every option; what each
  means is in [Parameters](reference/parameters).

## Examples

```bash
# everything: stack, subtract the continuum, compress, measure
./run_vista.sh --input input_list.txt --output-name co32 --verbose \
               --line-freq 345.79599 --line-window 500 --velocity-range 900 --extract

# only the stack (for imaging, or to measure later)
./run_vista.sh --input input_list.txt --output-name co32 --line-freq 345.79599 --line-window 500

# the measurement again (seconds), with limits on the centre and the width
./run_vista.sh --input input_list.txt --output-name co32 --line-freq 345.79599 --line-window 500 \
               --only measure --extract --v0-range=-200:200 --fwhm-range 100:600

# two lines, each with its own window; the continuum also kept out between them
./run_vista.sh --input input_list.txt --line-freq 345.79599 344.98857 \
               --line-window 350 350 --exclude-between --velocity-range 1250 --extract

# the continuum left in the data and fitted with the line
./run_vista.sh --input input_list.txt --line-freq 345.79599 --continuum fit --extract

# no line: the stack of the continuum, and its flux density and size
./run_vista.sh --input input_list.txt --output-name dust --extract

# only the continuum again, with the size measured on the line
./run_vista.sh --input input_list.txt --output-name co32 --line-freq 345.79599 --line-window 500 \
               --only measure --extract --measure continuum --continuum-model line

# the stack as a Measurement Set
./run_vista.sh export co32.zarr co32.ms
```

**Windows and ranges.** A number is a half-width (`500`: −500 to +500 km/s),
a pair is `lo:hi` (or `"lo hi"` in quotes). With several lines give one per
line, in the order of `--line-freq`, or one for all of them. A pair whose low
end is negative needs the `=` form or the quotes: `--v0-range=-200:200`,
`--v0-range "-200 200"`.

## Input, output, how it runs (`ViSta`)

| Option | Default | |
|---|---|---|
| `--input FILE` | | the [input list](getting_started/input_list) (required) |
| `--output-name NAME` | `stacked` | name and path of every file written ([Output files](extraction/outputs)) |
| `--output-format {zarr,ms}` | `zarr` | the stack: a Zarr store or a Measurement Set ([The stack on disk](stacking/combination)) |
| `--device {cpu,gpu}` | `cpu` | where the kernel runs |
| `--readers N`, `--writers N`, `--threads N` | from the cores | reader processes, writer threads (Zarr), kernel threads ([Performance](reference/performance)) |
| `--gpu-batch N` | from the memory | GPU: chunks per launch |
| `--scratch DIR` | | write the stack on this disk, then move it |
| `--verbose`, `--debug` | | progress and results; also the details and the time of every stage |
| `--log-file FILE` | | also write the log to this file |
| `--only STEP [STEP ...]` | all | `stack`, `continuum` (the subtraction again on the existing stack), `measure` |
| `--tag NAME` | | name of the measurement: its files are `<name>_<tag>_*` |

## What makes the stack (`stack`)

| Option | Default | |
|---|---|---|
| `--line-freq FREQ [FREQ ...]` | | rest frequency of the line, or of several lines: GHz, or with the unit attached (`345.796GHz`, `345796MHz`); the first is the line of the stack. Without it: no line, the stack of the continuum |
| `--line-window W [W ...]` | 600 | window of every line, km/s; lines whose windows overlap are fitted together |
| `--velocity-range KMS` | the widest band of one dataset | band of the stack around the first line, −KMS..+KMS ([Spectral grid](stacking/spectral_grid)) |
| `--nchan N` | | a number of channels instead |
| `--channel-width HZ` | coarsest rest-frame channel | common channel width |
| `--frame NAME` | `LSRK` | frame of the redshifts: the inputs are always brought to it ([Rest frame](stacking/restframing)) |
| `--data-column NAME` | `auto` | column read: `auto` prefers `CORRECTED_DATA` |
| `--continuum {subtract,fit,none}` | `subtract` | subtracted while stacking, fitted with the lines, or left as it is ([Continuum](stacking/continuum)) |
| `--continuum-order {0,1,bic}` | 0 | with `subtract`: constant, straight line, or chosen by the BIC |
| `--exclude-between` | | also leave out of the continuum fit the channels between the lines |
| `--companions FILE` | | other sources in the fields, `<ms_path> <RA> <Dec> [redshift]` per line: fitted with the target and subtracted ([Companions](stacking/companions)) |

## What is measured (`extract`)

The options below need `--extract` (without it the command stops and says so).

| Option | Default | |
|---|---|---|
| `--extract` | | measure the stack |
| `--annuli BMIN BMAX N` | 3 3000 18 | annuli of baseline length, kλ ([Compression](stacking/compression)) |
| `--weighting-scheme {natural,democratic}` | `natural` | [Weighting](extraction/weighting) |
| `--no-z-norm`, `--z-ref Z` | on, median | transport to a common redshift |
| `--no-pb-corr` | on | primary-beam correction |
| `--flux-norm`, `--flux-ref X` | off, median | divide by the last column of the input list |
| `--v0-range LO:HI [...]` | the window | allowed centres, km/s; `lo = hi` fixes it ([Measuring the line](extraction/measure_line)) |
| `--fwhm-range LO:HI [...]` | two channels to the window | allowed FWHM; `lo = hi` fixes it |
| `--model {auto,point,gauss,continuum,ARCSEC}` | `auto` | the source model of the lines, or a fixed size; `continuum`: the one measured on the continuum |
| `--continuum-model {auto,point,gauss,line,ARCSEC}` | `auto` | the source model of the continuum; `line`: the size measured on the line |
| `--measure {line,continuum}` | both | measure only one of the two |
| `--model-prior ARCSEC DEX` | | soft tie of the size |
| `--theta-range MIN:MAX` | 0:15 | allowed size of the Gaussian, arcsec; a minimum above 0 leaves the point source out |
| `--flux-measure {auto,gauss,area}` | `auto` | the flux quoted (both are always given) |
| `--shape-nsigma X` | 2 | threshold of the shape test of `auto` |
| `--window W [W ...]` | the line windows | window of the area flux |
| `--fit-span KMS` | 1500 | channels farther than this from every line are not fitted; `none`: every channel |
| `--min-sources-per-annulus N` | 2 | annuli with fewer sources are left out |
| `--n-bootstrap N`, `--n-null N`, `--seed N` | 500, 1000, 42 | bootstrap, null realisations, random seed |
| `--figures-unit NAME`, `--no-figures` | `Jy` | unit of the data on the figures; no figures |
