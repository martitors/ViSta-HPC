# Changelog

## v3.0.0

**One object, three calls.** `ViSta(input_list, output_name=...)` holds the
input, the output and how the run uses the machine; `.stack(...)` says what
makes the stack; `.extract(...)` what is measured on it; `.run()` runs them.
Only the input list and the rest frequency are needed, `.help()` lists every
option, and a value that does not make sense is refused when it is set,
before any data is read. The command line (`run_vista.sh`,
`python -m vista`) has the same options, one flag each.

**Outputs.** Every file starts with `output_name`: the stack (Zarr by
default, or a Measurement Set), its statistics, the results as JSON, the
figures.

**The line.** Searched in centre and width inside `v0_range` and
`fwhm_range`, fitted on all channels and baseline lengths at once. Two
fluxes, the area of the Gaussian fit and the area under the spectrum in
the window, and the one to quote chosen from their agreement. Several
lines in one run: those whose windows overlap are fitted together.

**The continuum.** Subtracted by default, and measured on its own. Its
source model never contradicts the line: its own Gaussian if the data ask
for it (Δχ² ≥ 9), otherwise the size of the line if the line is extended,
otherwise a point. `v.stack()` without `line_freq` stacks the continuum
alone, of datasets that share no line.

**Companions.** `v.stack(companions="companions.txt")` removes the other
sources in the fields of the targets: fitted together with the target, as
point sources at their positions, and subtracted before the continuum
subtraction; their continuum, and their line when they have a redshift
([Companions](../stacking/companions)).

**Errors.** For the line and for the continuum, the larger of the thermal
error (from the scatter of the visibilities) and the bootstrap over the
sources; they hold from 5 sources on (see
[Validation](../extraction/validation), written by
`benchmarks/validation.py`).

**Results.** `print(result)` says in words what was measured and why each
choice was made; `result["flux"]`, `result["size"]`, `result["continuum"]`,
... give every quantity with its value, error and unit; `result.plot()`
draws the figures (`"line"`, `"line_uv"`, `"continuum"`,
`"continuum_uv"`, `"noise"`, `"image"`); `vista.compare` puts several
measurements side by side.

**Resources.** Readers, writers and kernel threads are taken from the cores
the process may use, and the GPU batch from the free memory.

**GPU.** The GPU kernel computes the phase shift with the same exact
geometry as the CPU kernel (it used a small-angle formula, wrong by about
1° of phase for a source 20″ from the phase centre on a 1 km baseline at
100 GHz). `benchmarks/gpu_check.py` compares the two kernels and a whole
stack on the two devices.

**Documentation and examples.** Every page opens with what it is about in
one sentence, then what is done, why, and what to check. The example
notebook is divided in situations, each on its own simulation.

## Earlier versions

- **v2.5**: the spectral frame of the redshifts (LSRK) by default; stacking,
  continuum subtraction and compression in one pass; the noise measured from
  the data; this documentation.
- **v2.4**: the stack as a Zarr store written in parallel; every computation
  in float64.
- **v2.3**: a faster and lighter stacking engine; the log.
- **v2.1**: thread-safe reading, chunks sized to the memory, only the
  channels needed read from disk, `CORRECTED_DATA` preferred, calibrator
  scans excluded.
