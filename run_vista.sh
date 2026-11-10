#!/bin/bash
# ============================================================================
# run_vista.sh — stack and, with --extract, measure
#
# The command line of vista.ViSta: every option of ViSta(), stack() and
# extract() is a flag here (python -m vista --help lists them all).  Without
# --extract only the stack is written.
#
#   the stack (continuum subtracted), its statistics and the measurement:
#   ./run_vista.sh --input input_list.txt --line-freq 345.796GHz --line-window 500 --extract
#
#   only the stack (to image it, or to measure it later):
#   ./run_vista.sh --input input_list.txt --line-freq 345.796GHz --line-window 500
#
#   two lines, each with its own window, the continuum also kept out between them
#   (windows that overlap are fitted together, the others each on its own):
#   ./run_vista.sh --input input_list.txt --line-freq 345.796GHz 344.989GHz \
#                  --line-window 400 300 --exclude-between --extract
#
#   the measurement again (seconds), with limits on the centre and the width:
#   ./run_vista.sh --input input_list.txt --line-freq 345.796GHz --line-window 500 \
#                  --only measure --extract --v0-range=-200:200 --fwhm-range 100:600
#
#   the stack as a Measurement Set, for imaging:
#   ./run_vista.sh export stacked.zarr stacked.ms
#
# Everything written starts with --output-name (default "stacked"): the stack
# (stacked.zarr), the statistics, the results (stacked_results.json) and the
# figures (stacked_profile.png, _uv.png, _noise.png, stacked_continuum_*);
# with --tag T the results and figures are stacked_T_*.
# ============================================================================
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m vista "$@"
