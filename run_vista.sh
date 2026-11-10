#!/bin/bash
# ============================================================================
# run_vista.sh — stack, subtract the continuum, compress and measure
#
# The command line of vista.ViSta: every option of the Python interface is a
# flag here (python -m vista --help lists them all).
#
#   ./run_vista.sh --input input_list.txt --line-freq 345.796 --line-window 500
#
#   two lines, each with its own window, the continuum also kept out between them:
#   ./run_vista.sh --input input_list.txt --line-freq 345.796 344.989 \
#                  --line-window 400 300 --exclude-between
#
#   the measurement again (seconds), with limits on the centre and the width:
#   ./run_vista.sh --input input_list.txt --line-freq 345.796 --only measure \
#                  --v0-range=-200:200 --fwhm-range 100:600 --verbose
#
#   the stack as a Measurement Set, for imaging:
#   ./run_vista.sh export stacked.zarr stacked.ms
#
# Everything written starts with --output-name (default "stacked"): the stack
# (stacked.zarr), the statistics, the results (stacked_results.json) and the
# figures (stacked_profile.png, _uv.png, _noise.png, stacked_continuum_*).
# ============================================================================
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$here${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m vista "$@"
