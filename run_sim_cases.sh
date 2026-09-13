#!/bin/bash
# ============================================================================
# run_sim_cases.sh — stack once, then run the continuum cases side by side
#
#   run_sim/
#     stacked_simgauss.ms          the stack, built once
#     case_none/                   no continuum fitted, line only
#     case_joint/                  line and continuum fitted together
#     case_subtract/               continuum subtracted first, two fits
#
# The Measurement Set is built once and linked into each case directory, so
# the expensive step is not repeated.  The two cases that read DATA also share
# the compressed statistics, for the same reason; only 'subtract' needs its
# own, because it reads CORRECTED_DATA and MODEL_DATA instead.
#
# Usage
#   ./run_sim_cases.sh                               # all three cases
#   ./run_sim_cases.sh --cases none,joint            # only some
#   ./run_sim_cases.sh --skip-stack                  # the stack already exists
#
# Options: --input --rest-freq --tag --base --cases --match-tol-arcsec
#          --v-window --velocity-range --width --nboot --unit
#          --bins "MIN MAX N" --fit-span --theta-max --min-src-bin --model
#          --extra "..."  (further single-word options for run_vista.sh)
# ============================================================================
set -u

INPUT="sim_gauss_input.txt"
RESTFREQ="345.7959899"            # CO(3-2)
TAG="simgauss"
BASE="$PWD/run_sim"
CASES="none,joint,subtract"
MATCH_TOL="0"                     # simulations share one field: no grouping
VWINDOW="-600 600"
VELOCITY_RANGE="2000"
NBOOT="500"
UNIT="Jy"
BINS=""                           # "MIN MAX N" in klambda
FIT_SPAN=""                       # half-width of the shape fit, km/s
THETA_MAX=""                      # upper bound on the fitted size, arcsec
MIN_SRC_BIN=""                    # annuli with fewer sources are dropped
MODEL=""                          # gauss | point | gauss2
WIDTH=""                          # common rest-frame channel width, Hz
IN_COLUMN=""                      # column read from each input MS
EXTRA=""                          # further single-word options for run_vista.sh
SKIP_STACK=0

while [ $# -gt 0 ]; do
    case "$1" in
        --input)          INPUT="$2"; shift 2;;
        --rest-freq)      RESTFREQ="$2"; shift 2;;
        --tag)            TAG="$2"; shift 2;;
        --base)           BASE="$2"; shift 2;;
        --cases)          CASES="$2"; shift 2;;
        --match-tol-arcsec) MATCH_TOL="$2"; shift 2;;
        --v-window)       VWINDOW="$2"; shift 2;;
        --velocity-range) VELOCITY_RANGE="$2"; shift 2;;
        --nboot)          NBOOT="$2"; shift 2;;
        --unit)           UNIT="$2"; shift 2;;
        --bins)           BINS="$2"; shift 2;;
        --fit-span)       FIT_SPAN="$2"; shift 2;;
        --theta-max)      THETA_MAX="$2"; shift 2;;
        --min-src-bin)    MIN_SRC_BIN="$2"; shift 2;;
        --model)          MODEL="$2"; shift 2;;
        --width)          WIDTH="$2"; shift 2;;
        --in-column)      IN_COLUMN="$2"; shift 2;;
        --extra)          EXTRA="$2"; shift 2;;
        --skip-stack)     SKIP_STACK=1; shift;;
        -h|--help)        sed -n '2,25p' "$0"; exit 0;;
        *) echo "unknown option: $1" >&2; exit 2;;
    esac
done

HERE="$(cd "$(dirname "$0")" && pwd)"
DRIVER="$HERE/run_vista.sh"
[ -x "$DRIVER" ] || { echo "run_vista.sh not found next to this script" >&2; exit 1; }
[ -f "$INPUT" ]  || { echo "input list not found: $INPUT" >&2; exit 1; }
INPUT="$(cd "$(dirname "$INPUT")" && pwd)/$(basename "$INPUT")"

mkdir -p "$BASE"
MS="$BASE/stacked_${TAG}.ms"

COMMON=(--input "$INPUT" --rest-freq "$RESTFREQ" --tag "$TAG"
        --v-window "$VWINDOW" --match-tol-arcsec "$MATCH_TOL"
        --nboot "$NBOOT" --unit "$UNIT")
# quoted values must be forwarded as single array elements, or the spaces in
# "3 400 12" would be split into three arguments
[ -n "$BINS" ]        && COMMON+=(--bins "$BINS")
[ -n "$FIT_SPAN" ]    && COMMON+=(--fit-span "$FIT_SPAN")
[ -n "$THETA_MAX" ]   && COMMON+=(--theta-max "$THETA_MAX")
[ -n "$MIN_SRC_BIN" ] && COMMON+=(--min-src-bin "$MIN_SRC_BIN")
[ -n "$MODEL" ]       && COMMON+=(--model "$MODEL")
[ -n "$WIDTH" ]       && COMMON+=(--width "$WIDTH")
[ -n "$IN_COLUMN" ]   && COMMON+=(--in-column "$IN_COLUMN")
[ -n "$EXTRA" ]       && COMMON+=($EXTRA)

# ---------------------------------------------------------------------------
# 1. the stack, once
# ---------------------------------------------------------------------------
if [ "$SKIP_STACK" = 0 ]; then
    echo "############ stacking ############"
    "$DRIVER" "${COMMON[@]}" --out-dir "$BASE" \
              --velocity-range "$VELOCITY_RANGE" --only stack || exit 1
else
    echo "############ stacking skipped ############"
fi
[ -d "$MS" ] || { echo "stacked MS not found: $MS" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 2. one directory per case
# ---------------------------------------------------------------------------
FAILED=""

run_case () {
    local name="$1" dir="$BASE/case_$1"
    shift
    mkdir -p "$dir"
    # the MS is shared: link it in rather than stacking again
    [ -e "$dir/stacked_${TAG}.ms" ] || ln -s "$MS" "$dir/stacked_${TAG}.ms"
    echo
    echo "############ case: $name ############"
    if ! "$DRIVER" "${COMMON[@]}" --out-dir "$dir" --skip-stack "$@"; then
        echo "!!!! case $name FAILED" >&2
        FAILED="$FAILED $name"
    fi
}

IFS=',' read -ra WANTED <<< "$CASES"
for case_name in "${WANTED[@]}"; do
    case "$case_name" in
        none)
            run_case none
            ;;
        joint)
            # same column and same binning as 'none': reuse its statistics
            if [ -f "$BASE/case_none/${TAG}_line_stats.npy" ]; then
                mkdir -p "$BASE/case_joint"
                cp -n "$BASE/case_none/${TAG}_line_stats.npy" \
                      "$BASE/case_joint/" 2>/dev/null
                run_case joint --contsub after --skip-compress
            else
                run_case joint --contsub after
            fi
            ;;
        subtract)
            # writes MODEL_DATA and CORRECTED_DATA into the shared MS;
            # DATA is left untouched, so the other cases are unaffected
            run_case subtract --contsub before
            ;;
        *) echo "unknown case: $case_name" >&2; exit 2;;
    esac
done

# ---------------------------------------------------------------------------
# 3. the three answers side by side
# ---------------------------------------------------------------------------
if [ -n "$FAILED" ]; then
    echo
    echo "!!!! these cases failed:$FAILED" >&2
fi

echo
echo "############ summary ############"
python - "$BASE" "$TAG" <<'PY'
import glob, json, os, sys

base, tag = sys.argv[1], sys.argv[2]
rows = []
for path in sorted(glob.glob(os.path.join(base, "case_*", f"{tag}_*_results.json"))):
    with open(path) as fh:
        r = json.load(fh)
    case = os.path.basename(os.path.dirname(path)).replace("case_", "")
    what = os.path.basename(path).replace(f"{tag}_", "").replace("_results.json", "")
    rows.append((case, what, r))

if not rows:
    sys.exit("no results found")

head = f"{'case':10s} {'component':10s} {'N':>4s} {'flux':>12s} {'+-':>10s} " \
       f"{'theta':>8s} {'+-':>7s} {'chi2r':>7s} {'|Im|/sig':>9s}"
print(head)
print("-" * len(head))
for case, what, r in rows:
    print(f"{case:10s} {what:10s} {r['n_sources']:>4d} "
          f"{r['flux_total']:>12.5g} {r['flux_total_error']:>10.3g} "
          f"{r['theta_fwhm_arcsec']:>8.3f} {r['theta_error_arcsec']:>7.3f} "
          f"{r['chi2_reduced']:>7.2f} "
          f"{r.get('null_test_max_imaginary_over_sigma', float('nan')):>9.1f}")
    block = r.get("continuum")
    if block:
        print(f"{'':10s} {'continuum':10s} {'':>4s} "
              f"{block['flux_total']:>12.5g} {block['flux_total_error']:>10.3g} "
              f"{block['theta_fwhm_arcsec']:>8.3f} "
              f"{block['theta_error_arcsec']:>7.3f} "
              f"{block['chi2_reduced']:>7.2f}")
    shape = r.get("line_shape")
    if shape:
        print(f"{'':10s} {'  shape':10s}  v0={shape['centroid_kms']:+.0f} km/s  "
              f"FWHM={shape['fwhm_kms']:.0f} km/s")
print()
print("line flux in [unit] km/s, continuum flux density in [unit], "
      "theta in arcsec")
PY

echo
echo "done."
[ -n "$FAILED" ] && echo "  FAILED  :$FAILED"
echo "  stack   : $MS"
echo "  cases   : $BASE/case_*"
echo "  figures : $BASE/case_*/${TAG}_*.png"
