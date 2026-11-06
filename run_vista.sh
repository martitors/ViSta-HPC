#!/bin/bash
# ============================================================================
# run_vista.sh — stack, post-process and extract, end to end
#
#   1. STACK     ViSta: rest-frame, re-centre, regrid and combine the input
#                datasets into one stack (Zarr store, or Measurement Set)
#   2. CONTSUB   subtract the continuum, spectral window by spectral window
#                (optional: skip it and fit line and continuum together)
#   3. COMPRESS  compress the stack into sufficient statistics
#   4. FIT       extract the stacked flux and the source size in the uv plane
#
# Every step can be run on its own, so a failed or refined step does not force
# the previous ones to be repeated.  Steps 1-3 touch the stack; step 4 works on a
# few MB of statistics and is cheap to repeat.
#
# When the stack is built in the same run, steps 2 and 3 are done while it is
# written, chunk by chunk (one pass over the data instead of three; same
# stack, same statistics).  DATA is always kept, so after a look at the
# spectrum the subtraction can be redone with another window without stacking
# again:  --skip-stack --exclude LO HI (or --exclude-v ...).  --two-pass runs
# the three steps one after the other instead.
#
# Usage
#   ./run_vista.sh --input input_list.txt --rest-freq 345.7959899   # GHz
#   ./run_vista.sh --input input_list.txt --rest-freq 345.7959899 \
#                  --contsub after --weighting democratic --norm-flux
#   ./run_vista.sh --input input_list.txt --rest-freq 345.7959899 \
#                  --only fit --weighting natural        # refit, nothing else
#   ./run_vista.sh --help
#
# The input list is the plain ViSta one:
#   <ms_path>  <redshift>  <RA>  <Dec>  [FIELD_ID]  [SPW_IDS]  [NORM]
# NORM is optional and is only used by --norm-flux.
# The continuum is handled in one of three ways.  By default nothing is
# subtracted and no continuum term is fitted: the data are treated as line
# only, over the velocity window of interest.  Otherwise:
#   --contsub before   remove it from the stack first (it goes to MODEL_DATA,
#                      the line to CORRECTED_DATA), then fit line and
#                      continuum separately, each with its own figures.
#   --contsub after    leave it in and fit it together with the line, the line
#                      sitting on a constant term whose amplitude is the
#                      continuum flux density: one fit, one figure.
# ============================================================================
set -u

# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------
INPUT=""                    # ViSta input list                  (required)
RESTFREQ=""                 # rest frequency of the line, GHz   (required)
TAG=""                      # name of the run; default from the rest frequency
OUTDIR="."                  # where everything is written

# --- what to do with the continuum (a scientific choice) -------------------
# none   : the default.  No subtraction and no continuum term: everything is
#          treated as line, over the velocity window of interest.
# before : --contsub before.  Subtract it from the stack first, spectral
#          window by spectral window, then fit line and continuum separately.
# after  : --contsub after.   Leave it in and fit it together with the line,
#          one fit and one figure.
CONTINUUM="none"

# --- which steps to actually run (all of them by default) ------------------
SKIP_STACK=0
SKIP_CONTSUB=0              # the subtraction was already done: do not redo it
SKIP_COMPRESS=0
SKIP_FIT=0
TWO_PASS=0                  # 1 = stack first, then contsub and compress (re-reading it)

# --- 1. stacking -----------------------------------------------------------
VELOCITY_RANGE=2000         # half-width of the output band, km/s ("" to use NCHAN)
NCHAN=""                    # fixed number of output channels, alternative
WIDTH=""                    # common rest-frame channel width, Hz; empty =
                            # widest rest-framed channel of the sample
CHUNK_ROWS=5000             # rows per chunk (bounds the memory in use)
SCRATCH="${TMPDIR:-}"       # build the stack on fast local disk, then move it
IN_COLUMN="auto"            # column read from each input MS: auto prefers
                            # CORRECTED_DATA, or name one (DATA, ...)
FRAME=""                    # frame of the redshifts (LSRK, BARY): the inputs
                            # are corrected to it row by row; empty = none
DEVICE="cpu"                # cpu | gpu (needs a CUDA build of the kernel)
READERS=2                   # reader processes
THREADS=""                  # OpenMP threads; empty = OMP_NUM_THREADS or all cores
GPU_BATCH=8                 # GPU: chunks per launch
TIMING=0                    # 1 = log how long each stage took
FORMAT="zarr"               # zarr (parallel writes, compressed) | ms
WRITERS=4                   # zarr: threads writing the stack
IO_THREADS=4                # zarr: threads reading/writing in contsub and compress
EXPORT_MS=0                 # 1 = also write the stack as a Measurement Set at the end
VERBOSE="true"              # true | debug

# --- 2. continuum subtraction ---------------------------------------------
CONT_ORDER=0                # 0 constant, 1 linear, -1 chosen per spw by BIC
EXCLUDE_LO=""               # asymmetric exclusion window, km/s
EXCLUDE_HI=""               # (default: the line window below)

# --- 3. compression --------------------------------------------------------
BINS="3 3000 18"            # b_min[klambda] b_max[klambda] n_bins
MAX_AMP=""                  # discard |V| above this; NaN/inf always discarded
CHUNK_MB=512

# --- 4. extraction and fit -------------------------------------------------
VWINDOW="-600 600"          # velocity window of the line, km/s
SECOND_RESTFREQ=""          # second line in the same band, GHz
SECOND_VWINDOW="-500 500"
EXCLUDE_V=""                # extra "LO HI" pairs kept out of the line-free channels

WEIGHTING="natural"         # natural | democratic
NORM_Z=1                    # transport every source to a common redshift
ZREF=""                     # that redshift; empty = sample median
NORM_FLUX=0                 # rescale by norm_ref/NORM (last column of the list)
NORMREF=""                  # that reference; empty = sample median
ALREADY_NORMALISED=0        # the amplitudes on disk already carry alpha

METHOD="joint"              # joint | template | window
MODE="search"               # joint: search (centre and width free) | detect (fixed)
V0=""                       # joint: line centre, km/s (detect: fixed; search: start)
FWHM=""                     # joint: line FWHM, km/s (detect: required)
SIZE="auto"                 # joint: auto | point | gauss
NNULL=1000                  # joint: realisations of the search on noise
PLOT_REBIN=1                # joint: channels per bin in the profile figure
MODEL="gauss"               # gauss | point | gauss2
THETA_MAX=15
THETA_FIXED=""
THETA_PRIOR_FROM=""         # results JSON to take a size prior from
THETA_PRIOR_SIGMA=0.15
BMAX_KL=""
MIN_SRC_BIN=2
FIT_SPAN=1500
SHAPE_TIE="free"
NBOOT=500
SEED=42
UNIT=""                     # name of the amplitude unit on the plot axes
V_LIMITS=""                 # velocity range of the spectrum plot, "LO HI"
MATCH_TOL=2.0
PLOT=1

usage () {
    sed -n '2,30p' "$0"
    cat <<'EOF'

Options
  --input FILE            ViSta input list                          (required)
  --rest-freq GHZ         rest frequency of the target line         (required)
  --tag NAME              name of the run (default: from --rest-freq)
  --out-dir DIR           output directory                          (default .)

  --contsub {before,after}        how to handle the continuum.  Without it,
                                  the default, nothing is subtracted and no
                                  continuum term is fitted: the data are
                                  treated as line only.
                                    before: subtract it from the stack, then
                                      fit line and continuum separately, each
                                      with its own figures
                                    after:  keep it and fit it together with
                                      the line, one fit and one figure

  --skip-stack                    the stack already exists
  --skip-contsub                  the subtraction was already done
  --skip-compress                 the statistics already exist
  --skip-fit                      stop after the statistics
  --only STEP                     run only one of stack|contsub|compress|fit
  --two-pass                      stack first, then subtract and compress
                                  reading the stack back (default: steps 2
                                  and 3 during the stacking, one pass)

  --velocity-range KMS    half-width of the stacked band            (default 2000)
  --nchan N               fixed number of output channels instead
  --width HZ              common rest-frame channel width, in Hz.  Default:
                          the widest rest-framed channel of the sample, the
                          finest grid every dataset supports.  A finer value
                          is refused.
  --chunk-rows N          rows per chunk while reading              (default 5000)
  --scratch DIR           build the stack here, then move it        (default $TMPDIR)
  --in-column NAME        column read from each input MS.  'auto' (default)
                          prefers CORRECTED_DATA and falls back to DATA; give
                          DATA explicitly when CORRECTED_DATA holds something
                          you do not want, as in a corrupted simulation.
  --frame NAME            frame of the redshifts, LSRK or BARY: the inputs
                          (ALMA, VLA: TOPO) are corrected to it row by row.
                          Default: no correction, with a warning for TOPO
  --device {cpu,gpu}      where the stacking kernel runs            (default cpu)
  --readers N             reader processes                          (default 2)
  --threads N             OpenMP threads of the CPU kernel  (default: all cores)
  --gpu-batch N           GPU: chunks per launch                    (default 8)
  --timing                log how long reading, computing and writing took
  --format {zarr,ms}      format of the stack: zarr (default; parallel,
                          compressed writes) or a Measurement Set
  --writers N             zarr: threads writing the stack           (default 4)
  --io-threads N          zarr: I/O threads of contsub and compress (default 4)
  --export-ms             at the end, also write the stack as a Measurement
                          Set (stacked_<tag>.ms), e.g. for tclean
  --debug                 detailed log (metadata, queues, buffers)

  --cont-order {0,1,-1}   continuum: constant, linear, or BIC       (default 0)
  --exclude LO HI         asymmetric velocity window to exclude from the
                          continuum fit (default: the line window)

  --bins "MIN MAX N"      radial binning, klambda                   (default "3 3000 18")
  --max-amplitude X       discard visibilities above |V| = X
  --chunk-mb X            read chunk size, MB                       (default 512)

  --v-window "LO HI"      velocity window of the line               (default "-600 600")
  --second-rest-freq GHZ  second line in the same band
  --second-v-window "LO HI"                                         (default "-500 500")
  --exclude-v "LO HI ..." extra intervals kept out of the line-free channels

  --weighting {natural,democratic}                                  (default natural)
  --norm-z | --no-norm-z          rescale to a common redshift      (default on)
  --z-ref Z                       that redshift (default: median)
  --norm-flux | --no-norm-flux    rescale by norm_ref/NORM          (default off)
  --norm-ref X                    that reference (default: median)
  --already-normalised            the data already carry alpha

  --method {joint,template,window}  how the line is measured       (default joint)
                                  joint: one fit of all channels and annuli,
                                  noise measured from the data, profile,
                                  significance (vista.extract.measure);
                                  template, window: the chain of Sec. 5.3
  --mode {search,detect}          joint: centre and width free (search, the
                                  significance accounts for it) or fixed to
                                  --v0/--fwhm (detect: matched filter)
  --v0 KMS                        joint: line centre (default 0)
  --fwhm KMS                      joint: line FWHM (detect: required)
  --size {auto,point,gauss}       joint: source size                (default auto)
  --n-null N                      joint: noise realisations          (default 1000)
  --plot-rebin N                  joint: channels per bin in the profile figure
  --model {gauss,point,gauss2}    source model                      (default gauss)
  --theta-max ARCSEC                                                (default 15)
  --theta-fixed ARCSEC            freeze the size, fit the flux only
  --theta-prior-from FILE.json    take a size prior from a previous fit
  --theta-prior-sigma DEX                                           (default 0.15)
  --b-max-kl X                    ignore annuli beyond this baseline
  --min-src-bin N                 drop annuli with fewer sources     (default 2)
  --fit-span KMS                  half-width of the shape fit        (default 1500)
  --shape-tie {free,centroid,centroid+width}                        (default free)
  --nboot N                       bootstrap realisations             (default 500)
  --seed N                                                          (default 42)
  --match-tol-arcsec X            same-source tolerance              (default 2.0)
  --unit NAME                     amplitude unit on the plot axes, e.g. Jy
  --v-limits "LO HI"              velocity range of the spectrum plot
  --no-plot                       skip the diagnostic figures

With --contsub the line is measured on CORRECTED_DATA and the continuum on
MODEL_DATA, in two separate fits.  With --no-contsub nothing is subtracted and
a single fit measures line and continuum together, the line sitting on a
constant term whose amplitude is the continuum flux density.
EOF
}

# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------
while [ $# -gt 0 ]; do
    case "$1" in
        --input)              INPUT="$2"; shift 2;;
        --rest-freq)          RESTFREQ="$2"; shift 2;;
        --tag)                TAG="$2"; shift 2;;
        --out-dir)            OUTDIR="$2"; shift 2;;

        --contsub)
            case "$2" in
                before) CONTINUUM="before"; shift 2;;
                after)  CONTINUUM="after";  shift 2;;
                none)   CONTINUUM="none";   shift 2;;
                *) echo "--contsub wants before|after (or none)" >&2; exit 2;;
            esac;;
        --no-contsub)         CONTINUUM="none"; shift;;

        --skip-stack)         SKIP_STACK=1; shift;;
        --skip-contsub)       SKIP_CONTSUB=1; shift;;
        --skip-compress)      SKIP_COMPRESS=1; shift;;
        --skip-fit)           SKIP_FIT=1; shift;;
        --two-pass)           TWO_PASS=1; shift;;
        --only)
            SKIP_STACK=1; SKIP_CONTSUB=1; SKIP_COMPRESS=1; SKIP_FIT=1
            case "$2" in
                stack)    SKIP_STACK=0;;
                contsub)  SKIP_CONTSUB=0;;
                compress) SKIP_COMPRESS=0;;
                fit)      SKIP_FIT=0;;
                *) echo "--only wants stack|contsub|compress|fit" >&2; exit 2;;
            esac
            shift 2;;

        --velocity-range)     VELOCITY_RANGE="$2"; shift 2;;
        --nchan)              NCHAN="$2"; VELOCITY_RANGE=""; shift 2;;
        --width)              WIDTH="$2"; shift 2;;
        --chunk-rows)         CHUNK_ROWS="$2"; shift 2;;
        --scratch)            SCRATCH="$2"; shift 2;;
        --in-column)          IN_COLUMN="$2"; shift 2;;
        --frame)              FRAME="$2"; shift 2;;
        --device)             DEVICE="$2"; shift 2;;
        --readers)            READERS="$2"; shift 2;;
        --threads)            THREADS="$2"; shift 2;;
        --gpu-batch)          GPU_BATCH="$2"; shift 2;;
        --timing)             TIMING=1; shift;;
        --format)             FORMAT="$2"; shift 2;;
        --writers)            WRITERS="$2"; shift 2;;
        --io-threads)         IO_THREADS="$2"; shift 2;;
        --export-ms)          EXPORT_MS=1; shift;;
        --debug)              VERBOSE="debug"; shift;;

        --cont-order)         CONT_ORDER="$2"; shift 2;;
        --exclude)            EXCLUDE_LO="$2"; EXCLUDE_HI="$3"; shift 3;;

        --bins)               BINS="$2"; shift 2;;
        --max-amplitude)      MAX_AMP="$2"; shift 2;;
        --chunk-mb)           CHUNK_MB="$2"; shift 2;;

        --v-window)           VWINDOW="$2"; shift 2;;
        --second-rest-freq)   SECOND_RESTFREQ="$2"; shift 2;;
        --second-v-window)    SECOND_VWINDOW="$2"; shift 2;;
        --exclude-v)          EXCLUDE_V="$2"; shift 2;;

        --weighting)          WEIGHTING="$2"; shift 2;;
        --norm-z)             NORM_Z=1; shift;;
        --no-norm-z)          NORM_Z=0; shift;;
        --z-ref)              ZREF="$2"; shift 2;;
        --norm-flux)          NORM_FLUX=1; shift;;
        --no-norm-flux)       NORM_FLUX=0; shift;;
        --norm-ref)           NORMREF="$2"; shift 2;;
        --already-normalised) ALREADY_NORMALISED=1; shift;;

        --method)             METHOD="$2"; shift 2;;
        --mode)               MODE="$2"; shift 2;;
        --v0)                 V0="$2"; shift 2;;
        --fwhm)               FWHM="$2"; shift 2;;
        --size)               SIZE="$2"; shift 2;;
        --n-null)             NNULL="$2"; shift 2;;
        --plot-rebin)         PLOT_REBIN="$2"; shift 2;;
        --model)              MODEL="$2"; shift 2;;
        --theta-max)          THETA_MAX="$2"; shift 2;;
        --theta-fixed)        THETA_FIXED="$2"; shift 2;;
        --theta-prior-from)   THETA_PRIOR_FROM="$2"; shift 2;;
        --theta-prior-sigma)  THETA_PRIOR_SIGMA="$2"; shift 2;;
        --b-max-kl)           BMAX_KL="$2"; shift 2;;
        --min-src-bin)        MIN_SRC_BIN="$2"; shift 2;;
        --fit-span)           FIT_SPAN="$2"; shift 2;;
        --shape-tie)          SHAPE_TIE="$2"; shift 2;;
        --nboot)              NBOOT="$2"; shift 2;;
        --seed)               SEED="$2"; shift 2;;
        --match-tol-arcsec)   MATCH_TOL="$2"; shift 2;;
        --no-plot)            PLOT=0; shift;;
        --unit)               UNIT="$2"; shift 2;;
        --v-limits)           V_LIMITS="$2"; shift 2;;

        -h|--help)            usage; exit 0;;
        *) echo "unknown option: $1" >&2; echo "try --help" >&2; exit 2;;
    esac
done

[ -n "$INPUT" ]    || { echo "--input is required"    >&2; exit 2; }
[ -n "$RESTFREQ" ] || { echo "--rest-freq is required" >&2; exit 2; }
[ -f "$INPUT" ]    || { echo "input list not found: $INPUT" >&2; exit 2; }

[ -n "$TAG" ] || TAG=$(awk "BEGIN{printf \"line%.0f\", $RESTFREQ*1000}")
mkdir -p "$OUTDIR"

case "$FORMAT" in
    zarr) MS="$OUTDIR/stacked_${TAG}.zarr";;
    ms)   MS="$OUTDIR/stacked_${TAG}.ms";;
    *) echo "--format wants zarr|ms" >&2; exit 2;;
esac
STATS_LINE="$OUTDIR/${TAG}_line_stats.npy"
STATS_CONT="$OUTDIR/${TAG}_cont_stats.npy"
RESTHZ=$(awk "BEGIN{printf \"%.6e\", $RESTFREQ*1e9}")

echo "============================================================"
echo "  ViSta  ${TAG}   nu_rest = ${RESTFREQ} GHz"
echo "  input     : $INPUT"
echo "  stack     : $MS ($FORMAT)"
case "$CONTINUUM" in
    before) CONT_NOTE="subtracted from the stack, fitted separately";;
    after)  CONT_NOTE="kept, fitted jointly with the line";;
    none)   CONT_NOTE="not fitted, data treated as line only";;
esac
echo "  continuum : $CONT_NOTE"
echo "  weighting : $WEIGHTING"
echo "  in column : $IN_COLUMN"
echo "  device    : $DEVICE"
echo "============================================================"

# ---------------------------------------------------------------------------
# one pass: continuum subtraction and compression while the stack is written
# ---------------------------------------------------------------------------
ONE_PASS=0
INLINE_CONTSUB=0
if [ "$SKIP_STACK" = 0 ] && [ "$SKIP_COMPRESS" = 0 ] && [ "$TWO_PASS" = 0 ]; then
    if [ "$CONTINUUM" != before ]; then
        ONE_PASS=1
    elif [ "$SKIP_CONTSUB" = 0 ]; then
        ONE_PASS=1; INLINE_CONTSUB=1
    fi
fi
INLINE_JSON=""
if [ "$ONE_PASS" = 1 ]; then
    INLINE_JSON=$(python - "$RESTFREQ" "$VWINDOW" "$SECOND_RESTFREQ" "$SECOND_VWINDOW" \
        "$EXCLUDE_V" "$INLINE_CONTSUB" "$CONT_ORDER" "$EXCLUDE_LO" "$EXCLUDE_HI" \
        "$BINS" "$MAX_AMP" "$STATS_LINE" "$STATS_CONT" <<'PY'
import json, sys
(rest, vw, rest2, vw2, excl, contsub, order, lo, hi, bins, max_amp,
 stats_line, stats_cont) = sys.argv[1:14]
vals = [float(x) for x in excl.split()] if excl.strip() else []
if len(vals) % 2:
    sys.exit("--exclude-v needs pairs LO HI")
print(json.dumps({
    "rest_freq_ghz": float(rest), "v_window_kms": [float(x) for x in vw.split()],
    "second_rest_freq_ghz": float(rest2) if rest2 else None,
    "second_v_window_kms": [float(x) for x in vw2.split()],
    "exclude_v_kms": [vals[i:i + 2] for i in range(0, len(vals), 2)],
    "contsub": contsub == "1", "order": int(order),
    "exclude_lo": float(lo) if lo else None, "exclude_hi": float(hi) if hi else None,
    "bins": [float(x) for x in bins.split()],
    "max_amplitude": float(max_amp) if max_amp else None,
    "stats_line": stats_line, "stats_continuum": stats_cont if contsub == "1" else None}))
PY
) || exit 2
fi

# ---------------------------------------------------------------------------
# 1. stacking
# ---------------------------------------------------------------------------
if [ "$SKIP_STACK" = 0 ]; then
    if [ "$ONE_PASS" = 1 ]; then
        echo; echo "== [1/4] stacking, with $([ "$INLINE_CONTSUB" = 1 ] && echo "continuum subtraction and ")compression in the same pass =="
    else
        echo; echo "== [1/4] stacking =="
    fi
    python - "$INPUT" "$MS" "$RESTHZ" "$CHUNK_ROWS" "$VELOCITY_RANGE" \
             "$NCHAN" "$SCRATCH" "$WIDTH" "$IN_COLUMN" "$DEVICE" "$READERS" \
             "$THREADS" "$GPU_BATCH" "$TIMING" "$VERBOSE" "$FORMAT" "$WRITERS" "$FRAME" \
             "$INLINE_JSON" \
             <<'PY' 2>&1 | tee "$OUTDIR/${TAG}_stack.log"
import json
import sys
from vista import ViSta

(input_file, ms_out, restfreq, chunk_rows, vrange, nchan, scratch,
 width, in_column, device, readers, threads, gpu_batch, timing,
 verbose, out_format, writers, frame, inline) = sys.argv[1:20]

kwargs = {}
if vrange:
    kwargs["velocity_range_kms"] = float(vrange)
elif nchan:
    kwargs["nchan_out"] = int(nchan)
if scratch:
    kwargs["scratch_dir"] = scratch
if width:
    kwargs["channel_width_hz"] = float(width)
kwargs["data_column"] = in_column or "auto"
kwargs["device"] = device
kwargs["n_readers"] = int(readers)
kwargs["gpu_batch"] = int(gpu_batch)
kwargs["timing"] = timing == "1"
kwargs["output_format"] = out_format
kwargs["n_writers"] = int(writers)
if frame:
    kwargs["frame"] = frame
if threads:
    kwargs["n_threads"] = int(threads)
if inline:
    from vista.extract import BinningConfig, ContinuumConfig, LineConfig
    c = json.loads(inline)
    line = LineConfig(rest_freq_ghz=c["rest_freq_ghz"],
                      v_window_kms=tuple(c["v_window_kms"]),
                      second_rest_freq_ghz=c["second_rest_freq_ghz"],
                      second_v_window_kms=tuple(c["second_v_window_kms"]),
                      exclude_v_kms=[tuple(x) for x in c["exclude_v_kms"]])
    kwargs["line"] = line
    if c["contsub"]:
        # as `python -m vista.extract contsub`: the line window, unless --exclude
        kwargs["continuum"] = ContinuumConfig(
            order=c["order"],
            exclude_kms=max(abs(v) for v in line.v_window_kms),
            exclude_kms_lo=c["exclude_lo"], exclude_kms_hi=c["exclude_hi"])
    kwargs["stats_line"] = c["stats_line"]
    kwargs["stats_continuum"] = c["stats_continuum"]
    kwargs["binning"] = BinningConfig(b_min_klambda=c["bins"][0],
                                      b_max_klambda=c["bins"][1],
                                      n_bins=int(c["bins"][2]))
    kwargs["max_amplitude"] = c["max_amplitude"]

ViSta(input_file=input_file, chunk_rows=int(chunk_rows),
      verbose="debug" if verbose == "debug" else True).run(
    output=ms_out, central_freq=float(restfreq), **kwargs)
PY
    [ ${PIPESTATUS[0]} -eq 0 ] || { echo "stacking failed" >&2; exit 1; }
    if [ "$ONE_PASS" = 1 ]; then
        SKIP_COMPRESS=1
        [ "$INLINE_CONTSUB" = 1 ] && SKIP_CONTSUB=1
    fi
else
    echo; echo "== [1/4] stacking skipped =="
fi

# the stack is only needed by the two steps that read it
if { [ "$SKIP_CONTSUB" = 0 ] && [ "$CONTINUUM" = before ]; } \
   || [ "$SKIP_COMPRESS" = 0 ]; then
    [ -d "$MS" ] || { echo "stack not found: $MS" >&2; exit 1; }
fi

# ---------------------------------------------------------------------------
# 2. continuum subtraction
# ---------------------------------------------------------------------------
CONT_ARGS=()
if [ -n "$EXCLUDE_LO" ]; then
    CONT_ARGS+=(--exclude-kms-lo "$EXCLUDE_LO" --exclude-kms-hi "$EXCLUDE_HI")
fi
SECOND_ARGS=()
if [ -n "$SECOND_RESTFREQ" ]; then
    SECOND_ARGS+=(--second-rest-freq "$SECOND_RESTFREQ"
                  --second-v-window $SECOND_VWINDOW)
fi
EXCL_ARGS=()
[ -n "$EXCLUDE_V" ] && EXCL_ARGS+=(--exclude-v $EXCLUDE_V)

if [ "$CONTINUUM" = before ] && [ "$SKIP_CONTSUB" = 0 ]; then
    echo; echo "== [2/4] continuum subtraction (order=$CONT_ORDER) =="
    python -u -m vista.extract contsub "$MS" --io-threads "$IO_THREADS" \
        --rest-freq "$RESTFREQ" --v-window $VWINDOW \
        "${SECOND_ARGS[@]}" "${EXCL_ARGS[@]}" "${CONT_ARGS[@]}" \
        --order "$CONT_ORDER" \
        2>&1 | tee "$OUTDIR/${TAG}_contsub.log"
    [ ${PIPESTATUS[0]} -eq 0 ] || { echo "continuum subtraction failed" >&2; exit 1; }
elif [ "$CONTINUUM" = before ] && [ "$INLINE_CONTSUB" = 1 ]; then
    echo; echo "== [2/4] continuum subtraction: done during the stacking =="
elif [ "$CONTINUUM" = before ]; then
    echo; echo "== [2/4] continuum subtraction skipped (already done) =="
elif [ "$CONTINUUM" = after ]; then
    echo; echo "== [2/4] no subtraction: line and continuum will be fitted"
    echo "         together on DATA =="
else
    echo; echo "== [2/4] no subtraction and no continuum fit: DATA is treated"
    echo "         as line only =="
fi

# ---------------------------------------------------------------------------
# 3. compression into sufficient statistics
# ---------------------------------------------------------------------------
COMPRESS_ARGS=(--bins $BINS --chunk-mb "$CHUNK_MB")
[ -n "$MAX_AMP" ] && COMPRESS_ARGS+=(--max-amplitude "$MAX_AMP")
if [ "$CONTINUUM" = before ]; then
    COMPRESS_ARGS+=(--line-column CORRECTED_DATA
                    --out-continuum "$STATS_CONT")
else
    COMPRESS_ARGS+=(--line-column DATA)
fi

if [ "$SKIP_COMPRESS" = 0 ]; then
    echo; echo "== [3/4] compression into sufficient statistics =="
    python -u -m vista.extract compress "$MS" --input "$INPUT" --io-threads "$IO_THREADS" \
        --rest-freq "$RESTFREQ" --v-window $VWINDOW \
        "${SECOND_ARGS[@]}" "${EXCL_ARGS[@]}" \
        --out-line "$STATS_LINE" "${COMPRESS_ARGS[@]}" \
        2>&1 | tee "$OUTDIR/${TAG}_compress.log"
    [ ${PIPESTATUS[0]} -eq 0 ] || { echo "compression failed" >&2; exit 1; }
elif [ "$ONE_PASS" = 1 ]; then
    echo; echo "== [3/4] compression: done during the stacking =="
else
    echo; echo "== [3/4] compression skipped =="
fi

# ---------------------------------------------------------------------------
# 4. extraction and fit
# ---------------------------------------------------------------------------
export_ms () {
    if [ "$EXPORT_MS" = 1 ] && [ "$FORMAT" = zarr ] && [ -d "$MS" ]; then
        echo; echo "== export of the stack to a Measurement Set =="
        python -u -m vista.extract export-ms "$MS" "${MS%.zarr}.ms" \
            2>&1 | tee "$OUTDIR/${TAG}_export.log"
    fi
}

if [ "$SKIP_FIT" = 1 ]; then
    echo; echo "== [4/4] fit skipped =="
    export_ms
    echo "done: $MS"
    exit 0
fi

echo; echo "== [4/4] extraction and fit =="

FIT_ARGS=(--weighting "$WEIGHTING" --model "$MODEL"
          --theta-max "$THETA_MAX" --min-sources-per-bin "$MIN_SRC_BIN"
          --fit-span-kms "$FIT_SPAN" --shape-tie "$SHAPE_TIE"
          --n-bootstrap "$NBOOT" --seed "$SEED"
          --match-tol-arcsec "$MATCH_TOL")
[ "$NORM_Z"  = 0 ] && FIT_ARGS+=(--no-redshift-rescaling)
[ -n "$ZREF" ]     && FIT_ARGS+=(--z-ref "$ZREF")
[ "$NORM_FLUX" = 1 ] && FIT_ARGS+=(--flux-normalisation)
[ -n "$NORMREF" ]  && FIT_ARGS+=(--norm-ref "$NORMREF")
[ "$ALREADY_NORMALISED" = 1 ] && FIT_ARGS+=(--already-normalised)
[ -n "$THETA_FIXED" ] && FIT_ARGS+=(--theta-fixed "$THETA_FIXED")
[ -n "$BMAX_KL" ]     && FIT_ARGS+=(--b-max-kl "$BMAX_KL")
[ "$PLOT" = 1 ]       && FIT_ARGS+=(--plot)
[ -n "$UNIT" ]        && FIT_ARGS+=(--unit "$UNIT")
[ -n "$V_LIMITS" ]    && FIT_ARGS+=(--v-limits $V_LIMITS)
JOINT_ARGS=()
if [ "$METHOD" = joint ]; then
    [ "$MODEL" = point ] && [ "$SIZE" = auto ] && SIZE=point
    JOINT_ARGS+=(--mode "$MODE" --size "$SIZE" --n-null "$NNULL" --plot-rebin "$PLOT_REBIN")
    [ -n "$V0" ]   && JOINT_ARGS+=(--v0 "$V0")
    [ -n "$FWHM" ] && JOINT_ARGS+=(--fwhm "$FWHM")
fi

case "$CONTINUUM" in
before)
    # ---- two independent fits, on two columns, with their own figures -----
    echo "-- continuum, from the line-free channels of MODEL_DATA"
    python -u -m vista.extract flux "$STATS_CONT" --input "$INPUT" \
        --rest-freq "$RESTFREQ" --v-window $VWINDOW \
        "${FIT_ARGS[@]}" --method continuum \
        --out "$OUTDIR/${TAG}_continuum" \
        2>&1 | tee "$OUTDIR/${TAG}_fit_continuum.log"

    LINE_ARGS=("${FIT_ARGS[@]}" --method "$METHOD" "${JOINT_ARGS[@]}")
    if [ -n "$THETA_PRIOR_FROM" ]; then
        LINE_ARGS+=(--theta-prior-from "$THETA_PRIOR_FROM"
                    --theta-prior-sigma "$THETA_PRIOR_SIGMA")
    fi
    echo "-- line, from CORRECTED_DATA"
    python -u -m vista.extract flux "$STATS_LINE" --input "$INPUT" \
        --rest-freq "$RESTFREQ" --v-window $VWINDOW \
        "${SECOND_ARGS[@]}" "${LINE_ARGS[@]}" \
        --out "$OUTDIR/${TAG}_line" \
        2>&1 | tee "$OUTDIR/${TAG}_fit_line.log"
    ;;

after)
    # ---- one fit, line on top of a constant continuum, one figure ---------
    echo "-- line and continuum together, from DATA"
    python -u -m vista.extract flux "$STATS_LINE" --input "$INPUT" \
        --rest-freq "$RESTFREQ" --v-window $VWINDOW \
        "${SECOND_ARGS[@]}" "${FIT_ARGS[@]}" --method "$METHOD" "${JOINT_ARGS[@]}" \
        --joint-continuum \
        --out "$OUTDIR/${TAG}_joint" \
        2>&1 | tee "$OUTDIR/${TAG}_fit_joint.log"
    ;;

none)
    # ---- line only: whatever continuum is left is assumed negligible ------
    echo "-- line only, from DATA (no continuum term)"
    python -u -m vista.extract flux "$STATS_LINE" --input "$INPUT" \
        --rest-freq "$RESTFREQ" --v-window $VWINDOW \
        "${SECOND_ARGS[@]}" "${FIT_ARGS[@]}" --method "$METHOD" "${JOINT_ARGS[@]}" \
        --out "$OUTDIR/${TAG}_line" \
        2>&1 | tee "$OUTDIR/${TAG}_fit_line.log"
    ;;
esac

# ---------------------------------------------------------------------------
# checks worth reading before believing anything
# ---------------------------------------------------------------------------
echo
echo "== checks =="
python - "$OUTDIR" "$TAG" <<'PY'
import glob, json, os, sys
outdir, tag = sys.argv[1], sys.argv[2]
keys = [("flux_total", "F"), ("flux_total_error", "+-"),
        ("theta_fwhm_arcsec", "theta"), ("theta_error_arcsec", "+-"),
        ("chi2_reduced", "chi2r"),
        ("null_test_max_imaginary_over_sigma", "|Im|/sigma"),
        ("n_bins_in_fit", "bins"), ("n_bootstrap_failed", "boot failed")]
for path in sorted(glob.glob(os.path.join(outdir, f"{tag}_*_results.json"))):
    with open(path) as fh:
        r = json.load(fh)
    print(f"\n{os.path.basename(path)}  "
          f"({r['n_sources']} sources, {r['weighting_scheme']})")
    if "detection" in r:                 # --method joint
        ln, d, n, w = r["line"], r["detection"], r["noise"], r["window"]
        print(f"  F={ln['flux']:.4g} +-{ln['flux_error']:.3g} (thermal "
              f"{ln['flux_error_thermal']:.3g}, bootstrap {ln['flux_error_bootstrap']:.3g})  "
              f"peak={ln['peak']:.4g}  v0={ln['v0_kms']:+.0f}  FWHM={ln['fwhm_kms']:.0f} km/s  "
              f"size={ln['size_model']}  chi2r={ln['chi2_reduced']:.2f}")
        print(f"  detection: " + (f"S/N={d['snr']:.2f} (fixed shape)" if d["mode"] == "detect"
              else f"{d['significance_sigma']:.1f} sigma (search S/N={d['snr']:.2f}, "
                   f"p={d['p_value']:.2g})")
              + f"   window [{w['user_kms'][0]:.0f},{w['user_kms'][1]:.0f}]: "
                f"{w['flux']:.4g} +-{w['flux_error']:.3g}")
        print(f"  noise: kappa (scatter/weights)={n['kappa_scatter_over_weights']:.3f}  "
              f"imaginary/noise={n.get('imaginary_rms_over_noise', float('nan')):.3f}  "
              f"line-free/noise={n.get('line_free_rms_over_noise', float('nan')):.3f}  "
              f"channel correlation={n['channel_correlation'][:2]}")
        for key, what in (("imaginary_rms_over_noise", "imaginary part"),
                          ("line_free_rms_over_noise", "line-free channels")):
            if isinstance(n.get(key), float) and abs(n[key] - 1) > 0.15:
                print(f"  WARNING: the {what} scatter {n[key]:.2f} x the thermal noise")
        if abs(d.get("snr_imaginary", 0)) > 3:
            print(f"  WARNING: the imaginary part has S/N {d['snr_imaginary']:.1f} with the "
                  f"line template: check the centring")
        continue
    print("  " + "  ".join(
        f"{label}={r[key]:.4g}" if isinstance(r.get(key), (int, float))
        else f"{label}=n/a" for key, label in keys))
    if r.get("theta_at_bound"):
        print("  WARNING: the size ran into theta_max, the profile does not "
              "constrain it")
    if isinstance(r.get("null_test_max_imaginary_over_sigma"), float) \
            and r["null_test_max_imaginary_over_sigma"] > 3:
        print("  WARNING: the imaginary part is not consistent with zero, "
              "check the centring")
    for block in ("second_line", "continuum"):
        if block in r:
            b = r[block]
            print(f"  {block}: F={b['flux_total']:.4g} "
                  f"+-{b['flux_total_error']:.3g}  "
                  f"theta={b['theta_fwhm_arcsec']:.3f}\"")
PY

export_ms

echo
echo "done."
echo "  stack      : $MS"
[ "$EXPORT_MS" = 1 ] && [ "$FORMAT" = zarr ] && echo "  stack (MS) : ${MS%.zarr}.ms"
echo "  statistics : $STATS_LINE$([ "$CONTINUUM" = before ] && echo ", $STATS_CONT")"
echo "  results    : $OUTDIR/${TAG}_*_results.json"
[ "$PLOT" = 1 ] && echo "  figures    : $OUTDIR/${TAG}_*.png"

exit 0
