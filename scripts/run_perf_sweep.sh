#!/usr/bin/env bash
# Run every blas_parallel (epblas-parallel overlay) perf_* executable in
# the epblas-parallel build tree at OMP=1, pinned to P-core 0, and append
# the structured output to a TSV.
#
# TSV columns: target routine key size iters epblas_parallel_ns migrated_ns ratio
#   epblas_parallel_ns = the C epblas-parallel overlay's wall time (ns/call,
#                        smaller = faster) for this row
#   migrated_ns        = the migrated Fortran reference's ns/call for this row
#   ratio              = epblas_parallel_ns / migrated_ns (< 1.0 = parallel faster)
#
# Scope: epblas-parallel overlay only. epblas-openblas comparisons live under
# reports/cmp5/ (see reports/cmp5/run_cmp5.sh).
#
# Usage:
#   scripts/run_perf_sweep.sh [BUILD_DIR]
#     BUILD_DIR defaults to ./build (the epblas-parallel top-level build dir).
#     Inside it, perf_* executables for every built target live under
#     tests/epblas-parallel/ — distinguished by their prefix (e=kind10,
#     q=kind16, m=multifloats).
#
# Env knobs:
#   TIMEOUT  per-routine wall-clock cap in seconds (default 300)
#
# Survives single-routine crashes (heap corruption, assertion failures, etc.)
# so the rest of the sweep continues.
set -u

BUILD_DIR="${1:-./build}"
TDIR="$BUILD_DIR/tests/epblas-parallel"
if [[ ! -d "$TDIR" ]]; then
    echo "[fatal] no test dir at $TDIR — pass the epblas-parallel build dir as \$1" >&2
    exit 1
fi

OUTDIR="${OUTDIR:-reports}"
TIMEOUT="${TIMEOUT:-300}"
mkdir -p "$OUTDIR"
JSON="$OUTDIR/perf_sweep.json"
TSV="$OUTDIR/perf_sweep.tsv"
LOG="$OUTDIR/perf_sweep.log"
: > "$JSON"
: > "$TSV"
: > "$LOG"

echo -e "target\troutine\tkey\tsize\titers\tepblas_parallel_ns\tmigrated_ns\tratio" >> "$TSV"

for target in e q m; do
    case "$target" in
        e) tname=kind10 ;;
        q) tname=kind16 ;;
        m) tname=multifloats ;;
    esac
    # All targets share the same build dir; perf binaries are
    # distinguished by their first-letter prefix.
    shopt -s nullglob
    perf_bins=("$TDIR"/perf_${target}*)
    shopt -u nullglob
    if [[ ${#perf_bins[@]} -eq 0 ]]; then
        echo "[skip] $tname: no perf_${target}* binaries in $TDIR" >&2 | tee -a "$LOG"
        continue
    fi
    for exe in "${perf_bins[@]}"; do
        [[ -x "$exe" ]] || continue
        name=$(basename "$exe")
        echo "[run] $tname/$name" >&2
        # Run with a per-routine wall-clock cap (TIMEOUT env, default 300s).
        # Use a temp file to decouple the subprocess from the pipe so a
        # crash in the perf executable doesn't kill the parent shell.
        TMP=$(mktemp)
        if BLAS_PERF_JSON="$JSON" OMP_NUM_THREADS=1 \
              timeout "$TIMEOUT" taskset -c 0 "$exe" > "$TMP" 2>>"$LOG"; then
            : ok
        else
            echo "[fail] $tname/$name exit=$?" >> "$LOG"
        fi
        awk -v t="$tname" -v r="$name" '
            /^#/ {next}
            NF >= 6 {
                gsub(/x$/, "", $7);
                printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", t, $1, $2, $3, $4, $5, $6, $7;
            }' "$TMP" >> "$TSV"
        rm -f "$TMP"
    done
done

echo "wrote $TSV, $JSON; log at $LOG"
