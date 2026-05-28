#!/usr/bin/env bash
# Run the 4-variant cmp5 perf sweep for kind10:
#   - epopenblas binaries (ep_perf_*)        at OMP=1 and OMP=4
#   - parallel-blas binaries (perf_*)        at OMP=1 and OMP=4
# Pinned to P-core 0. Per-routine wall-clock cap via TIMEOUT env.
#
# Output: cmp5_raw.tsv (alongside this script) in the format the existing
# aggregate.py expects:
#   run_id  run_binary  omp  taskset  routine  key  size  iters  subject_GFs  migrated_GFs
#
# subject_GFs = GF/s of the C-implemented routine under test in this row
#   (which C overlay it is — epopenblas or parallel-blas — is read from run_id).
# migrated_GFs = GF/s of the migrated Fortran reference (the migrator-translated
#   _serial symbol; same symbol in both binaries, so should agree across runs).
#
# Usage: bash reports/cmp5/run_cmp5.sh
# Env:
#   STAGE_E       build dir holding both tests/epblas-openblas and
#                 tests/epblas-parallel (default <repo>/build)
#   TIMEOUT       per-binary wall-clock cap in seconds (default 300)
#   BLAS_PERF_ITERS  iters knob forwarded to the perf binaries (default 200)
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE_E="${STAGE_E:-${HERE}/../../build}"
TIMEOUT="${TIMEOUT:-300}"
export BLAS_PERF_ITERS="${BLAS_PERF_ITERS:-200}"

EP_DIR="${STAGE_E}/tests/epblas-openblas"
PAR_DIR="${STAGE_E}/tests/epblas-parallel"

RAW="${HERE}/cmp5_raw.tsv"
LOG="${HERE}/cmp5.log"

: > "$RAW"
: > "$LOG"
echo -e "run_id\trun_binary\tomp\ttaskset\troutine\tkey\tsize\titers\tsubject_GFs\tmigrated_GFs" >> "$RAW"

# Collect routine names — both halves must implement the same set, so use
# the epopenblas binary list as the source of truth and only run a
# parallel-blas binary if it exists with the same routine name.
mapfile -t ep_bins < <(find "$EP_DIR" -maxdepth 1 -type f -executable -name "ep_perf_*" | sort)
if (( ${#ep_bins[@]} == 0 )); then
    echo "[fatal] no ep_perf_* executables under $EP_DIR" >&2 | tee -a "$LOG"
    exit 1
fi

is_l3() {
    case "$1" in
        egemm|egemmtr|esymm|esyrk|esyr2k|etrmm|etrsm) return 0 ;;
        ygemm|ygemmtr|ysymm|ysyrk|ysyr2k|ytrmm|ytrsm) return 0 ;;
        yhemm|yherk|yher2k) return 0 ;;
    esac
    return 1
}

run_one() {
    local run_id="$1" run_bin_tag="$2" omp="$3" bin="$4" routine="$5"
    local TMP; TMP=$(mktemp)
    # L3 routines do O(N^3) work per call — keep iters low so each binary
    # finishes in a reasonable time at the larger sizes. L1/L2 stays at
    # BLAS_PERF_ITERS=200 to match the prior cmp5 sweep.
    local iters="$BLAS_PERF_ITERS"
    if is_l3 "$routine"; then iters=10; fi
    echo "[run] $run_id/$routine iters=$iters" >&2
    if OMP_NUM_THREADS="$omp" BLAS_PERF_ITERS="$iters" \
           timeout "$TIMEOUT" taskset -c 0 "$bin" \
           > "$TMP" 2>>"$LOG"; then
        :
    else
        echo "[fail] $run_id/$routine exit=$?" >> "$LOG"
    fi
    awk -v rid="$run_id" -v tag="$run_bin_tag" -v omp="$omp" '
        /^#/ {next}
        NF >= 6 {
            gsub(/x$/, "", $7);
            # perf binary stdout cols: routine key size iters subject_GFs migrated_GFs ratio
            # (subject = epopenblas or parallel-blas — read from rid)
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n",
                rid, tag, omp, 0, $1, $2, $3, $4, $5, $6;
        }' "$TMP" >> "$RAW"
    rm -f "$TMP"
}

for ep_bin in "${ep_bins[@]}"; do
    name=$(basename "$ep_bin")          # ep_perf_<routine>
    routine="${name#ep_perf_}"
    par_bin="$PAR_DIR/perf_${routine}"
    if [[ ! -x "$par_bin" ]]; then
        echo "[skip] no parallel-blas perf for $routine ($par_bin)" >> "$LOG"
        continue
    fi
    run_one "epopenblas-omp1"    "ep_"  1 "$ep_bin"  "$routine"
    run_one "epopenblas-omp4"    "ep_"  4 "$ep_bin"  "$routine"
    run_one "parallel-blas-omp1" "par_" 1 "$par_bin" "$routine"
    run_one "parallel-blas-omp4" "par_" 4 "$par_bin" "$routine"
done

echo "wrote $RAW; log $LOG"
