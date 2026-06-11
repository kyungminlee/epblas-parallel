#!/usr/bin/env bash
# Focused 5-way cmp5 sweep for the kind16 packed SYRK/SYR2K port (qsyrk,
# qsyr2k). Same methodology as run_cmp5.sh / run_cmp5_task96.sh (interleaved
# min-of-N, one physical core per thread, per-loop time budget) but scoped to
# the two routines touched by the packed-port rewrite, so the regression check
# finishes quickly — kind10 and the untouched kind16 routines are byte-identical
# to their committed baseline.
#
# Raw TSV is written under the gitignored workspace (never into bench/), in
# run_cmp5.sh's column format:
#   run_id  run_binary  omp  taskset  routine  key  size  iters  subject_ns  migrated_ns
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE_E="${STAGE_E:-${HERE}/../../build}"
OUT_DIR="${OUT_DIR:-${HERE}/../../workspace/files/gap5}"
TIMEOUT="${TIMEOUT:-300}"
export BLAS_PERF_ITERS="${BLAS_PERF_ITERS:-100}"
export BLAS_PERF_TIME_BUDGET="${BLAS_PERF_TIME_BUDGET:-0.3}"
export BLAS_PERF_WARMUP="${BLAS_PERF_WARMUP:-2}"
REPS="${REPS:-5}"

EP_DIR="${STAGE_E}/tests/epblas-openblas"
PAR_DIR="${STAGE_E}/tests/epblas-parallel"
RAW="${OUT_DIR}/cmp5_syrk2k_raw.tsv"
LOG="${OUT_DIR}/cmp5_syrk2k.log"

ROUTINES=(qsyrk qsyr2k)

mkdir -p "$OUT_DIR"
: > "$RAW"; : > "$LOG"
echo -e "run_id\trun_binary\tomp\ttaskset\troutine\tkey\tsize\titers\tsubject_ns\tmigrated_ns" >> "$RAW"

run_one() {
    local run_id="$1" tag="$2" omp="$3" bin="$4" routine="$5"
    local TMP; TMP=$(mktemp)
    local cpulist="2"; (( omp > 1 )) && cpulist="2-$((1 + omp))"
    echo "[run] $run_id/$routine taskset=$cpulist" >&2
    local status=0
    OMP_NUM_THREADS="$omp" timeout "$TIMEOUT" taskset -c "$cpulist" "$bin" \
        > "$TMP" 2>>"$LOG" || status=$?
    (( status != 0 )) && echo "[partial] $run_id/$routine exit=$status" >> "$LOG"
    awk -v rid="$run_id" -v tag="$tag" -v omp="$omp" -v ts="$cpulist" '
        /^#/ {next}
        NF >= 6 { printf "%s\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n",
            rid, tag, omp, ts, $1, $2, $3, $4, $5, $6; }' "$TMP" >> "$RAW"
    rm -f "$TMP"
}

for rep in $(seq 1 "$REPS"); do
    echo "[rep] $rep/$REPS" | tee -a "$LOG" >&2
    for routine in "${ROUTINES[@]}"; do
        ep_bin="$EP_DIR/ep_perf_${routine}"
        par_bin="$PAR_DIR/perf_${routine}"
        [[ -x "$ep_bin" && -x "$par_bin" ]] || { echo "[skip] $routine (missing binary)" >> "$LOG"; continue; }
        run_one "epopenblas-omp1"    "ep_"  1 "$ep_bin"  "$routine"
        run_one "epopenblas-omp4"    "ep_"  4 "$ep_bin"  "$routine"
        run_one "parallel-blas-omp1" "par_" 1 "$par_bin" "$routine"
        run_one "parallel-blas-omp4" "par_" 4 "$par_bin" "$routine"
    done
done
echo "wrote $RAW ($REPS reps)"
