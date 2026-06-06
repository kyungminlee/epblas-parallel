#!/usr/bin/env bash
# 3-way cmp5 regression sweep for the multifloats parallelization-gap work.
# Double-double has NO OpenBLAS leg, so this measures par-omp1, par-omp4, and the
# migrated serial reference (emitted as migrated_ns by each perf driver).
# Interleaved min-of-N, one physical core per thread, per-loop time budget.
#
# Usage: ROUTINES="masum mdot ..." [REPS=5] ./run_cmp5_gap.sh
# Emits cmp5_gap_raw.tsv:
#   run_id  omp  taskset  routine  key  size  iters  subject_ns  migrated_ns
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE_E="${STAGE_E:-${HERE}/../../build}"
TIMEOUT="${TIMEOUT:-300}"
export BLAS_PERF_ITERS="${BLAS_PERF_ITERS:-200}"
export BLAS_PERF_TIME_BUDGET="${BLAS_PERF_TIME_BUDGET:-0.3}"
export BLAS_PERF_WARMUP="${BLAS_PERF_WARMUP:-2}"
REPS="${REPS:-5}"

PAR_DIR="${STAGE_E}/tests/epblas-parallel"
RAW="${RAW:-${HERE}/cmp5_gap_raw.tsv}"
LOG="${LOG:-${HERE}/cmp5_gap.log}"
ROUTINES="${ROUTINES:?set ROUTINES='masum mdot ...'}"

: > "$RAW"; : > "$LOG"
echo -e "run_id\tomp\ttaskset\troutine\tkey\tsize\titers\tsubject_ns\tmigrated_ns" >> "$RAW"

run_one() {
    local run_id="$1" omp="$2" bin="$3" routine="$4"
    local TMP; TMP=$(mktemp)
    local cpulist="0"; (( omp > 1 )) && cpulist="0-$((omp - 1))"
    echo "[run] $run_id/$routine taskset=$cpulist" >&2
    local status=0
    OMP_NUM_THREADS="$omp" timeout "$TIMEOUT" taskset -c "$cpulist" "$bin" \
        > "$TMP" 2>>"$LOG" || status=$?
    (( status != 0 )) && echo "[partial] $run_id/$routine exit=$status" >> "$LOG"
    awk -v rid="$run_id" -v omp="$omp" -v ts="$cpulist" '
        /^#/ {next}
        NF >= 6 { printf "%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n",
            rid, omp, ts, $1, $2, $3, $4, $5, $6; }' "$TMP" >> "$RAW"
    rm -f "$TMP"
}

for rep in $(seq 1 "$REPS"); do
    echo "[rep] $rep/$REPS" | tee -a "$LOG" >&2
    for routine in $ROUTINES; do
        par_bin="$PAR_DIR/perf_${routine}"
        [[ -x "$par_bin" ]] || { echo "[skip] $routine (missing $par_bin)" >> "$LOG"; continue; }
        run_one "par-omp1" 1 "$par_bin" "$routine"
        run_one "par-omp4" 4 "$par_bin" "$routine"
    done
done
echo "wrote $RAW ($REPS reps)"
