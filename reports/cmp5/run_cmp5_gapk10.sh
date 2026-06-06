#!/usr/bin/env bash
# Focused 5-way cmp5 sweep for the four kind10 gap routines
# (enrm2/eynrm2 norms + etpsv/ytpsv packed solves) being threaded.
# Same methodology as run_cmp5_task96.sh; emits a raw.tsv in run_cmp5.sh's
# column format. Pass RAW=... to name the output (before vs after).
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE_E="${STAGE_E:-${HERE}/../../build}"
TIMEOUT="${TIMEOUT:-300}"
export BLAS_PERF_ITERS="${BLAS_PERF_ITERS:-200}"
export BLAS_PERF_TIME_BUDGET="${BLAS_PERF_TIME_BUDGET:-0.3}"
export BLAS_PERF_WARMUP="${BLAS_PERF_WARMUP:-2}"
REPS="${REPS:-3}"

EP_DIR="${STAGE_E}/tests/epblas-openblas"
PAR_DIR="${STAGE_E}/tests/epblas-parallel"
RAW="${RAW:-${HERE}/cmp5_gapk10_raw.tsv}"
LOG="${HERE}/cmp5_gapk10.log"

ROUTINES=(enrm2 eynrm2 etpsv ytpsv)

: > "$RAW"; : > "$LOG"
echo -e "run_id\trun_binary\tomp\ttaskset\troutine\tkey\tsize\titers\tsubject_ns\tmigrated_ns" >> "$RAW"

run_one() {
    local run_id="$1" tag="$2" omp="$3" bin="$4" routine="$5"
    local TMP; TMP=$(mktemp)
    local cpulist="0"; (( omp > 1 )) && cpulist="0-$((omp - 1))"
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
