#!/usr/bin/env bash
# Focused 5-way interleaved min-of-N for a few routines: par {omp1,omp4} +
# ob {omp1,omp4} + migrated. Emits gap5_raw.tsv:
#   run_id omp routine key size iters subject_ns migrated_ns
# leg = par|ob read from run_id. min-merge in agg_gap5.py.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE_E="${STAGE_E:-${HERE}/../../build}"
TIMEOUT="${TIMEOUT:-300}"
export BLAS_PERF_ITERS="${BLAS_PERF_ITERS:-200}"
export BLAS_PERF_TIME_BUDGET="${BLAS_PERF_TIME_BUDGET:-0.3}"
export BLAS_PERF_WARMUP="${BLAS_PERF_WARMUP:-2}"
REPS="${REPS:-5}"
PAR_DIR="${STAGE_E}/tests/epblas-parallel"
EP_DIR="${STAGE_E}/tests/epblas-openblas"
RAW="${RAW:-${HERE}/gap5_raw.tsv}"
LOG="${LOG:-${HERE}/gap5.log}"
ROUTINES="${ROUTINES:?set ROUTINES='mtrmv ...'}"

# Relink the exact binaries this sweep runs BEFORE timing. Editing a source
# and rebuilding only `--target qblas_parallel` leaves the perf_* executables
# linked against the stale archive: CMake relinks a library's *consumers*
# only when you build the consumer target, never as a side effect of building
# the library. Building perf_$r / ep_perf_$r pulls in the rebuilt archive as a
# dependency, so the whole chain is current and the sweep can't time a stale
# binary. Unknown targets (a par-only or ob-only routine) are skipped; a real
# compile/link failure aborts. Set SKIP_BUILD=1 to time a prebuilt state.
if [[ -z "${SKIP_BUILD:-}" ]]; then
    for routine in $ROUTINES; do
        for t in "perf_${routine}" "ep_perf_${routine}"; do
            out=$(cmake --build "${STAGE_E}" --target "$t" 2>&1) || {
                if grep -qiE 'unknown target|No rule to make target' <<<"$out"; then
                    echo "[build] skip $t (no such target)" >&2
                else
                    echo "$out" >&2
                    echo "[build] FAILED building $t — refusing to time stale binaries (SKIP_BUILD=1 to override)" >&2
                    exit 1
                fi
            }
        done
    done
fi

: > "$RAW"; : > "$LOG"
echo -e "run_id\tomp\troutine\tkey\tsize\titers\tsubject_ns\tmigrated_ns" >> "$RAW"
run_one() {
    local run_id="$1" omp="$2" bin="$3" routine="$4"
    local TMP; TMP=$(mktemp)
    local cpulist="0"; (( omp > 1 )) && cpulist="0-$((omp - 1))"
    local status=0
    OMP_NUM_THREADS="$omp" timeout "$TIMEOUT" taskset -c "$cpulist" "$bin" \
        > "$TMP" 2>>"$LOG" || status=$?
    (( status != 0 )) && echo "[partial] $run_id/$routine exit=$status" >> "$LOG"
    awk -v rid="$run_id" -v omp="$omp" '
        /^#/ {next}
        NF >= 6 { printf "%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\n",
            rid, omp, $1, $2, $3, $4, $5, $6; }' "$TMP" >> "$RAW"
    rm -f "$TMP"
}
for rep in $(seq 1 "$REPS"); do
    echo "[rep] $rep/$REPS" >&2
    for routine in $ROUTINES; do
        par_bin="$PAR_DIR/perf_${routine}"; ep_bin="$EP_DIR/ep_perf_${routine}"
        [[ -x "$par_bin" ]] && { run_one "par-omp1" 1 "$par_bin" "$routine"; run_one "par-omp4" 4 "$par_bin" "$routine"; }
        [[ -x "$ep_bin" ]]  && { run_one "ob-omp1"  1 "$ep_bin"  "$routine"; run_one "ob-omp4"  4 "$ep_bin"  "$routine"; }
    done
done
echo "wrote $RAW ($REPS reps)"
