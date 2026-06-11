#!/usr/bin/env bash
# Focused 5-way interleaved measurement for a few routines: par {omp1,omp4} +
# ob {omp1,omp4} + migrated. Emits gap5_raw.tsv:
#   run_id omp routine key size iters subject_ns migrated_ns rep
# leg = par|ob read from run_id; `rep` lets agg_gap5.py PAIR par/ob per rep.
#
# Accuracy protocol (see reports/cmp5/BENCH_PROTOCOL.md):
#   * The dominant error here is CPU frequency (turbo/DVFS), not random noise —
#     a cell's absolute ns swings ~40% with turbo on an i7-8700. We attack it two
#     ways: (1) best-effort frequency PIN (performance governor + no-turbo) when
#     the sysfs knobs are writable; (2) when they aren't (no root), we PAIR the
#     two arms we compare back-to-back within each rep so they run at ~the same
#     frequency, and agg takes the statistic on the per-rep RATIOS (frequency
#     cancels in the ratio). Hence the leg order below is par1,ob1,par4,ob4 —
#     par1/ob1 and par4/ob4 are the adjacent pairs.
#   * Cross-process is structural (both libs export the same BLAS symbols, so
#     they can't be linked into one process); per-rep is the finest pairing
#     available. Never run two pinned sweeps at once (contention poisons it).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE_E="${STAGE_E:-${HERE}/../../build}"
TIMEOUT="${TIMEOUT:-300}"
export BLAS_PERF_ITERS="${BLAS_PERF_ITERS:-200}"
export BLAS_PERF_TIME_BUDGET="${BLAS_PERF_TIME_BUDGET:-0.3}"
export BLAS_PERF_WARMUP="${BLAS_PERF_WARMUP:-2}"
REPS="${REPS:-11}"
PAR_DIR="${STAGE_E}/tests/epblas-parallel"
EP_DIR="${STAGE_E}/tests/epblas-openblas"
# Benchmark working files (raw sweeps, meta, scoreboard) live in the gitignored
# scratch tree; the tracked harness scripts stay in reports/cmp5/.
OUT_DIR="${OUT_DIR:-${HERE}/../../workspace/files/gap5}"
mkdir -p "$OUT_DIR"
RAW="${RAW:-${OUT_DIR}/gap5_raw.tsv}"
LOG="${LOG:-${OUT_DIR}/gap5.log}"
META="${RAW%.tsv}.meta"
ROUTINES="${ROUTINES:?set ROUTINES='mtrmv ...'}"

# ---- frequency control (best-effort; needs root to actually pin) -----------
# Pin every online CPU's governor to `performance` and disable turbo so the
# floor is a single fixed frequency. If the sysfs knobs aren't writable (the
# usual unprivileged case) we DON'T fail — we record freq_pinned=0 so agg knows
# to lean on the per-rep ratio (which cancels frequency) rather than absolutes.
FREQ_PINNED=0
declare -a _SAVED_GOV=()
_turbo_path=""; _turbo_saved=""
freq_restore() {
    local i=0 cpu
    for cpu in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
        [[ -w "$cpu" && -n "${_SAVED_GOV[$i]:-}" ]] && echo "${_SAVED_GOV[$i]}" > "$cpu" 2>/dev/null
        ((i++))
    done
    [[ -n "$_turbo_path" && -w "$_turbo_path" && -n "$_turbo_saved" ]] && echo "$_turbo_saved" > "$_turbo_path" 2>/dev/null
}
freq_pin() {
    local ok=1 i=0 cpu
    for cpu in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor; do
        if [[ -w "$cpu" ]]; then
            _SAVED_GOV[$i]="$(cat "$cpu")"; echo performance > "$cpu" 2>/dev/null || ok=0
        else ok=0; fi
        ((i++))
    done
    # turbo: intel_pstate/no_turbo (1=off) or cpufreq/boost (0=off)
    if [[ -e /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
        _turbo_path=/sys/devices/system/cpu/intel_pstate/no_turbo
        if [[ -w "$_turbo_path" ]]; then _turbo_saved="$(cat "$_turbo_path")"; echo 1 > "$_turbo_path" 2>/dev/null || ok=0; else ok=0; fi
    elif [[ -e /sys/devices/system/cpu/cpufreq/boost ]]; then
        _turbo_path=/sys/devices/system/cpu/cpufreq/boost
        if [[ -w "$_turbo_path" ]]; then _turbo_saved="$(cat "$_turbo_path")"; echo 0 > "$_turbo_path" 2>/dev/null || ok=0; else ok=0; fi
    fi
    FREQ_PINNED=$ok
    if (( ok )); then
        trap freq_restore EXIT INT TERM
        echo "[freq] PINNED: governor=performance, turbo=off (restore on exit)" >&2
    else
        echo "[freq] NOT pinned (sysfs not writable). For exact absolutes, run once:" >&2
        echo "[freq]   ! sudo cpupower frequency-set -g performance && echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo" >&2
        echo "[freq] Proceeding unprivileged: per-rep ratio cancels frequency, absolutes are turbo-variant." >&2
    fi
}
freq_pin

# ---- relink the exact binaries this sweep runs BEFORE timing ----------------
# Editing a source and rebuilding only the library leaves perf_* linked against
# the stale archive (CMake relinks a library's consumers only when you build the
# consumer target). Building perf_$r / ep_perf_$r pulls the rebuilt archive in,
# so the sweep can't time a stale binary. SKIP_BUILD=1 times a prebuilt state.
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

# ---- frequency warmup: ramp to steady state before the first timed call -----
# A cold core ramps turbo over the first ~tens of ms; without a warmup the
# earliest reps read a different frequency than the rest. Burn ~0.4s on cpu0.
taskset -c 0 bash -c 'a=0; for ((i=0;i<30000000;i++)); do ((a+=i)); done' >/dev/null 2>&1 || true

: > "$RAW"; : > "$LOG"
{
    echo "freq_pinned=$FREQ_PINNED"
    echo "reps=$REPS"
    echo "iters=$BLAS_PERF_ITERS time_budget=$BLAS_PERF_TIME_BUDGET warmup=$BLAS_PERF_WARMUP"
    echo "routines=$ROUTINES"
} > "$META"
echo -e "run_id\tomp\troutine\tkey\tsize\titers\tsubject_ns\tmigrated_ns\trep" >> "$RAW"
run_one() {
    local run_id="$1" omp="$2" bin="$3" routine="$4" rep="$5"
    local TMP; TMP=$(mktemp)
    local cpulist="0"; (( omp > 1 )) && cpulist="0-$((omp - 1))"
    local status=0
    OMP_NUM_THREADS="$omp" timeout "$TIMEOUT" taskset -c "$cpulist" "$bin" \
        > "$TMP" 2>>"$LOG" || status=$?
    (( status != 0 )) && echo "[partial] $run_id/$routine rep=$rep exit=$status" >> "$LOG"
    awk -v rid="$run_id" -v omp="$omp" -v rep="$rep" '
        /^#/ {next}
        NF >= 6 { printf "%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\n",
            rid, omp, $1, $2, $3, $4, $5, $6, rep; }' "$TMP" >> "$RAW"
    rm -f "$TMP"
}
for rep in $(seq 1 "$REPS"); do
    echo "[rep] $rep/$REPS" >&2
    for routine in $ROUTINES; do
        par_bin="$PAR_DIR/perf_${routine}"; ep_bin="$EP_DIR/ep_perf_${routine}"
        # Order pairs the arms agg compares (par1|ob1, par4|ob4) back-to-back so
        # each pair runs at ~the same frequency within the rep.
        [[ -x "$par_bin" ]] && run_one "par-omp1" 1 "$par_bin" "$routine" "$rep"
        [[ -x "$ep_bin"  ]] && run_one "ob-omp1"  1 "$ep_bin"  "$routine" "$rep"
        [[ -x "$par_bin" ]] && run_one "par-omp4" 4 "$par_bin" "$routine" "$rep"
        [[ -x "$ep_bin"  ]] && run_one "ob-omp4"  4 "$ep_bin"  "$routine" "$rep"
    done
done
echo "wrote $RAW ($REPS reps, freq_pinned=$FREQ_PINNED); meta $META"
