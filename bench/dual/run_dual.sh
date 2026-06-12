#!/usr/bin/env bash
# run_dual.sh — end-to-end dual-link (option-2) perf sweep for one precision family.
#
# Pipeline (all in-process, single-binary, par/ob/mig timed interleaved per rep):
#   1. (re)build the par/ob/mig CMake archives           [defeats the stale-archive trap]
#   2. namespace them into lib_{par,ob,mig}_ns.a          [bench/dual/nsbuild.sh]
#   3. generate one dual driver per routine               [scripts/gen_dual_harnesses.py]
#   4. compile each driver against the ns archives + refblas (lsame_/xerbla_)
#   5. run OMP=1 (core 2) and OMP=4 (cores 2-5) at REPS, writing <rt>.omp{1,4}.txt
#   6. aggregate into the scoreboard                      [bench/dual/agg_dual.py]
#
# Usage:
#   bench/dual/run_dual.sh <family> [routine,routine,...]
#     family   : e | q | m   (kind10 | kind16 | multifloats)
#     routines : comma list (default: ALL routines the generator supports for the family)
# Env: REPS (default 40), CORE1 (default 2), CORE4 (default 2-5),
#      OUT (results dir, default workspace/files/gap5/nsbench/results),
#      SKIP_BUILD=1 to reuse existing archives, NOAGG=1 to skip the scoreboard.
#
# Raw data lands in the gitignored workspace; never under bench/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
FAM="${1:?usage: run_dual.sh <e|q|m> [routine,...]}"
ROUTINES_CSV="${2:-}"
BUILD="${BUILD:-$ROOT/build}"
NSDIR="${NSDIR:-$ROOT/workspace/files/gap5/nsbench}"
OUT="${OUT:-$NSDIR/results}"
DRV="$NSDIR/drivers"
REPS="${REPS:-40}"
CORE1="${CORE1:-2}"
CORE4="${CORE4:-2-5}"
mkdir -p "$OUT" "$DRV"

case "$FAM" in
  e) LIB=eblas; CC=gcc; EXT=c;   LINK="-lgfortran -lm";            REF=kind10 ;;
  q) LIB=qblas; CC=gcc; EXT=c;   LINK="-lgfortran -lquadmath -lm"; REF=kind16 ;;
  m) LIB=mblas; CC=g++; EXT=cpp; LINK="-lgfortran -lquadmath -lm"; REF=multifloats ;;
  *) echo "unknown family '$FAM' (want e|q|m)" >&2; exit 2 ;;
esac

shopt -s nullglob
refblas=( "$BUILD/_deps/eplinalg-$REF/lib/lib${LIB}"*-gfortran-*.a "$BUILD/_deps/eplinalg-$REF/lib/libblas-gfortran-"*.a )
[[ ${#refblas[@]} -ge 1 ]] || { echo "refblas archive not found under build/_deps/eplinalg-$REF/lib" >&2; exit 1; }
REFA="${refblas[0]}"

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
  echo "### [1/6] build par/ob/mig archives (anti-stale) ###"
  cmake --build "$BUILD" --target "${LIB}_parallel" "${LIB}_openblas" "${LIB}_migrated_build"
  echo "### [2/6] namespace archives ###"
  OUT="$NSDIR" BUILD="$BUILD" "$HERE/nsbuild.sh" "$FAM"
fi
PAR="$NSDIR/lib_par_ns.a"; OB="$NSDIR/lib_ob_ns.a"; MIG="$NSDIR/lib_mig_ns.a"
for a in "$PAR" "$OB" "$MIG"; do [[ -f "$a" ]] || { echo "missing $a (run without SKIP_BUILD)" >&2; exit 1; }; done

echo "### [3/6] generate drivers ###"
# --family keeps the ALL-routines sweep single-family (e and q both emit .c).
GENARGS=(--family "$FAM" --outdir "$DRV")
[[ -n "$ROUTINES_CSV" ]] && GENARGS+=(--routines "$ROUTINES_CSV")
GENLIST="$(python3 "$ROOT/scripts/gen_dual_harnesses.py" "${GENARGS[@]}" --list)"

# Which routines to run: explicit list, or exactly what the generator emitted.
if [[ -n "$ROUTINES_CSV" ]]; then
  ROUTINES="${ROUTINES_CSV//,/ }"
else
  ROUTINES="$GENLIST"
fi

CFLAGS=(-O3 -DNDEBUG -ffp-contract=fast -march=native -fopenmp)
for r in $ROUTINES; do
  src="$DRV/dual_$r.$EXT"
  [[ -f "$src" ]] || { echo "  skip $r (no driver $src)"; continue; }
  bin="$DRV/dual_$r"
  echo "### [4/6] compile $r ###"
  "$CC" "${CFLAGS[@]}" "$src" "$PAR" "$OB" "$MIG" "$REFA" $LINK -o "$bin"
  echo "### [5/6] run $r  (reps=$REPS, omp1 core $CORE1 / omp4 cores $CORE4) ###"
  OMP_NUM_THREADS=1 taskset -c "$CORE1" "$bin" "$REPS" > "$OUT/$r.omp1.txt"
  OMP_NUM_THREADS=4 taskset -c "$CORE4" "$bin" "$REPS" > "$OUT/$r.omp4.txt"
done

if [[ "${NOAGG:-0}" != 1 ]]; then
  echo "### [6/6] scoreboard ###"
  python3 "$HERE/agg_dual.py" "$OUT"
fi
echo "done -> $OUT"
