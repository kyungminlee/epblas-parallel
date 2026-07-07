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
  e) LIB=eyblas; CC=gcc; EXT=c;   LINK="-lgfortran -lm";            REF=kind10 ;;
  q) LIB=qxblas; CC=gcc; EXT=c;   LINK="-lgfortran -lquadmath -lm"; REF=kind16 ;;
  m) LIB=mwblas; CC=g++; EXT=cpp; LINK="-lgfortran -lquadmath -lm"; REF=multifloats ;;
  *) echo "unknown family '$FAM' (want e|q|m)" >&2; exit 2 ;;
esac

shopt -s nullglob
# lsame_/xerbla_ (referenced as bare externals by the mig leg's netlib objects)
# live in libblas-gfortran, NOT the per-precision lib${LIB} reference.
refblas=( "$BUILD/_deps/eplinalg-$REF/lib/libblas-gfortran-"*.a )
[[ ${#refblas[@]} -ge 1 ]] || { echo "refblas (libblas-gfortran) not found under build/_deps/eplinalg-$REF/lib" >&2; exit 1; }
REFAS=( "${refblas[0]}" )
# The multifloats mig leg is gfortran netlib that USES the Fortran `multifloats`
# module (double-double ops __multifloats_MOD_dd_*), which in turn calls the
# C++ companions (fmoddd/matmuldd_*). Both live ONLY in the `_mf` subproject's
# archives — built with gcc-15 + fat-LTO (LTO 15.1), so they cannot be linked
# into our gcc-16 driver (lto1 rejects the bytecode-version mismatch). We can't
# use build/_mf/{fsrc/libmultifloatsf.a,src/libmultifloats-lto-gcc-15.a} directly.
# Instead we recompile that runtime NON-LTO from the same generated sources
# (build_mf_runtime, below) into a private archive and link THAT.
if [[ "$FAM" == m ]]; then
  MFRT="$NSDIR/lib_mfrt_nolto.a"
  REFAS+=( "$MFRT" )
fi

# Recompile the multifloats dd-module + C++ companions NON-LTO so the gcc-16
# driver can link the gfortran-15-pinned `_mf` runtime. Compilers default to the
# -15 toolchain that produced the reference (keeps the mig leg byte-faithful);
# override via MF_FC / MF_CXX.
build_mf_runtime() {
  local fc="${MF_FC:-gfortran-15}" cxx="${MF_CXX:-g++-15}"
  local fsrc="$BUILD/_mf/fsrc/generated/multifloats-quad.f90"
  local csrc="$BUILD/_deps/multifloats_fetch-src/src"
  [[ -f "$fsrc" ]] || { echo "multifloats module source not found: $fsrc" >&2; exit 1; }
  [[ -f "$csrc/multifloats_math.cc" ]] || { echo "multifloats companion source not found under $csrc" >&2; exit 1; }
  local tmp="$NSDIR/_mfrt"; mkdir -p "$tmp"
  "$fc"  -O3 -fno-lto -fPIC -J"$tmp" -c "$fsrc"                   -o "$tmp/mfquad.o"
  "$cxx" -O3 -fno-lto -fPIC -I"$csrc/../include" -I"$csrc" -c "$csrc/multifloats_math.cc" -o "$tmp/mfmath.o"
  "$cxx" -O3 -fno-lto -fPIC -I"$csrc/../include" -I"$csrc" -c "$csrc/multifloats_io.cc"   -o "$tmp/mfio.o"
  rm -f "$MFRT"; ar rcs "$MFRT" "$tmp/mfquad.o" "$tmp/mfmath.o" "$tmp/mfio.o"
}

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
  echo "### [1/6] build par/ob/mig archives (anti-stale) ###"
  cmake --build "$BUILD" --target "${LIB}_parallel" "${LIB}_openblas" "${LIB}_migrated_build"
  echo "### [2/6] namespace archives ###"
  OUT="$NSDIR" BUILD="$BUILD" "$HERE/nsbuild.sh" "$FAM"
  if [[ "$FAM" == m ]]; then
    echo "### [2b/6] recompile multifloats runtime non-LTO ###"
    build_mf_runtime
  fi
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
  "$CC" "${CFLAGS[@]}" "$src" "$PAR" "$OB" "$MIG" "${REFAS[@]}" $LINK -o "$bin"
  echo "### [5/6] run $r  (reps=$REPS, omp1 core $CORE1 / omp4 cores $CORE4) ###"
  OMP_NUM_THREADS=1 taskset -c "$CORE1" "$bin" "$REPS" > "$OUT/$r.omp1.txt"
  OMP_NUM_THREADS=4 taskset -c "$CORE4" "$bin" "$REPS" > "$OUT/$r.omp4.txt"
done

if [[ "${NOAGG:-0}" != 1 ]]; then
  echo "### [6/6] scoreboard ###"
  python3 "$HERE/agg_dual.py" "$OUT"
fi
echo "done -> $OUT"
