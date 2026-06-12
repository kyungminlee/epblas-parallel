#!/usr/bin/env bash
# nsbuild.sh — build namespaced par/ob/mig static archives for one precision
# family, so all three can be linked into ONE dual-link bench binary without the
# shared-helper symbol collisions the cross-process harness was built to dodge.
#
# Every GLOBALLY-DEFINED symbol in each source archive is renamed with a leg
# prefix (objcopy --redefine-syms); undefined externals are left alone so each
# leg keeps resolving its own libgfortran/libquadmath helpers independently.
#   par : <sym>            -> par_<sym>     (public entry qsyrk_  -> par_qsyrk_)
#   ob  : <sym>            -> ob_<sym>      (             qsyrk_  -> ob_qsyrk_)
#   mig : <sym>_migrated_  -> mig_<sym>_    (    qsyrk_migrated_  -> mig_qsyrk_)
#
# Output (gitignored workspace): $OUT/lib_{par,ob,mig}_ns.a + *_map.txt
#
# Usage:  bench/dual/nsbuild.sh <family>          # family in {e,q,m}
#   e -> kind10 (long double / _Complex long double)
#   q -> kind16 (__float128 / __complex128)
#   m -> multifloats (double-double)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
FAM="${1:?usage: nsbuild.sh <e|q|m>}"
BUILD="${BUILD:-$ROOT/build}"
OUT="${OUT:-$ROOT/workspace/files/gap5/nsbench}"
mkdir -p "$OUT"

case "$FAM" in
  e) PARG="epblas_parallel_kind10";      OBG="epblas_openblas_kind10";      LIB=eblas ;;
  q) PARG="epblas_parallel_kind16";      OBG="epblas_openblas_kind16";      LIB=qblas ;;
  m) PARG="epblas_parallel_multifloats"; OBG="epblas_openblas_multifloats"; LIB=mblas ;;
  *) echo "unknown family '$FAM' (want e|q|m)" >&2; exit 2 ;;
esac

# Resolve the three source archives (glob over the gfortran-NN version suffix).
shopt -s nullglob
par_src=( "$BUILD/src/epblas-parallel/$PARG/lib${LIB}_parallel"-gfortran-*.a )
ob_src=(  "$BUILD/src/epblas-openblas/$OBG/lib${LIB}_openblas"-gfortran-*.a )
mig_src="$BUILD/tests/lib${LIB}_migrated.a"
[[ ${#par_src[@]} -eq 1 ]] || { echo "par archive not unique: ${par_src[*]:-<none>}" >&2; exit 1; }
[[ ${#ob_src[@]}  -eq 1 ]] || { echo "ob archive not unique: ${ob_src[*]:-<none>}"  >&2; exit 1; }
[[ -f "$mig_src" ]]        || { echo "mig archive missing: $mig_src" >&2; exit 1; }

# Build a redefine-syms map from an archive's GLOBAL DEFINED symbols.
#   $1 archive  $2 prefix  $3 1=strip-_migrated_-and-re-suffix  $4 out map
gen_map() {
    local arch="$1" pfx="$2" mig="$3" out="$4"
    nm -g --defined-only "$arch" | awk -v p="$pfx" -v mig="$mig" '
        NF < 3 { next }                       # skip "<archive>:" / blank / undefined
        { s = $3
          if (mig == "1") { sub(/_migrated_$/, "_", s); print $3, p s }
          else            { print $3, p s } }' | sort -u > "$out"
}

ns_one() {
    local src="$1" pfx="$2" mig="$3"
    local map="$OUT/${pfx}map.txt" dst="$OUT/lib_${pfx%_}_ns.a"
    gen_map "$src" "$pfx" "$mig" "$map"
    cp -f "$src" "$dst"
    objcopy --redefine-syms="$map" "$dst"
    printf "  %-14s %4d syms -> %s\n" "$(basename "$src")" "$(wc -l < "$map")" "$(basename "$dst")"
}

echo "nsbuild family=$FAM ($LIB)  ->  $OUT"
ns_one "${par_src[0]}" "par_" 0
ns_one "${ob_src[0]}"  "ob_"  0
ns_one "$mig_src"      "mig_" 1
echo "done."
