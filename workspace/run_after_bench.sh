#!/usr/bin/env bash
# After-refactor kind16 L3 bench — IDENTICAL params to the baseline capture
# (workspace/refactor_bench_before.txt). Overlay ns/call vs migrated ns/call.
set -u
cd /home/kyungminlee/code/epblas-parallel/build/tests/epblas-parallel
OUT=/home/kyungminlee/code/epblas-parallel/workspace/refactor_bench_after.txt
: > "$OUT"
bins=$(ls perf_q* perf_x* 2>/dev/null | grep -vE '_blocked$' \
       | grep -E 'perf_(q|x)(gemm|gemmtr|symm|hemm|syrk|herk|syr2k|her2k|trmm|trsm)$')
for omp in 1 4; do
  for b in $bins; do
    echo "### $b OMP=$omp" >> "$OUT"
    OMP_NUM_THREADS=$omp BLAS_PERF_SIZES=128,256 BLAS_PERF_ITERS=4 BLAS_PERF_WARMUP=1 \
        timeout 180 ./$b 2>>"$OUT" | grep -vE '^#' >> "$OUT"
  done
done
echo "=== after-bench captured ===" >> "$OUT"
