/*
 * xgemmtr_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 xgemmtr overlay is split across:
 *
 *   xgemmtr_serial.c    The pure single-thread complex triangular GEMM update
 *                       (no OpenMP). Owns the per-column compute cores and the
 *                       trans-char decode, plus the public `xgemmtr_serial_`
 *                       Fortran entry. Safe to call from inside another
 *                       routine's own parallel region.
 *
 *   xgemmtr_parallel.c  The public Fortran entry `xgemmtr_` — threading
 *                       orchestration only. Fans the triangular column range
 *                       across an OpenMP team (per-column static schedule);
 *                       falls back to serial when called from inside a
 *                       parallel region or below the threading threshold.
 *
 * kind16 is arithmetic-bound (__complex128 lowers to libquadmath calls), so
 * the overlay uses the unblocked reference ZGEMMTR algorithm with no packing.
 * Conjugation under 'C' is applied at element access. Each column j of C is
 * computed independently and writes only the UPLO triangle, so per-column
 * static scheduling is race-free and bitwise-identical to the serial sweep.
 * The shared surface is just the trans decode and the two per-column cores: a
 * beta-only pass (used on the alpha==0 / K==0 early exit) and the full compute
 * pass (which folds its own beta handling inline).
 */
#ifndef EPBLAS_PARALLEL_KIND16_XGEMMTR_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XGEMMTR_KERNEL_H

#include <stddef.h>
#include <quadmath.h>   /* __complex128 */

typedef __complex128 xgemmtr_T;

/* Normalize a Fortran trans char to a code (upper-cased; 'C' kept distinct
 * from 'T' so conjugation can be applied for complex input). */
int xgemmtr_trans_code(const char *p);

/* Beta-only pass over columns [j0,j1): the body run on the alpha==0 / K==0
 * early-exit path. Scales (or zeros) the UPLO triangle of each column by beta.
 * Only reached when beta != one. */
void xgemmtr_beta_core(
    int j0, int j1, int N, int upper,
    xgemmtr_T beta,
    xgemmtr_T *c, int ldc);

/* Full compute pass over columns [j0,j1):
 *
 *   C[:,j] = beta * C[:,j] + alpha * op(A) * op(B)[:,j]   (UPLO triangle only)
 *
 * Folds the per-column beta scaling inline, matching the original loop body.
 * trans_a/trans_b select axpy vs. inner-product form; conj_a/conj_b apply
 * conjugation at element access. Each column is owned by exactly one call, so
 * the partition is race-free. */
void xgemmtr_compute_core(
    int j0, int j1, int N, int upper, int K,
    int trans_a, int trans_b, int conj_a, int conj_b,
    xgemmtr_T alpha, xgemmtr_T beta,
    const xgemmtr_T *a, int lda,
    const xgemmtr_T *b, int ldb,
    xgemmtr_T *c, int ldc);

/* Pure-serial Fortran entry. No OpenMP anywhere on this call path; safe to
 * invoke from inside another function's `#pragma omp parallel` region. Keeps
 * the exact Fortran-ABI signature of xgemmtr_. */
void xgemmtr_serial_(
    const char *uplo, const char *transa, const char *transb,
    const int *n_, const int *k_,
    const xgemmtr_T *alpha_,
    const xgemmtr_T *a, const int *lda_,
    const xgemmtr_T *b, const int *ldb_,
    const xgemmtr_T *beta_,
    xgemmtr_T *c, const int *ldc_,
    size_t uplo_len, size_t ta_len, size_t tb_len);

#endif /* EPBLAS_PARALLEL_KIND16_XGEMMTR_KERNEL_H */
