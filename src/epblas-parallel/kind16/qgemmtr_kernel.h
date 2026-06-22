/*
 * qgemmtr_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 qgemmtr overlay is split across:
 *
 *   qgemmtr_serial.c    The pure single-thread triangular GEMM update (no
 *                       OpenMP). Owns the per-column compute cores and the
 *                       trans-char decode, plus the public `qgemmtr_serial_`
 *                       Fortran entry. Safe to call from inside another
 *                       routine's own parallel region.
 *
 *   qgemmtr_parallel.c  The public Fortran entry `qgemmtr_` — threading
 *                       orchestration only. Fans the triangular column range
 *                       across an OpenMP team (per-column static schedule);
 *                       falls back to serial when called from inside a
 *                       parallel region or below the threading threshold.
 *
 * kind16 is arithmetic-bound (__float128 lowers to libquadmath calls), so the
 * overlay uses the unblocked reference DGEMMTR algorithm with no packing. Each
 * column j of C is computed independently and writes only the UPLO triangle,
 * so per-column static scheduling is race-free and bitwise-identical to the
 * serial sweep. The shared surface is just the trans decode and the two
 * per-column cores: a beta-only pass (used on the alpha==0 / K==0 early exit)
 * and the full compute pass (which folds its own beta handling inline).
 */
#ifndef EPBLAS_PARALLEL_KIND16_QGEMMTR_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_QGEMMTR_KERNEL_H

#include <stddef.h>

typedef __float128 qgemmtr_T;

/* Normalize a Fortran trans char to a code ('C' ≡ 'T' for real input). */
ptrdiff_t qgemmtr_trans_code(const char *p);

/* Beta-only pass over columns [j0,j1): the body run on the alpha==0 / K==0
 * early-exit path. Scales (or zeros) the UPLO triangle of each column by beta.
 * Only reached when beta != one. */
void qgemmtr_beta_core(
    ptrdiff_t j0, ptrdiff_t j1, ptrdiff_t N, int upper,
    qgemmtr_T beta,
    qgemmtr_T *c, ptrdiff_t ldc);

/* Full compute pass over columns [j0,j1):
 *
 *   C[:,j] = beta * C[:,j] + alpha * op(A) * op(B)[:,j]   (UPLO triangle only)
 *
 * Folds the per-column beta scaling inline, matching the original loop body.
 * Each column is owned by exactly one call, so the partition is race-free. */
void qgemmtr_compute_core(
    ptrdiff_t j0, ptrdiff_t j1, ptrdiff_t N, int upper, ptrdiff_t K,
    int ta, int tb,
    qgemmtr_T alpha, qgemmtr_T beta,
    const qgemmtr_T *a, ptrdiff_t lda,
    const qgemmtr_T *b, ptrdiff_t ldb,
    qgemmtr_T *c, ptrdiff_t ldc);

/* Pure-serial by-value entry. No OpenMP anywhere on this call path; safe to
 * invoke from inside another function's `#pragma omp parallel` region. Shares
 * the ptrdiff_t core ABI of qgemmtr_core. */
void qgemmtr_serial(
    char uplo, char transa, char transb,
    ptrdiff_t N, ptrdiff_t K,
    const qgemmtr_T *alpha_,
    const qgemmtr_T *a, ptrdiff_t lda,
    const qgemmtr_T *b, ptrdiff_t ldb,
    const qgemmtr_T *beta_,
    qgemmtr_T *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND16_QGEMMTR_KERNEL_H */
