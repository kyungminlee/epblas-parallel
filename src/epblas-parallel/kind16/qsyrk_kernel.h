/*
 * qsyrk_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 qsyrk overlay is split across:
 *
 *   qsyrk_serial.c    The pure single-thread symmetric rank-k update (no
 *                     OpenMP). Owns the per-column compute core, the
 *                     uplo/trans decode, and the public `qsyrk_serial_`
 *                     entry. Safe to invoke from inside another routine's
 *                     own `#pragma omp parallel` region.
 *
 *   qsyrk_parallel.c  The public Fortran entry `qsyrk_` — threading
 *                     orchestration only. Fans the independent columns of C
 *                     across an OpenMP team; falls back to running the loop
 *                     serially when invoked from inside a parallel region.
 *
 * kind16 is arithmetic-bound (__float128 lowers to libquadmath calls), so
 * the overlay uses the unblocked Netlib reference (DSYRK shape) with no
 * packing — the shared surface is the decode plus the per-column core.
 *
 * Each column j of C touches a distinct UPLO-triangle slice [i_lo,i_hi) and
 * reads only A, so columns are fully independent — per-column static
 * scheduling is bitwise-identical to the serial sweep.
 */
#ifndef EPBLAS_PARALLEL_KIND16_QSYRK_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_QSYRK_KERNEL_H

#include <stddef.h>

typedef __float128 qsyrk_T;

/* Decode a Fortran uplo char to upper-cased 'U'/'L'. */
char qsyrk_uplo(const char *p);

/* Decode a Fortran trans char to a code ('C' ≡ 'T' for real input). */
int qsyrk_trans_code(const char *p, size_t len);

/* Compute columns [j0,j1) of C. The body of each j-iteration is exactly one
 * iteration of the original omp-parallel-for: the beta scaling of column j's
 * UPLO slice [i_lo,i_hi) followed by the rank-k (TR=='N') or inner-product
 * (TR=='T') accumulation. No OpenMP pragmas — pure sequential per-column
 * work; callers partition the column range if they want parallelism. */
void qsyrk_core(
    int j0, int j1,
    char UPLO, int TR, int N, int K,
    qsyrk_T alpha, qsyrk_T beta,
    const qsyrk_T *a, int lda,
    qsyrk_T *c, int ldc);

/* Pure-serial Fortran entry. No OpenMP anywhere on this call path. */
void qsyrk_serial_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const qsyrk_T *alpha_,
    const qsyrk_T *a, const int *lda_,
    const qsyrk_T *beta_,
    qsyrk_T *c, const int *ldc_,
    size_t uplo_len, size_t trans_len);

#endif /* EPBLAS_PARALLEL_KIND16_QSYRK_KERNEL_H */
