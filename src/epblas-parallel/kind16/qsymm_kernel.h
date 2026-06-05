/*
 * qsymm_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 qsymm overlay is split across:
 *
 *   qsymm_serial.c    The pure single-thread symmetric multiply (no OpenMP).
 *                     Owns the per-column compute core, the uplo-char decode,
 *                     and the public `qsymm_serial_` entry. Called by qsymm_
 *                     as its serial branch and usable from inside another
 *                     routine's own parallel region.
 *
 *   qsymm_parallel.c  The public Fortran entry `qsymm_` — threading
 *                     orchestration only (omp-parallel-for over columns of C).
 *                     Delegates to the serial core per column; runs serially
 *                     when invoked from inside a parallel region.
 *
 * kind16 is arithmetic-bound (__float128 lowers to libquadmath calls), so the
 * overlay uses the unblocked reference algorithm — the shared surface is just
 * the uplo decode and the per-column compute core. Each column j of C is
 * independent, so a per-column static partition is bitwise-identical to the
 * serial sweep.
 */
#ifndef EPBLAS_PARALLEL_KIND16_QSYMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_QSYMM_KERNEL_H

#include <stddef.h>

typedef __float128 qsymm_T;

/* Normalize a Fortran uplo/side char to its uppercase code. */
char qsymm_uplo(const char *p);

/* Compute columns [j0,j1) of C — each column is one iteration of the
 * reference omp loop: beta-scale C[:,j] then accumulate the symmetric
 * matvec for the given SIDE/UPLO. No OpenMP pragmas; each column is owned
 * by exactly one caller, so the work is race-free under a column partition. */
void qsymm_core(
    int j0, int j1,
    char SIDE, char UPLO,
    int M, int N,
    qsymm_T alpha, qsymm_T beta,
    const qsymm_T *a, int lda,
    const qsymm_T *b, int ldb,
    qsymm_T *c, int ldc);

/* Pure-serial Fortran entry. No OpenMP anywhere on this call path; safe to
 * invoke from inside another function's `#pragma omp parallel` region. Keeps
 * the exact Fortran-ABI signature of qsymm_. */
void qsymm_serial_(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const qsymm_T *alpha_,
    const qsymm_T *a, const int *lda_,
    const qsymm_T *b, const int *ldb_,
    const qsymm_T *beta_,
    qsymm_T *c, const int *ldc_,
    size_t side_len, size_t uplo_len);

#endif /* EPBLAS_PARALLEL_KIND16_QSYMM_KERNEL_H */
