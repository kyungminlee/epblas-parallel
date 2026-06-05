/*
 * xhemm_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 xhemm overlay is split across:
 *
 *   xhemm_serial.c    The pure single-thread Hermitian multiply (no OpenMP).
 *                     Owns the per-column compute core, the uplo-char decode,
 *                     and the public `xhemm_serial_` entry. Called by xhemm_
 *                     as its serial branch and usable from inside another
 *                     routine's own parallel region.
 *
 *   xhemm_parallel.c  The public Fortran entry `xhemm_` — threading
 *                     orchestration only (omp-parallel-for over columns of C).
 *                     Delegates to the serial core per column; runs serially
 *                     when invoked from inside a parallel region.
 *
 * Off-diagonal entries reflect via the Hermitian property
 * A(k,i) = conj(A(i,k)); the diagonal of A is real — only its real part is
 * read. kind16 is arithmetic-bound (__complex128 lowers to libquadmath
 * calls), so the overlay uses the unblocked reference algorithm (matches
 * Netlib ZHEMM column ordering). Each column j of C is independent, so a
 * per-column static partition is bitwise-identical to the serial sweep.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XHEMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XHEMM_KERNEL_H

#include <stddef.h>
#include <quadmath.h>   /* __complex128 */

typedef __complex128 xhemm_T;

/* Normalize a Fortran uplo/side char to its uppercase code. */
char xhemm_uplo(const char *p);

/* Compute columns [j0,j1) of C — each column is one iteration of the
 * reference omp loop: beta-scale C[:,j] then accumulate the Hermitian matvec
 * for the given SIDE/UPLO (conj reflection off-diagonal, real-part-only
 * diagonal). No OpenMP pragmas; each column is owned by exactly one caller,
 * so the work is race-free under a column partition. */
void xhemm_core(
    int j0, int j1,
    char SIDE, char UPLO,
    int M, int N,
    xhemm_T alpha, xhemm_T beta,
    const xhemm_T *a, int lda,
    const xhemm_T *b, int ldb,
    xhemm_T *c, int ldc);

/* Pure-serial Fortran entry. No OpenMP anywhere on this call path; safe to
 * invoke from inside another function's `#pragma omp parallel` region. Keeps
 * the exact Fortran-ABI signature of xhemm_. */
void xhemm_serial_(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const xhemm_T *alpha_,
    const xhemm_T *a, const int *lda_,
    const xhemm_T *b, const int *ldb_,
    const xhemm_T *beta_,
    xhemm_T *c, const int *ldc_,
    size_t side_len, size_t uplo_len);

#endif /* EPBLAS_PARALLEL_KIND16_XHEMM_KERNEL_H */
