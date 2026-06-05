/*
 * xsymm_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 xsymm overlay is split across:
 *
 *   xsymm_serial.c    The pure single-thread complex symmetric multiply (no
 *                     OpenMP). Owns the per-column compute core, the uplo-char
 *                     decode, and the public `xsymm_serial_` entry. Called by
 *                     xsymm_ as its serial branch and usable from inside
 *                     another routine's own parallel region.
 *
 *   xsymm_parallel.c  The public Fortran entry `xsymm_` — threading
 *                     orchestration only (omp-parallel-for over columns of C).
 *                     Delegates to the serial core per column; runs serially
 *                     when invoked from inside a parallel region.
 *
 * NOT Hermitian — no conjugate. For Hermitian see xhemm. kind16 is
 * arithmetic-bound (__complex128 lowers to libquadmath calls), so the overlay
 * uses the unblocked reference algorithm. Each column j of C is independent,
 * so a per-column static partition is bitwise-identical to the serial sweep.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XSYMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XSYMM_KERNEL_H

#include <stddef.h>
#include <quadmath.h>   /* __complex128 */

typedef __complex128 xsymm_T;

/* Normalize a Fortran uplo/side char to its uppercase code. */
char xsymm_uplo(const char *p);

/* Compute columns [j0,j1) of C — each column is one iteration of the
 * reference omp loop: beta-scale C[:,j] then accumulate the symmetric
 * matvec for the given SIDE/UPLO. No OpenMP pragmas; each column is owned
 * by exactly one caller, so the work is race-free under a column partition. */
void xsymm_core(
    int j0, int j1,
    char SIDE, char UPLO,
    int M, int N,
    xsymm_T alpha, xsymm_T beta,
    const xsymm_T *a, int lda,
    const xsymm_T *b, int ldb,
    xsymm_T *c, int ldc);

/* Pure-serial Fortran entry. No OpenMP anywhere on this call path; safe to
 * invoke from inside another function's `#pragma omp parallel` region. Keeps
 * the exact Fortran-ABI signature of xsymm_. */
void xsymm_serial_(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const xsymm_T *alpha_,
    const xsymm_T *a, const int *lda_,
    const xsymm_T *b, const int *ldb_,
    const xsymm_T *beta_,
    xsymm_T *c, const int *ldc_,
    size_t side_len, size_t uplo_len);

#endif /* EPBLAS_PARALLEL_KIND16_XSYMM_KERNEL_H */
