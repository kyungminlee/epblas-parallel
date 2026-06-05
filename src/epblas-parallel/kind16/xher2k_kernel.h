/*
 * xher2k_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 xher2k overlay is split across:
 *
 *   xher2k_serial.c    The pure single-thread Hermitian rank-2k (no OpenMP).
 *                      Owns the uplo decode and the per-column compute core,
 *                      plus the public `xher2k_serial_` entry. Safe to call
 *                      from inside another routine's parallel region.
 *
 *   xher2k_parallel.c  The public Fortran entry `xher2k_` — threading
 *                      orchestration only (one omp-parallel-for over the
 *                      columns of C). Each column is independent, so the
 *                      static per-column partition is race-free and
 *                      bitwise-identical to the serial path.
 *
 * alpha is COMPLEX, beta is REAL, and the diagonal of C stays real (the
 * imaginary part is dropped at every touch — see xher2k_core). kind16 is
 * arithmetic-bound (__complex128 lowers to libquadmath calls), so the overlay
 * uses the unblocked reference with no packing — the shared surface is the
 * uplo decode and the per-column compute core.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XHER2K_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XHER2K_KERNEL_H

#include <stddef.h>
#include <quadmath.h>   /* __complex128 */

typedef __complex128 xher2k_TC;
typedef __float128   xher2k_TR;

/* Uppercase a Fortran uplo/trans char (de-static'd original `up`). */
char xher2k_uplo(const char *p);

/* Compute columns [j0,j1) of C — full per-column rank-2k work: the beta
 * scaling of the UPLO triangle slice (with the Hermitian diagonal kept real)
 * followed by both rank-update terms (TRANS branch handled inside, exactly as
 * the original omp loop body). The diagonal element (i == j) takes only the
 * real part of its rank-update contribution. Each column is independent, so
 * any [j0,j1) sub-range is race-free.
 *
 * UPLO is the decoded uplo char ('L'/'U'); TR is the decoded trans char
 * ('N' or 'C'). alpha_conj is conjq(alpha). Caller guarantees
 * alpha != 0 && K > 0. */
void xher2k_core(
    int j0, int j1, int N, int K,
    char UPLO, char TR,
    xher2k_TC alpha, xher2k_TC alpha_conj, xher2k_TR beta,
    const xher2k_TC *a, int lda,
    const xher2k_TC *b, int ldb,
    xher2k_TC *c, int ldc);

/* Pure-serial Fortran entry. No OpenMP on this call path; keeps the exact
 * Fortran-ABI signature of xher2k_ so callers already inside a parallel
 * region can swap the symbol name only. */
void xher2k_serial_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const xher2k_TC *alpha_,
    const xher2k_TC *restrict a, const int *lda_,
    const xher2k_TC *restrict b, const int *ldb_,
    const xher2k_TR *beta_,
    xher2k_TC *restrict c, const int *ldc_,
    size_t uplo_len, size_t trans_len);

#endif /* EPBLAS_PARALLEL_KIND16_XHER2K_KERNEL_H */
