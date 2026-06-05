/*
 * xsyr2k_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 xsyr2k overlay is split across:
 *
 *   xsyr2k_serial.c    The pure single-thread complex symmetric rank-2k (no
 *                      OpenMP). Owns the uplo decode and the per-column
 *                      compute core, plus the public `xsyr2k_serial_` entry.
 *                      Safe to call from inside another routine's parallel
 *                      region.
 *
 *   xsyr2k_parallel.c  The public Fortran entry `xsyr2k_` — threading
 *                      orchestration only (one omp-parallel-for over the
 *                      columns of C). Each column is independent, so the
 *                      static per-column partition is race-free and
 *                      bitwise-identical to the serial path.
 *
 * C is N×N complex symmetric (NOT Hermitian). kind16 is arithmetic-bound
 * (__complex128 lowers to libquadmath calls), so the overlay uses the
 * unblocked Netlib reference with no packing — the shared surface is just the
 * uplo decode and the per-column compute core.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XSYR2K_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XSYR2K_KERNEL_H

#include <stddef.h>
#include <quadmath.h>   /* __complex128 */

typedef __complex128 xsyr2k_T;

/* Uppercase a Fortran uplo/trans char (de-static'd original `up`). */
char xsyr2k_uplo(const char *p);

/* Compute columns [j0,j1) of C — full per-column rank-2k work: the beta
 * scaling of the UPLO triangle slice followed by both rank-update terms
 * (TRANS branch handled inside, exactly as the original omp loop body).
 * Each column is independent, so any [j0,j1) sub-range is race-free.
 *
 * UPLO is the decoded uplo char ('L'/'U'); TR is the decoded trans char
 * ('N' or 'T', with 'C' already folded to 'T'). Caller guarantees
 * alpha != 0 && K > 0. */
void xsyr2k_core(
    int j0, int j1, int N, int K,
    char UPLO, char TR,
    xsyr2k_T alpha, xsyr2k_T beta,
    const xsyr2k_T *a, int lda,
    const xsyr2k_T *b, int ldb,
    xsyr2k_T *c, int ldc);

/* Pure-serial Fortran entry. No OpenMP on this call path; keeps the exact
 * Fortran-ABI signature of xsyr2k_ so callers already inside a parallel
 * region can swap the symbol name only. */
void xsyr2k_serial_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const xsyr2k_T *alpha_,
    const xsyr2k_T *restrict a, const int *lda_,
    const xsyr2k_T *restrict b, const int *ldb_,
    const xsyr2k_T *beta_,
    xsyr2k_T *restrict c, const int *ldc_,
    size_t uplo_len, size_t trans_len);

#endif /* EPBLAS_PARALLEL_KIND16_XSYR2K_KERNEL_H */
