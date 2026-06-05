/*
 * xsyrk_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 xsyrk overlay is split across:
 *
 *   xsyrk_serial.c    The pure single-thread complex symmetric rank-k update
 *                     (no OpenMP). Owns the per-column compute core, the
 *                     uplo/trans decode, and the public `xsyrk_serial_`
 *                     entry. Safe to invoke from inside another routine's
 *                     own `#pragma omp parallel` region.
 *
 *   xsyrk_parallel.c  The public Fortran entry `xsyrk_` — threading
 *                     orchestration only. Fans the independent columns of C
 *                     across an OpenMP team; falls back to running the loop
 *                     serially when invoked from inside a parallel region.
 *
 * C is N×N symmetric (NOT Hermitian — no conjugate); only the UPLO triangle
 * is referenced/written. kind16 complex is arithmetic-bound (__complex128
 * lowers to libquadmath calls), so the overlay uses the unblocked Netlib
 * reference with no packing — the shared surface is the decode plus the
 * per-column core.
 *
 * Each column j of C touches a distinct UPLO-triangle slice [i_lo,i_hi) and
 * reads only A, so columns are fully independent — per-column static
 * scheduling is bitwise-identical to the serial sweep.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XSYRK_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XSYRK_KERNEL_H

#include <stddef.h>
#include <quadmath.h>   /* __complex128 */

typedef __complex128 xsyrk_T;

/* Decode a Fortran uplo char to upper-cased 'U'/'L'. */
char xsyrk_uplo(const char *p);

/* Decode a Fortran trans char to upper-cased 'N'/'T'. */
char xsyrk_trans(const char *p);

/* Compute columns [j0,j1) of C. The body of each j-iteration is exactly one
 * iteration of the original omp-parallel-for: the beta scaling of column j's
 * UPLO slice [i_lo,i_hi) followed by the rank-k (TR=='N') or inner-product
 * (TR=='T') accumulation. No OpenMP pragmas — pure sequential per-column
 * work; callers partition the column range if they want parallelism. */
void xsyrk_core(
    int j0, int j1,
    char UPLO, char TR, int N, int K,
    xsyrk_T alpha, xsyrk_T beta,
    const xsyrk_T *a, int lda,
    xsyrk_T *c, int ldc);

/* Pure-serial Fortran entry. No OpenMP anywhere on this call path. */
void xsyrk_serial_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const xsyrk_T *alpha_,
    const xsyrk_T *a, const int *lda_,
    const xsyrk_T *beta_,
    xsyrk_T *c, const int *ldc_,
    size_t uplo_len, size_t trans_len);

#endif /* EPBLAS_PARALLEL_KIND16_XSYRK_KERNEL_H */
