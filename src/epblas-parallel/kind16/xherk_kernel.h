/*
 * xherk_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 xherk overlay is split across:
 *
 *   xherk_serial.c    The pure single-thread Hermitian rank-k update (no
 *                     OpenMP). Owns the per-column compute core, the
 *                     uplo/trans decode, and the public `xherk_serial_`
 *                     entry. Safe to invoke from inside another routine's
 *                     own `#pragma omp parallel` region.
 *
 *   xherk_parallel.c  The public Fortran entry `xherk_` — threading
 *                     orchestration only. Fans the independent columns of C
 *                     across an OpenMP team; falls back to running the loop
 *                     serially when invoked from inside a parallel region.
 *
 * C is N×N Hermitian; only the UPLO triangle is referenced/written and its
 * diagonal is forced real (the imag part is zeroed, matching Netlib ZHERK).
 * alpha and beta are REAL (kind=16). kind16 complex is arithmetic-bound
 * (__complex128 lowers to libquadmath calls), so the overlay uses the
 * unblocked Netlib reference with no packing — the shared surface is the
 * decode plus the per-column core.
 *
 * Each column j of C touches a distinct UPLO-triangle slice [i_lo,i_hi) and
 * reads only A, so columns are fully independent — per-column static
 * scheduling is bitwise-identical to the serial sweep. The real-diagonal
 * rule (zero the imag part of cj[j]) is applied per column inside the core.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XHERK_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XHERK_KERNEL_H

#include <stddef.h>
#include <quadmath.h>   /* __complex128, __float128 */

typedef __complex128 xherk_TC;   /* matrices A, C */
typedef __float128   xherk_TR;   /* scalars alpha, beta */

/* Decode a Fortran uplo char to upper-cased 'U'/'L'. */
char xherk_uplo(const char *p);

/* Decode a Fortran trans char to upper-cased 'N'/'C'. */
char xherk_trans(const char *p);

/* Compute columns [j0,j1) of C. The body of each j-iteration is exactly one
 * iteration of the original omp-parallel-for: the beta scaling of column j's
 * UPLO slice [i_lo,i_hi) (diagonal kept real), followed by the rank-k
 * (TR=='N') or inner-product (TR=='C') accumulation with the conjugation and
 * real-diagonal rules preserved. No OpenMP pragmas — pure sequential
 * per-column work; callers partition the column range for parallelism. */
void xherk_core(
    int j0, int j1,
    char UPLO, char TR, int N, int K,
    xherk_TR alpha, xherk_TR beta,
    const xherk_TC *a, int lda,
    xherk_TC *c, int ldc);

/* Pure-serial Fortran entry. No OpenMP anywhere on this call path. */
void xherk_serial_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const xherk_TR *alpha_,
    const xherk_TC *a, const int *lda_,
    const xherk_TR *beta_,
    xherk_TC *c, const int *ldc_,
    size_t uplo_len, size_t trans_len);

#endif /* EPBLAS_PARALLEL_KIND16_XHERK_KERNEL_H */
