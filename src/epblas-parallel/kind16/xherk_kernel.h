/*
 * xherk_kernel.h — internal surface shared by the two translation units the
 * kind16 complex (__complex128 / interleaved __float128) xherk overlay is split
 * across:
 *
 *   xherk_serial.c    The pure single-thread Hermitian rank-k update (no
 *                     OpenMP). Owns the fused serial packed-GEMM driver plus the
 *                     internal serial entry `xherk_serial`. Called directly by
 *                     xherk_ as its OOM fallback / nesting delegate.
 *
 *   xherk_parallel.c  The public Fortran entries (`xherk_` LP64 + `xherk_64_`
 *                     ILP64) — threading only. Fans the same pieces across an
 *                     OpenMP team (one shared B-pack under `omp single`, each
 *                     thread an M-row slice of the output, UPLO-clipped per
 *                     N-band). Delegates to xherk_serial when called from inside
 *                     another routine's parallel region (the libgomp
 *                     barrier-wedge guard).
 *
 * Faithful port of OpenBLAS ZHERK as the SINGLE-PRODUCT special case of the
 * xher2k packed-GEMM nest: one packed GEMM (Bp packed from the same A, the
 * conjugated side chosen by TRANS) clipped to the UPLO triangle, run ONCE per
 * (is,js) tile. The diagonal NR×NR block is merged singly and the true diagonal
 * realified (Hermitian C). The triangular β pre-pass (qblas_xherk_beta_{u,l},
 * real beta) and the diagonal-aware kernel (qblas_xherk_kernel_{u,l}) both live
 * in the shared complex L3 substrate (xl3_complex.h).
 *
 *   C := alpha·A·Aᴴ + beta·C   (trans='N', A is N×K)
 *   C := alpha·Aᴴ·A + beta·C   (trans='C', A is K×N)
 *
 * alpha and beta are REAL; the diagonal of C stays real on output. Conjugation
 * is absorbed at pack time: TRANS='N' conjugates the Bp side, TRANS='C'
 * conjugates the Ap side.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XHERK_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XHERK_KERNEL_H

#include <stddef.h>
#include <quadmath.h>

typedef __complex128 xherk_TC;   /* complex operands: A, C */
typedef __float128   xherk_TR;   /* real scalars alpha, beta; interleaved storage */

/* Pure-serial by-value entry (no OpenMP). Shares the ptrdiff_t core ABI so
 * callers already inside a parallel region can swap the symbol name only. */
void xherk_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const xherk_TR *alpha_,
    const xherk_TC *a, ptrdiff_t lda,
    const xherk_TR *beta_,
    xherk_TC *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND16_XHERK_KERNEL_H */
