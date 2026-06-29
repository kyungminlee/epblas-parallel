/*
 * xsyrk_kernel.h — internal surface shared by the two translation units the
 * kind16 complex (__complex128 / interleaved __float128) xsyrk overlay is
 * split across:
 *
 *   xsyrk_serial.c    The pure single-thread symmetric rank-k update (no
 *                     OpenMP). Owns the fused serial packed-GEMM driver plus
 *                     the internal serial entry `xsyrk_serial`. Called directly
 *                     by xsyrk_ as its OOM fallback / nesting delegate.
 *
 *   xsyrk_parallel.c  The public Fortran entries (`xsyrk_` LP64 + `xsyrk_64_`
 *                     ILP64) — threading only. Fans the same pieces across an
 *                     OpenMP team (one shared B-pack under `omp single`, each
 *                     thread an M-row slice of the output, UPLO-clipped per
 *                     N-band). Delegates to xsyrk_serial when called from inside
 *                     another routine's parallel region (the libgomp
 *                     barrier-wedge guard).
 *
 * Faithful port of OpenBLAS ZSYRK as the SINGLE-PRODUCT special case of the
 * xsyr2k packed-GEMM nest: one packed GEMM (Bp packed from the same A) clipped
 * to the UPLO triangle, run ONCE per (is,js) tile. The diagonal NR×NR block is
 * merged singly (no symmetric mirror — a single product is already symmetric on
 * the diagonal). The triangular β pre-pass (qblas_xsyrk_beta_{u,l}) and the
 * diagonal-aware kernel (qblas_xsyrk_kernel_{u,l}) both live in the shared
 * complex L3 substrate (xl3_complex.h).
 *
 *   C := alpha·A·Aᵀ + beta·C   (trans='N', A is N×K)
 *   C := alpha·Aᵀ·A + beta·C   (trans='T', A is K×N)
 *
 * Complex SYMMETRIC variant — no conjugation, A == Aᵀ (NOT Hermitian; see
 * xherk). alpha and beta are complex. Only the UPLO triangle of C is touched.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XSYRK_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XSYRK_KERNEL_H

#include <stddef.h>
#include <quadmath.h>

/* Public ABI is the genuine complex type, mirroring kind10 ysyrk: the complex
 * matrices A/C and the complex scalars alpha/beta are all __complex128 (TC).
 * Internally the math reinterprets the TC pointers as the interleaved (re,im)
 * __float128 (TR) storage the OpenBLAS driver expects (×2 per complex element;
 * ld* in COMPLEX elements). */
typedef __complex128 xsyrk_TC;   /* complex operands: A, C, alpha, beta */
typedef __float128   xsyrk_TR;   /* internal interleaved (re,im) storage */

/* Pure-serial by-value entry (no OpenMP). Shares the ptrdiff_t core ABI so
 * callers already inside a parallel region can swap the symbol name only. */
void xsyrk_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const xsyrk_TC *alpha_,
    const xsyrk_TC *a, ptrdiff_t lda,
    const xsyrk_TC *beta_,
    xsyrk_TC *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND16_XSYRK_KERNEL_H */
