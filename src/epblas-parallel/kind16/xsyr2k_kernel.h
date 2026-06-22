/*
 * xsyr2k_kernel.h — internal surface shared by the two translation units the
 * kind16 complex (__complex128 / interleaved __float128) xsyr2k overlay is
 * split across:
 *
 *   xsyr2k_serial.c    The pure single-thread symmetric rank-2k update (no
 *                      OpenMP). Owns the fused serial packed-GEMM driver plus
 *                      the internal serial entry `xsyr2k_serial`. Called
 *                      directly by xsyr2k_ as its OOM fallback / nesting
 *                      delegate.
 *
 *   xsyr2k_parallel.c  The public Fortran entry `xsyr2k_` — threading only.
 *                      Fans the same pieces across an OpenMP team (two shared
 *                      B-packs under `omp single`, each thread an M-row slice
 *                      of the output, UPLO-clipped per N-band). Delegates to
 *                      xsyr2k_serial when called from inside another routine's
 *                      parallel region (the libgomp barrier-wedge guard).
 *
 * Faithful port of OpenBLAS ZSYR2K (interface/syr2k.c dispatch →
 * driver/level3/level3_syr2k.c blocking nest → driver/level3/syr2k_kernel.c
 * diagonal kernel, the latter transcribed into the shared xl3_complex.c
 * substrate as qblas_ysyr2k_kernel_{u,l}):
 *
 *   C := alpha·(A·Bᵀ + B·Aᵀ) + beta·C   (trans='N', A,B are N×K)
 *   C := alpha·(Aᵀ·B + Bᵀ·A) + beta·C   (trans='T', A,B are K×N)
 *
 * Complex SYMMETRIC variant — no conjugation, A == Aᵀ (NOT Hermitian). Only
 * the UPLO triangle of C is read or written. Each (is,js) tile does TWO kernel
 * passes: pass 1 (Ap=A-pack, Bp=B-pack, flag=1) GEMMs the diagonal NR×NR block
 * to a subbuffer and merges it symmetrically, covering both A·Bᵀ and B·Aᵀ on
 * the diagonal; pass 2 (Ap=B-pack, Bp=A-pack, flag=0) adds the off-diagonal
 * B·Aᵀ strips. The triangular β pre-pass and the diagonal kernel both live in
 * the shared complex L3 substrate (xl3_complex.h) — the same one xgemm/xsymm/
 * xsyrk use — because the kernel indexes the packed buffers at arbitrary
 * (possibly odd) offsets, valid only under OpenBLAS's contiguous-odd-tail
 * packing.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XSYR2K_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XSYR2K_KERNEL_H

#include <stddef.h>

/* Arrays are interleaved (re,im) pairs of __float128, indexed as the ob
 * driver does (×2 per complex element); ld* are in COMPLEX elements. */
typedef __float128 xsyr2k_T;

/* Pure-serial by-value entry (no OpenMP). Shares the ptrdiff_t core ABI of
 * xsyr2k_core so callers already inside a parallel region can swap the symbol
 * name only; mirrors the kind10 ysyr2k_serial by-value shape. The a/b/c/alpha/
 * beta pointers are interleaved (re,im) __float128; alpha_/beta_ stay pointers. */
void xsyr2k_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const xsyr2k_T *alpha_,
    const xsyr2k_T *a, ptrdiff_t lda,
    const xsyr2k_T *b, ptrdiff_t ldb,
    const xsyr2k_T *beta_,
    xsyr2k_T *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND16_XSYR2K_KERNEL_H */
