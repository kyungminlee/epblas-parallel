/*
 * xher2k_kernel.h — internal surface shared by the two translation units the
 * kind16 complex (__complex128 / interleaved __float128) xher2k overlay is
 * split across:
 *
 *   xher2k_serial.c    The pure single-thread Hermitian rank-2k update (no
 *                      OpenMP). Owns the fused serial packed-GEMM driver plus
 *                      the internal serial entry `xher2k_serial`. Called
 *                      directly by xher2k_ as its OOM fallback / nesting
 *                      delegate.
 *
 *   xher2k_parallel.c  The public Fortran entry `xher2k_` — threading only.
 *                      Fans the same pieces across an OpenMP team (two shared
 *                      B-packs under `omp single`, each thread an M-row slice
 *                      of the output, UPLO-clipped per N-band). Delegates to
 *                      xher2k_serial when called from inside another routine's
 *                      parallel region (the libgomp barrier-wedge guard).
 *
 * Faithful port of OpenBLAS ZHER2K (interface/syr2k.c with the HEMM macro →
 * driver/level3/zher2k_k.c blocking nest → driver/level3/zher2k_kernel.c
 * diagonal kernel, the latter transcribed into the shared xl3_complex.c
 * substrate as qblas_xher2k_kernel_{u,l}):
 *
 *   C := alpha·A·Bᴴ + conj(alpha)·B·Aᴴ + beta·C   (trans='N', A,B are N×K)
 *   C := alpha·Aᴴ·B + conj(alpha)·Bᴴ·A + beta·C   (trans='C', A,B are K×N)
 *
 * alpha is COMPLEX, beta is REAL, C is HERMITIAN: only the UPLO triangle is
 * read/written and the diagonal stays real on output (the β pre-pass
 * qblas_xherk_beta_{u,l} unconditionally clears the diagonal imaginary part).
 * Each (is,js) tile does TWO kernel passes: pass 1 (Ap=A-pack, Bp=B-pack,
 * flag=1) GEMMs the diagonal NR×NR block to a subbuffer and merges it
 * Hermitian-symmetrically, covering both A·Bᴴ and conj(alpha)·B·Aᴴ on the
 * diagonal; pass 2 (Ap=B-pack, Bp=A-pack, conj(alpha) via −alphai, flag=0)
 * adds the off-diagonal B·Aᴴ strips. Conjugation is absorbed at pack time:
 * TRANS='N' conjugates the Bp side, TRANS='C' conjugates the Ap side. The
 * triangular β pre-pass and the diagonal kernel both live in the shared
 * complex L3 substrate (xl3_complex.h) — the same one xgemm/xsymm/xsyrk use —
 * because the kernel indexes the packed buffers at arbitrary (possibly odd)
 * offsets, valid only under OpenBLAS's contiguous-odd-tail packing.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XHER2K_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XHER2K_KERNEL_H

#include <stddef.h>
#include <quadmath.h>

/* Public ABI is the genuine complex/real type pair, mirroring kind10 yher2k's
 * EPBLAS_FACADE_SYR2K(yher2k, TC, TR, TC): the complex matrices A/B/C and the
 * complex alpha are __complex128 (TC), the real beta is __float128 (TR).
 *
 * Internally the math is a faithful OpenBLAS port that indexes operands as the
 * interleaved (re,im) __float128 storage the ob driver expects (×2 per complex
 * element; ld* in COMPLEX elements). Each entry reinterpret-casts its TC
 * pointers to xher2k_TR (= __float128) at the top — a complex value is exactly
 * two contiguous reals — and runs the interleaved kernel unchanged. So TC is
 * always complex and TR always real; xher2k_TR doubles as the real beta operand
 * and the internal interleaved storage element. */
typedef __complex128 xher2k_TC;   /* complex operands: A, B, C, alpha */
typedef __float128   xher2k_TR;   /* real beta + internal interleaved storage */

/* Pure-serial by-value entry (no OpenMP). Shares the ptrdiff_t core ABI of
 * xher2k_core so callers already inside a parallel region can swap the symbol
 * name only; mirrors xgemm_serial. alpha_/beta_ stay pointers. */
void xher2k_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const xher2k_TC *alpha_,
    const xher2k_TC *a, ptrdiff_t lda,
    const xher2k_TC *b, ptrdiff_t ldb,
    const xher2k_TR *beta_,
    xher2k_TC *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND16_XHER2K_KERNEL_H */
