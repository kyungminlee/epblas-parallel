/*
 * etri_kernel.h — the ob-convention L3 GEMM substrate shared by the kind10
 * (REAL(KIND=10) / 80-bit long double) triangular routines (etrsm, etrmm).
 *
 * This is a private MR=NR=2 GEMM micro-kernel plus its matching ncopy/tcopy
 * packers and a zero-then-accumulate "store" variant — faithful ports of
 * OpenBLAS kernel/generic/{gemmkernel_2x2,gemm_ncopy_2,gemm_tcopy_2}.c (the
 * same code the openblas overlay carries in eblas_l3_real.c).
 *
 * Why a PRIVATE substrate rather than par's egemm primitives: a triangular
 * routine's solve/trmm micro-kernel and its trailing GEMM share one packed
 * diagonal-block buffer at MR/NR granularity, and OpenBLAS's
 * contiguous-odd-tail packing convention is baked into the packer ↔
 * solve/trmm ↔ kernel triad. par's egemm packs odd tails zero-padded at
 * stride MR instead, so its kernel reads those bytes differently (proven
 * mismatch on every odd m/n/k). This substrate is therefore self-consistent
 * with the triangular packers (etrsm_pack.c, etrmm_pack.c). The
 * layout-AGNOSTIC block-size policy is still shared with egemm
 * (egemm_choose_blocks / egemm_beta_prepass / blas_round_up).
 *
 * Both etrsm and etrmm depend on this substrate: etrsm pairs it with the
 * diagonal solve kernel (etrsm_kernel.c), etrmm with the TRMM kernel
 * (etrmm_kernel.c).
 */
#ifndef EPBLAS_PARALLEL_KIND10_ETRI_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_ETRI_KERNEL_H

#include <stddef.h>

typedef long double etri_TR;

/* C += alpha·Ap·Bp over one packed (bm,bn,bk) tile (accumulate). */
void etri_gemm_kernel(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk, etri_TR alpha,
                      const etri_TR *Ap, const etri_TR *Bp,
                      etri_TR *C, ptrdiff_t ldc);

/* C -= Ap·Bp over one packed tile — the alpha = -1 trailing update the
 * triangular solve/trmm drivers issue. Negate-specialized: the per-element
 * store is a bare subtract (no alpha multiply), matching OpenBLAS's
 * constant-propagated kernel. Use this instead of etri_gemm_kernel(...,-1,...)
 * on the hot trailing-GEMM path. */
void etri_gemm_kernel_msub(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                           const etri_TR *Ap, const etri_TR *Bp,
                           etri_TR *C, ptrdiff_t ldc);

/* C := alpha·Ap·Bp over one packed tile (overwrite): zero C then accumulate.
 * Mirrors OpenBLAS's GEMM_KERNEL(beta=0) call inside the TRMM driver. */
void etri_kernel_store(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk, etri_TR alpha,
                       const etri_TR *Ap, const etri_TR *Bp,
                       etri_TR *C, ptrdiff_t ldc);

/* Pack a plain (non-triangular) A/B slab into the packed layout. */
void etri_ncopy(ptrdiff_t m, ptrdiff_t n, const etri_TR *a, ptrdiff_t lda,
                etri_TR *b);
void etri_tcopy(ptrdiff_t m, ptrdiff_t n, const etri_TR *a, ptrdiff_t lda,
                etri_TR *b);

#endif /* EPBLAS_PARALLEL_KIND10_ETRI_KERNEL_H */
