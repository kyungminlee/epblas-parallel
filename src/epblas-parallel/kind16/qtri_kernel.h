/*
 * qtri_kernel.h — the ob-convention L3 GEMM substrate shared by the kind16
 * (REAL(KIND=16) / __float128) packed rank-k routines (qsyrk, qsyr2k).
 *
 * This is a private MR=NR=2 GEMM micro-kernel plus its matching ncopy/tcopy
 * packers and a zero-then-accumulate "store" variant — a faithful __float128
 * port of kind10's etri_kernel.c (itself a port of OpenBLAS
 * kernel/generic/{gemmkernel_2x2,gemm_ncopy_2,gemm_tcopy_2}.c).
 *
 * Why a PRIVATE substrate rather than par's qgemm primitives: a SYRK/SYR2K
 * diagonal kernel GEMMs strict-triangle remainders at arbitrary (possibly
 * odd) row/column offsets into the packed buffers, and OpenBLAS's
 * contiguous-odd-tail packing convention is baked into the packer ↔ kernel
 * pair. par's qgemm packs odd tails zero-padded at stride MR instead, so its
 * kernel reads those bytes differently. This substrate is self-consistent
 * with the syrk/syr2k diagonal kernels. The layout-AGNOSTIC block-size policy
 * is still shared with qgemm (qgemm_choose_blocks / qgemm_round_up).
 */
#ifndef EPBLAS_PARALLEL_KIND16_QTRI_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_QTRI_KERNEL_H

#include <stddef.h>

typedef __float128 qtri_T;

/* C += alpha·Ap·Bp over one packed (bm,bn,bk) tile (accumulate). */
void qtri_gemm_kernel(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk, qtri_T alpha,
                      const qtri_T *Ap, const qtri_T *Bp,
                      qtri_T *C, ptrdiff_t ldc);

/* C := alpha·Ap·Bp over one packed tile (overwrite): zero C then accumulate.
 * Mirrors OpenBLAS's GEMM_KERNEL(beta=0) call inside the TRMM driver. */
void qtri_kernel_store(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk, qtri_T alpha,
                       const qtri_T *Ap, const qtri_T *Bp,
                       qtri_T *C, ptrdiff_t ldc);

/* Pack a plain (non-triangular) A/B slab into the packed layout. */
void qtri_ncopy(ptrdiff_t m, ptrdiff_t n, const qtri_T *a, ptrdiff_t lda,
                qtri_T *b);
void qtri_tcopy(ptrdiff_t m, ptrdiff_t n, const qtri_T *a, ptrdiff_t lda,
                qtri_T *b);

/* Area-balanced M-row partition for a triangular (SYRK/SYR2K) output.
 * Returns thread `tid`'s contiguous row range [*m_lo, *m_hi) of [0, N) such
 * that the TRIANGULAR work — row i carries (N-i) active columns for uplo 'U',
 * (i+1) for 'L' — is split evenly across `nth` threads. A naive equal-row-count
 * split caps the rank-k NoTrans speedup at ~16/7 ≈ 2.29× on 4 threads because
 * the thread owning the fat end of the triangle carries 7/16 of the work; this
 * weights band widths by area instead (narrow at the fat end, wide at the thin
 * end). Boundaries are floored to MR so each thread's packed panels stay
 * MR-aligned; the result tiles [0, N) with no gaps or overlaps. */
void qtri_row_bounds(int uplo, ptrdiff_t N, ptrdiff_t nth, ptrdiff_t tid,
                     ptrdiff_t mr, ptrdiff_t *m_lo, ptrdiff_t *m_hi);

#endif /* EPBLAS_PARALLEL_KIND16_QTRI_KERNEL_H */
