/*
 * qgemm_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 qgemm overlay is split across:
 *
 *   qgemm_serial.c    The pure single-thread GEMM (no OpenMP). Owns ALL
 *                     the math — packers, MR×NR micro-kernel, beta
 *                     pre-pass, block-size policy — and the public
 *                     `qgemm_serial_` entry. Called directly by the L3
 *                     routines (qtrsm, …) that run qgemm trailing updates
 *                     inside their OWN parallel region, and by qgemm_ as
 *                     its serial branch.
 *
 *   qgemm_parallel.c  The public Fortran entry `qgemm_` — threading
 *                     orchestration only. Delegates to qgemm_serial_ when
 *                     called from inside a parallel region; otherwise fans
 *                     these same kernel pieces across an OpenMP team
 *                     (M-axis split, shared packed Bp).
 *
 * Structure mirrors the kind10 egemm overlay: GotoBLAS / OpenBLAS
 * three-level cache blocking (NC × KC × MC), copy-and-conquer packing
 * (op(A), op(B) absorbed into Ap/Bp), register-blocked MR×NR outer-product
 * micro-kernel. Although __float128 arithmetic lowers to libquadmath soft-
 * float calls, the blocked packed path still wins: the four independent
 * MR×NR accumulator chains let the out-of-order engine overlap those
 * independent calls, and packing kills the strided B traffic of the naive
 * reference. (Measured: the ob blocked clone beats the unblocked reference
 * ~13% serial.)
 *
 * Everything here is internal to the overlay. `qgemm_serial_` keeps the
 * exact Fortran-ABI signature of qgemm_ so callers already inside a
 * parallel region can swap the symbol name only.
 */
#ifndef EPBLAS_PARALLEL_KIND16_QGEMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_QGEMM_KERNEL_H

#include <stddef.h>

typedef __float128 qgemm_TR;

/* Register-tile dims (compile-time constants; small so the four soft-float
 * accumulators stay register-resident across the K-loop). */
#define QGEMM_MR 2
#define QGEMM_NR 2



/* Cache-block sizes (env-overridable QBLAS_MC/KC/NC) with OpenBLAS-style
 * adaptive MC when K fits one panel. */
void qgemm_choose_blocks(ptrdiff_t k, ptrdiff_t *MC, ptrdiff_t *KC, ptrdiff_t *NC);

/* C := beta*C pre-pass over the full M×N tile (handles K==0 / alpha==0). */
void qgemm_beta_prepass(ptrdiff_t m, ptrdiff_t n, qgemm_TR beta, qgemm_TR *c, ptrdiff_t ldc);

/* Packers (panel-packed, OpenBLAS-style). */
void qgemm_pack_A(const qgemm_TR *restrict A, ptrdiff_t lda,
                  ptrdiff_t ic, ptrdiff_t pc, ptrdiff_t ib, ptrdiff_t pb, ptrdiff_t ta,
                  qgemm_TR *restrict Ap);
void qgemm_pack_B(const qgemm_TR *restrict B, ptrdiff_t ldb,
                  ptrdiff_t pc, ptrdiff_t jc, ptrdiff_t pb, ptrdiff_t jb, ptrdiff_t tb,
                  qgemm_TR *restrict Bp);

/* Drive one packed (ib,jb,pb) macro-tile via MR×NR sub-tiles. */
void qgemm_macro_kernel(ptrdiff_t ib, ptrdiff_t jb, ptrdiff_t pb, qgemm_TR alpha,
                        const qgemm_TR *restrict Ap, const qgemm_TR *restrict Bp,
                        qgemm_TR *restrict C, ptrdiff_t ldc);

/* TA='T',TB='N' path: one C-column j2 (stride-1 dot, no packing). For
 * __float128 this is the ONLY TN path — it beats the blocked packed kernel at
 * every K and size because packing buys nothing without SIMD and A^T already
 * streams both operands stride-1 along the contraction index. The C-column
 * loop is threaded over j2 in the qgemm_ parallel entry. */
void qgemm_fast_col(ptrdiff_t j2, ptrdiff_t m, ptrdiff_t k, qgemm_TR alpha,
                    const qgemm_TR *a, ptrdiff_t lda, const qgemm_TR *b, ptrdiff_t ldb,
                    qgemm_TR *c, ptrdiff_t ldc);

/* Pure-serial by-value entry. No OpenMP anywhere on this call path; safe to
 * invoke from inside another function's `#pragma omp parallel` region. Shares
 * the ptrdiff_t core ABI of qgemm_core so callers already inside a parallel
 * region can swap the symbol name only. */
void qgemm_serial(
    char transa, char transb,
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
    const qgemm_TR *alpha_,
    const qgemm_TR *a, ptrdiff_t lda,
    const qgemm_TR *b, ptrdiff_t ldb,
    const qgemm_TR *beta_,
    qgemm_TR *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND16_QGEMM_KERNEL_H */
