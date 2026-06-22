/*
 * qsyrk_kernel.h — internal surface shared by the two translation units the
 * kind16 (REAL(KIND=16) / __float128) qsyrk overlay is split across:
 *
 *   qsyrk_serial.c    The pure single-thread rank-k update (no OpenMP). Owns
 *                     the SYRK-specific math — the triangular β pre-pass and
 *                     the diagonal-aware writeback kernel — plus the public
 *                     Fortran-ABI serial entry `qsyrk_serial_`. Called directly
 *                     by qsyrk_ as its serial branch / OOM fallback / nesting
 *                     delegate.
 *
 *   qsyrk_parallel.c  The public Fortran entry `qsyrk_` — threading only.
 *                     Fans the same pieces across an OpenMP team (shared Bp
 *                     packed under `omp single`, each thread an M-row slice
 *                     of the output, UPLO-clipped per N-band). Delegates to
 *                     qsyrk_serial_ when called from inside another routine's
 *                     parallel region.
 *
 * SYRK is a faithful __float128 port of OpenBLAS DSYRK: C := alpha·A·A^T +
 * beta·C (or A^T·A). Only the UPLO triangle of C is touched. It reuses the
 * SHARED ob-convention L3 substrate (qtri_gemm_kernel / qtri_ncopy /
 * qtri_tcopy from qtri_kernel.h) — NOT par's qgemm primitives. SYRK's diagonal
 * kernel splits each MC×NC block around the global diagonal and GEMMs the
 * strict-triangle remainders at arbitrary (possibly odd) row/column offsets
 * into the packed buffers; that sub-block indexing only lands on valid strip
 * boundaries under OpenBLAS's contiguous-odd-tail packing, which the qtri
 * substrate provides and par's zero-padded qgemm layout does not. The
 * layout-agnostic block-size policy is still shared with qgemm
 * (qgemm_choose_blocks / qgemm_round_up).
 *
 * Why packed rather than the old unblocked register-tile: at small N (≈64) the
 * unblocked rank-k trailed ob's packed kernel ~3-4% on the NoTrans serial
 * corner; packing the operands matches ob's data-movement structure and closes
 * it (the libquadmath arithmetic floor is identical either way).
 */
#ifndef EPBLAS_PARALLEL_KIND16_QSYRK_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_QSYRK_KERNEL_H

#include <stddef.h>

typedef __float128 qsyrk_T;

#define QSYRK_MR 2
#define QSYRK_NR 2

/* Triangular β pre-pass: C := β·C over the UPLO triangle of C only (the
 * off-UPLO triangle is left untouched — the SYRK fuzz sentinel checks this). */
void qsyrk_beta_u(ptrdiff_t n, qsyrk_T beta, qsyrk_T *c, ptrdiff_t ldc);
void qsyrk_beta_l(ptrdiff_t n, qsyrk_T beta, qsyrk_T *c, ptrdiff_t ldc);

/* Diagonal-aware writeback kernel for one packed (m,n,k) block whose top-left
 * corner sits at global diagonal offset (row_base - col_base). Off-diagonal
 * strict-triangle remainders are full GEMMs (qtri_gemm_kernel); the diagonal
 * NR×NR blocks go through a subbuffer so only the UPLO triangle merges into C. */
void qsyrk_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, qsyrk_T alpha,
                    const qsyrk_T *a, const qsyrk_T *b,
                    qsyrk_T *c, ptrdiff_t ldc, ptrdiff_t offset);
void qsyrk_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, qsyrk_T alpha,
                    const qsyrk_T *a, const qsyrk_T *b,
                    qsyrk_T *c, ptrdiff_t ldc, ptrdiff_t offset);

/* Transpose path (trans='T'): accumulate the UPLO triangle of output column j
 * of C := alpha·A^T·A + C via the netlib-style stride-1 register inner product.
 * A is K×N so both dot operands stream contiguously over the K axis; unlike the
 * NoTrans outer product (which RMW-streams each C column K times), this writes
 * each C(i,j) once, so there is nothing for packing to amortise and the clean
 * dot loop ties/beats the reference. β is assumed already applied to C (by
 * qsyrk_beta_{u,l}). Per-column so the parallel entry can `omp for` over j
 * (cyclic) for balanced triangular load with no shared pack or barrier. */
void qsyrk_trans_col(ptrdiff_t j, int uplo, ptrdiff_t N, ptrdiff_t K,
                     qsyrk_T alpha, const qsyrk_T *a, ptrdiff_t lda,
                     qsyrk_T *c, ptrdiff_t ldc);

/* Pure-serial by-value entry (no OpenMP); the single-thread packed driver.
 * Shares the ptrdiff_t core ABI so callers already inside a parallel region can
 * swap the symbol name only. Safe to call from inside another routine's
 * parallel region. */
void qsyrk_serial(
    char uplo, char trans,
    ptrdiff_t N, ptrdiff_t K,
    const qsyrk_T *alpha_,
    const qsyrk_T *a, ptrdiff_t lda,
    const qsyrk_T *beta_,
    qsyrk_T *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND16_QSYRK_KERNEL_H */
