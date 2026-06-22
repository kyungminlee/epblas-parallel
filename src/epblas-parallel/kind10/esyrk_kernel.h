/*
 * esyrk_kernel.h — internal surface shared by the two translation units the
 * kind10 (REAL(KIND=10) / 80-bit long double) esyrk overlay is split across:
 *
 *   esyrk_serial.c    The pure single-thread rank-k update (no OpenMP). Owns
 *                     the SYRK-specific math — the triangular β pre-pass and
 *                     the diagonal-aware writeback kernel — plus the public
 *                     Fortran-ABI serial entry `esyrk_serial`. Called directly
 *                     by esyrk_ as its serial branch / OOM fallback / nesting
 *                     delegate.
 *
 *   esyrk_parallel.c  The public Fortran entry `esyrk_` — threading only.
 *                     Fans the same pieces across an OpenMP team (shared Bp
 *                     packed under `omp single`, each thread an M-row slice
 *                     of the output, UPLO-clipped per N-band). Delegates to
 *                     esyrk_serial when called from inside another routine's
 *                     parallel region (the libgomp barrier-wedge guard).
 *
 * SYRK is a faithful port of OpenBLAS DSYRK: C := alpha·A·A^T + beta·C (or
 * A^T·A). Only the UPLO triangle of C is touched. It reuses the SHARED
 * ob-convention L3 substrate (etri_gemm_kernel / etri_ncopy / etri_tcopy from
 * etri_kernel.h) — NOT par's egemm primitives. SYRK's diagonal kernel splits
 * each MC×NC block around the global diagonal and GEMMs the strict-triangle
 * remainders at arbitrary (possibly odd) row/column offsets into the packed
 * buffers; that sub-block indexing only lands on valid strip boundaries under
 * OpenBLAS's contiguous-odd-tail packing, which the etri substrate provides
 * and par's zero-padded egemm layout does not. The layout-agnostic block-size
 * policy is still shared with egemm (egemm_choose_blocks / egemm_round_up).
 */
#ifndef EPBLAS_PARALLEL_KIND10_ESYRK_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_ESYRK_KERNEL_H

#include <stddef.h>

typedef long double esyrk_T;

#define ESYRK_MR 2
#define ESYRK_NR 2

/* Triangular β pre-pass: C := β·C over the UPLO triangle of C only (the
 * off-UPLO triangle is left untouched — the SYRK fuzz sentinel checks this). */
void esyrk_beta_u(ptrdiff_t n, esyrk_T beta, esyrk_T *c, ptrdiff_t ldc);
void esyrk_beta_l(ptrdiff_t n, esyrk_T beta, esyrk_T *c, ptrdiff_t ldc);

/* Diagonal-aware writeback kernel for one packed (m,n,k) block whose top-left
 * corner sits at global diagonal offset (row_base - col_base). Off-diagonal
 * strict-triangle remainders are full GEMMs (etri_gemm_kernel); the diagonal
 * NR×NR blocks go through a subbuffer so only the UPLO triangle merges into C. */
void esyrk_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, esyrk_T alpha,
                    const esyrk_T *a, const esyrk_T *b,
                    esyrk_T *c, ptrdiff_t ldc, ptrdiff_t offset);
void esyrk_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, esyrk_T alpha,
                    const esyrk_T *a, const esyrk_T *b,
                    esyrk_T *c, ptrdiff_t ldc, ptrdiff_t offset);

/* Pure-serial by-value core (no OpenMP). Same shape as esyrk_core. */
void esyrk_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const esyrk_T *alpha_,
    const esyrk_T *a, ptrdiff_t lda,
    const esyrk_T *beta_,
    esyrk_T *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND10_ESYRK_KERNEL_H */
