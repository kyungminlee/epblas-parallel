/*
 * esyr2k_kernel.h — internal surface shared by the two translation units the
 * kind10 (REAL(KIND=10) / 80-bit long double) esyr2k overlay is split across:
 *
 *   esyr2k_serial.c    The pure single-thread rank-2k update (no OpenMP). Owns
 *                      the SYR2K-specific diagonal-aware writeback kernel and
 *                      the fused serial driver, plus the public Fortran-ABI
 *                      serial entry `esyr2k_serial`. Called directly by
 *                      esyr2k_ as its serial branch / OOM fallback / nesting
 *                      delegate.
 *
 *   esyr2k_parallel.c  The public Fortran entry `esyr2k_` — threading only.
 *                      Fans the same pieces across an OpenMP team (the two
 *                      shared B-packs under `omp single`, each thread an M-row
 *                      slice of the output, UPLO-clipped per N-band). Delegates
 *                      to esyr2k_serial when called from inside another
 *                      routine's parallel region (the libgomp barrier-wedge
 *                      guard).
 *
 * SYR2K is a faithful port of OpenBLAS DSYR2K:
 *   C := alpha·(A·B^T + B·A^T) + beta·C   (trans='N', A,B are N×K)
 *   C := alpha·(A^T·B + B^T·A) + beta·C   (trans='T', A,B are K×N)
 * Only the UPLO triangle of C is touched. Each (is,js) tile does TWO kernel
 * passes: pass 1 with (Ap=A-pack, Bp=B-pack) and flag=1 (the diagonal NR×NR
 * block is GEMMed to a subbuffer and merged symmetrically, covering both the
 * A·B^T and B·A^T contributions on the diagonal); pass 2 with (Ap=B-pack,
 * Bp=A-pack) and flag=0 (only the strict-triangle strips, accumulating B·A^T
 * off the diagonal).
 *
 * Like esyrk, this is built on the SHARED ob-convention L3 substrate
 * (etri_gemm_kernel / etri_ncopy / etri_tcopy from etri_kernel.h) — NOT par's
 * egemm primitives — because the diagonal kernel indexes the packed buffers at
 * arbitrary (possibly odd) row/column offsets, valid only under OpenBLAS's
 * contiguous-odd-tail packing. The triangular β pre-pass is the SYRK helper
 * (esyrk_beta_{u,l} from esyrk_kernel.h), reused verbatim as ob does. The
 * layout-agnostic block-size policy is shared with egemm (egemm_choose_blocks
 * / egemm_round_up).
 */
#ifndef EPBLAS_PARALLEL_KIND10_ESYR2K_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_ESYR2K_KERNEL_H

#include <stddef.h>

typedef long double esyr2k_TR;

#define ESYR2K_MR 2
#define ESYR2K_NR 2

/* Diagonal-aware writeback kernel for one packed (m,n,k) block whose top-left
 * corner sits at global diagonal offset `offset` (row_base - col_base). The
 * strict-triangle rectangular remainders are full GEMMs (etri_gemm_kernel).
 * When flag != 0 the diagonal NR×NR block is GEMMed into a zeroed subbuffer and
 * its UPLO triangle merged symmetrically (subbuf + subbuf^T) into C — this is
 * pass 1, which folds both A·B^T and B·A^T on the diagonal. flag == 0 (pass 2)
 * skips the diagonal block; the off-diagonal B·A^T strips land in C directly. */
void esyr2k_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, esyr2k_TR alpha,
                     const esyr2k_TR *a, const esyr2k_TR *b,
                     esyr2k_TR *c, ptrdiff_t ldc, ptrdiff_t offset, ptrdiff_t flag);
void esyr2k_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, esyr2k_TR alpha,
                     const esyr2k_TR *a, const esyr2k_TR *b,
                     esyr2k_TR *c, ptrdiff_t ldc, ptrdiff_t offset, ptrdiff_t flag);

/* Pure-serial by-value core (no OpenMP). */
void esyr2k_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const esyr2k_TR *alpha_,
    const esyr2k_TR *a, ptrdiff_t lda,
    const esyr2k_TR *b, ptrdiff_t ldb,
    const esyr2k_TR *beta_,
    esyr2k_TR *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND10_ESYR2K_KERNEL_H */
