/*
 * qsyr2k_kernel.h — internal surface shared by the two translation units the
 * kind16 (REAL(KIND=16) / __float128) qsyr2k overlay is split across:
 *
 *   qsyr2k_serial.c    The pure single-thread rank-2k update (no OpenMP). Owns
 *                      the SYR2K-specific diagonal-aware writeback kernel and
 *                      the fused serial driver, plus the public Fortran-ABI
 *                      serial entry `qsyr2k_serial_`. Called directly by
 *                      qsyr2k_ as its serial branch / OOM fallback / nesting
 *                      delegate.
 *
 *   qsyr2k_parallel.c  The public Fortran entry `qsyr2k_` — threading only.
 *                      Fans the same pieces across an OpenMP team (the two
 *                      shared B-packs under `omp single`, each thread an M-row
 *                      slice of the output, UPLO-clipped per N-band). Delegates
 *                      to qsyr2k_serial_ when called from inside another
 *                      routine's parallel region.
 *
 * SYR2K is a faithful __float128 port of OpenBLAS DSYR2K:
 *   C := alpha·(A·B^T + B·A^T) + beta·C   (trans='N', A,B are N×K)
 *   C := alpha·(A^T·B + B^T·A) + beta·C   (trans='T', A,B are K×N)
 * Only the UPLO triangle of C is touched. Each (is,js) tile does TWO kernel
 * passes: pass 1 with (Ap=A-pack, Bp=B-pack) and flag=1 (the diagonal NR×NR
 * block is GEMMed to a subbuffer and merged symmetrically, covering both the
 * A·B^T and B·A^T contributions on the diagonal); pass 2 with (Ap=B-pack,
 * Bp=A-pack) and flag=0 (only the strict-triangle strips, accumulating B·A^T
 * off the diagonal).
 *
 * Like qsyrk, this is built on the SHARED ob-convention L3 substrate
 * (qtri_gemm_kernel / qtri_ncopy / qtri_tcopy from qtri_kernel.h) — NOT par's
 * qgemm primitives — because the diagonal kernel indexes the packed buffers at
 * arbitrary (possibly odd) row/column offsets, valid only under OpenBLAS's
 * contiguous-odd-tail packing. The triangular β pre-pass is the SYRK helper
 * (qsyrk_beta_{u,l} from qsyrk_kernel.h), reused verbatim as ob does. The
 * layout-agnostic block-size policy is shared with qgemm (qgemm_choose_blocks
 * / blas_round_up).
 */
#ifndef EPBLAS_PARALLEL_KIND16_QSYR2K_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_QSYR2K_KERNEL_H

#include <stddef.h>

typedef __float128 qsyr2k_TR;

#define QSYR2K_MR 2
#define QSYR2K_NR 2

/* Diagonal-aware writeback kernel for one packed (m,n,k) block whose top-left
 * corner sits at global diagonal offset `offset` (row_base - col_base). The
 * strict-triangle rectangular remainders are full GEMMs (qtri_gemm_kernel).
 * When flag != 0 the diagonal NR×NR block is GEMMed into a zeroed subbuffer and
 * its UPLO triangle merged symmetrically (subbuf + subbuf^T) into C — this is
 * pass 1, which folds both A·B^T and B·A^T on the diagonal. flag == 0 (pass 2)
 * skips the diagonal block; the off-diagonal B·A^T strips land in C directly. */
void qsyr2k_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, qsyr2k_TR alpha,
                     const qsyr2k_TR *a, const qsyr2k_TR *b,
                     qsyr2k_TR *c, ptrdiff_t ldc, ptrdiff_t offset, bool flag);
void qsyr2k_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, qsyr2k_TR alpha,
                     const qsyr2k_TR *a, const qsyr2k_TR *b,
                     qsyr2k_TR *c, ptrdiff_t ldc, ptrdiff_t offset, bool flag);

/* Transpose path (trans='T'): accumulate the UPLO triangle of output column j
 * of C := alpha·(A^T·B + B^T·A) + C via the netlib-style stride-1 register
 * inner product. A,B are K×N so all dot operands stream contiguously over the K
 * axis; like qsyrk this writes each C(i,j) once, so packing buys nothing and the
 * clean dot loop ties/beats the reference. β is assumed already applied to C (by
 * qsyrk_beta_{u,l}). Per-column so the parallel entry can `omp for` over j
 * (cyclic) for balanced triangular load with no shared packs or barrier. */
void qsyr2k_trans_col(ptrdiff_t j, char uplo, ptrdiff_t n, ptrdiff_t k,
                      qsyr2k_TR alpha, const qsyr2k_TR *a, ptrdiff_t lda,
                      const qsyr2k_TR *b, ptrdiff_t ldb,
                      qsyr2k_TR *c, ptrdiff_t ldc);

/* Pure-serial by-value entry (no OpenMP); the single-thread packed driver.
 * Shares the ptrdiff_t core ABI so callers already inside a parallel region can
 * swap the symbol name only. Safe to call from inside another routine's
 * parallel region. */
void qsyr2k_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const qsyr2k_TR *alpha_,
    const qsyr2k_TR *a, ptrdiff_t lda,
    const qsyr2k_TR *b, ptrdiff_t ldb,
    const qsyr2k_TR *beta_,
    qsyr2k_TR *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND16_QSYR2K_KERNEL_H */
