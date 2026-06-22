/*
 * wgemmtr_kernel.h — internal kernel surface shared by the two translation
 * units the multifloats (complex double-double / complex64x2) wgemmtr overlay
 * is split across:
 *
 *   wgemmtr_serial.cpp    The pure single-thread triangular GEMM update (no
 *                         OpenMP). Owns the block-size policy, the scalar
 *                         diagonal-triangle update, and the two per-block
 *                         cores, plus the public `wgemmtr_serial` entry. The
 *                         off-diagonal rectangle routes through wgemm_serial.
 *
 *   wgemmtr_parallel.cpp  The public Fortran entry `wgemmtr_` — threading
 *                         orchestration only. Fans the column-disjoint jc-
 *                         block loop across an OpenMP team (race-free,
 *                         bitwise-identical to the serial sweep); delegates to
 *                         wgemmtr_serial when nested or below threshold.
 *
 * Leaf names are wgemmtr_-prefixed so they keep external linkage without
 * colliding with the other routines' helpers in the same archive.
 */
#ifndef EPBLAS_PARALLEL_MULTIFLOATS_WGEMMTR_KERNEL_H
#define EPBLAS_PARALLEL_MULTIFLOATS_WGEMMTR_KERNEL_H

#include <cstddef>
#include <multifloats.h>

using wgemmtr_T = multifloats::complex64x2;

/* Block size over the column axis (env-overridable WGEMMTR_NB). */
std::ptrdiff_t wgemmtr_block_nb(void);

/* Threading threshold (N below this stays serial). */
#define WGEMMTR_OMP_MIN 32

/* Beta-only pass over columns [j0,j1): scales (or zeros) the UPLO triangle of
 * each column by beta. The body of the alpha==0 / K==0 early-exit loop. */
void wgemmtr_beta_core(std::ptrdiff_t j0, std::ptrdiff_t j1, std::ptrdiff_t N, bool upper,
                       wgemmtr_T beta, wgemmtr_T *c, std::ptrdiff_t ldc);

/* One jc-block of the full update: beta-scale the triangle slice for columns
 * [jc, jc+jb), add the scalar jb×jb diagonal triangle, then route the off-
 * diagonal rectangle through wgemm_serial. Each jc-block is column-disjoint →
 * race-free. ta/tb are kept distinct (N/T/C) for the complex case. */
void wgemmtr_block_core(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t N, std::ptrdiff_t K,
                        wgemmtr_T alpha, wgemmtr_T beta,
                        const wgemmtr_T *a, std::ptrdiff_t lda,
                        const wgemmtr_T *b, std::ptrdiff_t ldb,
                        wgemmtr_T *c, std::ptrdiff_t ldc,
                        bool upper, char ta, char tb);

/* Pure-serial Fortran entry. No OpenMP on this path; same ABI as wgemmtr_. */
extern "C" void wgemmtr_serial(
    const char *uplo, const char *transa, const char *transb,
    const int *n_, const int *k_,
    const wgemmtr_T *alpha_,
    const wgemmtr_T *a, const int *lda_,
    const wgemmtr_T *b, const int *ldb_,
    const wgemmtr_T *beta_,
    wgemmtr_T *c, const int *ldc_,
    std::size_t uplo_len, std::size_t ta_len, std::size_t tb_len);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_WGEMMTR_KERNEL_H */
