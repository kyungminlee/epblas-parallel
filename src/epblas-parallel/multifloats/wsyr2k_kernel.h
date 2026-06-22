/*
 * wsyr2k_kernel.h — internal kernel surface shared by the two translation
 * units the multifloats (complex double-double / complex64x2) wsyr2k overlay
 * is split across. wsyr2k is complex SYMMETRIC (not Hermitian — see wher2k).
 *
 *   wsyr2k_serial.cpp   The pure single-thread complex-symmetric rank-2k update
 *                       (no OpenMP). Owns ALL the numerics: the AVX2 SIMD
 *                       diagonal kernels (TR='N' rank-2 and TR='T' dot forms),
 *                       the scalar diagonal fallback, the block-size policy,
 *                       the per-block worker `wsyr2k_block` (beta-scale +
 *                       diagonal update + the two trailing gemm calls via
 *                       wgemm_serial — no nested OpenMP), the per-column
 *                       triangle scaler `wsyr2k_scale_col` (the alpha==0 / K==0
 *                       early-exit), and the public `wsyr2k_serial` entry.
 *
 *   wsyr2k_parallel.cpp The public Fortran entry `wsyr2k_` — threading
 *                       orchestration only. Fans the jc BLOCK loop across an
 *                       OpenMP team (schedule(dynamic,1)); each block owns a
 *                       disjoint set of C columns so the diagonal update and
 *                       its two trailing gemm calls write disjoint regions —
 *                       race-free and bitwise-identical to the serial sweep.
 *                       The alpha==0 / K==0 early exit fans the per-column
 *                       triangle scale (schedule(static)). Delegates to
 *                       wsyr2k_serial when nested.
 *
 * Leaf names are wsyr2k_-prefixed so they keep external linkage without
 * colliding with the other routines' helpers in the same archive.
 */
#ifndef EPBLAS_PARALLEL_MULTIFLOATS_WSYR2K_KERNEL_H
#define EPBLAS_PARALLEL_MULTIFLOATS_WSYR2K_KERNEL_H

#include <cstddef>
#include <multifloats.h>

using wsyr2k_T = multifloats::complex64x2;

/* Threading threshold (N below this stays serial). */
#define WSYR2K_OMP_MIN 32

/* Block size over the diagonal axis (env WSYR2K_NB). */
std::ptrdiff_t wsyr2k_block_nb(void);

/* Scale column j's UPLO triangle of C by beta (the alpha==0 / K==0 early
 * exit). Handles beta==0 (zero-fill) and beta!=1; the caller skips beta==1. */
void wsyr2k_scale_col(std::ptrdiff_t j, std::ptrdiff_t N, char UPLO, wsyr2k_T beta,
                      wsyr2k_T *c, std::ptrdiff_t ldc);

/* One diagonal block [jc, jc+jb) of the rank-2k update: beta-scale the block's
 * own triangle columns, accumulate the diagonal block (SIMD or scalar), then
 * the two trailing gemm calls (routed through wgemm_serial — no nested
 * OpenMP). Each block writes a disjoint column range of C → race-free. */
void wsyr2k_block(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t N, std::ptrdiff_t K, char UPLO, char TR,
                  wsyr2k_T alpha, wsyr2k_T beta,
                  const wsyr2k_T *a, std::ptrdiff_t lda, const wsyr2k_T *b, std::ptrdiff_t ldb,
                  wsyr2k_T *c, std::ptrdiff_t ldc);

/* Pure-serial Fortran entry. No OpenMP on this path; same ABI as wsyr2k_. */
extern "C" void wsyr2k_serial(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const wsyr2k_T *alpha_,
    const wsyr2k_T *a, const int *lda_,
    const wsyr2k_T *b, const int *ldb_,
    const wsyr2k_T *beta_,
    wsyr2k_T *c, const int *ldc_,
    std::size_t uplo_len, std::size_t trans_len);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_WSYR2K_KERNEL_H */
