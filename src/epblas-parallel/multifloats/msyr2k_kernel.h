/*
 * msyr2k_kernel.h — internal kernel surface shared by the two translation
 * units the multifloats (double-double / float64x2) msyr2k overlay is split
 * across:
 *
 *   msyr2k_serial.cpp   The pure single-thread symmetric rank-2k update (no
 *                       OpenMP). Owns ALL the numerics: the AVX2 SIMD diagonal
 *                       kernels (TR='N' rank-2 and TR='T' dot forms), the
 *                       scalar diagonal fallback, the block-size policy, the
 *                       per-block worker `msyr2k_block` (beta-scale + diagonal
 *                       update + the two trailing gemm calls via mgemm_serial —
 *                       no nested OpenMP), the per-column triangle scaler
 *                       `msyr2k_scale_col` (the alpha==0 / K==0 early-exit),
 *                       and the public `msyr2k_serial` entry.
 *
 *   msyr2k_parallel.cpp The public Fortran entry `msyr2k_` — threading
 *                       orchestration only. Fans the jc BLOCK loop across an
 *                       OpenMP team (schedule(dynamic,1)); each block owns a
 *                       disjoint set of C columns so the diagonal update and
 *                       its two trailing gemm calls write disjoint regions —
 *                       race-free and bitwise-identical to the serial sweep.
 *                       The alpha==0 / K==0 early exit fans the per-column
 *                       triangle scale (schedule(static)). Delegates to
 *                       msyr2k_serial when nested.
 *
 * The serial and parallel paths drive numerics through the SAME per-block
 * worker, so a static/dynamic partition is bitwise-identical to the serial
 * sweep (the per-block accumulation order is independent of the block schedule).
 *
 * Leaf names are msyr2k_-prefixed so they keep external linkage without
 * colliding with the other routines' helpers in the same archive.
 */
#ifndef EPBLAS_PARALLEL_MULTIFLOATS_MSYR2K_KERNEL_H
#define EPBLAS_PARALLEL_MULTIFLOATS_MSYR2K_KERNEL_H

#include <cstddef>
#include <multifloats.h>

using msyr2k_T = multifloats::float64x2;

/* Threading threshold (N below this stays serial). */
#define MSYR2K_OMP_MIN 32

/* Block size over the diagonal axis (env MSYR2K_NB). */
int msyr2k_block_nb(void);

/* Scale column j's UPLO triangle of C by beta (the alpha==0 / K==0 early
 * exit). Handles beta==0 (zero-fill) and beta!=1; the caller skips beta==1. */
void msyr2k_scale_col(int j, int N, char UPLO, msyr2k_T beta,
                      msyr2k_T *c, int ldc);

/* One diagonal block [jc, jc+jb) of the rank-2k update: beta-scale the block's
 * own triangle columns, accumulate the diagonal block (SIMD or scalar), then
 * the two trailing gemm calls (routed through mgemm_serial — no nested
 * OpenMP). Each block writes a disjoint column range of C → race-free. */
void msyr2k_block(int jc, int jb, int N, int K, char UPLO, char TR,
                  msyr2k_T alpha, msyr2k_T beta,
                  const msyr2k_T *a, int lda, const msyr2k_T *b, int ldb,
                  msyr2k_T *c, int ldc);

/* Pure-serial Fortran entry. No OpenMP on this path; same ABI as msyr2k_. */
extern "C" void msyr2k_serial(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const msyr2k_T *alpha_,
    const msyr2k_T *a, const int *lda_,
    const msyr2k_T *b, const int *ldb_,
    const msyr2k_T *beta_,
    msyr2k_T *c, const int *ldc_,
    std::size_t uplo_len, std::size_t trans_len);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_MSYR2K_KERNEL_H */
