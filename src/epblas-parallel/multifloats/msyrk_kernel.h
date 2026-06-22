/*
 * msyrk_kernel.h — internal kernel surface shared by the two translation
 * units the multifloats (double-double / float64x2) msyrk overlay is split
 * across:
 *
 *   msyrk_serial.cpp    The pure single-thread symmetric rank-k update (no
 *                       OpenMP). Owns ALL the numerics: the AVX2 SIMD diagonal
 *                       kernels (TR='N' rank-1 and TR='T' dot forms), the
 *                       scalar diagonal fallback, the block-size policy, the
 *                       per-block worker `msyrk_block` (beta-scale + diagonal
 *                       update + trailing gemm via mgemm_serial — no nested
 *                       OpenMP), the per-column triangle scaler `msyrk_scale_col`
 *                       (the alpha==0 / K==0 early-exit), and the public
 *                       `msyrk_serial` entry.
 *
 *   msyrk_parallel.cpp  The public Fortran entry `msyrk_` — threading
 *                       orchestration only. Fans the jc BLOCK loop across an
 *                       OpenMP team (schedule(dynamic,1)); each block owns a
 *                       disjoint set of C columns so the diagonal update and
 *                       its trailing gemm write disjoint regions — race-free
 *                       and bitwise-identical to the serial sweep. The
 *                       alpha==0 / K==0 early exit fans the per-column triangle
 *                       scale (schedule(static)). Delegates to msyrk_serial
 *                       when nested.
 *
 * The serial and parallel paths drive numerics through the SAME per-block
 * worker, so a static/dynamic partition is bitwise-identical to the serial
 * sweep (the per-block accumulation order is independent of the block schedule).
 *
 * Leaf names are msyrk_-prefixed so they keep external linkage without
 * colliding with the other routines' helpers in the same archive.
 */
#ifndef EPBLAS_PARALLEL_MULTIFLOATS_MSYRK_KERNEL_H
#define EPBLAS_PARALLEL_MULTIFLOATS_MSYRK_KERNEL_H

#include <cstddef>
#include <multifloats.h>

using msyrk_T = multifloats::float64x2;

/* Threading threshold (N below this stays serial). */
#define MSYRK_OMP_MIN 32

/* Block size over the diagonal axis (env MSYRK_NB). */
std::ptrdiff_t msyrk_block_nb(void);

/* Scale column j's UPLO triangle of C by beta (the alpha==0 / K==0 early
 * exit). Handles beta==0 (zero-fill) and beta!=1; the caller skips beta==1. */
void msyrk_scale_col(std::ptrdiff_t j, std::ptrdiff_t N, char UPLO, msyrk_T beta,
                     msyrk_T *c, std::ptrdiff_t ldc);

/* One diagonal block [jc, jc+jb) of the rank-k update: beta-scale the block's
 * own triangle columns, accumulate the diagonal block (SIMD or scalar), then
 * the trailing gemm (routed through mgemm_serial — no nested OpenMP). Each
 * block writes a disjoint column range of C → race-free across blocks. */
void msyrk_block(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t N, std::ptrdiff_t K, char UPLO, char TR,
                 msyrk_T alpha, msyrk_T beta,
                 const msyrk_T *a, std::ptrdiff_t lda, msyrk_T *c, std::ptrdiff_t ldc);

/* Pure-serial Fortran entry. No OpenMP on this path; same ABI as msyrk_. */
extern "C" void msyrk_serial(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const msyrk_T *alpha_,
    const msyrk_T *a, const int *lda_,
    const msyrk_T *beta_,
    msyrk_T *c, const int *ldc_,
    std::size_t uplo_len, std::size_t trans_len);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_MSYRK_KERNEL_H */
