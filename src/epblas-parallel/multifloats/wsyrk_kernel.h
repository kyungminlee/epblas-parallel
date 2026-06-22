/*
 * wsyrk_kernel.h — internal kernel surface shared by the two translation
 * units the multifloats (complex double-double / complex64x2) wsyrk overlay is
 * split across:
 *
 *   wsyrk_serial.cpp    The pure single-thread complex symmetric rank-k update
 *                       (no OpenMP). Owns ALL the numerics: the AVX2 SIMD
 *                       diagonal kernels (TR='N' rank-1 and TR='T' dot forms,
 *                       cmul/cadd over 4 SoA arrays), the scalar diagonal
 *                       fallback, the block-size policy, the per-block worker
 *                       `wsyrk_block` (beta-scale + diagonal update + trailing
 *                       gemm via wgemm_serial — no nested OpenMP), the
 *                       per-column triangle scaler `wsyrk_scale_col` (the
 *                       alpha==0 / K==0 early-exit), and the public
 *                       `wsyrk_serial` entry.
 *
 *   wsyrk_parallel.cpp  The public Fortran entry `wsyrk_` — threading
 *                       orchestration only. Fans the jc BLOCK loop across an
 *                       OpenMP team (schedule(dynamic,1)); each block owns a
 *                       disjoint set of C columns so the diagonal update and
 *                       its trailing gemm write disjoint regions — race-free
 *                       and bitwise-identical to the serial sweep. The
 *                       alpha==0 / K==0 early exit fans the per-column triangle
 *                       scale (schedule(static)). Delegates to wsyrk_serial
 *                       when nested.
 *
 * Symmetric rank-k for complex input keeps TR ∈ {'N','T'} (no conjugate form);
 * the trailing gemm uses the plain transpose ('T'), so the slice workers take
 * TR verbatim.
 *
 * Leaf names are wsyrk_-prefixed so they keep external linkage without
 * colliding with the other routines' helpers in the same archive.
 */
#ifndef EPBLAS_PARALLEL_MULTIFLOATS_WSYRK_KERNEL_H
#define EPBLAS_PARALLEL_MULTIFLOATS_WSYRK_KERNEL_H

#include <cstddef>
#include <multifloats.h>

using wsyrk_T = multifloats::complex64x2;

/* Threading threshold (N below this stays serial). */
#define WSYRK_OMP_MIN 32

/* Block size over the diagonal axis (env WSYRK_NB). */
std::ptrdiff_t wsyrk_block_nb(void);

/* Scale column j's UPLO triangle of C by beta (the alpha==0 / K==0 early
 * exit). Handles beta==0 (zero-fill) and beta!=1; the caller skips beta==1. */
void wsyrk_scale_col(std::ptrdiff_t j, std::ptrdiff_t N, char UPLO, wsyrk_T beta,
                     wsyrk_T *c, std::ptrdiff_t ldc);

/* One diagonal block [jc, jc+jb) of the rank-k update: beta-scale the block's
 * own triangle columns, accumulate the diagonal block (SIMD or scalar), then
 * the trailing gemm (routed through wgemm_serial — no nested OpenMP). Each
 * block writes a disjoint column range of C → race-free across blocks. */
void wsyrk_block(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t N, std::ptrdiff_t K, char UPLO, char TR,
                 wsyrk_T alpha, wsyrk_T beta,
                 const wsyrk_T *a, std::ptrdiff_t lda, wsyrk_T *c, std::ptrdiff_t ldc);

/* Pure-serial Fortran entry. No OpenMP on this path; same ABI as wsyrk_. */
extern "C" void wsyrk_serial(
    char uplo, char trans,
    std::ptrdiff_t N, std::ptrdiff_t K,
    const wsyrk_T *alpha_,
    const wsyrk_T *a, std::ptrdiff_t lda,
    const wsyrk_T *beta_,
    wsyrk_T *c, std::ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_WSYRK_KERNEL_H */
