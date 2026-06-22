/*
 * wherk_kernel.h — internal kernel surface shared by the two translation
 * units the multifloats (complex double-double / complex64x2) wherk overlay is
 * split across:
 *
 *   wherk_serial.cpp    The pure single-thread Hermitian rank-k update (no
 *                       OpenMP). alpha/beta REAL, A/C complex, the diagonal of
 *                       C stays real. Owns ALL the numerics: the AVX2 SIMD
 *                       diagonal kernels (TR='N' rank-1 with conjugated panel /
 *                       TR='C' dot, real-alpha scaling, real-diag preservation),
 *                       the scalar diagonal fallback, the block-size policy, the
 *                       per-block worker `wherk_block` (beta-scale + diagonal
 *                       update + trailing gemm via wgemm_serial with conjugate
 *                       transpose — no nested OpenMP), the per-column triangle
 *                       scaler `wherk_scale_col` and the diagonal-imaginary
 *                       zeroer `wherk_zero_diag_im` (the alpha==0 / K==0 early
 *                       exit), and the public `wherk_serial` entry.
 *
 *   wherk_parallel.cpp  The public Fortran entry `wherk_` — threading
 *                       orchestration only. Fans the jc BLOCK loop across an
 *                       OpenMP team (schedule(dynamic,1)); each block owns a
 *                       disjoint set of C columns so the diagonal update and
 *                       its trailing gemm write disjoint regions — race-free
 *                       and bitwise-identical to the serial sweep. The
 *                       alpha==0 / K==0 early exit either zeroes the diagonal
 *                       imaginary parts (beta==1, serial) or fans the per-column
 *                       triangle scale (schedule(static)). Delegates to
 *                       wherk_serial when nested.
 *
 * Hermitian rank-k keeps TR ∈ {'N','C'} (conjugate transpose); the trailing
 * gemm uses 'C', and the diagonal cells keep their original imaginary part on
 * unpack (Netlib herk semantics). The slice workers take TR verbatim.
 *
 * Leaf names are wherk_-prefixed so they keep external linkage without
 * colliding with the other routines' helpers in the same archive.
 */
#ifndef EPBLAS_PARALLEL_MULTIFLOATS_WHERK_KERNEL_H
#define EPBLAS_PARALLEL_MULTIFLOATS_WHERK_KERNEL_H

#include <cstddef>
#include <multifloats.h>

using wherk_R = multifloats::float64x2;
using wherk_T = multifloats::complex64x2;

/* Threading threshold (N below this stays serial). */
#define WHERK_OMP_MIN 32

/* Block size over the diagonal axis (env WHERK_NB). */
std::ptrdiff_t wherk_block_nb(void);

/* Zero the imaginary part of the diagonal cell C[j,j] (the beta==1 early
 * exit — Hermitian C has a real diagonal). */
void wherk_zero_diag_im(std::ptrdiff_t j, wherk_T *c, std::ptrdiff_t ldc);

/* Scale column j's UPLO triangle of C by the real beta (the alpha==0 / K==0
 * early exit, beta!=1). Handles beta==0 (zero-fill); the diagonal cell becomes
 * real (beta·re, 0). The caller handles beta==1 via wherk_zero_diag_im. */
void wherk_scale_col(std::ptrdiff_t j, std::ptrdiff_t N, char UPLO, wherk_R beta,
                     wherk_T *c, std::ptrdiff_t ldc);

/* One diagonal block [jc, jc+jb) of the rank-k update: beta-scale the block's
 * own triangle columns (real-diag preservation), accumulate the diagonal block
 * (SIMD or scalar), then the trailing conjugate-transpose gemm (routed through
 * wgemm_serial — no nested OpenMP). Each block writes a disjoint column range
 * of C → race-free across blocks. */
void wherk_block(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t N, std::ptrdiff_t K, char UPLO, char TR,
                 wherk_R alpha, wherk_R beta,
                 const wherk_T *a, std::ptrdiff_t lda, wherk_T *c, std::ptrdiff_t ldc);

/* Pure-serial Fortran entry. No OpenMP on this path; same ABI as wherk_. */
extern "C" void wherk_serial(
    char uplo, char trans,
    std::ptrdiff_t N, std::ptrdiff_t K,
    const wherk_R *alpha_,
    const wherk_T *a, std::ptrdiff_t lda,
    const wherk_R *beta_,
    wherk_T *c, std::ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_WHERK_KERNEL_H */
