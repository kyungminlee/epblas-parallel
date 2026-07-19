/*
 * wher2k_kernel.h — internal kernel surface shared by the two translation
 * units the multifloats (complex double-double / complex64x2) wher2k overlay
 * is split across. wher2k is complex HERMITIAN (cf. wsyr2k, symmetric):
 * alpha complex, beta REAL, A/B/C complex, and the diagonal of C stays real.
 *
 *   wher2k_serial.cpp   The pure single-thread Hermitian rank-2k update (no
 *                       OpenMP). Owns ALL the numerics: the AVX2 SIMD diagonal
 *                       kernels (TRANS='N' rank-2 with conjugated panels /
 *                       TRANS='C' dot, complex-alpha + conj(alpha) scaling,
 *                       real-diag preservation), the scalar diagonal fallback,
 *                       the block-size policy, the per-block worker
 *                       `wher2k_block` (beta-scale + diagonal update + the two
 *                       trailing conjugate-transpose gemm calls via
 *                       wgemm_serial — no nested OpenMP), the per-column
 *                       triangle scaler `wher2k_scale_col` and the diagonal-
 *                       imaginary zeroer `wher2k_zero_diag_im` (the alpha==0 /
 *                       K==0 early exit), and the public `wher2k_serial` entry.
 *
 *   wher2k_parallel.cpp The public Fortran entry `wher2k_` — threading
 *                       orchestration only. Fans the jc BLOCK loop across an
 *                       OpenMP team (schedule(dynamic,1)); each block owns a
 *                       disjoint set of C columns so the diagonal update and
 *                       its two trailing gemm calls write disjoint regions —
 *                       race-free and bitwise-identical to the serial sweep.
 *                       The alpha==0 / K==0 early exit either zeroes the
 *                       diagonal imaginary parts (beta==1, serial) or fans the
 *                       per-column triangle scale (schedule(static)).
 *                       Delegates to wher2k_serial when nested.
 *
 * Hermitian rank-2k keeps TRANS ∈ {'N','C'} (conjugate transpose); the trailing
 * gemms use 'C', and the diagonal cells keep their original imaginary part on
 * unpack (Netlib her2k semantics — the diagonal accumulates only the real part
 * of the update, the stored imaginary part is forced to zero by the beta pass).
 *
 * Leaf names are wher2k_-prefixed so they keep external linkage without
 * colliding with the other routines' helpers in the same archive.
 */
#ifndef EPBLAS_PARALLEL_MULTIFLOATS_WHER2K_KERNEL_H
#define EPBLAS_PARALLEL_MULTIFLOATS_WHER2K_KERNEL_H

#include <cstddef>
#include <multifloats.h>

using wher2k_TR = multifloats::float64x2;
using wher2k_TC = multifloats::complex64x2;

/* Threading threshold (N below this stays serial). */
#define WHER2K_OMP_MIN 32

/* Block size over the diagonal axis (compile-time constant). */
std::ptrdiff_t wher2k_block_nb(void);

/* Zero the imaginary part of the diagonal cell C[j,j] (the beta==1 early
 * exit — Hermitian C has a real diagonal). */
void wher2k_zero_diag_im(std::ptrdiff_t j, wher2k_TC *c, std::ptrdiff_t ldc);

/* Scale column j's UPLO triangle of C by the real beta (the alpha==0 / K==0
 * early exit, beta!=1). Handles beta==0 (zero-fill); the diagonal cell becomes
 * real (beta·re, 0). The caller handles beta==1 via wher2k_zero_diag_im. */
void wher2k_scale_col(std::ptrdiff_t j, std::ptrdiff_t n, char UPLO, wher2k_TR beta,
                      wher2k_TC *c, std::ptrdiff_t ldc);

/* One diagonal block [jc, jc+jb) of the rank-2k update: beta-scale the block's
 * own triangle columns (real-diag preservation), accumulate the diagonal block
 * (SIMD or scalar), then the two trailing conjugate-transpose gemm calls
 * (routed through wgemm_serial — no nested OpenMP). Each block writes a
 * disjoint column range of C → race-free across blocks. */
void wher2k_block(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t n, std::ptrdiff_t k, char UPLO, char TRANS,
                  wher2k_TC alpha, wher2k_TR beta,
                  const wher2k_TC *a, std::ptrdiff_t lda, const wher2k_TC *b, std::ptrdiff_t ldb,
                  wher2k_TC *c, std::ptrdiff_t ldc);

/* Pure-serial by-value entry. No OpenMP on this path; shares the ptrdiff_t
 * core ABI of wher2k_core. */
extern "C" void wher2k_serial(
    char uplo, char trans,
    std::ptrdiff_t n, std::ptrdiff_t k,
    const wher2k_TC *alpha_,
    const wher2k_TC *a, std::ptrdiff_t lda,
    const wher2k_TC *b, std::ptrdiff_t ldb,
    const wher2k_TR *beta_,
    wher2k_TC *c, std::ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_WHER2K_KERNEL_H */
