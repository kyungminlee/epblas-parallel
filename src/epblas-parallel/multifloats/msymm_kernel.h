/*
 * msymm_kernel.h — internal kernel surface shared by the two translation
 * units the multifloats (double-double / float64x2) msymm overlay is split
 * across:
 *
 *   msymm_serial.cpp    The pure single-thread symmetric matrix multiply (no
 *                       OpenMP). Owns ALL the numerics: the AVX2 SIMD diagonal
 *                       kernels (SIDE='L' 4-column rank-1, SIDE='R' 4-row dot),
 *                       the scalar diagonal fallbacks, the block-size policy,
 *                       the two per-block workers `msymm_block_L` /
 *                       `msymm_block_R` (beta-scale the block + the leading and
 *                       trailing gemm wings via mgemm_serial + the diagonal
 *                       update — no nested OpenMP), the per-column scaler
 *                       `msymm_scale_col` (the alpha==0 early exit), and the
 *                       public `msymm_serial` entry.
 *
 *   msymm_parallel.cpp  The public Fortran entry `msymm_` — threading
 *                       orchestration only. Fans the block loop across an
 *                       OpenMP team (schedule(dynamic,1)): SIDE='L' over the
 *                       row blocks ic, SIDE='R' over the column blocks jc. Each
 *                       block writes a disjoint slab of C (rows for L, columns
 *                       for R) so the work is race-free and bitwise-identical
 *                       to the serial sweep. Delegates to msymm_serial when
 *                       nested.
 *
 * Leaf names are msymm_-prefixed so they keep external linkage without
 * colliding with the other routines' helpers in the same archive.
 */
#ifndef EPBLAS_PARALLEL_MULTIFLOATS_MSYMM_KERNEL_H
#define EPBLAS_PARALLEL_MULTIFLOATS_MSYMM_KERNEL_H

#include <cstddef>
#include <multifloats.h>

using msymm_T = multifloats::float64x2;

/* Threading threshold (the threaded axis below this stays serial). */
#define MSYMM_OMP_MIN 32

/* Block size over the threaded axis (env MSYMM_NB). */
std::ptrdiff_t msymm_block_nb(void);

/* Scale column j of C (rows 0..M) by beta — the alpha==0 early exit. Handles
 * beta==0 (zero-fill). The caller skips this entirely for beta==1. */
void msymm_scale_col(std::ptrdiff_t j, std::ptrdiff_t M, msymm_T beta, msymm_T *c, std::ptrdiff_t ldc);

/* SIDE='L' row block [ic, ic+ib): beta-scale the block's rows across all N
 * columns, then the leading gemm wing, the symmetric diagonal block, and the
 * trailing gemm wing (both wings routed through mgemm_serial — no nested
 * OpenMP). Writes only rows [ic, ic+ib) of C → row-disjoint across blocks. */
void msymm_block_L(std::ptrdiff_t ic, std::ptrdiff_t ib, std::ptrdiff_t M, std::ptrdiff_t N, char UPLO,
                   msymm_T alpha, msymm_T beta,
                   const msymm_T *a, std::ptrdiff_t lda, const msymm_T *b, std::ptrdiff_t ldb,
                   msymm_T *c, std::ptrdiff_t ldc);

/* SIDE='R' column block [jc, jc+jb): beta-scale the block's columns over all M
 * rows, then the leading gemm wing, the symmetric diagonal block, and the
 * trailing gemm wing. Writes only columns [jc, jc+jb) of C → column-disjoint
 * across blocks. */
void msymm_block_R(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t M, std::ptrdiff_t N, char UPLO,
                   msymm_T alpha, msymm_T beta,
                   const msymm_T *a, std::ptrdiff_t lda, const msymm_T *b, std::ptrdiff_t ldb,
                   msymm_T *c, std::ptrdiff_t ldc);

/* Pure-serial by-value core. No OpenMP on this path; internal (no Fortran ABI). */
extern "C" void msymm_serial(
    char side, char uplo,
    std::ptrdiff_t M, std::ptrdiff_t N,
    const msymm_T *alpha_,
    const msymm_T *a, std::ptrdiff_t lda,
    const msymm_T *b, std::ptrdiff_t ldb,
    const msymm_T *beta_,
    msymm_T *c, std::ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_MSYMM_KERNEL_H */
