/*
 * whemm_kernel.h — internal kernel surface shared by the two translation
 * units the multifloats (complex double-double / complex64x2) whemm overlay is
 * split across. whemm is complex HERMITIAN (not symmetric — see wsymm).
 *
 *   whemm_serial.cpp    The pure single-thread complex-Hermitian matrix
 *                       multiply (no OpenMP). Owns ALL the numerics: the AVX2
 *                       SIMD diagonal kernels (SIDE='L' 4-column rank-1,
 *                       SIDE='R' 4-row dot via the complex AoS<->SoA shuffles),
 *                       the scalar diagonal fallbacks, the block-size policy,
 *                       the two per-block workers `whemm_block_L` /
 *                       `whemm_block_R` (beta-scale the block + the leading and
 *                       trailing gemm wings via wgemm_serial + the diagonal
 *                       update — no nested OpenMP), the per-column scaler
 *                       `whemm_scale_col` (the alpha==0 early exit), and the
 *                       public `whemm_serial` entry. Hermitian specifics: A's
 *                       off-diagonal use is conjugated, A(i,i) is forced real,
 *                       and the trailing/leading wings use conjugate-transpose
 *                       'C' (not 'T').
 *
 *   whemm_parallel.cpp  The public Fortran entry `whemm_` — threading
 *                       orchestration only. Fans the block loop across an
 *                       OpenMP team (schedule(dynamic,1)): SIDE='L' over the
 *                       row blocks ic, SIDE='R' over the column blocks jc. Each
 *                       block writes a disjoint slab of C — race-free and
 *                       bitwise-identical to the serial sweep. Delegates to
 *                       whemm_serial when nested.
 *
 * Leaf names are whemm_-prefixed so they keep external linkage without
 * colliding with the other routines' helpers in the same archive.
 */
#ifndef EPBLAS_PARALLEL_MULTIFLOATS_WHEMM_KERNEL_H
#define EPBLAS_PARALLEL_MULTIFLOATS_WHEMM_KERNEL_H

#include <cstddef>
#include <multifloats.h>

using whemm_T = multifloats::complex64x2;

/* Threading threshold (the threaded axis below this stays serial). */
#define WHEMM_OMP_MIN 32

/* Block size over the threaded axis (compile-time constant). */
std::ptrdiff_t whemm_block_nb(void);

/* Scale column j of C (rows 0..M) by beta — the alpha==0 early exit. Handles
 * beta==0 (zero-fill). The caller skips this entirely for beta==1. */
void whemm_scale_col(std::ptrdiff_t j, std::ptrdiff_t m, whemm_T beta, whemm_T *c, std::ptrdiff_t ldc);

/* SIDE='L' row block [ic, ic+ib): beta-scale the block's rows across all N
 * columns, then the leading gemm wing, the Hermitian diagonal block, and the
 * trailing gemm wing (both wings routed through wgemm_serial — no nested
 * OpenMP). Writes only rows [ic, ic+ib) of C → row-disjoint across blocks. */
void whemm_block_L(std::ptrdiff_t ic, std::ptrdiff_t ib, std::ptrdiff_t m, std::ptrdiff_t n, char UPLO,
                   whemm_T alpha, whemm_T beta,
                   const whemm_T *a, std::ptrdiff_t lda, const whemm_T *b, std::ptrdiff_t ldb,
                   whemm_T *c, std::ptrdiff_t ldc);

/* SIDE='R' column block [jc, jc+jb): beta-scale the block's columns over all M
 * rows, then the leading gemm wing, the Hermitian diagonal block, and the
 * trailing gemm wing. Writes only columns [jc, jc+jb) of C → column-disjoint
 * across blocks. */
void whemm_block_R(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t m, std::ptrdiff_t n, char UPLO,
                   whemm_T alpha, whemm_T beta,
                   const whemm_T *a, std::ptrdiff_t lda, const whemm_T *b, std::ptrdiff_t ldb,
                   whemm_T *c, std::ptrdiff_t ldc);

/* Pure-serial Fortran entry. No OpenMP on this path; same ABI as whemm_. */
extern "C" void whemm_serial(
    char side, char uplo,
    std::ptrdiff_t m, std::ptrdiff_t n,
    const whemm_T *alpha_,
    const whemm_T *a, std::ptrdiff_t lda,
    const whemm_T *b, std::ptrdiff_t ldb,
    const whemm_T *beta_,
    whemm_T *c, std::ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_WHEMM_KERNEL_H */
