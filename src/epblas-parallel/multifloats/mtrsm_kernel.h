/*
 * mtrsm_kernel.h — internal kernel surface shared by the two translation
 * units the multifloats (double-double / float64x2) mtrsm overlay is split
 * across:
 *
 *   mtrsm_serial.cpp    The pure single-thread triangular solve (no OpenMP).
 *                       Owns ALL the numerics: the scalar column/row "core"
 *                       kernels, the AVX2 SIMD diagonal kernels (SIDE='L' and
 *                       SIDE='R'), the block-size policy, and the blocked
 *                       SIDE='L' chunk worker (whose trailing update routes
 *                       through mgemm_serial). Exposes per-slice workers
 *                       (mtrsm_L_slice / mtrsm_R_slice) plus the public
 *                       `mtrsm_serial` entry.
 *
 *   mtrsm_parallel.cpp  The public Fortran entry `mtrsm_` — threading
 *                       orchestration only. Fans the free axis (B's columns
 *                       for SIDE='L', B's rows for SIDE='R') across an OpenMP
 *                       team; each thread owns a disjoint slice and runs a
 *                       per-slice worker with no cross-thread synchronization.
 *                       Delegates to mtrsm_serial when nested or below the
 *                       threading threshold.
 *
 * The serial and parallel paths drive numerics through the SAME per-slice
 * workers, so a static partition is bitwise-identical to the serial sweep.
 *
 * Leaf names are mtrsm_-prefixed so they keep external linkage without
 * colliding with the other routines' helpers in the same archive.
 */
#ifndef EPBLAS_PARALLEL_MULTIFLOATS_MTRSM_KERNEL_H
#define EPBLAS_PARALLEL_MULTIFLOATS_MTRSM_KERNEL_H

#include <cstddef>
#include <multifloats.h>

using mtrsm_T = multifloats::float64x2;

/* Threading threshold (free axis below this stays serial). */
#define MTRSM_OMP_N_MIN 32

/* Block size over the M axis for the blocked SIDE='L' path (env MTRSM_NB). */
std::ptrdiff_t mtrsm_block_nb(void);

/* B := 0 over the full M×N tile (the alpha==0 early-exit). */
void mtrsm_zero_B(std::ptrdiff_t m, std::ptrdiff_t n, mtrsm_T *b, std::ptrdiff_t ldb);

/* One column slice [j_start, j_end) of a SIDE='L' solve. UPLO/TR select the
 * variant; use_blocked picks the blocked path (mgemm_serial trailing update +
 * SIMD diagonal) vs the scalar unblocked core. Each slice owns a disjoint
 * column range → race-free, bitwise-identical to the serial sweep. */
void mtrsm_L_slice(char UPLO, char TR, std::ptrdiff_t use_blocked,
                   std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, std::ptrdiff_t nb, mtrsm_T alpha,
                   const mtrsm_T *a, std::ptrdiff_t lda, mtrsm_T *b, std::ptrdiff_t ldb, bool nounit);

/* One row slice [row_lo, row_hi) of a SIDE='R' solve (SIMD 4-row chunks +
 * scalar tail). Each slice owns a disjoint row range → race-free. */
void mtrsm_R_slice(char UPLO, char TR, std::ptrdiff_t row_lo, std::ptrdiff_t row_hi,
                   std::ptrdiff_t n, mtrsm_T alpha,
                   const mtrsm_T *a, std::ptrdiff_t lda, mtrsm_T *b, std::ptrdiff_t ldb, bool nounit);

/* Pure-serial by-value core. No OpenMP on this path; internal (no Fortran ABI). */
extern "C" void mtrsm_serial(
    char side, char uplo, char transa, char diag,
    std::ptrdiff_t m, std::ptrdiff_t n,
    const mtrsm_T *alpha_,
    const mtrsm_T *a, std::ptrdiff_t lda,
    mtrsm_T *b, std::ptrdiff_t ldb);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_MTRSM_KERNEL_H */
