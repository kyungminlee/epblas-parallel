/*
 * wtrsm_kernel.h — internal kernel surface shared by the two translation
 * units the multifloats (complex double-double / complex64x2) wtrsm overlay
 * is split across:
 *
 *   wtrsm_serial.cpp    The pure single-thread triangular solve (no OpenMP).
 *                       Owns ALL the numerics: the scalar column/row "core"
 *                       kernels, the AVX2 SIMD diagonal kernels (SIDE='L' and
 *                       SIDE='R'), the block-size policy, and the blocked
 *                       SIDE='L' chunk worker (whose trailing update routes
 *                       through wgemm_serial). Exposes per-slice workers
 *                       (wtrsm_L_slice / wtrsm_R_slice) plus the public
 *                       `wtrsm_serial` entry.
 *
 *   wtrsm_parallel.cpp  The public Fortran entry `wtrsm_` — threading
 *                       orchestration only. Fans the free axis (B's columns
 *                       for SIDE='L', B's rows for SIDE='R') across an OpenMP
 *                       team; each thread owns a disjoint slice and runs a
 *                       per-slice worker with no cross-thread synchronization.
 *                       Delegates to wtrsm_serial when nested or below the
 *                       threading threshold.
 *
 * The serial and parallel paths drive numerics through the SAME per-slice
 * workers, so a static partition is bitwise-identical to the serial sweep.
 *
 * Unlike the real (mtrsm) twin, TRANSA is kept as 'N'/'T'/'C' DISTINCT — for
 * complex, transpose ('T') and conjugate transpose ('C') differ, so the slice
 * workers take TR verbatim and map it (with UPLO) to the 6-way variant.
 *
 * Leaf names are wtrsm_-prefixed so they keep external linkage without
 * colliding with the other routines' helpers in the same archive.
 */
#ifndef EPBLAS_PARALLEL_MULTIFLOATS_WTRSM_KERNEL_H
#define EPBLAS_PARALLEL_MULTIFLOATS_WTRSM_KERNEL_H

#include <cstddef>
#include <multifloats.h>

using wtrsm_T = multifloats::complex64x2;

/* Threading threshold (free axis below this stays serial). */
#define WTRSM_OMP_N_MIN 32

/* Block size over the M axis for the blocked SIDE='L' path (env WTRSM_NB). */
std::ptrdiff_t wtrsm_block_nb(void);

/* B := 0 over the full M×N tile (the alpha==0 early-exit). */
void wtrsm_zero_B(std::ptrdiff_t M, std::ptrdiff_t N, wtrsm_T *b, std::ptrdiff_t ldb);

/* One column slice [j_start, j_end) of a SIDE='L' solve. UPLO/TR select the
 * variant (TR ∈ {'N','T','C'}); use_blocked picks the blocked path
 * (wgemm_serial trailing update + SIMD diagonal) vs the scalar unblocked core.
 * Each slice owns a disjoint column range → race-free, bitwise-identical to
 * the serial sweep. */
void wtrsm_L_slice(char UPLO, char TR, std::ptrdiff_t use_blocked,
                   std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t M, std::ptrdiff_t nb, wtrsm_T alpha,
                   const wtrsm_T *a, std::ptrdiff_t lda, wtrsm_T *b, std::ptrdiff_t ldb, bool nounit);

/* One row slice [row_lo, row_hi) of a SIDE='R' solve (SIMD 4-row chunks +
 * scalar tail). TR ∈ {'N','T','C'}. Each slice owns a disjoint row range →
 * race-free. */
void wtrsm_R_slice(char UPLO, char TR, std::ptrdiff_t row_lo, std::ptrdiff_t row_hi,
                   std::ptrdiff_t N, wtrsm_T alpha,
                   const wtrsm_T *a, std::ptrdiff_t lda, wtrsm_T *b, std::ptrdiff_t ldb, bool nounit);

/* Pure-serial Fortran entry. No OpenMP on this path; same ABI as wtrsm_. */
extern "C" void wtrsm_serial(
    char side, char uplo, char transa, char diag,
    std::ptrdiff_t M, std::ptrdiff_t N,
    const wtrsm_T *alpha_,
    const wtrsm_T *a, std::ptrdiff_t lda,
    wtrsm_T *b, std::ptrdiff_t ldb);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_WTRSM_KERNEL_H */
