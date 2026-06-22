/*
 * wtrmm_kernel.h — internal kernel surface shared by the two translation
 * units the multifloats (complex double-double / complex64x2) wtrmm overlay
 * is split across:
 *
 *   wtrmm_serial.cpp    The pure single-thread triangular multiply (no
 *                       OpenMP). Owns ALL the numerics: the scalar column
 *                       (SIDE='L') and row (SIDE='R') "core" kernels, the
 *                       AVX2 SIMD diagonal kernels (SIDE='L' packed-SoA and
 *                       SIDE='R' 4-row chunks), the block-size policy, and
 *                       the blocked chunk workers for BOTH sides — whose
 *                       trailing-matrix update routes through wgemm_serial
 *                       (no nested OpenMP). Exposes the per-slice workers
 *                       wtrmm_L_slice / wtrmm_R_slice plus the public
 *                       `wtrmm_serial` entry.
 *
 *   wtrmm_parallel.cpp  The public Fortran entry `wtrmm_` — threading
 *                       orchestration only. Fans the free axis (B's columns
 *                       for SIDE='L', B's rows for SIDE='R') across an OpenMP
 *                       team; each thread owns a disjoint slice and runs a
 *                       per-slice worker with no cross-thread synchronization.
 *                       Delegates to wtrmm_serial when nested or below the
 *                       threading threshold.
 *
 * The serial and parallel paths drive numerics through the SAME per-slice
 * workers, so a static partition is bitwise-identical to the serial sweep.
 *
 * Unlike the real (mtrmm) twin, TRANSA is kept as 'N'/'T'/'C' DISTINCT — for
 * complex, transpose ('T') and conjugate transpose ('C') differ, so the slice
 * workers take TR verbatim and map it (with UPLO) to the 6-way variant.
 *
 * Leaf names are wtrmm_-prefixed so they keep external linkage without
 * colliding with the other routines' helpers in the same archive.
 */
#ifndef EPBLAS_PARALLEL_MULTIFLOATS_WTRMM_KERNEL_H
#define EPBLAS_PARALLEL_MULTIFLOATS_WTRMM_KERNEL_H

#include <cstddef>
#include <multifloats.h>

using wtrmm_T = multifloats::complex64x2;

/* Threading threshold (free axis below this stays serial). */
#define WTRMM_OMP_MIN 32

/* Block size over the triangular axis for the blocked paths (env WTRMM_NB). */
std::ptrdiff_t wtrmm_block_nb(void);

/* B := 0 over the full M×N tile (the alpha==0 early-exit). */
void wtrmm_zero_B(std::ptrdiff_t M, std::ptrdiff_t N, wtrmm_T *b, std::ptrdiff_t ldb);

/* One column slice [j_start, j_end) of a SIDE='L' multiply. UPLO/TR select the
 * variant (TR ∈ {'N','T','C'}); use_blocked picks the blocked path
 * (wgemm_serial trailing update + SIMD diagonal) vs the unblocked diagonal
 * core. Each slice owns a disjoint column range → race-free, bitwise-identical
 * to the serial sweep. */
void wtrmm_L_slice(char UPLO, char TR, std::ptrdiff_t use_blocked,
                   std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t M, std::ptrdiff_t nb, wtrmm_T alpha,
                   const wtrmm_T *a, std::ptrdiff_t lda, wtrmm_T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit);

/* One row slice [row_lo, row_hi) of a SIDE='R' multiply. UPLO/TR select the
 * variant; use_blocked picks the blocked path (wgemm_serial trailing update +
 * SIMD diagonal) vs the unblocked SIMD/scalar diagonal over the slice rows.
 * Each slice owns a disjoint row range → race-free. */
void wtrmm_R_slice(char UPLO, char TR, std::ptrdiff_t use_blocked,
                   std::ptrdiff_t row_lo, std::ptrdiff_t row_hi, std::ptrdiff_t N, std::ptrdiff_t nb, wtrmm_T alpha,
                   const wtrmm_T *a, std::ptrdiff_t lda, wtrmm_T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit);

/* Pure-serial Fortran entry. No OpenMP on this path; same ABI as wtrmm_. */
extern "C" void wtrmm_serial(
    char side, char uplo, char transa, char diag,
    std::ptrdiff_t M, std::ptrdiff_t N,
    const wtrmm_T *alpha_,
    const wtrmm_T *a, std::ptrdiff_t lda,
    wtrmm_T *b, std::ptrdiff_t ldb);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_WTRMM_KERNEL_H */
