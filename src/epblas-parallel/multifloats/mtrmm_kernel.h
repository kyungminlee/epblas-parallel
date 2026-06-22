/*
 * mtrmm_kernel.h — internal kernel surface shared by the two translation
 * units the multifloats (double-double / float64x2) mtrmm overlay is split
 * across:
 *
 *   mtrmm_serial.cpp    The pure single-thread triangular multiply (no
 *                       OpenMP). Owns ALL the numerics: the scalar column
 *                       (SIDE='L') and row (SIDE='R') "core" kernels, the
 *                       AVX2 SIMD diagonal kernels (SIDE='L' packed-SoA and
 *                       SIDE='R' 4-row chunks), the block-size policy, and
 *                       the blocked chunk workers for BOTH sides — whose
 *                       trailing-matrix update routes through mgemm_serial
 *                       (no nested OpenMP). Exposes the per-slice workers
 *                       mtrmm_L_slice / mtrmm_R_slice plus the public
 *                       `mtrmm_serial` entry.
 *
 *   mtrmm_parallel.cpp  The public Fortran entry `mtrmm_` — threading
 *                       orchestration only. Fans the free axis (B's columns
 *                       for SIDE='L', B's rows for SIDE='R') across an OpenMP
 *                       team; each thread owns a disjoint slice and runs a
 *                       per-slice worker with no cross-thread synchronization.
 *                       Delegates to mtrmm_serial when nested or below the
 *                       threading threshold.
 *
 * The serial and parallel paths drive numerics through the SAME per-slice
 * workers, so a static partition is bitwise-identical to the serial sweep.
 *
 * As for the real (mtrsm) twin, TRANSA's conjugate-transpose collapses to
 * plain transpose ('C' ≡ 'T' for real DD); the slice workers take the already
 * normalized TR ∈ {'N','T'} and map it (with UPLO) to the 4-way variant.
 *
 * Leaf names are mtrmm_-prefixed so they keep external linkage without
 * colliding with the other routines' helpers in the same archive.
 */
#ifndef EPBLAS_PARALLEL_MULTIFLOATS_MTRMM_KERNEL_H
#define EPBLAS_PARALLEL_MULTIFLOATS_MTRMM_KERNEL_H

#include <cstddef>
#include <multifloats.h>

using mtrmm_T = multifloats::float64x2;

/* Threading threshold (free axis below this stays serial). */
#define MTRMM_OMP_MIN 32

/* Block size over the triangular axis for the blocked paths (env MTRMM_NB). */
std::ptrdiff_t mtrmm_block_nb(void);

/* B := 0 over the full M×N tile (the alpha==0 early-exit). */
void mtrmm_zero_B(std::ptrdiff_t M, std::ptrdiff_t N, mtrmm_T *b, std::ptrdiff_t ldb);

/* One column slice [j_start, j_end) of a SIDE='L' multiply. UPLO/TR select the
 * variant (TR ∈ {'N','T'}); use_blocked picks the blocked path (mgemm_serial
 * trailing update + SIMD diagonal) vs the unblocked diagonal core. Each slice
 * owns a disjoint column range → race-free, bitwise-identical to the serial
 * sweep. */
void mtrmm_L_slice(char UPLO, char TR, std::ptrdiff_t use_blocked,
                   std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t M, std::ptrdiff_t nb, mtrmm_T alpha,
                   const mtrmm_T *a, std::ptrdiff_t lda, mtrmm_T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit);

/* One row slice [row_lo, row_hi) of a SIDE='R' multiply. UPLO/TR select the
 * variant; use_blocked picks the blocked path (mgemm_serial trailing update +
 * SIMD diagonal) vs the unblocked SIMD/scalar diagonal over the slice rows.
 * Each slice owns a disjoint row range → race-free. */
void mtrmm_R_slice(char UPLO, char TR, std::ptrdiff_t use_blocked,
                   std::ptrdiff_t row_lo, std::ptrdiff_t row_hi, std::ptrdiff_t N, std::ptrdiff_t nb, mtrmm_T alpha,
                   const mtrmm_T *a, std::ptrdiff_t lda, mtrmm_T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit);

/* Pure-serial Fortran entry. No OpenMP on this path; same ABI as mtrmm_. */
extern "C" void mtrmm_serial(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const mtrmm_T *alpha_,
    const mtrmm_T *a, const int *lda_,
    mtrmm_T *b, const int *ldb_,
    std::size_t side_len, std::size_t uplo_len,
    std::size_t transa_len, std::size_t diag_len);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_MTRMM_KERNEL_H */
