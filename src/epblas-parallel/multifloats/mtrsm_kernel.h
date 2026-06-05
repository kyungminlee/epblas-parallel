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
int mtrsm_block_nb(void);

/* B := 0 over the full M×N tile (the alpha==0 early-exit). */
void mtrsm_zero_B(int M, int N, mtrsm_T *b, int ldb);

/* One column slice [j_start, j_end) of a SIDE='L' solve. UPLO/TR select the
 * variant; use_blocked picks the blocked path (mgemm_serial trailing update +
 * SIMD diagonal) vs the scalar unblocked core. Each slice owns a disjoint
 * column range → race-free, bitwise-identical to the serial sweep. */
void mtrsm_L_slice(char UPLO, char TR, int use_blocked,
                   int j_start, int j_end, int M, int nb, mtrsm_T alpha,
                   const mtrsm_T *a, int lda, mtrsm_T *b, int ldb, int nounit);

/* One row slice [row_lo, row_hi) of a SIDE='R' solve (SIMD 4-row chunks +
 * scalar tail). Each slice owns a disjoint row range → race-free. */
void mtrsm_R_slice(char UPLO, char TR, int row_lo, int row_hi,
                   int N, mtrsm_T alpha,
                   const mtrsm_T *a, int lda, mtrsm_T *b, int ldb, int nounit);

/* Pure-serial Fortran entry. No OpenMP on this path; same ABI as mtrsm_. */
extern "C" void mtrsm_serial(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const mtrsm_T *alpha_,
    const mtrsm_T *a, const int *lda_,
    mtrsm_T *b, const int *ldb_,
    std::size_t side_len, std::size_t uplo_len,
    std::size_t transa_len, std::size_t diag_len);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_MTRSM_KERNEL_H */
