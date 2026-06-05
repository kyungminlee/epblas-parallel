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
int wtrmm_block_nb(void);

/* B := 0 over the full M×N tile (the alpha==0 early-exit). */
void wtrmm_zero_B(int M, int N, wtrmm_T *b, int ldb);

/* One column slice [j_start, j_end) of a SIDE='L' multiply. UPLO/TR select the
 * variant (TR ∈ {'N','T','C'}); use_blocked picks the blocked path
 * (wgemm_serial trailing update + SIMD diagonal) vs the unblocked diagonal
 * core. Each slice owns a disjoint column range → race-free, bitwise-identical
 * to the serial sweep. */
void wtrmm_L_slice(char UPLO, char TR, int use_blocked,
                   int j_start, int j_end, int M, int nb, wtrmm_T alpha,
                   const wtrmm_T *a, int lda, wtrmm_T *b, int ldb, int nounit);

/* One row slice [row_lo, row_hi) of a SIDE='R' multiply. UPLO/TR select the
 * variant; use_blocked picks the blocked path (wgemm_serial trailing update +
 * SIMD diagonal) vs the unblocked SIMD/scalar diagonal over the slice rows.
 * Each slice owns a disjoint row range → race-free. */
void wtrmm_R_slice(char UPLO, char TR, int use_blocked,
                   int row_lo, int row_hi, int N, int nb, wtrmm_T alpha,
                   const wtrmm_T *a, int lda, wtrmm_T *b, int ldb, int nounit);

/* Pure-serial Fortran entry. No OpenMP on this path; same ABI as wtrmm_. */
extern "C" void wtrmm_serial(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const wtrmm_T *alpha_,
    const wtrmm_T *a, const int *lda_,
    wtrmm_T *b, const int *ldb_,
    std::size_t side_len, std::size_t uplo_len,
    std::size_t transa_len, std::size_t diag_len);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_WTRMM_KERNEL_H */
