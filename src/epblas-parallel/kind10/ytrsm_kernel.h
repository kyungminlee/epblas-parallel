/*
 * ytrsm_kernel.h — internal shared declarations for the kind10 complex
 * (COMPLEX(KIND=10) / _Complex long double) triangular-solve overlay,
 * split across two translation units:
 *
 *   ytrsm_serial.c   — all the math (column/row-range cores + the blocked
 *                      worker, whose SIDE='L' trailing updates call
 *                      ygemm_serial) plus the pure-serial Fortran-ABI entry
 *                      `ytrsm_serial`. No `#pragma omp`.
 *   ytrsm_parallel.c — the public Fortran entry `ytrsm_`: threading
 *                      orchestration only, with an `omp_in_parallel()`
 *                      guard that delegates to `ytrsm_serial` when called
 *                      from inside another routine's parallel region.
 *
 * The cores are range-parameterized — [j_start, j_end) over the column
 * axis (SIDE='L') or [i_start, i_end) over the row axis (SIDE='R'). The
 * serial entry calls them over the full range; the parallel driver
 * partitions the range across a team and calls the same cores per chunk.
 * The TRANSA='T' and 'C' (conjugate-transpose) variants share one core
 * each, selected by a runtime conj_flag.
 *
 * Nested calls must run serial: opening a nested OpenMP region here trips
 * the libgomp barrier wedge. The
 * entry guard is the single nesting gate; the cores never touch OpenMP.
 */
#ifndef EPBLAS_PARALLEL_KIND10_YTRSM_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_YTRSM_KERNEL_H

#include <stddef.h>
#include <complex.h>

typedef _Complex long double ytrsm_T;

/* Blocked SIDE='L' variant selector. */
enum ytrsm_variant { YLLN, YLUN, YLLT, YLUT, YLLC, YLUC };

/* Env-tunable block size for the SIDE='L' blocked path (YTRSM_NB). */
ptrdiff_t ytrsm_nb(void);

/* ── SIDE='L' column-range cores: serial work over columns
 *    [j_start, j_end) of B; A is M×M. ──────────────────────────── */
void ytrsm_lln_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, ytrsm_T alpha,
                    const ytrsm_T *a, ptrdiff_t lda, ytrsm_T *b, ptrdiff_t ldb, ptrdiff_t nounit);
void ytrsm_lun_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, ytrsm_T alpha,
                    const ytrsm_T *a, ptrdiff_t lda, ytrsm_T *b, ptrdiff_t ldb, ptrdiff_t nounit);
/* (L, L, T or C): conj_flag selects 'C' over 'T'. */
void ytrsm_llTC_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, ytrsm_T alpha,
                     const ytrsm_T *a, ptrdiff_t lda, ytrsm_T *b, ptrdiff_t ldb,
                     ptrdiff_t nounit, ptrdiff_t conj_flag);
/* (L, U, T or C). */
void ytrsm_luTC_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, ytrsm_T alpha,
                     const ytrsm_T *a, ptrdiff_t lda, ytrsm_T *b, ptrdiff_t ldb,
                     ptrdiff_t nounit, ptrdiff_t conj_flag);

/* ── SIDE='R' row-range cores: serial work over rows [i_start, i_end)
 *    of B; A is N×N. ────────────────────────────────────────────── */
void ytrsm_rln_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, ytrsm_T alpha,
                    const ytrsm_T *a, ptrdiff_t lda, ytrsm_T *b, ptrdiff_t ldb, ptrdiff_t nounit);
void ytrsm_run_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, ytrsm_T alpha,
                    const ytrsm_T *a, ptrdiff_t lda, ytrsm_T *b, ptrdiff_t ldb, ptrdiff_t nounit);
void ytrsm_rlTC_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, ytrsm_T alpha,
                     const ytrsm_T *a, ptrdiff_t lda, ytrsm_T *b, ptrdiff_t ldb,
                     ptrdiff_t nounit, ptrdiff_t conj_flag);
void ytrsm_ruTC_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, ytrsm_T alpha,
                     const ytrsm_T *a, ptrdiff_t lda, ytrsm_T *b, ptrdiff_t ldb,
                     ptrdiff_t nounit, ptrdiff_t conj_flag);

/* Per-thread serial blocked-TRSM on a column slice [j_start, j_end) of B
 * (SIDE='L'). Trailing updates call ygemm_serial. */
void ytrsm_blocked_chunk(enum ytrsm_variant V, ptrdiff_t j_start, ptrdiff_t j_end,
                         ptrdiff_t M, ptrdiff_t nb, ytrsm_T alpha,
                         const ytrsm_T *a, ptrdiff_t lda, ytrsm_T *b, ptrdiff_t ldb,
                         ptrdiff_t nounit);

/* Per-thread serial blocked-TRSM on a row band [i_start, i_end) of B
 * (SIDE='R'). Blocks the triangular column (N) axis: the bulk cross-block
 * update runs through ygemm_serial, only the jb×jb diagonal block goes
 * through the naive R cores. (upper, trans, conj) select the variant. */
void ytrsm_R_blocked_chunk(ptrdiff_t upper, ptrdiff_t trans, ptrdiff_t conj,
                           ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, ptrdiff_t nb, ytrsm_T alpha,
                           const ytrsm_T *a, ptrdiff_t lda, ytrsm_T *b, ptrdiff_t ldb,
                           ptrdiff_t nounit);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same signature as ytrsm_. */
void ytrsm_serial(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const ptrdiff_t *m_, const ptrdiff_t *n_,
    const ytrsm_T *alpha_,
    const ytrsm_T *a, const ptrdiff_t *lda_,
    ytrsm_T *b, const ptrdiff_t *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len);

#endif /* EPBLAS_PARALLEL_KIND10_YTRSM_KERNEL_H */
