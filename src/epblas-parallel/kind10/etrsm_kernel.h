/*
 * etrsm_kernel.h — internal shared declarations for the kind10 etrsm
 * (REAL(KIND=10) / long double) triangular-solve overlay, split across
 * two translation units:
 *
 *   etrsm_serial.c   — all the math (column/row-range cores + the blocked
 *                      worker) plus the pure-serial Fortran-ABI entry
 *                      `etrsm_serial`. No `#pragma omp`.
 *   etrsm_parallel.c — the public Fortran entry `etrsm_`: threading
 *                      orchestration only, with an `omp_in_parallel()`
 *                      guard that delegates to `etrsm_serial` when called
 *                      from inside another routine's parallel region.
 *
 * The cores are range-parameterized — [j_start, j_end) over the column
 * axis (SIDE='L') or [i_start, i_end) over the row axis (SIDE='R'). The
 * serial entry calls them over the full range; the parallel driver
 * partitions the range across a team and calls the same cores per chunk.
 *
 * Nested calls must run serial: opening a nested OpenMP region here trips
 * the libgomp barrier wedge (see memory project-etrsm-omp4-wedge). The
 * entry guard is the single nesting gate; the cores themselves never
 * touch OpenMP.
 */
#ifndef EPBLAS_PARALLEL_KIND10_ETRSM_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_ETRSM_KERNEL_H

#include <stddef.h>

typedef long double etrsm_T;

/* Blocked SIDE='L' variant selector. */
enum etrsm_variant { LLN, LUN, LLT, LUT };

/* Env-tunable block size for the SIDE='L' blocked path (ETRSM_NB). */
int etrsm_nb(void);

/* ── SIDE='L' column-range cores: serial work over columns
 *    [j_start, j_end) of B; A is M×M. ──────────────────────────── */
void etrsm_lln_core(int j_start, int j_end, int M, etrsm_T alpha,
                    const etrsm_T *a, int lda, etrsm_T *b, int ldb, int nounit);
void etrsm_lun_core(int j_start, int j_end, int M, etrsm_T alpha,
                    const etrsm_T *a, int lda, etrsm_T *b, int ldb, int nounit);
void etrsm_llt_core(int j_start, int j_end, int M, etrsm_T alpha,
                    const etrsm_T *a, int lda, etrsm_T *b, int ldb, int nounit);
void etrsm_lut_core(int j_start, int j_end, int M, etrsm_T alpha,
                    const etrsm_T *a, int lda, etrsm_T *b, int ldb, int nounit);

/* ── SIDE='R' row-range cores: serial work over rows [i_start, i_end)
 *    of B; A is N×N. ────────────────────────────────────────────── */
void etrsm_rln_core(int i_start, int i_end, int N, etrsm_T alpha,
                    const etrsm_T *a, int lda, etrsm_T *b, int ldb, int nounit);
void etrsm_run_core(int i_start, int i_end, int N, etrsm_T alpha,
                    const etrsm_T *a, int lda, etrsm_T *b, int ldb, int nounit);
void etrsm_rlt_core(int i_start, int i_end, int N, etrsm_T alpha,
                    const etrsm_T *a, int lda, etrsm_T *b, int ldb, int nounit);
void etrsm_rut_core(int i_start, int i_end, int N, etrsm_T alpha,
                    const etrsm_T *a, int lda, etrsm_T *b, int ldb, int nounit);

/* Per-thread serial blocked-TRSM on a column slice [j_start, j_end) of B
 * (SIDE='L'). Trailing updates call egemm_serial. */
void etrsm_blocked_chunk(enum etrsm_variant V, int j_start, int j_end,
                         int M, int nb, etrsm_T alpha,
                         const etrsm_T *a, int lda, etrsm_T *b, int ldb,
                         int nounit);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same signature as etrsm_. */
void etrsm_serial(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const etrsm_T *alpha_,
    const etrsm_T *a, const int *lda_,
    etrsm_T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len);

#endif /* EPBLAS_PARALLEL_KIND10_ETRSM_KERNEL_H */
