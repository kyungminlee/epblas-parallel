/*
 * ytrmm_kernel.h — internal shared declarations for the kind10 complex
 * (COMPLEX(KIND=10) / _Complex long double) triangular-multiply overlay,
 * split across two translation units:
 *
 *   ytrmm_serial.c   — all the math (column/row-range cores + the blocked
 *                      SIDE='L' and SIDE='R' workers, whose trailing
 *                      updates call ygemm_serial) plus the pure-serial
 *                      Fortran-ABI entry `ytrmm_serial`. No `#pragma omp`.
 *   ytrmm_parallel.c — the public Fortran entry `ytrmm_`: threading
 *                      orchestration only, with an `omp_in_parallel()`
 *                      guard that delegates to `ytrmm_serial` when called
 *                      from inside another routine's parallel region.
 *
 * The cores are range-parameterized — [j_start, j_end) over the column
 * axis (SIDE='L') or [i_start, i_end) over the row axis (SIDE='R'). The
 * serial entry calls the blocked workers / cores over the full range; the
 * parallel driver partitions the range across a team. The TRANSA='T' and
 * 'C' variants share one core each, selected by a runtime conj_flag.
 *
 * Nested calls must run serial: opening a nested OpenMP region here trips
 * the libgomp barrier wedge (see memory project-etrsm-omp4-wedge).
 */
#ifndef EPBLAS_PARALLEL_KIND10_YTRMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_YTRMM_KERNEL_H

#include <stddef.h>
#include <complex.h>

typedef _Complex long double ytrmm_T;

/* Blocked variant selectors (distinct names for the L and R families). */
enum ytrmm_variant_L { YLLN, YLUN, YLLT, YLUT, YLLC, YLUC };
enum ytrmm_variant_R { YRLN, YRUN, YRLT, YRUT, YRLC, YRUC };

/* Env-tunable block size for the blocked paths (YTRMM_NB). */
int ytrmm_nb(void);

/* ── SIDE='L' column-range cores: serial work over columns
 *    [j_start, j_end) of B; A is M×M. ──────────────────────────── */
void ytrmm_lln_core(int j_start, int j_end, int M, ytrmm_T alpha,
                    const ytrmm_T *a, int lda, ytrmm_T *b, int ldb, int nounit);
void ytrmm_lun_core(int j_start, int j_end, int M, ytrmm_T alpha,
                    const ytrmm_T *a, int lda, ytrmm_T *b, int ldb, int nounit);
void ytrmm_llTC_core(int j_start, int j_end, int M, ytrmm_T alpha,
                     const ytrmm_T *a, int lda, ytrmm_T *b, int ldb,
                     int nounit, int conj_flag);
void ytrmm_luTC_core(int j_start, int j_end, int M, ytrmm_T alpha,
                     const ytrmm_T *a, int lda, ytrmm_T *b, int ldb,
                     int nounit, int conj_flag);

/* ── SIDE='R' row-range cores: serial work over rows [i_start, i_end)
 *    of B; A is N×N. ────────────────────────────────────────────── */
void ytrmm_rln_core(int i_start, int i_end, int N, ytrmm_T alpha,
                    const ytrmm_T *a, int lda, ytrmm_T *b, int ldb, int nounit);
void ytrmm_run_core(int i_start, int i_end, int N, ytrmm_T alpha,
                    const ytrmm_T *a, int lda, ytrmm_T *b, int ldb, int nounit);
void ytrmm_rlTC_core(int i_start, int i_end, int N, ytrmm_T alpha,
                     const ytrmm_T *a, int lda, ytrmm_T *b, int ldb,
                     int nounit, int conj_flag);
void ytrmm_ruTC_core(int i_start, int i_end, int N, ytrmm_T alpha,
                     const ytrmm_T *a, int lda, ytrmm_T *b, int ldb,
                     int nounit, int conj_flag);

/* Per-thread serial blocked-TRMM workers. SIDE='L' partitions B's columns
 * [j_start, j_end); SIDE='R' partitions B's rows [i_start, i_end).
 * Trailing updates call ygemm_serial. */
void ytrmm_blocked_chunk_L(enum ytrmm_variant_L V, int j_start, int j_end,
                           int M, int nb, ytrmm_T alpha,
                           const ytrmm_T *a, int lda, ytrmm_T *b, int ldb,
                           int nounit);
void ytrmm_blocked_chunk_R(enum ytrmm_variant_R V, int i_start, int i_end,
                           int N, int nb, ytrmm_T alpha,
                           const ytrmm_T *a, int lda, ytrmm_T *b, int ldb,
                           int nounit);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same signature as ytrmm_. */
void ytrmm_serial(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const ytrmm_T *alpha_,
    const ytrmm_T *a, const int *lda_,
    ytrmm_T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len);

#endif /* EPBLAS_PARALLEL_KIND10_YTRMM_KERNEL_H */
