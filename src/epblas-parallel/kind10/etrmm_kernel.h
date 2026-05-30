/*
 * etrmm_kernel.h — internal shared declarations for the kind10 real
 * (REAL(KIND=10) / long double) triangular matrix-multiply overlay, split
 * across two translation units:
 *
 *   etrmm_serial.c   — all the math: the eight scalar column/row-range
 *                      cores (lln/lun/llt/lut, rln/run/rlt/rut), the blocked
 *                      per-chunk workers (diagonal core + egemm_serial
 *                      trailing), and the pure-serial Fortran-ABI entry
 *                      `etrmm_serial`. No `#pragma omp`.
 *   etrmm_parallel.c — the public Fortran entry `etrmm_`: threading only
 *                      (coarse-N/M `omp parallel` partition of the cores for
 *                      the unblocked path; one team per blocked dispatch),
 *                      with an `omp_in_parallel()` guard that delegates to
 *                      `etrmm_serial` when called from inside another
 *                      routine's parallel region.
 *
 * The cores process a column slice [j_start, j_end) (SIDE='L') or row slice
 * [i_start, i_end) (SIDE='R') of B; the serial entry runs the full range,
 * the parallel driver partitions it across one team. The blocked chunk
 * workers run their trailing update through egemm_serial — opening a nested
 * egemm team would trip the libgomp barrier wedge (project-etrsm-omp4-wedge).
 */
#ifndef EPBLAS_PARALLEL_KIND10_ETRMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_ETRMM_KERNEL_H

#include <stddef.h>

typedef long double etrmm_T;

enum etrmm_variant_L { ETRMM_LLN, ETRMM_LUN, ETRMM_LLT, ETRMM_LUT };
enum etrmm_variant_R { ETRMM_RLN, ETRMM_RUN, ETRMM_RLT, ETRMM_RUT };

/* Env-tunable block size (ETRMM_NB, default 64). */
int etrmm_nb(void);

/* SIDE='L' column-range scalar cores (process B columns [j_start, j_end)). */
void etrmm_lln_core(int j_start, int j_end, int M, etrmm_T alpha,
                    const etrmm_T *a, int lda, etrmm_T *b, int ldb, int nounit);
void etrmm_lun_core(int j_start, int j_end, int M, etrmm_T alpha,
                    const etrmm_T *a, int lda, etrmm_T *b, int ldb, int nounit);
void etrmm_llt_core(int j_start, int j_end, int M, etrmm_T alpha,
                    const etrmm_T *a, int lda, etrmm_T *b, int ldb, int nounit);
void etrmm_lut_core(int j_start, int j_end, int M, etrmm_T alpha,
                    const etrmm_T *a, int lda, etrmm_T *b, int ldb, int nounit);

/* SIDE='R' row-range scalar cores (process B rows [i_start, i_end)). */
void etrmm_rln_core(int i_start, int i_end, int N, etrmm_T alpha,
                    const etrmm_T *a, int lda, etrmm_T *b, int ldb, int nounit);
void etrmm_run_core(int i_start, int i_end, int N, etrmm_T alpha,
                    const etrmm_T *a, int lda, etrmm_T *b, int ldb, int nounit);
void etrmm_rlt_core(int i_start, int i_end, int N, etrmm_T alpha,
                    const etrmm_T *a, int lda, etrmm_T *b, int ldb, int nounit);
void etrmm_rut_core(int i_start, int i_end, int N, etrmm_T alpha,
                    const etrmm_T *a, int lda, etrmm_T *b, int ldb, int nounit);

/* Per-chunk blocked workers (diagonal core + egemm_serial trailing). */
void etrmm_blocked_chunk_L(enum etrmm_variant_L V, int j_start, int j_end,
                           int M, int nb, etrmm_T alpha,
                           const etrmm_T *a, int lda, etrmm_T *b, int ldb,
                           int nounit);
void etrmm_blocked_chunk_R(enum etrmm_variant_R V, int i_start, int i_end,
                           int N, int nb, etrmm_T alpha,
                           const etrmm_T *a, int lda, etrmm_T *b, int ldb,
                           int nounit);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same signature as etrmm_. */
void etrmm_serial(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const etrmm_T *alpha_,
    const etrmm_T *a, const int *lda_,
    etrmm_T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len);

#endif /* EPBLAS_PARALLEL_KIND10_ETRMM_KERNEL_H */
