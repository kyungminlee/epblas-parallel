/*
 * qtrsm_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 qtrsm overlay is split across:
 *
 *   qtrsm_serial.c    The pure single-thread triangular solve (no OpenMP).
 *                     Owns the uplo decode, all eight range-parameterized
 *                     solve cores (declared below), and the public
 *                     `qtrsm_serial_` entry — the qtrsm_ algorithm forced
 *                     fully serial. Cores are called directly by the
 *                     parallel entries' threaded wrappers, and by
 *                     qtrsm_serial_ over the full [0,N)/[0,M) range.
 *
 *   qtrsm_parallel.c  The public Fortran entries `qtrsm_` (column/row
 *                     parallel, one fork-join, with the qtrsv-loop fast
 *                     path) and `qtrsm_blocked_` (LAPACK-blocked inside a
 *                     SINGLE parallel region, driving trailing updates
 *                     through qgemm_serial_). Threading orchestration only;
 *                     all numerics live in the shared cores.
 *
 * The within-core solve is loop-carried (serial), but DIFFERENT columns
 * (SIDE='L') / rows (SIDE='R') are independent — so the column/row
 * partition is exactly what makes threading the cores valid. The cores are
 * range-safe: each takes a [j_start,j_end) (L) or [i_start,i_end) (R)
 * sub-range and touches only that slice of B.
 */
#ifndef EPBLAS_PARALLEL_KIND16_QTRSM_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_QTRSM_KERNEL_H

#include <stddef.h>

typedef __float128 qtrsm_T;

/* Decode a Fortran character arg: first char, upper-cased. */
char qtrsm_uplo(const char *p);

/* ── SIDE = 'L' column-range cores ──────────────────────────────────
 * Each solves columns [j_start,j_end) of B against the M×M triangular A. */
void qtrsm_lln_core(int j_start, int j_end, int M, qtrsm_T alpha,
                    const qtrsm_T *a, int lda, qtrsm_T *b, int ldb, int nounit);
void qtrsm_lun_core(int j_start, int j_end, int M, qtrsm_T alpha,
                    const qtrsm_T *a, int lda, qtrsm_T *b, int ldb, int nounit);
void qtrsm_llt_core(int j_start, int j_end, int M, qtrsm_T alpha,
                    const qtrsm_T *a, int lda, qtrsm_T *b, int ldb, int nounit);
void qtrsm_lut_core(int j_start, int j_end, int M, qtrsm_T alpha,
                    const qtrsm_T *a, int lda, qtrsm_T *b, int ldb, int nounit);

/* ── SIDE = 'R' row-range cores ─────────────────────────────────────
 * Each solves rows [i_start,i_end) of B against the N×N triangular A. */
void qtrsm_rln_core(int i_start, int i_end, int N, qtrsm_T alpha,
                    const qtrsm_T *a, int lda, qtrsm_T *b, int ldb, int nounit);
void qtrsm_run_core(int i_start, int i_end, int N, qtrsm_T alpha,
                    const qtrsm_T *a, int lda, qtrsm_T *b, int ldb, int nounit);
void qtrsm_rlt_core(int i_start, int i_end, int N, qtrsm_T alpha,
                    const qtrsm_T *a, int lda, qtrsm_T *b, int ldb, int nounit);
void qtrsm_rut_core(int i_start, int i_end, int N, qtrsm_T alpha,
                    const qtrsm_T *a, int lda, qtrsm_T *b, int ldb, int nounit);

/* Pure-serial Fortran entry. No OpenMP anywhere on this call path; produces
 * results bit-identical to qtrsm_ run single-threaded (each core called over
 * the full column/row range). Keeps the exact Fortran-ABI signature. */
void qtrsm_serial_(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const qtrsm_T *alpha_,
    const qtrsm_T *a, const int *lda_,
    qtrsm_T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len);

#endif /* EPBLAS_PARALLEL_KIND16_QTRSM_KERNEL_H */
