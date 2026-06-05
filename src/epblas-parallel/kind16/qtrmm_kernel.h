/*
 * qtrmm_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 (REAL(KIND=16) / __float128) qtrmm overlay is split
 * across:
 *
 *   qtrmm_serial.c    The pure single-thread triangular multiply (no
 *                     OpenMP). Owns the uplo-char decode and the eight
 *                     range-parameterized per-column / per-row compute
 *                     cores (declared here), plus the public
 *                     `qtrmm_serial_` Fortran entry that dispatches to a
 *                     core over the FULL [0,N) / [0,M) range.
 *
 *   qtrmm_parallel.c  The public Fortran entry `qtrmm_` — threading
 *                     orchestration only. Per (side,uplo,trans) it either
 *                     fans the matching core across an OpenMP team
 *                     (column slice for SIDE='L', row slice for SIDE='R')
 *                     or, when below the threshold / nested / single-
 *                     threaded, runs it serially over the full range.
 *
 * Each column (SIDE='L') / row (SIDE='R') of B is independent, so any
 * static partition of the range is bitwise-identical to the serial run —
 * the cores carry their own [start,end) bounds.
 */
#ifndef EPBLAS_PARALLEL_KIND16_QTRMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_QTRMM_KERNEL_H

#include <stddef.h>

typedef __float128 qtrmm_T;

/* Normalize a Fortran character to upper case (uplo / side / trans decode). */
char qtrmm_uplo(const char *p);

/* ── SIDE = 'L' column-range cores ────────────────────────────────
 * B := alpha · op(A) · B, A is M×M, B is M×N.
 * Each call owns a column slice [j_start, j_end) of B. */
void trmm_lln_core(int j_start, int j_end, int M, qtrmm_T alpha,
                   const qtrmm_T *a, int lda, qtrmm_T *b, int ldb, int nounit);
void trmm_lun_core(int j_start, int j_end, int M, qtrmm_T alpha,
                   const qtrmm_T *a, int lda, qtrmm_T *b, int ldb, int nounit);
void trmm_llt_core(int j_start, int j_end, int M, qtrmm_T alpha,
                   const qtrmm_T *a, int lda, qtrmm_T *b, int ldb, int nounit);
void trmm_lut_core(int j_start, int j_end, int M, qtrmm_T alpha,
                   const qtrmm_T *a, int lda, qtrmm_T *b, int ldb, int nounit);

/* ── SIDE = 'R' row-range cores ────────────────────────────────────
 * B := alpha · B · op(A), A is N×N, B is M×N.
 * Each call owns a row slice [i_start, i_end) of B. */
void trmm_rln_core(int i_start, int i_end, int N, qtrmm_T alpha,
                   const qtrmm_T *a, int lda, qtrmm_T *b, int ldb, int nounit);
void trmm_run_core(int i_start, int i_end, int N, qtrmm_T alpha,
                   const qtrmm_T *a, int lda, qtrmm_T *b, int ldb, int nounit);
void trmm_rlt_core(int i_start, int i_end, int N, qtrmm_T alpha,
                   const qtrmm_T *a, int lda, qtrmm_T *b, int ldb, int nounit);
void trmm_rut_core(int i_start, int i_end, int N, qtrmm_T alpha,
                   const qtrmm_T *a, int lda, qtrmm_T *b, int ldb, int nounit);

/* Pure-serial Fortran entry. No OpenMP anywhere on this call path; safe to
 * invoke from inside another function's `#pragma omp parallel` region. Keeps
 * the exact Fortran-ABI signature of qtrmm_. */
void qtrmm_serial_(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const qtrmm_T *alpha_,
    const qtrmm_T *a, const int *lda_,
    qtrmm_T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len);

#endif /* EPBLAS_PARALLEL_KIND16_QTRMM_KERNEL_H */
