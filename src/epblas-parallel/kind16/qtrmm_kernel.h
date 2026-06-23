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
#include <stdbool.h>

typedef __float128 qtrmm_TR;

/* Normalize a character to upper case (uplo / side / trans decode). */
char qtrmm_uplo(char c);

/* ── SIDE = 'L' column-range cores ────────────────────────────────
 * B := alpha · op(A) · B, A is M×M, B is M×N.
 * Each call owns a column slice [j_start, j_end) of B. */
void trmm_lln_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, qtrmm_TR alpha,
                   const qtrmm_TR *a, ptrdiff_t lda, qtrmm_TR *b, ptrdiff_t ldb, bool nounit);
void trmm_lun_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, qtrmm_TR alpha,
                   const qtrmm_TR *a, ptrdiff_t lda, qtrmm_TR *b, ptrdiff_t ldb, bool nounit);
void trmm_llt_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, qtrmm_TR alpha,
                   const qtrmm_TR *a, ptrdiff_t lda, qtrmm_TR *b, ptrdiff_t ldb, bool nounit);
void trmm_lut_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, qtrmm_TR alpha,
                   const qtrmm_TR *a, ptrdiff_t lda, qtrmm_TR *b, ptrdiff_t ldb, bool nounit);

/* ── SIDE = 'R' row-range cores ────────────────────────────────────
 * B := alpha · B · op(A), A is N×N, B is M×N.
 * Each call owns a row slice [i_start, i_end) of B. */
void trmm_rln_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, qtrmm_TR alpha,
                   const qtrmm_TR *a, ptrdiff_t lda, qtrmm_TR *b, ptrdiff_t ldb, bool nounit);
void trmm_run_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, qtrmm_TR alpha,
                   const qtrmm_TR *a, ptrdiff_t lda, qtrmm_TR *b, ptrdiff_t ldb, bool nounit);
void trmm_rlt_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, qtrmm_TR alpha,
                   const qtrmm_TR *a, ptrdiff_t lda, qtrmm_TR *b, ptrdiff_t ldb, bool nounit);
void trmm_rut_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, qtrmm_TR alpha,
                   const qtrmm_TR *a, ptrdiff_t lda, qtrmm_TR *b, ptrdiff_t ldb, bool nounit);

/* Pure-serial by-value entry. No OpenMP anywhere on this call path; safe to
 * invoke from inside another function's `#pragma omp parallel` region. Shares
 * the ptrdiff_t core ABI of qtrmm_core. */
void qtrmm_serial(
    char side, char uplo, char transa, char diag,
    ptrdiff_t m, ptrdiff_t n,
    const qtrmm_TR *alpha_,
    const qtrmm_TR *a, ptrdiff_t lda,
    qtrmm_TR *b, ptrdiff_t ldb);

#endif /* EPBLAS_PARALLEL_KIND16_QTRMM_KERNEL_H */
