/*
 * qtrsm_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 qtrsm overlay is split across:
 *
 *   qtrsm_serial.c    The pure single-thread triangular solve (no OpenMP).
 *                     Owns the uplo decode, all eight range-parameterized
 *                     solve cores (declared below), and the public
 *                     `qtrsm_serial` entry — the qtrsm_ algorithm forced
 *                     fully serial. Cores are called directly by the
 *                     parallel entries' threaded wrappers, and by
 *                     qtrsm_serial over the full [0,N)/[0,M) range.
 *
 *   qtrsm_parallel.c  The public Fortran entries `qtrsm_` (column/row
 *                     parallel, one fork-join, with the qtrsv-loop fast
 *                     path) and `qtrsm_blocked_` (LAPACK-blocked inside a
 *                     SINGLE parallel region, driving trailing updates
 *                     through qgemm_serial). Threading orchestration only;
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
#include <stdbool.h>

typedef __float128 qtrsm_TR;

/* Decode a Fortran character arg: upper-cased. */
char qtrsm_uplo(char c);

/* ── SIDE = 'L' column-range cores ──────────────────────────────────
 * Each solves columns [j_start,j_end) of B against the M×M triangular A. */
void qtrsm_lln_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, qtrsm_TR alpha,
                    const qtrsm_TR *a, ptrdiff_t lda, qtrsm_TR *b, ptrdiff_t ldb, bool nounit);
void qtrsm_lun_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, qtrsm_TR alpha,
                    const qtrsm_TR *a, ptrdiff_t lda, qtrsm_TR *b, ptrdiff_t ldb, bool nounit);
void qtrsm_llt_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, qtrsm_TR alpha,
                    const qtrsm_TR *a, ptrdiff_t lda, qtrsm_TR *b, ptrdiff_t ldb, bool nounit);
void qtrsm_lut_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, qtrsm_TR alpha,
                    const qtrsm_TR *a, ptrdiff_t lda, qtrsm_TR *b, ptrdiff_t ldb, bool nounit);

/* ── SIDE = 'R' row-range cores ─────────────────────────────────────
 * Each solves rows [i_start,i_end) of B against the N×N triangular A. */
void qtrsm_rln_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, qtrsm_TR alpha,
                    const qtrsm_TR *a, ptrdiff_t lda, qtrsm_TR *b, ptrdiff_t ldb, bool nounit);
void qtrsm_run_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, qtrsm_TR alpha,
                    const qtrsm_TR *a, ptrdiff_t lda, qtrsm_TR *b, ptrdiff_t ldb, bool nounit);
void qtrsm_rlt_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, qtrsm_TR alpha,
                    const qtrsm_TR *a, ptrdiff_t lda, qtrsm_TR *b, ptrdiff_t ldb, bool nounit);
void qtrsm_rut_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, qtrsm_TR alpha,
                    const qtrsm_TR *a, ptrdiff_t lda, qtrsm_TR *b, ptrdiff_t ldb, bool nounit);

/* Pure-serial by-value entry. No OpenMP anywhere on this call path; produces
 * results bit-identical to qtrsm_ run single-threaded (each core called over
 * the full column/row range). Shares the ptrdiff_t core ABI of qtrsm_core. */
void qtrsm_serial(
    char side, char uplo, char transa, char diag,
    ptrdiff_t m, ptrdiff_t n,
    const qtrsm_TR *alpha_,
    const qtrsm_TR *a, ptrdiff_t lda,
    qtrsm_TR *b, ptrdiff_t ldb);

#endif /* EPBLAS_PARALLEL_KIND16_QTRSM_KERNEL_H */
