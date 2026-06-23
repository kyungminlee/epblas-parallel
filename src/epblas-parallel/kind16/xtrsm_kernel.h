/*
 * xtrsm_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 complex xtrsm overlay is split across:
 *
 *   xtrsm_serial.c    The pure single-thread triangular solve (no OpenMP).
 *                     Owns the uplo decode, the conjugate / A_op helpers,
 *                     all eight range-parameterized solve cores (declared
 *                     below; the trans/conj variants take a conj_flag),
 *                     and the public `xtrsm_serial_` entry — the xtrsm_
 *                     algorithm forced fully serial. Cores are called
 *                     directly by the parallel entries' threaded wrappers,
 *                     and by xtrsm_serial_ over the full range.
 *
 *   xtrsm_parallel.c  The public Fortran entries `xtrsm_` (column/row
 *                     parallel, one fork-join, with the xtrsv-loop fast
 *                     path) and `xtrsm_blocked_` (LAPACK-blocked inside a
 *                     SINGLE parallel region, driving trailing updates
 *                     through xgemm_serial). Threading orchestration only.
 *
 * The within-core solve is loop-carried (serial), but DIFFERENT columns
 * (SIDE='L') / rows (SIDE='R') are independent — so the column/row
 * partition is exactly what makes threading the cores valid. The cores are
 * range-safe: each takes a [j_start,j_end) (L) or [i_start,i_end) (R)
 * sub-range and touches only that slice of B.
 *
 * TRANSA='C' (conjugate transpose) and 'T' (plain transpose) share the
 * llTC/luTC/rlTC/ruTC cores via the conj_flag parameter.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XTRSM_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XTRSM_KERNEL_H

#include <stddef.h>
#include <quadmath.h>  /* __complex128 */

typedef __complex128 xtrsm_TC;

/* Decode a Fortran character arg: upper-cased. */
char xtrsm_uplo(char c);

/* ── SIDE = 'L' column-range cores ──────────────────────────────────
 * Each solves columns [j_start,j_end) of B against the M×M triangular A.
 * The TC variants handle TRANSA in {'T','C'} via conj_flag (1 = conjugate). */
void xtrsm_lln_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, xtrsm_TC alpha,
                    const xtrsm_TC *a, ptrdiff_t lda, xtrsm_TC *b, ptrdiff_t ldb, bool nounit);
void xtrsm_lun_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, xtrsm_TC alpha,
                    const xtrsm_TC *a, ptrdiff_t lda, xtrsm_TC *b, ptrdiff_t ldb, bool nounit);
void xtrsm_lltc_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, xtrsm_TC alpha,
                     const xtrsm_TC *a, ptrdiff_t lda, xtrsm_TC *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag);
void xtrsm_lutc_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, xtrsm_TC alpha,
                     const xtrsm_TC *a, ptrdiff_t lda, xtrsm_TC *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag);

/* ── SIDE = 'R' row-range cores ─────────────────────────────────────
 * Each solves rows [i_start,i_end) of B against the N×N triangular A. */
void xtrsm_rln_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, xtrsm_TC alpha,
                    const xtrsm_TC *a, ptrdiff_t lda, xtrsm_TC *b, ptrdiff_t ldb, bool nounit);
void xtrsm_run_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, xtrsm_TC alpha,
                    const xtrsm_TC *a, ptrdiff_t lda, xtrsm_TC *b, ptrdiff_t ldb, bool nounit);
void xtrsm_rltc_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, xtrsm_TC alpha,
                     const xtrsm_TC *a, ptrdiff_t lda, xtrsm_TC *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag);
void xtrsm_rutc_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, xtrsm_TC alpha,
                     const xtrsm_TC *a, ptrdiff_t lda, xtrsm_TC *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag);

/* Pure-serial by-value entry. No OpenMP anywhere on this call path; produces
 * results bit-identical to xtrsm_ run single-threaded (each core called over
 * the full column/row range). Shares the ptrdiff_t core ABI of xtrsm_core. */
void xtrsm_serial(
    char side, char uplo, char transa, char diag,
    ptrdiff_t m, ptrdiff_t n,
    const xtrsm_TC *alpha_,
    const xtrsm_TC *a, ptrdiff_t lda,
    xtrsm_TC *b, ptrdiff_t ldb);

#endif /* EPBLAS_PARALLEL_KIND16_XTRSM_KERNEL_H */
