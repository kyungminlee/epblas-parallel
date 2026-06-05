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
 *                     through xgemm_serial_). Threading orchestration only.
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

typedef __complex128 xtrsm_T;

/* Decode a Fortran character arg: first char, upper-cased. */
char xtrsm_uplo(const char *p);

/* ── SIDE = 'L' column-range cores ──────────────────────────────────
 * Each solves columns [j_start,j_end) of B against the M×M triangular A.
 * The TC variants handle TRANSA in {'T','C'} via conj_flag (1 = conjugate). */
void xtrsm_lln_core(int j_start, int j_end, int M, xtrsm_T alpha,
                    const xtrsm_T *a, int lda, xtrsm_T *b, int ldb, int nounit);
void xtrsm_lun_core(int j_start, int j_end, int M, xtrsm_T alpha,
                    const xtrsm_T *a, int lda, xtrsm_T *b, int ldb, int nounit);
void xtrsm_llTC_core(int j_start, int j_end, int M, xtrsm_T alpha,
                     const xtrsm_T *a, int lda, xtrsm_T *b, int ldb,
                     int nounit, int conj_flag);
void xtrsm_luTC_core(int j_start, int j_end, int M, xtrsm_T alpha,
                     const xtrsm_T *a, int lda, xtrsm_T *b, int ldb,
                     int nounit, int conj_flag);

/* ── SIDE = 'R' row-range cores ─────────────────────────────────────
 * Each solves rows [i_start,i_end) of B against the N×N triangular A. */
void xtrsm_rln_core(int i_start, int i_end, int N, xtrsm_T alpha,
                    const xtrsm_T *a, int lda, xtrsm_T *b, int ldb, int nounit);
void xtrsm_run_core(int i_start, int i_end, int N, xtrsm_T alpha,
                    const xtrsm_T *a, int lda, xtrsm_T *b, int ldb, int nounit);
void xtrsm_rlTC_core(int i_start, int i_end, int N, xtrsm_T alpha,
                     const xtrsm_T *a, int lda, xtrsm_T *b, int ldb,
                     int nounit, int conj_flag);
void xtrsm_ruTC_core(int i_start, int i_end, int N, xtrsm_T alpha,
                     const xtrsm_T *a, int lda, xtrsm_T *b, int ldb,
                     int nounit, int conj_flag);

/* Pure-serial Fortran entry. No OpenMP anywhere on this call path; produces
 * results bit-identical to xtrsm_ run single-threaded (each core called over
 * the full column/row range). Keeps the exact Fortran-ABI signature. */
void xtrsm_serial_(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const xtrsm_T *alpha_,
    const xtrsm_T *a, const int *lda_,
    xtrsm_T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len);

#endif /* EPBLAS_PARALLEL_KIND16_XTRSM_KERNEL_H */
