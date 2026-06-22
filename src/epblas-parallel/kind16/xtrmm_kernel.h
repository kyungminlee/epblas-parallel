/*
 * xtrmm_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 complex (COMPLEX(KIND=16) / __complex128) xtrmm overlay
 * is split across:
 *
 *   xtrmm_serial.c    The pure single-thread triangular multiply (no
 *                     OpenMP). Owns the char decode, the conj/A_op helpers
 *                     (file-static — only the cores use them), the
 *                     range-parameterized compute cores (declared here),
 *                     and the public `xtrmm_serial_` Fortran entry that
 *                     dispatches to a core over the FULL [0,N) / [0,M)
 *                     range.
 *
 *   xtrmm_parallel.c  The public Fortran entry `xtrmm_` — threading
 *                     orchestration only. Per (side,uplo,trans) it either
 *                     fans the matching core across an OpenMP team
 *                     (column slice for SIDE='L', row slice for SIDE='R')
 *                     or, when below the threshold / nested / single-
 *                     threaded, runs it serially over the full range.
 *
 * TRANSA='C' (conjugate transpose) is a distinct case from 'T': the *_TC
 * cores carry a conj_flag selecting between A and conj(A). Each column
 * (SIDE='L') / row (SIDE='R') of B is independent, so any static partition
 * of the range is bitwise-identical to the serial run.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XTRMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XTRMM_KERNEL_H

#include <stddef.h>
#include <quadmath.h>  /* __complex128 */

typedef __complex128 xtrmm_T;

/* Normalize a Fortran character to upper case (uplo / side / trans decode). */
char xtrmm_uplo(char c);

/* ── SIDE = 'L' column-range cores ──────────────────────────────── */
void xtrmm_lln_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, xtrmm_T alpha,
                    const xtrmm_T *a, ptrdiff_t lda, xtrmm_T *b, ptrdiff_t ldb, int nounit);
void xtrmm_lun_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, xtrmm_T alpha,
                    const xtrmm_T *a, ptrdiff_t lda, xtrmm_T *b, ptrdiff_t ldb, int nounit);
void xtrmm_llTC_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, xtrmm_T alpha,
                     const xtrmm_T *a, ptrdiff_t lda, xtrmm_T *b, ptrdiff_t ldb,
                     int nounit, int conj_flag);
void xtrmm_luTC_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, xtrmm_T alpha,
                     const xtrmm_T *a, ptrdiff_t lda, xtrmm_T *b, ptrdiff_t ldb,
                     int nounit, int conj_flag);

/* ── SIDE = 'R' row-range cores ─────────────────────────────────── */
void xtrmm_rln_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, xtrmm_T alpha,
                    const xtrmm_T *a, ptrdiff_t lda, xtrmm_T *b, ptrdiff_t ldb, int nounit);
void xtrmm_run_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, xtrmm_T alpha,
                    const xtrmm_T *a, ptrdiff_t lda, xtrmm_T *b, ptrdiff_t ldb, int nounit);
void xtrmm_rlTC_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, xtrmm_T alpha,
                     const xtrmm_T *a, ptrdiff_t lda, xtrmm_T *b, ptrdiff_t ldb,
                     int nounit, int conj_flag);
void xtrmm_ruTC_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, xtrmm_T alpha,
                     const xtrmm_T *a, ptrdiff_t lda, xtrmm_T *b, ptrdiff_t ldb,
                     int nounit, int conj_flag);

/* Pure-serial by-value entry. No OpenMP anywhere on this call path; safe to
 * invoke from inside another function's `#pragma omp parallel` region. Shares
 * the ptrdiff_t core ABI of xtrmm_core. */
void xtrmm_serial(
    char side, char uplo, char transa, char diag,
    ptrdiff_t M, ptrdiff_t N,
    const xtrmm_T *alpha_,
    const xtrmm_T *a, ptrdiff_t lda,
    xtrmm_T *b, ptrdiff_t ldb);

#endif /* EPBLAS_PARALLEL_KIND16_XTRMM_KERNEL_H */
