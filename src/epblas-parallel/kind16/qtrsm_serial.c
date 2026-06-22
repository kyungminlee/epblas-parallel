/*
 * qtrsm_serial.c — kind16 (REAL(KIND=16) / __float128) triangular solve,
 * serial core.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the
 * uplo decode and the eight range-parameterized solve cores (declared in
 * qtrsm_kernel.h), plus the by-value `qtrsm_serial` entry — the qtrsm_
 * algorithm forced fully serial. No OpenMP anywhere on this call path; safe
 * to invoke from inside another function's `#pragma omp parallel` region.
 *
 * qtrsm_serial drives numerics through the same cores the parallel entry
 * threads (each called over the full column/row range), so the two paths
 * are bitwise-identical.
 */

#include "qtrsm_kernel.h"
#include "../common/blas_char.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#include <quadmath.h>

/* qtrsv-loop fast path thresholds (mirror qtrsm_parallel.c). */
#define QTRSM_QTRSV_LOOP_M_MIN       128
#define QTRSM_QTRSV_LOOP_NB_HINT     64

typedef qtrsm_T T;

extern void qtrsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t N,
    const T *restrict a, ptrdiff_t lda,
    T *restrict x, ptrdiff_t incx);

/* Maximum nrhs at which the qtrsv-loop fast path beats column-parallel
 * qtrsm. In the serial entry no team is available, so the heuristic floors
 * at 1. */
static ptrdiff_t qtrsm_qtrsv_loop_max(ptrdiff_t M) {
    const ptrdiff_t max_nt     = 1 - 1;
    const ptrdiff_t max_amdahl = M / QTRSM_QTRSV_LOOP_NB_HINT;
    ptrdiff_t v = (max_nt < max_amdahl) ? max_nt : max_amdahl;
    if (v < 1) v = 1;
    return v;
}

char qtrsm_uplo(char c) {
    return blas_up(c);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── SIDE = 'L' column-range cores ──────────────────────────────── */

void qtrsm_lln_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        if (alpha != 1.0Q) for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = 0; k < M; ++k) {
            if (B_(k, j) != 0.0Q) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (ptrdiff_t i = k + 1; i < M; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

void qtrsm_lun_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        if (alpha != 1.0Q) for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = M - 1; k >= 0; --k) {
            if (B_(k, j) != 0.0Q) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (ptrdiff_t i = 0; i < k; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

void qtrsm_llt_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t i = M - 1; i >= 0; --i) {
            T t = alpha * B_(i, j);
            for (ptrdiff_t k = i + 1; k < M; ++k) t -= A_(k, i) * B_(k, j);
            if (nounit) t /= A_(i, i);
            B_(i, j) = t;
        }
    }
}

void qtrsm_lut_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t i = 0; i < M; ++i) {
            T t = alpha * B_(i, j);
            for (ptrdiff_t k = 0; k < i; ++k) t -= A_(k, i) * B_(k, j);
            if (nounit) t /= A_(i, i);
            B_(i, j) = t;
        }
    }
}

/* ── SIDE = 'R' row-range cores ─────────────────────────────────── */

void qtrsm_rln_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = N - 1; j >= 0; --j) {
        if (alpha != 1.0Q) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = j + 1; k < N; ++k) {
            if (A_(k, j) != 0.0Q) {
                const T akj = A_(k, j);
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = 1.0Q / A_(j, j);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

void qtrsm_run_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = 0; j < N; ++j) {
        if (alpha != 1.0Q) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = 0; k < j; ++k) {
            if (A_(k, j) != 0.0Q) {
                const T akj = A_(k, j);
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = 1.0Q / A_(j, j);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

void qtrsm_rlt_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t k = 0; k < N; ++k) {
        if (nounit) {
            const T inv = 1.0Q / A_(k, k);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (ptrdiff_t j = k + 1; j < N; ++j) {
            if (A_(j, k) != 0.0Q) {
                const T ajk = A_(j, k);
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != 1.0Q) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

void qtrsm_rut_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t N, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t k = N - 1; k >= 0; --k) {
        if (nounit) {
            const T inv = 1.0Q / A_(k, k);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (ptrdiff_t j = 0; j < k; ++j) {
            if (A_(j, k) != 0.0Q) {
                const T ajk = A_(j, k);
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != 1.0Q) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

/* ── Pure-serial entry ────────────────────────────────────────────
 *
 * The SAME algorithm as the public qtrsm_ non-blocked path with all OpenMP
 * removed: each core is called over the full column range [0,N) (SIDE='L')
 * or row range [0,M) (SIDE='R'). The qtrsv-loop fast path is preserved (it
 * is serial: nrhs sequential qtrsv solves), with no omp_in_parallel guard
 * since this path never threads.
 */
void qtrsm_serial(
    char side, char uplo, char transa, char diag,
    ptrdiff_t M, ptrdiff_t N,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    T *b, ptrdiff_t ldb)
{
    const T alpha = *alpha_;
    const char SIDE   = qtrsm_uplo(side);
    const char UPLO   = qtrsm_uplo(uplo);
    char TR           = qtrsm_uplo(transa);
    if (TR == 'C') TR = 'T';   /* real type: conj-trans ≡ trans */
    const bool nounit = (qtrsm_uplo(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (alpha == 0.0Q) {
        for (ptrdiff_t j = 0; j < N; ++j)
            for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) = 0.0Q;
        return;
    }

    /* qtrsv-loop fast path (serial: nrhs sequential qtrsv solves). */
    {
        const ptrdiff_t xv_max = qtrsm_qtrsv_loop_max(M);
        if (SIDE == 'L' && N >= 1 && N <= xv_max && M >= QTRSM_QTRSV_LOOP_M_MIN) {
            if (alpha != 1.0Q) {
                for (ptrdiff_t j = 0; j < N; ++j)
                    for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) *= alpha;
            }
            for (ptrdiff_t j = 0; j < N; ++j) {
                qtrsv_core(uplo, transa, diag, M, a, lda, &B_(0, j), 1);
            }
            return;
        }
    }

    if (SIDE == 'L') {
        if (TR == 'N') {
            if (UPLO == 'L') qtrsm_lln_core(0, N, M, alpha, a, lda, b, ldb, nounit);
            else             qtrsm_lun_core(0, N, M, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') qtrsm_llt_core(0, N, M, alpha, a, lda, b, ldb, nounit);
            else             qtrsm_lut_core(0, N, M, alpha, a, lda, b, ldb, nounit);
        }
    } else {
        if (TR == 'N') {
            if (UPLO == 'L') qtrsm_rln_core(0, M, N, alpha, a, lda, b, ldb, nounit);
            else             qtrsm_run_core(0, M, N, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') qtrsm_rlt_core(0, M, N, alpha, a, lda, b, ldb, nounit);
            else             qtrsm_rut_core(0, M, N, alpha, a, lda, b, ldb, nounit);
        }
    }
}

#undef A_
#undef B_
