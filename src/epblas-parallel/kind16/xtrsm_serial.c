/*
 * xtrsm_serial.c — kind16 complex (COMPLEX(KIND=16) / __complex128)
 * triangular solve, serial core.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the
 * uplo decode, the conjugate / A_op helpers, the eight range-parameterized
 * solve cores (declared in xtrsm_kernel.h), plus the public `xtrsm_serial_`
 * Fortran entry — the xtrsm_ algorithm forced fully serial. No OpenMP
 * anywhere on this call path; safe to invoke from inside another function's
 * `#pragma omp parallel` region.
 *
 * xtrsm_serial_ drives numerics through the same cores the parallel entry
 * threads (each called over the full column/row range), so the two paths
 * are bitwise-identical.
 *
 * TRANSA='C' is handled as a distinct case from 'T' (conjugate vs plain
 * transpose) via the conj_flag passed to the TC cores.
 */

#include "xtrsm_kernel.h"
#include "../common/blas_char.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#include <quadmath.h>

/* xtrsv-loop fast path thresholds (mirror xtrsm_parallel.c). */
#define XTRSM_XTRSV_LOOP_M_MIN       128
#define XTRSM_XTRSV_LOOP_NB_HINT     64

typedef xtrsm_T T;

extern void xtrsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n,
    const T *restrict a, ptrdiff_t lda,
    T *restrict x, ptrdiff_t incx);

/* Maximum nrhs at which the xtrsv-loop fast path beats column-parallel
 * xtrsm. In the serial entry no team is available, so the heuristic floors
 * at 1. */
static ptrdiff_t xtrsm_xtrsv_loop_max(ptrdiff_t m) {
    const ptrdiff_t max_nt     = 1 - 1;
    const ptrdiff_t max_amdahl = m / XTRSM_XTRSV_LOOP_NB_HINT;
    ptrdiff_t v = (max_nt < max_amdahl) ? max_nt : max_amdahl;
    if (v < 1) v = 1;
    return v;
}

char xtrsm_uplo(char c) {
    return blas_up(c);
}

static const T ZERO = 0.0Q + 0.0Qi;
static const T ONE  = 1.0Q + 0.0Qi;

static inline T cconj(T a) { return conjq(a); }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

static inline T A_op(const T *a, ptrdiff_t lda, ptrdiff_t row, ptrdiff_t col, bool conj_flag) {
    return conj_flag ? cconj(A_(row, col)) : A_(row, col);
}

/* ── SIDE = 'L' column-range cores ──────────────────────────────── */

void xtrsm_lln_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        if (alpha != ONE) for (ptrdiff_t i = 0; i < m; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = 0; k < m; ++k) {
            if (B_(k, j) != ZERO) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (ptrdiff_t i = k + 1; i < m; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

void xtrsm_lun_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        if (alpha != ONE) for (ptrdiff_t i = 0; i < m; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = m - 1; k >= 0; --k) {
            if (B_(k, j) != ZERO) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (ptrdiff_t i = 0; i < k; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

void xtrsm_llTC_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, T alpha,
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t i = m - 1; i >= 0; --i) {
            T t = alpha * B_(i, j);
            for (ptrdiff_t k = i + 1; k < m; ++k) t -= A_op(a, lda, k, i, conj_flag) * B_(k, j);
            if (nounit) t /= A_op(a, lda, i, i, conj_flag);
            B_(i, j) = t;
        }
    }
}

void xtrsm_luTC_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, T alpha,
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t i = 0; i < m; ++i) {
            T t = alpha * B_(i, j);
            for (ptrdiff_t k = 0; k < i; ++k) t -= A_op(a, lda, k, i, conj_flag) * B_(k, j);
            if (nounit) t /= A_op(a, lda, i, i, conj_flag);
            B_(i, j) = t;
        }
    }
}

/* ── SIDE = 'R' row-range cores ─────────────────────────────────── */

void xtrsm_rln_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = n - 1; j >= 0; --j) {
        if (alpha != ONE) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = j + 1; k < n; ++k) {
            if (A_(k, j) != ZERO) {
                const T akj = A_(k, j);
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = ONE / A_(j, j);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

void xtrsm_run_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, T alpha,
                    const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = 0; j < n; ++j) {
        if (alpha != ONE) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = 0; k < j; ++k) {
            if (A_(k, j) != ZERO) {
                const T akj = A_(k, j);
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = ONE / A_(j, j);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

void xtrsm_rlTC_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, T alpha,
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag)
{
    for (ptrdiff_t k = 0; k < n; ++k) {
        if (nounit) {
            const T inv = ONE / A_op(a, lda, k, k, conj_flag);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (ptrdiff_t j = k + 1; j < n; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != ONE) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

void xtrsm_ruTC_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, T alpha,
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag)
{
    for (ptrdiff_t k = n - 1; k >= 0; --k) {
        if (nounit) {
            const T inv = ONE / A_op(a, lda, k, k, conj_flag);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (ptrdiff_t j = 0; j < k; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != ONE) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

/* ── Pure-serial entry ────────────────────────────────────────────
 *
 * The SAME algorithm as the public xtrsm_ non-blocked path with all OpenMP
 * removed: each core is called over the full column range [0,N) (SIDE='L')
 * or row range [0,M) (SIDE='R'). The xtrsv-loop fast path is preserved (it
 * is serial: nrhs sequential xtrsv solves), with no omp_in_parallel guard
 * since this path never threads.
 */
void xtrsm_serial(
    char side, char uplo, char transa, char diag,
    ptrdiff_t m, ptrdiff_t n,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    T *b, ptrdiff_t ldb)
{
    const T alpha = *alpha_;
    const char SIDE = xtrsm_uplo(side);
    const char UPLO = xtrsm_uplo(uplo);
    const char TR = xtrsm_uplo(transa);
    const bool nounit = (xtrsm_uplo(diag) != 'U');
    const bool cflag = (TR == 'C') ? 1 : 0;

    if (m == 0 || n == 0) return;

    if (alpha == ZERO) {
        for (ptrdiff_t j = 0; j < n; ++j)
            for (ptrdiff_t i = 0; i < m; ++i) B_(i, j) = ZERO;
        return;
    }

    /* xtrsv-loop fast path (serial: nrhs sequential xtrsv solves). */
    {
        const ptrdiff_t xv_max = xtrsm_xtrsv_loop_max(m);
        if (SIDE == 'L' && n >= 1 && n <= xv_max && m >= XTRSM_XTRSV_LOOP_M_MIN) {
            if (alpha != ONE) {
                for (ptrdiff_t j = 0; j < n; ++j)
                    for (ptrdiff_t i = 0; i < m; ++i) B_(i, j) *= alpha;
            }
            for (ptrdiff_t j = 0; j < n; ++j) {
                xtrsv_core(uplo, transa, diag, m, a, lda, &B_(0, j), 1);
            }
            return;
        }
    }

    if (SIDE == 'L') {
        if (TR == 'N') {
            if (UPLO == 'L') xtrsm_lln_core(0, n, m, alpha, a, lda, b, ldb, nounit);
            else             xtrsm_lun_core(0, n, m, alpha, a, lda, b, ldb, nounit);
        } else if (TR == 'T') {
            if (UPLO == 'L') xtrsm_llTC_core(0, n, m, alpha, a, lda, b, ldb, nounit, 0);
            else             xtrsm_luTC_core(0, n, m, alpha, a, lda, b, ldb, nounit, 0);
        } else { /* 'C' */
            if (UPLO == 'L') xtrsm_llTC_core(0, n, m, alpha, a, lda, b, ldb, nounit, cflag);
            else             xtrsm_luTC_core(0, n, m, alpha, a, lda, b, ldb, nounit, cflag);
        }
    } else {
        if (TR == 'N') {
            if (UPLO == 'L') xtrsm_rln_core(0, m, n, alpha, a, lda, b, ldb, nounit);
            else             xtrsm_run_core(0, m, n, alpha, a, lda, b, ldb, nounit);
        } else if (TR == 'T') {
            if (UPLO == 'L') xtrsm_rlTC_core(0, m, n, alpha, a, lda, b, ldb, nounit, 0);
            else             xtrsm_ruTC_core(0, m, n, alpha, a, lda, b, ldb, nounit, 0);
        } else {
            if (UPLO == 'L') xtrsm_rlTC_core(0, m, n, alpha, a, lda, b, ldb, nounit, cflag);
            else             xtrsm_ruTC_core(0, m, n, alpha, a, lda, b, ldb, nounit, cflag);
        }
    }
}

#undef A_
#undef B_
