/*
 * xtrsm_serial.c — kind16 complex (COMPLEX(KIND=16) / __complex128)
 * triangular solve, serial core.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the
 * uplo decode, the conjugate / A_op helpers, the eight range-parameterized
 * solve cores (declared in xtrsm_kernel.h), plus the by-value `xtrsm_serial`
 * entry — the xtrsm_ algorithm forced fully serial. No OpenMP
 * anywhere on this call path; safe to invoke from inside another function's
 * `#pragma omp parallel` region.
 *
 * xtrsm_serial drives numerics through the same cores the parallel entry
 * threads (each called over the full column/row range), so the two paths
 * are bitwise-identical.
 *
 * TRANSA='C' is handled as a distinct case from 'T' (conjugate vs plain
 * transpose) via the conj_flag passed to the TC cores.
 */

#include "xtrsm_kernel.h"
#include "../common/blas_char.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <ctype.h>
#include <quadmath.h>

/* xtrsv-loop fast path threshold (mirror xtrsm_parallel.c). */
#define XTRSM_XTRSV_LOOP_M_MIN       128

typedef xtrsm_TC TC;

extern void xtrsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n,
    const TC *restrict a, ptrdiff_t lda,
    TC *restrict x, ptrdiff_t incx);

char xtrsm_uplo(char c) {
    return blas_up(c);
}

static const TC ZERO = 0.0Q + 0.0Qi;
static const TC ONE  = 1.0Q + 0.0Qi;

static inline TC cconj(TC a) { return conjq(a); }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

static inline TC A_op(const TC *a, ptrdiff_t lda, ptrdiff_t row, ptrdiff_t col, bool conj_flag) {
    return conj_flag ? cconj(A_(row, col)) : A_(row, col);
}

/* ── SIDE = 'L' column-range cores ──────────────────────────────── */

void xtrsm_lln_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, TC alpha,
                    const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        if (alpha != ONE) for (ptrdiff_t i = 0; i < m; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = 0; k < m; ++k) {
            if (B_(k, j) != ZERO) {
                if (nounit) B_(k, j) /= A_(k, k);
                const TC bk = B_(k, j);
                for (ptrdiff_t i = k + 1; i < m; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

void xtrsm_lun_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, TC alpha,
                    const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        if (alpha != ONE) for (ptrdiff_t i = 0; i < m; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = m - 1; k >= 0; --k) {
            if (B_(k, j) != ZERO) {
                if (nounit) B_(k, j) /= A_(k, k);
                const TC bk = B_(k, j);
                for (ptrdiff_t i = 0; i < k; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

void xtrsm_lltc_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, TC alpha,
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t i = m - 1; i >= 0; --i) {
            TC t = alpha * B_(i, j);
            for (ptrdiff_t k = i + 1; k < m; ++k) t -= A_op(a, lda, k, i, conj_flag) * B_(k, j);
            if (nounit) t /= A_op(a, lda, i, i, conj_flag);
            B_(i, j) = t;
        }
    }
}

void xtrsm_lutc_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, TC alpha,
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t i = 0; i < m; ++i) {
            TC t = alpha * B_(i, j);
            for (ptrdiff_t k = 0; k < i; ++k) t -= A_op(a, lda, k, i, conj_flag) * B_(k, j);
            if (nounit) t /= A_op(a, lda, i, i, conj_flag);
            B_(i, j) = t;
        }
    }
}

/* ── SIDE = 'R' row-range cores ─────────────────────────────────── */

void xtrsm_rln_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, TC alpha,
                    const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = n - 1; j >= 0; --j) {
        if (alpha != ONE) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = j + 1; k < n; ++k) {
            if (A_(k, j) != ZERO) {
                const TC akj = A_(k, j);
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const TC inv = ONE / A_(j, j);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

void xtrsm_run_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, TC alpha,
                    const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = 0; j < n; ++j) {
        if (alpha != ONE) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (ptrdiff_t k = 0; k < j; ++k) {
            if (A_(k, j) != ZERO) {
                const TC akj = A_(k, j);
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const TC inv = ONE / A_(j, j);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

void xtrsm_rltc_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, TC alpha,
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag)
{
    for (ptrdiff_t k = 0; k < n; ++k) {
        if (nounit) {
            const TC inv = ONE / A_op(a, lda, k, k, conj_flag);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (ptrdiff_t j = k + 1; j < n; ++j) {
            const TC ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != ONE) for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

void xtrsm_rutc_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, TC alpha,
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag)
{
    for (ptrdiff_t k = n - 1; k >= 0; --k) {
        if (nounit) {
            const TC inv = ONE / A_op(a, lda, k, k, conj_flag);
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (ptrdiff_t j = 0; j < k; ++j) {
            const TC ajk = A_op(a, lda, j, k, conj_flag);
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
    const TC *alpha_,
    const TC *a, ptrdiff_t lda,
    TC *b, ptrdiff_t ldb)
{
    const TC alpha = *alpha_;
    const char SIDE = xtrsm_uplo(side);
    const char UPLO = xtrsm_uplo(uplo);
    const char TRANS = xtrsm_uplo(transa);
    const bool nounit = (xtrsm_uplo(diag) != 'U');
    const bool cflag = (TRANS == 'C') ? 1 : 0;

    if (m == 0 || n == 0) return;

    if (alpha == ZERO) {
        for (ptrdiff_t j = 0; j < n; ++j)
            for (ptrdiff_t i = 0; i < m; ++i) B_(i, j) = ZERO;
        return;
    }

    /* xtrsv-loop fast path (serial: nrhs sequential xtrsv solves). */
    {
        /* Serial entry: no team is available, so the xtrsv-loop cap
         * floors at n == 1 (the parallel twin computes a real
         * min(team, m/NB) bound). */
        if (SIDE == 'L' && n == 1 && m >= XTRSM_XTRSV_LOOP_M_MIN) {
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
        if (TRANS == 'N') {
            if (UPLO == 'L') xtrsm_lln_core(0, n, m, alpha, a, lda, b, ldb, nounit);
            else             xtrsm_lun_core(0, n, m, alpha, a, lda, b, ldb, nounit);
        } else if (TRANS == 'T') {
            if (UPLO == 'L') xtrsm_lltc_core(0, n, m, alpha, a, lda, b, ldb, nounit, 0);
            else             xtrsm_lutc_core(0, n, m, alpha, a, lda, b, ldb, nounit, 0);
        } else { /* 'C' */
            if (UPLO == 'L') xtrsm_lltc_core(0, n, m, alpha, a, lda, b, ldb, nounit, cflag);
            else             xtrsm_lutc_core(0, n, m, alpha, a, lda, b, ldb, nounit, cflag);
        }
    } else {
        if (TRANS == 'N') {
            if (UPLO == 'L') xtrsm_rln_core(0, m, n, alpha, a, lda, b, ldb, nounit);
            else             xtrsm_run_core(0, m, n, alpha, a, lda, b, ldb, nounit);
        } else if (TRANS == 'T') {
            if (UPLO == 'L') xtrsm_rltc_core(0, m, n, alpha, a, lda, b, ldb, nounit, 0);
            else             xtrsm_rutc_core(0, m, n, alpha, a, lda, b, ldb, nounit, 0);
        } else {
            if (UPLO == 'L') xtrsm_rltc_core(0, m, n, alpha, a, lda, b, ldb, nounit, cflag);
            else             xtrsm_rutc_core(0, m, n, alpha, a, lda, b, ldb, nounit, cflag);
        }
    }
}

#undef A_
#undef B_
