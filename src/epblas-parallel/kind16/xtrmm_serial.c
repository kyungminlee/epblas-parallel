/*
 * xtrmm_serial.c — kind16 complex (COMPLEX(KIND=16) / __complex128)
 * triangular multiply, serial core.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the
 * char decode, the file-static conj / A_op helpers (only the cores use
 * them), the range-parameterized compute cores (declared in
 * xtrmm_kernel.h), and the public `xtrmm_serial_` Fortran entry. No OpenMP
 * anywhere on this call path — safe to invoke from inside another
 * function's `#pragma omp parallel` region; callers are responsible for
 * partitioning if they want thread parallelism.
 *
 * Both xtrmm_serial_ and the parallel xtrmm_ drive numerics through the same
 * cores over identical [start,end) ranges, so the two paths are
 * bitwise-identical.
 */

#include "xtrmm_kernel.h"
#include "../common/blas_char.h"
#include <ctype.h>
#include <stdbool.h>
#include <quadmath.h>

typedef xtrmm_TC TC;

char xtrmm_uplo(char c) {
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

void xtrmm_lln_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, TC alpha,
                    const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t k = m - 1; k >= 0; --k) {
            if (B_(k, j) != ZERO) {
                TC temp = alpha * B_(k, j);
                for (ptrdiff_t i = m - 1; i > k; --i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

void xtrmm_lun_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, TC alpha,
                    const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t k = 0; k < m; ++k) {
            if (B_(k, j) != ZERO) {
                TC temp = alpha * B_(k, j);
                for (ptrdiff_t i = 0; i < k; ++i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

void xtrmm_lltc_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, TC alpha,
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t i = 0; i < m; ++i) {
            TC t = B_(i, j);
            if (nounit) t *= A_op(a, lda, i, i, conj_flag);
            for (ptrdiff_t k = i + 1; k < m; ++k)
                t += A_op(a, lda, k, i, conj_flag) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

void xtrmm_lutc_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, TC alpha,
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t i = m - 1; i >= 0; --i) {
            TC t = B_(i, j);
            if (nounit) t *= A_op(a, lda, i, i, conj_flag);
            for (ptrdiff_t k = 0; k < i; ++k)
                t += A_op(a, lda, k, i, conj_flag) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

/* ── SIDE = 'R' row-range cores ─────────────────────────────────── */

void xtrmm_rln_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, TC alpha,
                    const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = 0; j < n; ++j) {
        TC t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != ONE)
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (ptrdiff_t k = j + 1; k < n; ++k) {
            if (A_(k, j) != ZERO) {
                const TC akj = alpha * A_(k, j);
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

void xtrmm_run_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, TC alpha,
                    const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = n - 1; j >= 0; --j) {
        TC t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != ONE)
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (ptrdiff_t k = 0; k < j; ++k) {
            if (A_(k, j) != ZERO) {
                const TC akj = alpha * A_(k, j);
                for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

void xtrmm_rltc_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, TC alpha,
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag)
{
    for (ptrdiff_t k = n - 1; k >= 0; --k) {
        for (ptrdiff_t j = k + 1; j < n; ++j) {
            const TC ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                const TC scaled = alpha * ajk;
                for (ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) += scaled * B_(i, k);
            }
        }
        TC t = alpha;
        if (nounit) t *= A_op(a, lda, k, k, conj_flag);
        if (t != ONE)
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

void xtrmm_rutc_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, TC alpha,
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb,
                     bool nounit, bool conj_flag)
{
    for (ptrdiff_t k = 0; k < n; ++k) {
        for (ptrdiff_t j = 0; j < k; ++j) {
            const TC ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                const TC scaled = alpha * ajk;
                for (ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) += scaled * B_(i, k);
            }
        }
        TC t = alpha;
        if (nounit) t *= A_op(a, lda, k, k, conj_flag);
        if (t != ONE)
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

void xtrmm_serial(
    char side, char uplo, char transa, char diag,
    ptrdiff_t m, ptrdiff_t n,
    const TC *alpha_,
    const TC *a, ptrdiff_t lda,
    TC *b, ptrdiff_t ldb)
{
    const TC alpha = *alpha_;
    const char SIDE = xtrmm_uplo(side);
    const char UPLO = xtrmm_uplo(uplo);
    const char TRANS = xtrmm_uplo(transa);
    const bool nounit = (xtrmm_uplo(diag) != 'U');

    if (m == 0 || n == 0) return;

    if (alpha == ZERO) {
        for (ptrdiff_t j = 0; j < n; ++j)
            for (ptrdiff_t i = 0; i < m; ++i) B_(i, j) = ZERO;
        return;
    }

    if (SIDE == 'L') {
        if (TRANS == 'N') {
            if (UPLO == 'L') xtrmm_lln_core(0, n, m, alpha, a, lda, b, ldb, nounit);
            else             xtrmm_lun_core(0, n, m, alpha, a, lda, b, ldb, nounit);
        } else if (TRANS == 'T') {
            if (UPLO == 'L') xtrmm_lltc_core(0, n, m, alpha, a, lda, b, ldb, nounit, 0);
            else             xtrmm_lutc_core(0, n, m, alpha, a, lda, b, ldb, nounit, 0);
        } else {
            if (UPLO == 'L') xtrmm_lltc_core(0, n, m, alpha, a, lda, b, ldb, nounit, 1);
            else             xtrmm_lutc_core(0, n, m, alpha, a, lda, b, ldb, nounit, 1);
        }
    } else {
        if (TRANS == 'N') {
            if (UPLO == 'L') xtrmm_rln_core(0, m, n, alpha, a, lda, b, ldb, nounit);
            else             xtrmm_run_core(0, m, n, alpha, a, lda, b, ldb, nounit);
        } else if (TRANS == 'T') {
            if (UPLO == 'L') xtrmm_rltc_core(0, m, n, alpha, a, lda, b, ldb, nounit, 0);
            else             xtrmm_rutc_core(0, m, n, alpha, a, lda, b, ldb, nounit, 0);
        } else {
            if (UPLO == 'L') xtrmm_rltc_core(0, m, n, alpha, a, lda, b, ldb, nounit, 1);
            else             xtrmm_rutc_core(0, m, n, alpha, a, lda, b, ldb, nounit, 1);
        }
    }
}

#undef A_
#undef B_
