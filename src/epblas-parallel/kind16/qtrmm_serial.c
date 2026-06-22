/*
 * qtrmm_serial.c — kind16 (REAL(KIND=16) / __float128) triangular multiply,
 * serial core.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the
 * uplo-char decode and the eight range-parameterized compute cores
 * (declared in qtrmm_kernel.h), plus the public `qtrmm_serial_` Fortran
 * entry. No OpenMP anywhere on this call path — safe to invoke from inside
 * another function's `#pragma omp parallel` region; callers are responsible
 * for partitioning if they want thread parallelism.
 *
 * Both qtrmm_serial_ and the parallel qtrmm_ drive numerics through the same
 * cores over identical [start,end) ranges, so the two paths are
 * bitwise-identical.
 */

#include "qtrmm_kernel.h"
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>

typedef qtrmm_T T;

char qtrmm_uplo(char c) {
    return blas_up(c);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── SIDE = 'L' column-range cores ────────────────────────────────
 * B := alpha · op(A) · B, A is M×M, B is M×N.
 * Each thread owns a column slice [j_start, j_end) of B. */

/* B := alpha · L · B */
void trmm_lln_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, T alpha,
                   const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t k = m - 1; k >= 0; --k) {
            if (B_(k, j) != 0.0Q) {
                T temp = alpha * B_(k, j);
                for (ptrdiff_t i = m - 1; i > k; --i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

/* B := alpha · U · B */
void trmm_lun_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, T alpha,
                   const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t k = 0; k < m; ++k) {
            if (B_(k, j) != 0.0Q) {
                T temp = alpha * B_(k, j);
                for (ptrdiff_t i = 0; i < k; ++i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

/* B := alpha · Lᵀ · B */
void trmm_llt_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, T alpha,
                   const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t i = 0; i < m; ++i) {
            T t = B_(i, j);
            if (nounit) t *= A_(i, i);
            for (ptrdiff_t k = i + 1; k < m; ++k) t += A_(k, i) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

/* B := alpha · Uᵀ · B */
void trmm_lut_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, T alpha,
                   const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        for (ptrdiff_t i = m - 1; i >= 0; --i) {
            T t = B_(i, j);
            if (nounit) t *= A_(i, i);
            for (ptrdiff_t k = 0; k < i; ++k) t += A_(k, i) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

/* ── SIDE = 'R' row-range cores ────────────────────────────────────
 * B := alpha · B · op(A), A is N×N, B is M×N.
 * Each thread owns a row slice [i_start, i_end) of B. */

/* B := alpha · B · L */
void trmm_rln_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, T alpha,
                   const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = 0; j < n; ++j) {
        T t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != 1.0Q)
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (ptrdiff_t k = j + 1; k < n; ++k) {
            if (A_(k, j) != 0.0Q) {
                const T akj = alpha * A_(k, j);
                for (ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

/* B := alpha · B · U */
void trmm_run_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, T alpha,
                   const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t j = n - 1; j >= 0; --j) {
        T t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != 1.0Q)
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (ptrdiff_t k = 0; k < j; ++k) {
            if (A_(k, j) != 0.0Q) {
                const T akj = alpha * A_(k, j);
                for (ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

/* B := alpha · B · Lᵀ */
void trmm_rlt_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, T alpha,
                   const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t k = n - 1; k >= 0; --k) {
        for (ptrdiff_t j = k + 1; j < n; ++j) {
            if (A_(j, k) != 0.0Q) {
                const T ajk = alpha * A_(j, k);
                for (ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) += ajk * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t *= A_(k, k);
        if (t != 1.0Q)
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

/* B := alpha · B · Uᵀ */
void trmm_rut_core(ptrdiff_t i_start, ptrdiff_t i_end, ptrdiff_t n, T alpha,
                   const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, bool nounit)
{
    for (ptrdiff_t k = 0; k < n; ++k) {
        for (ptrdiff_t j = 0; j < k; ++j) {
            if (A_(j, k) != 0.0Q) {
                const T ajk = alpha * A_(j, k);
                for (ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) += ajk * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t *= A_(k, k);
        if (t != 1.0Q)
            for (ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

void qtrmm_serial(
    char side, char uplo, char transa, char diag,
    ptrdiff_t m, ptrdiff_t n,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    T *b, ptrdiff_t ldb)
{
    const T alpha = *alpha_;
    const char SIDE   = qtrmm_uplo(side);
    const char UPLO   = qtrmm_uplo(uplo);
    char TRANS           = qtrmm_uplo(transa);
    if (TRANS == 'C') TRANS = 'T';
    const bool nounit = (qtrmm_uplo(diag) != 'U');

    if (m == 0 || n == 0) return;

    if (alpha == 0.0Q) {
        for (ptrdiff_t j = 0; j < n; ++j)
            for (ptrdiff_t i = 0; i < m; ++i) B_(i, j) = 0.0Q;
        return;
    }

    if (SIDE == 'L') {
        if (TRANS == 'N') {
            if (UPLO == 'L') trmm_lln_core(0, n, m, alpha, a, lda, b, ldb, nounit);
            else             trmm_lun_core(0, n, m, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') trmm_llt_core(0, n, m, alpha, a, lda, b, ldb, nounit);
            else             trmm_lut_core(0, n, m, alpha, a, lda, b, ldb, nounit);
        }
    } else {
        if (TRANS == 'N') {
            if (UPLO == 'L') trmm_rln_core(0, m, n, alpha, a, lda, b, ldb, nounit);
            else             trmm_run_core(0, m, n, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') trmm_rlt_core(0, m, n, alpha, a, lda, b, ldb, nounit);
            else             trmm_rut_core(0, m, n, alpha, a, lda, b, ldb, nounit);
        }
    }
}

#undef A_
#undef B_
