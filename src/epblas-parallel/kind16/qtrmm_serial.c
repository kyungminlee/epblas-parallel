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
#include <ctype.h>
#include <quadmath.h>

typedef qtrmm_T T;

char qtrmm_uplo(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── SIDE = 'L' column-range cores ────────────────────────────────
 * B := alpha · op(A) · B, A is M×M, B is M×N.
 * Each thread owns a column slice [j_start, j_end) of B. */

/* B := alpha · L · B */
void trmm_lln_core(int j_start, int j_end, int M, T alpha,
                   const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int k = M - 1; k >= 0; --k) {
            if (B_(k, j) != 0.0Q) {
                T temp = alpha * B_(k, j);
                for (int i = M - 1; i > k; --i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

/* B := alpha · U · B */
void trmm_lun_core(int j_start, int j_end, int M, T alpha,
                   const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int k = 0; k < M; ++k) {
            if (B_(k, j) != 0.0Q) {
                T temp = alpha * B_(k, j);
                for (int i = 0; i < k; ++i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

/* B := alpha · Lᵀ · B */
void trmm_llt_core(int j_start, int j_end, int M, T alpha,
                   const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = 0; i < M; ++i) {
            T t = B_(i, j);
            if (nounit) t *= A_(i, i);
            for (int k = i + 1; k < M; ++k) t += A_(k, i) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

/* B := alpha · Uᵀ · B */
void trmm_lut_core(int j_start, int j_end, int M, T alpha,
                   const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = M - 1; i >= 0; --i) {
            T t = B_(i, j);
            if (nounit) t *= A_(i, i);
            for (int k = 0; k < i; ++k) t += A_(k, i) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

/* ── SIDE = 'R' row-range cores ────────────────────────────────────
 * B := alpha · B · op(A), A is N×N, B is M×N.
 * Each thread owns a row slice [i_start, i_end) of B. */

/* B := alpha · B · L */
void trmm_rln_core(int i_start, int i_end, int N, T alpha,
                   const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = 0; j < N; ++j) {
        T t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != 1.0Q)
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (int k = j + 1; k < N; ++k) {
            if (A_(k, j) != 0.0Q) {
                const T akj = alpha * A_(k, j);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

/* B := alpha · B · U */
void trmm_run_core(int i_start, int i_end, int N, T alpha,
                   const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = N - 1; j >= 0; --j) {
        T t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != 1.0Q)
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (int k = 0; k < j; ++k) {
            if (A_(k, j) != 0.0Q) {
                const T akj = alpha * A_(k, j);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

/* B := alpha · B · Lᵀ */
void trmm_rlt_core(int i_start, int i_end, int N, T alpha,
                   const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = N - 1; k >= 0; --k) {
        for (int j = k + 1; j < N; ++j) {
            if (A_(j, k) != 0.0Q) {
                const T ajk = alpha * A_(j, k);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += ajk * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t *= A_(k, k);
        if (t != 1.0Q)
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

/* B := alpha · B · Uᵀ */
void trmm_rut_core(int i_start, int i_end, int N, T alpha,
                   const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < k; ++j) {
            if (A_(j, k) != 0.0Q) {
                const T ajk = alpha * A_(j, k);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += ajk * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t *= A_(k, k);
        if (t != 1.0Q)
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

void qtrmm_serial_(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len)
{
    (void)side_len; (void)uplo_len; (void)transa_len; (void)diag_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_;
    const T alpha = *alpha_;
    const char SIDE   = qtrmm_uplo(side);
    const char UPLO   = qtrmm_uplo(uplo);
    char TR           = qtrmm_uplo(transa);
    if (TR == 'C') TR = 'T';
    const int nounit = (qtrmm_uplo(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (alpha == 0.0Q) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = 0.0Q;
        return;
    }

    if (SIDE == 'L') {
        if (TR == 'N') {
            if (UPLO == 'L') trmm_lln_core(0, N, M, alpha, a, lda, b, ldb, nounit);
            else             trmm_lun_core(0, N, M, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') trmm_llt_core(0, N, M, alpha, a, lda, b, ldb, nounit);
            else             trmm_lut_core(0, N, M, alpha, a, lda, b, ldb, nounit);
        }
    } else {
        if (TR == 'N') {
            if (UPLO == 'L') trmm_rln_core(0, M, N, alpha, a, lda, b, ldb, nounit);
            else             trmm_run_core(0, M, N, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') trmm_rlt_core(0, M, N, alpha, a, lda, b, ldb, nounit);
            else             trmm_rut_core(0, M, N, alpha, a, lda, b, ldb, nounit);
        }
    }
}

#undef A_
#undef B_
