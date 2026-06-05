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
#include <ctype.h>
#include <quadmath.h>

typedef xtrmm_T T;

char xtrmm_uplo(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static const T ZERO = 0.0Q + 0.0Qi;
static const T ONE  = 1.0Q + 0.0Qi;

static inline T cconj(T a) { return conjq(a); }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

static inline T A_op(const T *a, int lda, int row, int col, int conj_flag) {
    return conj_flag ? cconj(A_(row, col)) : A_(row, col);
}

/* ── SIDE = 'L' column-range cores ──────────────────────────────── */

void xtrmm_lln_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int k = M - 1; k >= 0; --k) {
            if (B_(k, j) != ZERO) {
                T temp = alpha * B_(k, j);
                for (int i = M - 1; i > k; --i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

void xtrmm_lun_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int k = 0; k < M; ++k) {
            if (B_(k, j) != ZERO) {
                T temp = alpha * B_(k, j);
                for (int i = 0; i < k; ++i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

void xtrmm_llTC_core(int j_start, int j_end, int M, T alpha,
                     const T *a, int lda, T *b, int ldb,
                     int nounit, int conj_flag)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = 0; i < M; ++i) {
            T t = B_(i, j);
            if (nounit) t *= A_op(a, lda, i, i, conj_flag);
            for (int k = i + 1; k < M; ++k)
                t += A_op(a, lda, k, i, conj_flag) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

void xtrmm_luTC_core(int j_start, int j_end, int M, T alpha,
                     const T *a, int lda, T *b, int ldb,
                     int nounit, int conj_flag)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = M - 1; i >= 0; --i) {
            T t = B_(i, j);
            if (nounit) t *= A_op(a, lda, i, i, conj_flag);
            for (int k = 0; k < i; ++k)
                t += A_op(a, lda, k, i, conj_flag) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

/* ── SIDE = 'R' row-range cores ─────────────────────────────────── */

void xtrmm_rln_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = 0; j < N; ++j) {
        T t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != ONE)
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (int k = j + 1; k < N; ++k) {
            if (A_(k, j) != ZERO) {
                const T akj = alpha * A_(k, j);
                for (int i = i_start; i < i_end; ++i) B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

void xtrmm_run_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = N - 1; j >= 0; --j) {
        T t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != ONE)
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (int k = 0; k < j; ++k) {
            if (A_(k, j) != ZERO) {
                const T akj = alpha * A_(k, j);
                for (int i = i_start; i < i_end; ++i) B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

void xtrmm_rlTC_core(int i_start, int i_end, int N, T alpha,
                     const T *a, int lda, T *b, int ldb,
                     int nounit, int conj_flag)
{
    for (int k = N - 1; k >= 0; --k) {
        for (int j = k + 1; j < N; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                const T scaled = alpha * ajk;
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += scaled * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t *= A_op(a, lda, k, k, conj_flag);
        if (t != ONE)
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

void xtrmm_ruTC_core(int i_start, int i_end, int N, T alpha,
                     const T *a, int lda, T *b, int ldb,
                     int nounit, int conj_flag)
{
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < k; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                const T scaled = alpha * ajk;
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += scaled * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t *= A_op(a, lda, k, k, conj_flag);
        if (t != ONE)
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

void xtrmm_serial_(
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
    const char SIDE = xtrmm_uplo(side);
    const char UPLO = xtrmm_uplo(uplo);
    const char TR = xtrmm_uplo(transa);
    const int nounit = (xtrmm_uplo(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = ZERO;
        return;
    }

    if (SIDE == 'L') {
        if (TR == 'N') {
            if (UPLO == 'L') xtrmm_lln_core(0, N, M, alpha, a, lda, b, ldb, nounit);
            else             xtrmm_lun_core(0, N, M, alpha, a, lda, b, ldb, nounit);
        } else if (TR == 'T') {
            if (UPLO == 'L') xtrmm_llTC_core(0, N, M, alpha, a, lda, b, ldb, nounit, 0);
            else             xtrmm_luTC_core(0, N, M, alpha, a, lda, b, ldb, nounit, 0);
        } else {
            if (UPLO == 'L') xtrmm_llTC_core(0, N, M, alpha, a, lda, b, ldb, nounit, 1);
            else             xtrmm_luTC_core(0, N, M, alpha, a, lda, b, ldb, nounit, 1);
        }
    } else {
        if (TR == 'N') {
            if (UPLO == 'L') xtrmm_rln_core(0, M, N, alpha, a, lda, b, ldb, nounit);
            else             xtrmm_run_core(0, M, N, alpha, a, lda, b, ldb, nounit);
        } else if (TR == 'T') {
            if (UPLO == 'L') xtrmm_rlTC_core(0, M, N, alpha, a, lda, b, ldb, nounit, 0);
            else             xtrmm_ruTC_core(0, M, N, alpha, a, lda, b, ldb, nounit, 0);
        } else {
            if (UPLO == 'L') xtrmm_rlTC_core(0, M, N, alpha, a, lda, b, ldb, nounit, 1);
            else             xtrmm_ruTC_core(0, M, N, alpha, a, lda, b, ldb, nounit, 1);
        }
    }
}

#undef A_
#undef B_
