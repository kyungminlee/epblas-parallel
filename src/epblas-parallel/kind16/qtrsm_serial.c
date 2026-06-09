/*
 * qtrsm_serial.c — kind16 (REAL(KIND=16) / __float128) triangular solve,
 * serial core.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the
 * uplo decode and the eight range-parameterized solve cores (declared in
 * qtrsm_kernel.h), plus the public `qtrsm_serial_` Fortran entry — the
 * qtrsm_ algorithm forced fully serial. No OpenMP anywhere on this call
 * path; safe to invoke from inside another function's `#pragma omp parallel`
 * region.
 *
 * qtrsm_serial_ drives numerics through the same cores the parallel entry
 * threads (each called over the full column/row range), so the two paths
 * are bitwise-identical.
 */

#include "qtrsm_kernel.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#include <quadmath.h>

/* qtrsv-loop fast path thresholds (mirror qtrsm_parallel.c). */
#define QTRSM_QTRSV_LOOP_M_MIN       128
#define QTRSM_QTRSV_LOOP_NB_HINT     64

typedef qtrsm_T T;

extern void qtrsv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n,
    const T *a, const int *lda,
    T *x, const int *incx,
    size_t uplo_len, size_t trans_len, size_t diag_len);

/* Maximum nrhs at which the qtrsv-loop fast path beats column-parallel
 * qtrsm. In the serial entry no team is available, so the heuristic floors
 * at 1. */
static int qtrsm_qtrsv_loop_max(int M) {
    const int max_nt     = 1 - 1;
    const int max_amdahl = M / QTRSM_QTRSV_LOOP_NB_HINT;
    int v = (max_nt < max_amdahl) ? max_nt : max_amdahl;
    if (v < 1) v = 1;
    return v;
}

char qtrsm_uplo(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── SIDE = 'L' column-range cores ──────────────────────────────── */

void qtrsm_lln_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        if (alpha != 1.0Q) for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
        for (int k = 0; k < M; ++k) {
            if (B_(k, j) != 0.0Q) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (int i = k + 1; i < M; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

void qtrsm_lun_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        if (alpha != 1.0Q) for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
        for (int k = M - 1; k >= 0; --k) {
            if (B_(k, j) != 0.0Q) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (int i = 0; i < k; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

void qtrsm_llt_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = M - 1; i >= 0; --i) {
            T t = alpha * B_(i, j);
            for (int k = i + 1; k < M; ++k) t -= A_(k, i) * B_(k, j);
            if (nounit) t /= A_(i, i);
            B_(i, j) = t;
        }
    }
}

void qtrsm_lut_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = 0; i < M; ++i) {
            T t = alpha * B_(i, j);
            for (int k = 0; k < i; ++k) t -= A_(k, i) * B_(k, j);
            if (nounit) t /= A_(i, i);
            B_(i, j) = t;
        }
    }
}

/* ── SIDE = 'R' row-range cores ─────────────────────────────────── */

void qtrsm_rln_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = N - 1; j >= 0; --j) {
        if (alpha != 1.0Q) for (int i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (int k = j + 1; k < N; ++k) {
            if (A_(k, j) != 0.0Q) {
                const T akj = A_(k, j);
                for (int i = i_start; i < i_end; ++i) B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = 1.0Q / A_(j, j);
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

void qtrsm_run_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = 0; j < N; ++j) {
        if (alpha != 1.0Q) for (int i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (int k = 0; k < j; ++k) {
            if (A_(k, j) != 0.0Q) {
                const T akj = A_(k, j);
                for (int i = i_start; i < i_end; ++i) B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = 1.0Q / A_(j, j);
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

void qtrsm_rlt_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = 0; k < N; ++k) {
        if (nounit) {
            const T inv = 1.0Q / A_(k, k);
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (int j = k + 1; j < N; ++j) {
            if (A_(j, k) != 0.0Q) {
                const T ajk = A_(j, k);
                for (int i = i_start; i < i_end; ++i) B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != 1.0Q) for (int i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

void qtrsm_rut_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = N - 1; k >= 0; --k) {
        if (nounit) {
            const T inv = 1.0Q / A_(k, k);
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (int j = 0; j < k; ++j) {
            if (A_(j, k) != 0.0Q) {
                const T ajk = A_(j, k);
                for (int i = i_start; i < i_end; ++i) B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != 1.0Q) for (int i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
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
void qtrsm_serial_(
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
    const char SIDE   = qtrsm_uplo(side);
    const char UPLO   = qtrsm_uplo(uplo);
    char TR           = qtrsm_uplo(transa);
    if (TR == 'C') TR = 'T';   /* real type: conj-trans ≡ trans */
    const int nounit = (qtrsm_uplo(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (alpha == 0.0Q) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = 0.0Q;
        return;
    }

    /* qtrsv-loop fast path (serial: nrhs sequential qtrsv solves). */
    {
        const int xv_max = qtrsm_qtrsv_loop_max(M);
        if (SIDE == 'L' && N >= 1 && N <= xv_max && M >= QTRSM_QTRSV_LOOP_M_MIN) {
            if (alpha != 1.0Q) {
                for (int j = 0; j < N; ++j)
                    for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
            }
            const int incx_one = 1;
            for (int j = 0; j < N; ++j) {
                qtrsv_(uplo, transa, diag, m_, a, lda_,
                       &B_(0, j), &incx_one, 1, 1, 1);
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
