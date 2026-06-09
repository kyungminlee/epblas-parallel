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
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#include <quadmath.h>

/* xtrsv-loop fast path thresholds (mirror xtrsm_parallel.c). */
#define XTRSM_XTRSV_LOOP_M_MIN       128
#define XTRSM_XTRSV_LOOP_NB_HINT     64

typedef xtrsm_T T;

extern void xtrsv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n,
    const T *a, const int *lda,
    T *x, const int *incx,
    size_t uplo_len, size_t trans_len, size_t diag_len);

/* Maximum nrhs at which the xtrsv-loop fast path beats column-parallel
 * xtrsm. In the serial entry no team is available, so the heuristic floors
 * at 1. */
static int xtrsm_xtrsv_loop_max(int M) {
    const int max_nt     = 1 - 1;
    const int max_amdahl = M / XTRSM_XTRSV_LOOP_NB_HINT;
    int v = (max_nt < max_amdahl) ? max_nt : max_amdahl;
    if (v < 1) v = 1;
    return v;
}

char xtrsm_uplo(const char *p) {
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

void xtrsm_lln_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        if (alpha != ONE) for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
        for (int k = 0; k < M; ++k) {
            if (B_(k, j) != ZERO) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (int i = k + 1; i < M; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

void xtrsm_lun_core(int j_start, int j_end, int M, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        if (alpha != ONE) for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
        for (int k = M - 1; k >= 0; --k) {
            if (B_(k, j) != ZERO) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (int i = 0; i < k; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

void xtrsm_llTC_core(int j_start, int j_end, int M, T alpha,
                     const T *a, int lda, T *b, int ldb,
                     int nounit, int conj_flag)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = M - 1; i >= 0; --i) {
            T t = alpha * B_(i, j);
            for (int k = i + 1; k < M; ++k) t -= A_op(a, lda, k, i, conj_flag) * B_(k, j);
            if (nounit) t /= A_op(a, lda, i, i, conj_flag);
            B_(i, j) = t;
        }
    }
}

void xtrsm_luTC_core(int j_start, int j_end, int M, T alpha,
                     const T *a, int lda, T *b, int ldb,
                     int nounit, int conj_flag)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = 0; i < M; ++i) {
            T t = alpha * B_(i, j);
            for (int k = 0; k < i; ++k) t -= A_op(a, lda, k, i, conj_flag) * B_(k, j);
            if (nounit) t /= A_op(a, lda, i, i, conj_flag);
            B_(i, j) = t;
        }
    }
}

/* ── SIDE = 'R' row-range cores ─────────────────────────────────── */

void xtrsm_rln_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = N - 1; j >= 0; --j) {
        if (alpha != ONE) for (int i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (int k = j + 1; k < N; ++k) {
            if (A_(k, j) != ZERO) {
                const T akj = A_(k, j);
                for (int i = i_start; i < i_end; ++i) B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = ONE / A_(j, j);
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

void xtrsm_run_core(int i_start, int i_end, int N, T alpha,
                    const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = 0; j < N; ++j) {
        if (alpha != ONE) for (int i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (int k = 0; k < j; ++k) {
            if (A_(k, j) != ZERO) {
                const T akj = A_(k, j);
                for (int i = i_start; i < i_end; ++i) B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = ONE / A_(j, j);
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

void xtrsm_rlTC_core(int i_start, int i_end, int N, T alpha,
                     const T *a, int lda, T *b, int ldb,
                     int nounit, int conj_flag)
{
    for (int k = 0; k < N; ++k) {
        if (nounit) {
            const T inv = ONE / A_op(a, lda, k, k, conj_flag);
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (int j = k + 1; j < N; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                for (int i = i_start; i < i_end; ++i) B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != ONE) for (int i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

void xtrsm_ruTC_core(int i_start, int i_end, int N, T alpha,
                     const T *a, int lda, T *b, int ldb,
                     int nounit, int conj_flag)
{
    for (int k = N - 1; k >= 0; --k) {
        if (nounit) {
            const T inv = ONE / A_op(a, lda, k, k, conj_flag);
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (int j = 0; j < k; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                for (int i = i_start; i < i_end; ++i) B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != ONE) for (int i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
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
void xtrsm_serial_(
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
    const char SIDE = xtrsm_uplo(side);
    const char UPLO = xtrsm_uplo(uplo);
    const char TR = xtrsm_uplo(transa);
    const int nounit = (xtrsm_uplo(diag) != 'U');
    const int cflag = (TR == 'C') ? 1 : 0;

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = ZERO;
        return;
    }

    /* xtrsv-loop fast path (serial: nrhs sequential xtrsv solves). */
    {
        const int xv_max = xtrsm_xtrsv_loop_max(M);
        if (SIDE == 'L' && N >= 1 && N <= xv_max && M >= XTRSM_XTRSV_LOOP_M_MIN) {
            if (alpha != ONE) {
                for (int j = 0; j < N; ++j)
                    for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
            }
            const int incx_one = 1;
            for (int j = 0; j < N; ++j) {
                xtrsv_(uplo, transa, diag, m_, a, lda_,
                       &B_(0, j), &incx_one, 1, 1, 1);
            }
            return;
        }
    }

    if (SIDE == 'L') {
        if (TR == 'N') {
            if (UPLO == 'L') xtrsm_lln_core(0, N, M, alpha, a, lda, b, ldb, nounit);
            else             xtrsm_lun_core(0, N, M, alpha, a, lda, b, ldb, nounit);
        } else if (TR == 'T') {
            if (UPLO == 'L') xtrsm_llTC_core(0, N, M, alpha, a, lda, b, ldb, nounit, 0);
            else             xtrsm_luTC_core(0, N, M, alpha, a, lda, b, ldb, nounit, 0);
        } else { /* 'C' */
            if (UPLO == 'L') xtrsm_llTC_core(0, N, M, alpha, a, lda, b, ldb, nounit, cflag);
            else             xtrsm_luTC_core(0, N, M, alpha, a, lda, b, ldb, nounit, cflag);
        }
    } else {
        if (TR == 'N') {
            if (UPLO == 'L') xtrsm_rln_core(0, M, N, alpha, a, lda, b, ldb, nounit);
            else             xtrsm_run_core(0, M, N, alpha, a, lda, b, ldb, nounit);
        } else if (TR == 'T') {
            if (UPLO == 'L') xtrsm_rlTC_core(0, M, N, alpha, a, lda, b, ldb, nounit, 0);
            else             xtrsm_ruTC_core(0, M, N, alpha, a, lda, b, ldb, nounit, 0);
        } else {
            if (UPLO == 'L') xtrsm_rlTC_core(0, M, N, alpha, a, lda, b, ldb, nounit, cflag);
            else             xtrsm_ruTC_core(0, M, N, alpha, a, lda, b, ldb, nounit, cflag);
        }
    }
}

#undef A_
#undef B_
