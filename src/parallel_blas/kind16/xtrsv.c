/*
 * xtrsv — kind16 complex triangular solve.
 *
 * Three public entries:
 *
 *   xtrsv_         — top-level dispatch. Routes stride-1 calls above
 *                    the 2·NB threshold into xtrsv_blocked_ (which
 *                    opens its own parallel region), otherwise falls
 *                    through to the unblocked Netlib serial body.
 *                    Skips the blocked-path dispatch when already
 *                    inside an OpenMP parallel region.
 *
 *   xtrsv_serial_  — pure serial unblocked Netlib body. No OpenMP
 *                    pragma anywhere on this call path. Safe to call
 *                    from inside another function's parallel region.
 *
 *   xtrsv_blocked_ — LAPACK-blocked algorithm wrapped in a SINGLE
 *                    `#pragma omp parallel` region. Threads cooperate
 *                    manually: thread 0 does each diagonal sub-solve
 *                    via xtrsv_serial_, then all threads partition
 *                    the trailing xgemv across the long axis (M for
 *                    TR='N', N for TR='T'/'C') and call xgemv_serial_
 *                    on their slice. Two barriers per diagonal step
 *                    (after sub-solve, after trailing update).
 *
 * The refactor replaces the previous shape — N/nb separate xgemv
 * fork-joins — with one fork-join total plus 2·N/nb barriers. Same
 * order of synchronization cost but threads stay pinned across the
 * walk, improving cache locality. Aligns with the "no OMP-using
 * function called from inside an OMP region" rule by routing all
 * inner work through the *_serial_ entries.
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef __complex128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#define XTRSV_BLOCKED_NB_DEFAULT 64

static int xtrsv_blocked_nb(void) {
    static int cached = 0;
    if (cached == 0) {
        const char *s = getenv("XTRSV_NB");
        int v = (s && *s) ? atoi(s) : 0;
        cached = (v > 0) ? v : XTRSV_BLOCKED_NB_DEFAULT;
    }
    return cached;
}

void xtrsv_blocked_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len);

void xtrsv_serial_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len);

void xtrsv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    const int N = *n_;
    const int incx = *incx_;

    if (N == 0) return;

    /* Block-parallel dispatch: stride-1 calls above the blocking
     * threshold route through the LAPACK-blocked path. Skip if already
     * inside an OMP region (caller is managing parallelism). */
#ifdef _OPENMP
    const int in_par = omp_in_parallel();
#else
    const int in_par = 0;
#endif
    if (incx == 1 && N >= 2 * xtrsv_blocked_nb() && !in_par) {
        xtrsv_blocked_(uplo, trans, diag, n_, a, lda_, x, incx_,
                       uplo_len, trans_len, diag_len);
        return;
    }

    xtrsv_serial_(uplo, trans, diag, n_, a, lda_, x, incx_,
                  uplo_len, trans_len, diag_len);
}

/* Pure-serial unblocked Netlib body. No OpenMP. */
void xtrsv_serial_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const int N = *n_;
    const int lda = *lda_, incx = *incx_;
    const char UPLO = up(uplo);
    const char TR   = up(trans);
    const char DIAG = up(diag);
    const int nounit = (DIAG != 'U');

    if (N == 0) return;

    const T zero = 0.0Q + 0.0Qi;

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (int i = 0; i < N; ++i) {
                    if (x[i] != zero) {
                        if (nounit) x[i] /= A_(i, i);
                        const T xi = x[i];
                        const T *ai = &A_(0, i);
                        for (int k = i + 1; k < N; ++k) x[k] -= xi * ai[k];
                    }
                }
            } else {
                for (int i = N - 1; i >= 0; --i) {
                    if (x[i] != zero) {
                        if (nounit) x[i] /= A_(i, i);
                        const T xi = x[i];
                        const T *ai = &A_(0, i);
                        for (int k = 0; k < i; ++k) x[k] -= xi * ai[k];
                    }
                }
            }
        } else {
            const int conj_a = (TR == 'C');
            if (UPLO == 'L') {
                for (int i = N - 1; i >= 0; --i) {
                    T t = x[i];
                    const T *ai = &A_(0, i);
                    if (conj_a) {
                        for (int k = i + 1; k < N; ++k) t -= conjq(ai[k]) * x[k];
                        if (nounit) t /= conjq(ai[i]);
                    } else {
                        for (int k = i + 1; k < N; ++k) t -= ai[k] * x[k];
                        if (nounit) t /= ai[i];
                    }
                    x[i] = t;
                }
            } else {
                for (int i = 0; i < N; ++i) {
                    T t = x[i];
                    const T *ai = &A_(0, i);
                    if (conj_a) {
                        for (int k = 0; k < i; ++k) t -= conjq(ai[k]) * x[k];
                        if (nounit) t /= conjq(ai[i]);
                    } else {
                        for (int k = 0; k < i; ++k) t -= ai[k] * x[k];
                        if (nounit) t /= ai[i];
                    }
                    x[i] = t;
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (int i = 0; i < N; ++i) {
                    const int ix = kx + i * incx;
                    if (x[ix] != zero) {
                        if (nounit) x[ix] /= A_(i, i);
                        const T xi = x[ix];
                        for (int k = i + 1; k < N; ++k) x[kx + k * incx] -= xi * A_(k, i);
                    }
                }
            } else {
                for (int i = N - 1; i >= 0; --i) {
                    const int ix = kx + i * incx;
                    if (x[ix] != zero) {
                        if (nounit) x[ix] /= A_(i, i);
                        const T xi = x[ix];
                        for (int k = 0; k < i; ++k) x[kx + k * incx] -= xi * A_(k, i);
                    }
                }
            }
        } else {
            const int conj_a = (TR == 'C');
            if (UPLO == 'L') {
                for (int i = N - 1; i >= 0; --i) {
                    T t = x[kx + i * incx];
                    for (int k = i + 1; k < N; ++k) {
                        const T aki = conj_a ? conjq(A_(k, i)) : A_(k, i);
                        t -= aki * x[kx + k * incx];
                    }
                    if (nounit) t /= (conj_a ? conjq(A_(i, i)) : A_(i, i));
                    x[kx + i * incx] = t;
                }
            } else {
                for (int i = 0; i < N; ++i) {
                    T t = x[kx + i * incx];
                    for (int k = 0; k < i; ++k) {
                        const T aki = conj_a ? conjq(A_(k, i)) : A_(k, i);
                        t -= aki * x[kx + k * incx];
                    }
                    if (nounit) t /= (conj_a ? conjq(A_(i, i)) : A_(i, i));
                    x[kx + i * incx] = t;
                }
            }
        }
    }
}

/* ── Block-parallel variant: single parallel region ─────────────────
 *
 * One `#pragma omp parallel` wraps the entire diagonal walk. Threads
 * cooperate manually inside the region:
 *
 *   - Thread 0 calls xtrsv_serial_ on each diagonal sub-block.
 *   - All threads partition the trailing xgemv across its long axis
 *     and call xgemv_serial_ on their slice.
 *   - Two `#pragma omp barrier`s per step (after sub-solve, after
 *     trailing update).
 *
 * Compared to the previous per-step xgemv_ fork-joins, this issues
 * one fork-join total + 2·N/nb barriers. Threads stay pinned across
 * the walk so the partial x vector each thread accesses can stay in
 * its cache, and there's no nested OMP because the inner calls all
 * route through the *_serial_ entries.
 */

extern void xgemv_serial_(
    const char *trans,
    const int *m, const int *n,
    const T *alpha,
    const T *a, const int *lda,
    const T *x, const int *incx,
    const T *beta,
    T *y, const int *incy,
    size_t trans_len);

void xtrsv_blocked_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    const int N = *n_;
    const int lda = *lda_, incx = *incx_;
    const int nb = xtrsv_blocked_nb();
    const char UPLO = up(uplo);
    const char TR   = up(trans);

    if (N == 0) return;
    if (incx != 1 || N < 2 * nb) {
        xtrsv_serial_(uplo, trans, diag, n_, a, lda_, x, incx_,
                      uplo_len, trans_len, diag_len);
        return;
    }

    const T neg_one = -1.0Q + 0.0Qi;
    const T one_v   =  1.0Q + 0.0Qi;
    const char NN[1] = {'N'};
    const char TT[1] = {(TR == 'C') ? 'C' : 'T'};
    const int one_i = 1;

#ifdef _OPENMP
    const int use_omp = (blas_omp_max_threads() > 1 && !omp_in_parallel());
#else
    const int use_omp = 0;
#endif

#ifdef _OPENMP
    #pragma omp parallel if(use_omp)
#endif
    {
        int tid = 0, nt = 1;
#ifdef _OPENMP
        if (use_omp) { tid = omp_get_thread_num(); nt = omp_get_num_threads(); }
#endif

        if (TR == 'N' && UPLO == 'L') {
            /* Forward: solve A11 x1 = b1, then x2 -= A21 x1, repeat. */
            for (int j = 0; j < N; j += nb) {
                int jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    xtrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), lda_,
                                  &x[j], &one_i, uplo_len, trans_len, diag_len);
                }
#ifdef _OPENMP
                if (use_omp) {
                    #pragma omp barrier
                }
#endif
                int mt = N - j - jb;
                if (mt > 0) {
                    int j2 = j + jb;
                    long long lo = (long long)mt * tid / nt;
                    long long hi = (long long)mt * (tid + 1) / nt;
                    int m_slice = (int)(hi - lo);
                    if (m_slice > 0) {
                        const int i_off = j2 + (int)lo;
                        xgemv_serial_(NN, &m_slice, &jb, &neg_one,
                                      &A_(i_off, j), lda_,
                                      &x[j], &one_i, &one_v,
                                      &x[i_off], &one_i, 1);
                    }
                }
#ifdef _OPENMP
                if (use_omp) {
                    #pragma omp barrier
                }
#endif
            }
        } else if (TR == 'N' && UPLO == 'U') {
            /* Backward: solve A22 x2 = b2, then x1 -= A12 x2, repeat. */
            int j = ((N - 1) / nb) * nb;
            while (j >= 0) {
                int jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    xtrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), lda_,
                                  &x[j], &one_i, uplo_len, trans_len, diag_len);
                }
#ifdef _OPENMP
                if (use_omp) {
                    #pragma omp barrier
                }
#endif
                if (j > 0) {
                    long long lo = (long long)j * tid / nt;
                    long long hi = (long long)j * (tid + 1) / nt;
                    int m_slice = (int)(hi - lo);
                    if (m_slice > 0) {
                        const int i_off = (int)lo;
                        xgemv_serial_(NN, &m_slice, &jb, &neg_one,
                                      &A_(i_off, j), lda_,
                                      &x[j], &one_i, &one_v,
                                      &x[i_off], &one_i, 1);
                    }
                }
#ifdef _OPENMP
                if (use_omp) {
                    #pragma omp barrier
                }
#endif
                j -= nb;
            }
        } else if ((TR == 'T' || TR == 'C') && UPLO == 'L') {
            /* L,L,T/C: iterate diagonal from bottom up.
             *  x[0:j] -= op(A[j:j+jb, 0:j]) * x[j:j+jb].
             *  xgemv(op, M=jb, N=j) on submatrix &A_(j, 0).
             *  Parallel axis is the output (N=j); partition that. */
            int j = ((N - 1) / nb) * nb;
            while (j >= 0) {
                int jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    xtrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), lda_,
                                  &x[j], &one_i, uplo_len, trans_len, diag_len);
                }
#ifdef _OPENMP
                if (use_omp) {
                    #pragma omp barrier
                }
#endif
                if (j > 0) {
                    long long lo = (long long)j * tid / nt;
                    long long hi = (long long)j * (tid + 1) / nt;
                    int n_slice = (int)(hi - lo);
                    if (n_slice > 0) {
                        const int n_off = (int)lo;
                        xgemv_serial_(TT, &jb, &n_slice, &neg_one,
                                      &A_(j, n_off), lda_,
                                      &x[j], &one_i, &one_v,
                                      &x[n_off], &one_i, 1);
                    }
                }
#ifdef _OPENMP
                if (use_omp) {
                    #pragma omp barrier
                }
#endif
                j -= nb;
            }
        } else {
            /* L,U,T/C: iterate top-down. */
            for (int j = 0; j < N; j += nb) {
                int jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    xtrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), lda_,
                                  &x[j], &one_i, uplo_len, trans_len, diag_len);
                }
#ifdef _OPENMP
                if (use_omp) {
                    #pragma omp barrier
                }
#endif
                int mt = N - j - jb;
                if (mt > 0) {
                    int j2 = j + jb;
                    long long lo = (long long)mt * tid / nt;
                    long long hi = (long long)mt * (tid + 1) / nt;
                    int n_slice = (int)(hi - lo);
                    if (n_slice > 0) {
                        const int n_off = j2 + (int)lo;
                        xgemv_serial_(TT, &jb, &n_slice, &neg_one,
                                      &A_(j, n_off), lda_,
                                      &x[j], &one_i, &one_v,
                                      &x[n_off], &one_i, 1);
                    }
                }
#ifdef _OPENMP
                if (use_omp) {
                    #pragma omp barrier
                }
#endif
            }
        }
    }
}

#undef A_
