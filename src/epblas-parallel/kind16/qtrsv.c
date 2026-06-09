/*
 * qtrsv — kind16 (__float128) triangular solve.
 *
 * Three public entries:
 *
 *   qtrsv_         — top-level dispatch. Routes stride-1 calls above
 *                    the 2·NB threshold into qtrsv_blocked_ (which
 *                    opens its own parallel region), otherwise falls
 *                    through to the unblocked Netlib serial body.
 *                    Skips the blocked-path dispatch when already
 *                    inside an OpenMP parallel region.
 *
 *   qtrsv_serial_  — pure serial unblocked Netlib body. No OpenMP
 *                    pragma anywhere on this call path. Safe to call
 *                    from inside another function's parallel region.
 *
 *   qtrsv_blocked_ — LAPACK-blocked algorithm wrapped in a SINGLE
 *                    `#pragma omp parallel` region. Threads cooperate
 *                    manually: thread 0 does each diagonal sub-solve
 *                    via qtrsv_serial_, then all threads partition
 *                    the trailing qgemv across the long axis (M for
 *                    TR='N', N for TR='T') and call qgemv_serial_ on
 *                    their slice. Two barriers per diagonal step.
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef __float128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#define QTRSV_BLOCKED_NB_DEFAULT 64

static int qtrsv_blocked_nb(void) {
    return QTRSV_BLOCKED_NB_DEFAULT;
}

void qtrsv_blocked_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len);

void qtrsv_serial_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len);

void qtrsv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    const int N = *n_;
    const int incx = *incx_;

    if (N == 0) return;

#ifdef _OPENMP
    const int in_par = omp_in_parallel();
#else
    const int in_par = 0;
#endif
    if (incx == 1 && N >= 2 * qtrsv_blocked_nb() && !in_par) {
        qtrsv_blocked_(uplo, trans, diag, n_, a, lda_, x, incx_,
                       uplo_len, trans_len, diag_len);
        return;
    }

    qtrsv_serial_(uplo, trans, diag, n_, a, lda_, x, incx_,
                  uplo_len, trans_len, diag_len);
}

/* Pure-serial unblocked Netlib body. No OpenMP. */
void qtrsv_serial_(
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
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
    const char DIAG = up(diag);
    const int nounit = (DIAG != 'U');

    if (N == 0) return;
    const T zero = 0.0Q;

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
            if (UPLO == 'L') {
                for (int i = N - 1; i >= 0; --i) {
                    T t = x[i];
                    const T *ai = &A_(0, i);
                    for (int k = i + 1; k < N; ++k) t -= ai[k] * x[k];
                    if (nounit) t /= ai[i];
                    x[i] = t;
                }
            } else {
                for (int i = 0; i < N; ++i) {
                    T t = x[i];
                    const T *ai = &A_(0, i);
                    for (int k = 0; k < i; ++k) t -= ai[k] * x[k];
                    if (nounit) t /= ai[i];
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
            if (UPLO == 'L') {
                for (int i = N - 1; i >= 0; --i) {
                    T t = x[kx + i * incx];
                    for (int k = i + 1; k < N; ++k) t -= A_(k, i) * x[kx + k * incx];
                    if (nounit) t /= A_(i, i);
                    x[kx + i * incx] = t;
                }
            } else {
                for (int i = 0; i < N; ++i) {
                    T t = x[kx + i * incx];
                    for (int k = 0; k < i; ++k) t -= A_(k, i) * x[kx + k * incx];
                    if (nounit) t /= A_(i, i);
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
 *   - Thread 0 calls qtrsv_serial_ on each diagonal sub-block.
 *   - All threads partition the trailing qgemv across its long axis
 *     and call qgemv_serial_ on their slice.
 *   - Two `#pragma omp barrier`s per step.
 *
 * Inner calls route through *_serial_ entries to avoid nested OMP.
 */

extern void qgemv_serial_(
    const char *trans,
    const int *m, const int *n,
    const T *alpha,
    const T *a, const int *lda,
    const T *x, const int *incx,
    const T *beta,
    T *y, const int *incy,
    size_t trans_len);

void qtrsv_blocked_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    const int N = *n_;
    const int lda = *lda_, incx = *incx_;
    const int nb = qtrsv_blocked_nb();
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';

    if (N == 0) return;
    if (incx != 1 || N < 2 * nb) {
        qtrsv_serial_(uplo, trans, diag, n_, a, lda_, x, incx_,
                      uplo_len, trans_len, diag_len);
        return;
    }

    const T neg_one = -1.0Q;
    const T one_v   =  1.0Q;
    const char NN[1] = {'N'};
    const char TT[1] = {'T'};
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
            for (int j = 0; j < N; j += nb) {
                int jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    qtrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), lda_,
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
                        qgemv_serial_(NN, &m_slice, &jb, &neg_one,
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
            int j = ((N - 1) / nb) * nb;
            while (j >= 0) {
                int jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    qtrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), lda_,
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
                        qgemv_serial_(NN, &m_slice, &jb, &neg_one,
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
        } else if (TR == 'T' && UPLO == 'L') {
            int j = ((N - 1) / nb) * nb;
            while (j >= 0) {
                int jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    qtrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), lda_,
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
                        qgemv_serial_(TT, &jb, &n_slice, &neg_one,
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
            /* TR == 'T' && UPLO == 'U' */
            for (int j = 0; j < N; j += nb) {
                int jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    qtrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), lda_,
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
                        qgemv_serial_(TT, &jb, &n_slice, &neg_one,
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
