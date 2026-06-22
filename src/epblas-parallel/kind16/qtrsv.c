/*
 * qtrsv — kind16 (__float128) triangular solve.
 *   A x = b           (TRANS='N')
 *   Aᵀ x = b          (TRANS='T'/'C')
 * where A is N×N triangular (UPLO, DIAG). x overwrites b in-place.
 *
 * Public entry is the by-value `qtrsv_core` (drives qtrsv_ / qtrsv_64_ via
 * EPBLAS_FACADE_TRMV). It routes stride-1 calls above the 2·NB threshold into
 * qtrsv_blocked_ (its own parallel region), else the unblocked Netlib serial
 * body; the blocked dispatch is skipped inside an existing parallel region.
 *
 *   qtrsv_serial_  — pure serial unblocked Netlib body (no OpenMP). Used for
 *                    each diagonal sub-solve.
 *   qtrsv_blocked_ — LAPACK-blocked: one `#pragma omp parallel`; thread 0 does
 *                    each diagonal sub-solve, then all threads partition the
 *                    trailing qgemv (long axis) and call qgemv_core on their
 *                    slice (qgemv's own fork gated off by omp_in_parallel()).
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

typedef __float128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#define QTRSV_BLOCKED_NB_DEFAULT 64

static ptrdiff_t qtrsv_blocked_nb(void) {
    return QTRSV_BLOCKED_NB_DEFAULT;
}

void qtrsv_blocked_(
    const char *uplo, const char *trans, const char *diag,
    const ptrdiff_t *n_,
    const T *restrict a, const ptrdiff_t *lda_,
    T *restrict x, const ptrdiff_t *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len);

void qtrsv_serial_(
    const char *uplo, const char *trans, const char *diag,
    const ptrdiff_t *n_,
    const T *restrict a, const ptrdiff_t *lda_,
    T *restrict x, const ptrdiff_t *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len);

void qtrsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t N,
    const T *restrict a, ptrdiff_t lda,
    T *restrict x, ptrdiff_t incx)
{
    if (N == 0) return;

#ifdef _OPENMP
    const ptrdiff_t in_par = omp_in_parallel();
#else
    const ptrdiff_t in_par = 0;
#endif
    const char uplo_c = uplo, trans_c = trans, diag_c = diag;
    if (incx == 1 && N >= 2 * qtrsv_blocked_nb() && !in_par
        && blas_omp_max_threads() > 1) {
        qtrsv_blocked_(&uplo_c, &trans_c, &diag_c, &N, a, &lda, x, &incx,
                       1, 1, 1);
        return;
    }

    qtrsv_serial_(&uplo_c, &trans_c, &diag_c, &N, a, &lda, x, &incx,
                  1, 1, 1);
}

/* Pure-serial unblocked Netlib body. No OpenMP. */
void qtrsv_serial_(
    const char *uplo, const char *trans, const char *diag,
    const ptrdiff_t *n_,
    const T *restrict a, const ptrdiff_t *lda_,
    T *restrict x, const ptrdiff_t *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const ptrdiff_t N = *n_;
    const ptrdiff_t lda = *lda_, incx = *incx_;
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
    const char DIAG = up(diag);
    const ptrdiff_t nounit = (DIAG != 'U');

    if (N == 0) return;
    const T zero = 0.0Q;

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (ptrdiff_t i = 0; i < N; ++i) {
                    if (x[i] != zero) {
                        if (nounit) x[i] /= A_(i, i);
                        const T xi = x[i];
                        const T *ai = &A_(0, i);
                        for (ptrdiff_t k = i + 1; k < N; ++k) x[k] -= xi * ai[k];
                    }
                }
            } else {
                for (ptrdiff_t i = N - 1; i >= 0; --i) {
                    if (x[i] != zero) {
                        if (nounit) x[i] /= A_(i, i);
                        const T xi = x[i];
                        const T *ai = &A_(0, i);
                        for (ptrdiff_t k = 0; k < i; ++k) x[k] -= xi * ai[k];
                    }
                }
            }
        } else {
            if (UPLO == 'L') {
                for (ptrdiff_t i = N - 1; i >= 0; --i) {
                    T t = x[i];
                    const T *ai = &A_(0, i);
                    for (ptrdiff_t k = i + 1; k < N; ++k) t -= ai[k] * x[k];
                    if (nounit) t /= ai[i];
                    x[i] = t;
                }
            } else {
                for (ptrdiff_t i = 0; i < N; ++i) {
                    T t = x[i];
                    const T *ai = &A_(0, i);
                    for (ptrdiff_t k = 0; k < i; ++k) t -= ai[k] * x[k];
                    if (nounit) t /= ai[i];
                    x[i] = t;
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (ptrdiff_t i = 0; i < N; ++i) {
                    const ptrdiff_t ix = kx + i * incx;
                    if (x[ix] != zero) {
                        if (nounit) x[ix] /= A_(i, i);
                        const T xi = x[ix];
                        for (ptrdiff_t k = i + 1; k < N; ++k) x[kx + k * incx] -= xi * A_(k, i);
                    }
                }
            } else {
                for (ptrdiff_t i = N - 1; i >= 0; --i) {
                    const ptrdiff_t ix = kx + i * incx;
                    if (x[ix] != zero) {
                        if (nounit) x[ix] /= A_(i, i);
                        const T xi = x[ix];
                        for (ptrdiff_t k = 0; k < i; ++k) x[kx + k * incx] -= xi * A_(k, i);
                    }
                }
            }
        } else {
            if (UPLO == 'L') {
                for (ptrdiff_t i = N - 1; i >= 0; --i) {
                    T t = x[kx + i * incx];
                    for (ptrdiff_t k = i + 1; k < N; ++k) t -= A_(k, i) * x[kx + k * incx];
                    if (nounit) t /= A_(i, i);
                    x[kx + i * incx] = t;
                }
            } else {
                for (ptrdiff_t i = 0; i < N; ++i) {
                    T t = x[kx + i * incx];
                    for (ptrdiff_t k = 0; k < i; ++k) t -= A_(k, i) * x[kx + k * incx];
                    if (nounit) t /= A_(i, i);
                    x[kx + i * incx] = t;
                }
            }
        }
    }
}

/* ── Block-parallel variant: single parallel region ─────────────────
 * One `#pragma omp parallel` wraps the diagonal walk. Thread 0 calls
 * qtrsv_serial_ on each diagonal sub-block; all threads partition the trailing
 * qgemv across its long axis and call qgemv_core on their slice. Two barriers
 * per step. Inner GEMV routes through qgemv_core (its own fork gated off by
 * omp_in_parallel()) to avoid nested OMP. */

extern void qgemv_core(
    char trans,
    ptrdiff_t m, ptrdiff_t n,
    const T *alpha,
    const T *a, ptrdiff_t lda,
    const T *x, ptrdiff_t incx,
    const T *beta,
    T *y, ptrdiff_t incy);

void qtrsv_blocked_(
    const char *uplo, const char *trans, const char *diag,
    const ptrdiff_t *n_,
    const T *restrict a, const ptrdiff_t *lda_,
    T *restrict x, const ptrdiff_t *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    const ptrdiff_t N = *n_;
    const ptrdiff_t lda = *lda_, incx = *incx_;
    const ptrdiff_t nb = qtrsv_blocked_nb();
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';

    if (N == 0) return;
    if (incx != 1 || N < 2 * nb) {
        const ptrdiff_t n_pt = *n_, lda_pt = *lda_, incx_pt = *incx_;
        qtrsv_serial_(uplo, trans, diag, &n_pt, a, &lda_pt, x, &incx_pt,
                      uplo_len, trans_len, diag_len);
        return;
    }

    const T neg_one = -1.0Q;
    const T one_v   =  1.0Q;
    const char NN[1] = {'N'};
    const char TT[1] = {'T'};
    const ptrdiff_t one_i = 1;

#ifdef _OPENMP
    const ptrdiff_t use_omp = (blas_omp_max_threads() > 1 && !omp_in_parallel());
#else
    const ptrdiff_t use_omp = 0;
#endif

#ifdef _OPENMP
    #pragma omp parallel if(use_omp)
#endif
    {
        ptrdiff_t tid = 0, nt = 1;
#ifdef _OPENMP
        if (use_omp) { tid = omp_get_thread_num(); nt = omp_get_num_threads(); }
#endif

        if (TR == 'N' && UPLO == 'L') {
            for (ptrdiff_t j = 0; j < N; j += nb) {
                ptrdiff_t jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    const ptrdiff_t lda_pt = *lda_;
                    qtrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), &lda_pt,
                                  &x[j], &one_i, uplo_len, trans_len, diag_len);
                }
#ifdef _OPENMP
                if (use_omp) {
                    #pragma omp barrier
                }
#endif
                ptrdiff_t mt = N - j - jb;
                if (mt > 0) {
                    ptrdiff_t j2 = j + jb;
                    ptrdiff_t lo = blas_part_bound(mt, tid, nt);
                    ptrdiff_t hi = blas_part_bound(mt, tid + 1, nt);
                    ptrdiff_t m_slice = (ptrdiff_t)(hi - lo);
                    if (m_slice > 0) {
                        const ptrdiff_t i_off = j2 + (ptrdiff_t)lo;
                        qgemv_core(NN[0], m_slice, jb, &neg_one,
                                   &A_(i_off, j), *lda_,
                                   &x[j], one_i, &one_v,
                                   &x[i_off], one_i);
                    }
                }
#ifdef _OPENMP
                if (use_omp) {
                    #pragma omp barrier
                }
#endif
            }
        } else if (TR == 'N' && UPLO == 'U') {
            ptrdiff_t j = ((N - 1) / nb) * nb;
            while (j >= 0) {
                ptrdiff_t jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    const ptrdiff_t lda_pt = *lda_;
                    qtrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), &lda_pt,
                                  &x[j], &one_i, uplo_len, trans_len, diag_len);
                }
#ifdef _OPENMP
                if (use_omp) {
                    #pragma omp barrier
                }
#endif
                if (j > 0) {
                    ptrdiff_t lo = blas_part_bound(j, tid, nt);
                    ptrdiff_t hi = blas_part_bound(j, tid + 1, nt);
                    ptrdiff_t m_slice = (ptrdiff_t)(hi - lo);
                    if (m_slice > 0) {
                        const ptrdiff_t i_off = (ptrdiff_t)lo;
                        qgemv_core(NN[0], m_slice, jb, &neg_one,
                                   &A_(i_off, j), *lda_,
                                   &x[j], one_i, &one_v,
                                   &x[i_off], one_i);
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
            ptrdiff_t j = ((N - 1) / nb) * nb;
            while (j >= 0) {
                ptrdiff_t jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    const ptrdiff_t lda_pt = *lda_;
                    qtrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), &lda_pt,
                                  &x[j], &one_i, uplo_len, trans_len, diag_len);
                }
#ifdef _OPENMP
                if (use_omp) {
                    #pragma omp barrier
                }
#endif
                if (j > 0) {
                    ptrdiff_t lo = blas_part_bound(j, tid, nt);
                    ptrdiff_t hi = blas_part_bound(j, tid + 1, nt);
                    ptrdiff_t n_slice = (ptrdiff_t)(hi - lo);
                    if (n_slice > 0) {
                        const ptrdiff_t n_off = (ptrdiff_t)lo;
                        qgemv_core(TT[0], jb, n_slice, &neg_one,
                                   &A_(j, n_off), *lda_,
                                   &x[j], one_i, &one_v,
                                   &x[n_off], one_i);
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
            for (ptrdiff_t j = 0; j < N; j += nb) {
                ptrdiff_t jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    const ptrdiff_t lda_pt = *lda_;
                    qtrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), &lda_pt,
                                  &x[j], &one_i, uplo_len, trans_len, diag_len);
                }
#ifdef _OPENMP
                if (use_omp) {
                    #pragma omp barrier
                }
#endif
                ptrdiff_t mt = N - j - jb;
                if (mt > 0) {
                    ptrdiff_t j2 = j + jb;
                    ptrdiff_t lo = blas_part_bound(mt, tid, nt);
                    ptrdiff_t hi = blas_part_bound(mt, tid + 1, nt);
                    ptrdiff_t n_slice = (ptrdiff_t)(hi - lo);
                    if (n_slice > 0) {
                        const ptrdiff_t n_off = j2 + (ptrdiff_t)lo;
                        qgemv_core(TT[0], jb, n_slice, &neg_one,
                                   &A_(j, n_off), *lda_,
                                   &x[j], one_i, &one_v,
                                   &x[n_off], one_i);
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

EPBLAS_FACADE_TRMV(qtrsv, T)

#undef A_
