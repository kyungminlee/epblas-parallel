/*
 * xtrsv — kind16 complex (__complex128) triangular solve.
 *   A x = b           (TRANS='N')
 *   Aᵀ x = b          (TRANS='T')
 *   Aᴴ x = b          (TRANS='C', conjugated)
 * where A is N×N triangular (UPLO, DIAG). x overwrites b in-place.
 *
 * Public entry is the by-value `xtrsv_core` (drives xtrsv_ / xtrsv_64_ via
 * EPBLAS_FACADE_TRMV). It routes stride-1 calls above the 2·NB threshold into
 * xtrsv_blocked_ (its own parallel region), else the unblocked Netlib serial
 * body; the blocked dispatch is skipped inside an existing parallel region.
 *
 *   xtrsv_serial_  — pure serial unblocked Netlib body (no OpenMP). Used for
 *                    each diagonal sub-solve.
 *   xtrsv_blocked_ — LAPACK-blocked: one `#pragma omp parallel`; thread 0 does
 *                    each diagonal sub-solve, then all threads partition the
 *                    trailing xgemv (long axis) and call xgemv_core on their
 *                    slice (xgemv's own fork gated off by omp_in_parallel()).
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

typedef __complex128 T;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#define XTRSV_BLOCKED_NB_DEFAULT 64

static ptrdiff_t xtrsv_blocked_nb(void) {
    return XTRSV_BLOCKED_NB_DEFAULT;
}

void xtrsv_blocked_(
    const char *uplo, const char *trans, const char *diag,
    const ptrdiff_t *n_,
    const T *restrict a, const ptrdiff_t *lda_,
    T *restrict x, const ptrdiff_t *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len);

void xtrsv_serial_(
    const char *uplo, const char *trans, const char *diag,
    const ptrdiff_t *n_,
    const T *restrict a, const ptrdiff_t *lda_,
    T *restrict x, const ptrdiff_t *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len);

void xtrsv_core(
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
    if (incx == 1 && N >= 2 * xtrsv_blocked_nb() && !in_par
        && blas_omp_max_threads() > 1) {
        xtrsv_blocked_(&uplo_c, &trans_c, &diag_c, &N, a, &lda, x, &incx,
                       1, 1, 1);
        return;
    }

    xtrsv_serial_(&uplo_c, &trans_c, &diag_c, &N, a, &lda, x, &incx,
                  1, 1, 1);
}

/* Pure-serial unblocked Netlib body. No OpenMP. */
void xtrsv_serial_(
    const char *uplo, const char *trans, const char *diag,
    const ptrdiff_t *n_,
    const T *restrict a, const ptrdiff_t *lda_,
    T *restrict x, const ptrdiff_t *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const ptrdiff_t N = *n_;
    const ptrdiff_t lda = *lda_, incx = *incx_;
    const char UPLO = blas_up(*uplo);
    const char TR   = blas_up(*trans);
    const char DIAG = blas_up(*diag);
    const bool nounit = (DIAG != 'U');

    if (N == 0) return;

    const T zero = 0.0Q + 0.0Qi;

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
            const bool conj_a = (TR == 'C');
            if (UPLO == 'L') {
                for (ptrdiff_t i = N - 1; i >= 0; --i) {
                    T t = x[i];
                    const T *ai = &A_(0, i);
                    if (conj_a) {
                        for (ptrdiff_t k = i + 1; k < N; ++k) t -= conjq(ai[k]) * x[k];
                        if (nounit) t /= conjq(ai[i]);
                    } else {
                        for (ptrdiff_t k = i + 1; k < N; ++k) t -= ai[k] * x[k];
                        if (nounit) t /= ai[i];
                    }
                    x[i] = t;
                }
            } else {
                for (ptrdiff_t i = 0; i < N; ++i) {
                    T t = x[i];
                    const T *ai = &A_(0, i);
                    if (conj_a) {
                        for (ptrdiff_t k = 0; k < i; ++k) t -= conjq(ai[k]) * x[k];
                        if (nounit) t /= conjq(ai[i]);
                    } else {
                        for (ptrdiff_t k = 0; k < i; ++k) t -= ai[k] * x[k];
                        if (nounit) t /= ai[i];
                    }
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
            const bool conj_a = (TR == 'C');
            if (UPLO == 'L') {
                for (ptrdiff_t i = N - 1; i >= 0; --i) {
                    T t = x[kx + i * incx];
                    for (ptrdiff_t k = i + 1; k < N; ++k) {
                        const T aki = conj_a ? conjq(A_(k, i)) : A_(k, i);
                        t -= aki * x[kx + k * incx];
                    }
                    if (nounit) t /= (conj_a ? conjq(A_(i, i)) : A_(i, i));
                    x[kx + i * incx] = t;
                }
            } else {
                for (ptrdiff_t i = 0; i < N; ++i) {
                    T t = x[kx + i * incx];
                    for (ptrdiff_t k = 0; k < i; ++k) {
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
 * One `#pragma omp parallel` wraps the diagonal walk. Thread 0 calls
 * xtrsv_serial_ on each diagonal sub-block; all threads partition the trailing
 * xgemv across its long axis and call xgemv_core on their slice. Two barriers
 * per step. Inner GEMV routes through xgemv_core (its own fork gated off by
 * omp_in_parallel()) to avoid nested OMP. */

extern void xgemv_core(
    char trans,
    ptrdiff_t m, ptrdiff_t n,
    const T *alpha,
    const T *a, ptrdiff_t lda,
    const T *x, ptrdiff_t incx,
    const T *beta,
    T *y, ptrdiff_t incy);

void xtrsv_blocked_(
    const char *uplo, const char *trans, const char *diag,
    const ptrdiff_t *n_,
    const T *restrict a, const ptrdiff_t *lda_,
    T *restrict x, const ptrdiff_t *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    const ptrdiff_t N = *n_;
    const ptrdiff_t lda = *lda_, incx = *incx_;
    const ptrdiff_t nb = xtrsv_blocked_nb();
    const char UPLO = blas_up(*uplo);
    const char TR   = blas_up(*trans);

    if (N == 0) return;
    if (incx != 1 || N < 2 * nb) {
        const ptrdiff_t n_pt = *n_, lda_pt = *lda_, incx_pt = *incx_;
        xtrsv_serial_(uplo, trans, diag, &n_pt, a, &lda_pt, x, &incx_pt,
                      uplo_len, trans_len, diag_len);
        return;
    }

    const T neg_one = -1.0Q + 0.0Qi;
    const T one_v   =  1.0Q + 0.0Qi;
    const char NN[1] = {'N'};
    const char TT[1] = {(TR == 'C') ? 'C' : 'T'};
    const ptrdiff_t one_i = 1;

#ifdef _OPENMP
    const bool use_omp = (blas_omp_should_thread());
#else
    const bool use_omp = 0;
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
            /* Forward: solve A11 x1 = b1, then x2 -= A21 x1, repeat. */
            for (ptrdiff_t j = 0; j < N; j += nb) {
                ptrdiff_t jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    const ptrdiff_t lda_pt = *lda_;
                    xtrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), &lda_pt,
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
                    ptrdiff_t m_slice = hi - lo;
                    if (m_slice > 0) {
                        const ptrdiff_t i_off = j2 + lo;
                        xgemv_core(NN[0], m_slice, jb, &neg_one,
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
            /* Backward: solve A22 x2 = b2, then x1 -= A12 x2, repeat. */
            ptrdiff_t j = ((N - 1) / nb) * nb;
            while (j >= 0) {
                ptrdiff_t jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    const ptrdiff_t lda_pt = *lda_;
                    xtrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), &lda_pt,
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
                    ptrdiff_t m_slice = hi - lo;
                    if (m_slice > 0) {
                        const ptrdiff_t i_off = lo;
                        xgemv_core(NN[0], m_slice, jb, &neg_one,
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
        } else if ((TR == 'T' || TR == 'C') && UPLO == 'L') {
            /* L,L,T/C: iterate diagonal from bottom up.
             *  x[0:j] -= op(A[j:j+jb, 0:j]) * x[j:j+jb].
             *  xgemv(op, M=jb, N=j) on submatrix &A_(j, 0).
             *  Parallel axis is the output (N=j); partition that. */
            ptrdiff_t j = ((N - 1) / nb) * nb;
            while (j >= 0) {
                ptrdiff_t jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    const ptrdiff_t lda_pt = *lda_;
                    xtrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), &lda_pt,
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
                    ptrdiff_t n_slice = hi - lo;
                    if (n_slice > 0) {
                        const ptrdiff_t n_off = lo;
                        xgemv_core(TT[0], jb, n_slice, &neg_one,
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
            /* L,U,T/C: iterate top-down. */
            for (ptrdiff_t j = 0; j < N; j += nb) {
                ptrdiff_t jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    const ptrdiff_t lda_pt = *lda_;
                    xtrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), &lda_pt,
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
                    ptrdiff_t n_slice = hi - lo;
                    if (n_slice > 0) {
                        const ptrdiff_t n_off = j2 + lo;
                        xgemv_core(TT[0], jb, n_slice, &neg_one,
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

EPBLAS_FACADE_TRMV(xtrsv, T)

#undef A_
