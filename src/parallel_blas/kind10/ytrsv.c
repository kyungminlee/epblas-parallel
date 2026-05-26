/*
 * ytrsv — kind10 complex triangular solve.
 *   A x = b      (TRANS='N')
 *   Aᵀ x = b     (TRANS='T')
 *   Aᴴ x = b     (TRANS='C')
 *
 * Three public entries:
 *
 *   ytrsv_         — top-level dispatch. Routes stride-1 calls above
 *                    the 2·NB threshold into ytrsv_blocked_; otherwise
 *                    falls through to the unblocked serial body.
 *                    Skips the blocked-path dispatch when already
 *                    inside an OpenMP parallel region.
 *
 *   ytrsv_serial_  — pure serial unblocked Netlib body. Carries the
 *                    Addendum 18 (backward LT inner) + Addendum 19
 *                    (U-T K-unroll) tuning. No OpenMP. Safe to call
 *                    from inside a parallel region.
 *
 *   ytrsv_blocked_ — LAPACK-blocked algorithm wrapped in a SINGLE
 *                    `#pragma omp parallel` region. Threads cooperate
 *                    manually: thread 0 does each diagonal sub-solve
 *                    via ytrsv_serial_, then all threads partition
 *                    the trailing ygemv across the long axis and call
 *                    ygemv_ on their slice. ygemv's OMP fork is gated
 *                    off by omp_in_parallel().
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef _Complex long double T;
static const T ZERO = 0.0L + 0.0Li;
static inline T cconj(T z) { return ~z; }

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#define YTRSV_BLOCKED_NB_DEFAULT 64

static int ytrsv_blocked_nb(void) {
    static int cached = 0;
    if (cached == 0) {
        const char *s = getenv("YTRSV_NB");
        int v = (s && *s) ? atoi(s) : 0;
        cached = (v > 0) ? v : YTRSV_BLOCKED_NB_DEFAULT;
    }
    return cached;
}

void ytrsv_blocked_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len);

void ytrsv_serial_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len);

void ytrsv_(
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
    if (incx == 1 && N >= 2 * ytrsv_blocked_nb() && !in_par
        && blas_omp_max_threads() > 1) {
        ytrsv_blocked_(uplo, trans, diag, n_, a, lda_, x, incx_,
                       uplo_len, trans_len, diag_len);
        return;
    }

    ytrsv_serial_(uplo, trans, diag, n_, a, lda_, x, incx_,
                  uplo_len, trans_len, diag_len);
}

void ytrsv_serial_(
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

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'L') {
                /* Forward subst with J-unroll-by-2: process columns i and
                 * i+1 jointly so the trailing-x AXPY loop loads/stores each
                 * x[k] once for BOTH columns' contributions. Halves x
                 * memory traffic on the complex AXPY inner. (Same trick as
                 * etrsv LN; doubles the saving here since complex x is
                 * 20 bytes vs 10 for real.) */
                int i = 0;
                for (; i + 1 < N; i += 2) {
                    if (nounit) x[i] /= A_(i, i);
                    const T xi = x[i];
                    x[i + 1] -= xi * A_(i + 1, i);
                    if (nounit) x[i + 1] /= A_(i + 1, i + 1);
                    const T xi1 = x[i + 1];
                    const T *a0 = &A_(0, i);
                    const T *a1 = &A_(0, i + 1);
                    for (int k = i + 2; k < N; ++k) {
                        x[k] = (x[k] - xi * a0[k]) - xi1 * a1[k];
                    }
                }
                if (i < N) {
                    if (nounit) x[i] /= A_(i, i);
                    const T xi = x[i];
                    const T *ai = &A_(0, i);
                    for (int k = i + 1; k < N; ++k) x[k] -= xi * ai[k];
                }
            } else {
                /* UPLO='U': back-subst with J-unroll-by-2 pair (i, i-1). */
                int i = N - 1;
                for (; i - 1 >= 0; i -= 2) {
                    if (nounit) x[i] /= A_(i, i);
                    const T xi = x[i];
                    x[i - 1] -= xi * A_(i - 1, i);
                    if (nounit) x[i - 1] /= A_(i - 1, i - 1);
                    const T xi1 = x[i - 1];
                    const T *a0 = &A_(0, i);
                    const T *a1 = &A_(0, i - 1);
                    for (int k = 0; k < i - 1; ++k) {
                        x[k] = (x[k] - xi * a0[k]) - xi1 * a1[k];
                    }
                }
                if (i >= 0) {
                    if (nounit) x[i] /= A_(i, i);
                    const T xi = x[i];
                    const T *ai = &A_(0, i);
                    for (int k = 0; k < i; ++k) x[k] -= xi * ai[k];
                }
            }
        } else {
            const int conj_a = (TR == 'C');
            if (UPLO == 'L') {
                /* Inner walk backward to match the outer's descent — under
                 * memory pressure the forward variant collapses to ~0.3×
                 * because x falls out of L1 between outer iters. See
                 * etrsv LTN / Addendum 18. */
                for (int i = N - 1; i >= 0; --i) {
                    T t = x[i];
                    const T *ai = &A_(0, i);
                    if (conj_a) {
                        for (int k = N - 1; k > i; --k) t -= cconj(ai[k]) * x[k];
                        if (nounit) t /= cconj(ai[i]);
                    } else {
                        for (int k = N - 1; k > i; --k) t -= ai[k] * x[k];
                        if (nounit) t /= ai[i];
                    }
                    x[i] = t;
                }
            } else {
                /* U-T/U-C: outer forward, inner forward — direction matches
                 * the Fortran reference. The non-conj path (U-T) carried a
                 * single-accumulator x87 fmul dep chain that landed at
                 * 0.82–0.90× of migrated at N=256/512; two-way K-unroll
                 * splits it into two parallel chains (t0,t1) and recovers
                 * to ~0.93×.
                 *
                 * The conj path (U-C) does NOT benefit from the same
                 * unroll — the extra fchs from `cconj()` evidently
                 * disrupts gcc's scheduling and U-C regresses from ~1.00×
                 * to ~0.91× when unrolled. Keep it single-accumulator. */
                if (conj_a) {
                    for (int i = 0; i < N; ++i) {
                        T t = x[i];
                        const T *ai = &A_(0, i);
                        for (int k = 0; k < i; ++k) t -= cconj(ai[k]) * x[k];
                        if (nounit) t /= cconj(ai[i]);
                        x[i] = t;
                    }
                } else {
                    for (int i = 0; i < N; ++i) {
                        T t0 = x[i], t1 = ZERO;
                        const T *ai = &A_(0, i);
                        int k = 0;
                        for (; k + 1 < i; k += 2) {
                            t0 -= ai[k]     * x[k];
                            t1 -= ai[k + 1] * x[k + 1];
                        }
                        if (k < i) t0 -= ai[k] * x[k];
                        T t = t0 + t1;
                        if (nounit) t /= ai[i];
                        x[i] = t;
                    }
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (int i = 0; i < N; ++i) {
                    const int ix = kx + i * incx;
                    if (x[ix] != ZERO) {
                        if (nounit) x[ix] /= A_(i, i);
                        const T xi = x[ix];
                        for (int k = i + 1; k < N; ++k) x[kx + k * incx] -= xi * A_(k, i);
                    }
                }
            } else {
                for (int i = N - 1; i >= 0; --i) {
                    const int ix = kx + i * incx;
                    if (x[ix] != ZERO) {
                        if (nounit) x[ix] /= A_(i, i);
                        const T xi = x[ix];
                        for (int k = 0; k < i; ++k) x[kx + k * incx] -= xi * A_(k, i);
                    }
                }
            }
        } else {
            const int conj_a = (TR == 'C');
            if (UPLO == 'L') {
                /* Inner walks backward to match Fortran reference; same
                 * cache-direction reasoning as the incx=1 LT/LC path
                 * (Addendum 18 / Rule 21). */
                for (int i = N - 1; i >= 0; --i) {
                    T t = x[kx + i * incx];
                    for (int k = N - 1; k > i; --k) {
                        const T aki = conj_a ? cconj(A_(k, i)) : A_(k, i);
                        t -= aki * x[kx + k * incx];
                    }
                    if (nounit) t /= (conj_a ? cconj(A_(i, i)) : A_(i, i));
                    x[kx + i * incx] = t;
                }
            } else {
                for (int i = 0; i < N; ++i) {
                    T t = x[kx + i * incx];
                    for (int k = 0; k < i; ++k) {
                        const T aki = conj_a ? cconj(A_(k, i)) : A_(k, i);
                        t -= aki * x[kx + k * incx];
                    }
                    if (nounit) t /= (conj_a ? cconj(A_(i, i)) : A_(i, i));
                    x[kx + i * incx] = t;
                }
            }
        }
    }
}

/* ── Block-parallel variant: single parallel region ─────────────────
 *
 * Same shape as etrsv_blocked_ / qtrsv_blocked_ (Addendum 29).
 * For TR='C', the trailing update passes "C" to ygemv (which does
 * sum_i conj(A(i,j)) * x(i)). One `#pragma omp parallel` wraps the
 * entire diagonal walk; two `#pragma omp barrier`s per step.
 */

extern void ygemv_(
    const char *trans,
    const int *m, const int *n,
    const T *alpha,
    const T *a, const int *lda,
    const T *x, const int *incx,
    const T *beta,
    T *y, const int *incy,
    size_t trans_len);

void ytrsv_blocked_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    const int N = *n_;
    const int lda = *lda_, incx = *incx_;
    const int nb = ytrsv_blocked_nb();
    const char UPLO = up(uplo);
    const char TR = up(trans);

    if (N == 0) return;
    if (incx != 1 || N < 2 * nb) {
        ytrsv_serial_(uplo, trans, diag, n_, a, lda_, x, incx_,
                      uplo_len, trans_len, diag_len);
        return;
    }

    const T neg_one = -1.0L + 0.0Li;
    const T one_v   =  1.0L + 0.0Li;
    const char NN[1] = {'N'};
    const char TT[1] = {'T'};
    const char CC[1] = {'C'};
    const char *gemv_tr = (TR == 'C') ? CC : TT;
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
                    ytrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), lda_,
                                  &x[j], &one_i, uplo_len, trans_len, diag_len);
                }
#ifdef _OPENMP
                if (use_omp) { _Pragma("omp barrier"); }
#endif
                int mt = N - j - jb;
                if (mt > 0) {
                    int j2 = j + jb;
                    long long lo = (long long)mt * tid / nt;
                    long long hi = (long long)mt * (tid + 1) / nt;
                    int m_slice = (int)(hi - lo);
                    if (m_slice > 0) {
                        const int i_off = j2 + (int)lo;
                        ygemv_(NN, &m_slice, &jb, &neg_one,
                               &A_(i_off, j), lda_,
                               &x[j], &one_i, &one_v,
                               &x[i_off], &one_i, 1);
                    }
                }
#ifdef _OPENMP
                if (use_omp) { _Pragma("omp barrier"); }
#endif
            }
        } else if (TR == 'N' && UPLO == 'U') {
            int j = ((N - 1) / nb) * nb;
            while (j >= 0) {
                int jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    ytrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), lda_,
                                  &x[j], &one_i, uplo_len, trans_len, diag_len);
                }
#ifdef _OPENMP
                if (use_omp) { _Pragma("omp barrier"); }
#endif
                if (j > 0) {
                    long long lo = (long long)j * tid / nt;
                    long long hi = (long long)j * (tid + 1) / nt;
                    int m_slice = (int)(hi - lo);
                    if (m_slice > 0) {
                        const int i_off = (int)lo;
                        ygemv_(NN, &m_slice, &jb, &neg_one,
                               &A_(i_off, j), lda_,
                               &x[j], &one_i, &one_v,
                               &x[i_off], &one_i, 1);
                    }
                }
#ifdef _OPENMP
                if (use_omp) { _Pragma("omp barrier"); }
#endif
                j -= nb;
            }
        } else if ((TR == 'T' || TR == 'C') && UPLO == 'L') {
            int j = ((N - 1) / nb) * nb;
            while (j >= 0) {
                int jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    ytrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), lda_,
                                  &x[j], &one_i, uplo_len, trans_len, diag_len);
                }
#ifdef _OPENMP
                if (use_omp) { _Pragma("omp barrier"); }
#endif
                if (j > 0) {
                    long long lo = (long long)j * tid / nt;
                    long long hi = (long long)j * (tid + 1) / nt;
                    int n_slice = (int)(hi - lo);
                    if (n_slice > 0) {
                        const int n_off = (int)lo;
                        ygemv_(gemv_tr, &jb, &n_slice, &neg_one,
                               &A_(j, n_off), lda_,
                               &x[j], &one_i, &one_v,
                               &x[n_off], &one_i, 1);
                    }
                }
#ifdef _OPENMP
                if (use_omp) { _Pragma("omp barrier"); }
#endif
                j -= nb;
            }
        } else {
            /* (TR == 'T' || TR == 'C') && UPLO == 'U' */
            for (int j = 0; j < N; j += nb) {
                int jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    ytrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), lda_,
                                  &x[j], &one_i, uplo_len, trans_len, diag_len);
                }
#ifdef _OPENMP
                if (use_omp) { _Pragma("omp barrier"); }
#endif
                int mt = N - j - jb;
                if (mt > 0) {
                    int j2 = j + jb;
                    long long lo = (long long)mt * tid / nt;
                    long long hi = (long long)mt * (tid + 1) / nt;
                    int n_slice = (int)(hi - lo);
                    if (n_slice > 0) {
                        const int n_off = j2 + (int)lo;
                        ygemv_(gemv_tr, &jb, &n_slice, &neg_one,
                               &A_(j, n_off), lda_,
                               &x[j], &one_i, &one_v,
                               &x[n_off], &one_i, 1);
                    }
                }
#ifdef _OPENMP
                if (use_omp) { _Pragma("omp barrier"); }
#endif
            }
        }
    }
}

#undef A_
