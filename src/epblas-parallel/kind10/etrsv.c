/*
 * etrsv — kind10 (REAL(KIND=10)) triangular solve.
 *   A x = b           (TRANS='N')
 *   Aᵀ x = b          (TRANS='T'/'C')
 * where A is N×N triangular (UPLO, DIAG). x overwrites b in-place.
 *
 * Three public entries:
 *
 *   etrsv_         — top-level dispatch. Routes stride-1 calls above
 *                    the 2·NB threshold into etrsv_blocked_; otherwise
 *                    falls through to the unblocked Netlib serial body.
 *                    Skips the blocked-path dispatch when already
 *                    inside an OpenMP parallel region.
 *
 *   etrsv_serial_  — pure serial unblocked Netlib body. K-unroll-by-2
 *                    + backward inner walks per Addenda 18/19. No
 *                    OpenMP. Safe to call from inside a parallel
 *                    region.
 *
 *   etrsv_blocked_ — LAPACK-blocked algorithm wrapped in a SINGLE
 *                    `#pragma omp parallel` region. Threads cooperate
 *                    manually: thread 0 does each diagonal sub-solve
 *                    via etrsv_serial_, then all threads partition
 *                    the trailing egemv across the long axis and call
 *                    egemv_ on their slice (egemv's own OMP fork is
 *                    gated off by omp_in_parallel()).
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

typedef long double T;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#define ETRSV_BLOCKED_NB_DEFAULT 64

static ptrdiff_t etrsv_blocked_nb(void) {
    return ETRSV_BLOCKED_NB_DEFAULT;
}

void etrsv_blocked_(
    const char *uplo, const char *trans, const char *diag,
    const ptrdiff_t *n_,
    const T *restrict a, const ptrdiff_t *lda_,
    T *restrict x, const ptrdiff_t *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len);

void etrsv_serial_(
    const char *uplo, const char *trans, const char *diag,
    const ptrdiff_t *n_,
    const T *restrict a, const ptrdiff_t *lda_,
    T *restrict x, const ptrdiff_t *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len);

void etrsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t N,
    const T *restrict a, ptrdiff_t lda,
    T *restrict x, ptrdiff_t incx)
{
    if (N == 0) return;

#ifdef _OPENMP
    const bool in_par = omp_in_parallel();
#else
    const bool in_par = 0;
#endif
    const char uplo_c = uplo, trans_c = trans, diag_c = diag;
    /* Threshold `N >= 3*NB` (not the usual 2*NB) — etrsv's per-op cost
     * is so low that the OMP fork-join + per-step barriers cost more
     * than the parallel work at N == 2*NB. At N=128 (=2*NB with NB=64)
     * the T-branch's K-unroll serial path was ~3 µs per call; OMP=4
     * dispatch regressed it to 0.80x. Bumping to 3*NB (192) keeps
     * N=128 on the (faster) serial path while N=256+ still goes
     * blocked-parallel and wins. */
    if (incx == 1 && N >= 3 * etrsv_blocked_nb() && !in_par
        && blas_omp_max_threads() > 1) {
        etrsv_blocked_(&uplo_c, &trans_c, &diag_c, &N, a, &lda, x, &incx,
                       1, 1, 1);
        return;
    }

    etrsv_serial_(&uplo_c, &trans_c, &diag_c, &N, a, &lda, x, &incx,
                  1, 1, 1);
}

/* Pure-serial unblocked Netlib body. No OpenMP. Inherits the
 * Addendum 18 (backward inner) + Addendum 19 (K-unroll-by-2 split
 * accumulators) tuning of the previous etrsv_. */
void etrsv_serial_(
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
    char TR = blas_up(*trans);
    if (TR == 'C') TR = 'T';
    const char DIAG = blas_up(*diag);
    const bool nounit = (DIAG != 'U');

    if (N == 0) return;

    const T zero = 0.0L;

    (void)zero;
    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'L') {
                /* Forward substitution: x[i] = (b[i] - sum_{k<i} A(i,k) x[k]) / A(i,i).
                 *
                 * J-unroll-by-2: process columns i and i+1 jointly so the
                 * trailing-x update loop loads/stores each x[k] once for
                 * BOTH columns' contributions. Halves x memory traffic on
                 * the AXPY-style inner — same trick as egemv N-branch.
                 * Inner becomes `x[k] = (x[k] - xi*a0[k]) - xi1*a1[k]`. */
                ptrdiff_t i = 0;
                for (; i + 1 < N; i += 2) {
                    if (nounit) x[i] /= A_(i, i);
                    const T xi = x[i];
                    /* Apply column i's contribution to x[i+1] before solving it. */
                    x[i + 1] -= xi * A_(i + 1, i);
                    if (nounit) x[i + 1] /= A_(i + 1, i + 1);
                    const T xi1 = x[i + 1];
                    const T *a0 = &A_(0, i);
                    const T *a1 = &A_(0, i + 1);
                    for (ptrdiff_t k = i + 2; k < N; ++k) {
                        x[k] = (x[k] - xi * a0[k]) - xi1 * a1[k];
                    }
                }
                if (i < N) {
                    if (nounit) x[i] /= A_(i, i);
                    const T xi = x[i];
                    const T *ai = &A_(0, i);
                    for (ptrdiff_t k = i + 1; k < N; ++k) x[k] -= xi * ai[k];
                }
            } else {
                /* UPLO='U': back substitution iterates i backward.
                 * x[i] = (b[i] - sum_{k>i} A(i,k) x[k]) / A(i,i).
                 *
                 * J-unroll-by-2 (same trick as LN branch, descending): pair
                 * (i, i-1) so the inner k = 0..i-2 walk loads/stores each
                 * x[k] once for both columns' contributions. */
                ptrdiff_t i = N - 1;
                for (; i - 1 >= 0; i -= 2) {
                    if (nounit) x[i] /= A_(i, i);
                    const T xi = x[i];
                    /* Apply column i's contribution to x[i-1] before solving it. */
                    x[i - 1] -= xi * A_(i - 1, i);
                    if (nounit) x[i - 1] /= A_(i - 1, i - 1);
                    const T xi1 = x[i - 1];
                    const T *a0 = &A_(0, i);
                    const T *a1 = &A_(0, i - 1);
                    for (ptrdiff_t k = 0; k < i - 1; ++k) {
                        x[k] = (x[k] - xi * a0[k]) - xi1 * a1[k];
                    }
                }
                if (i >= 0) {
                    if (nounit) x[i] /= A_(i, i);
                    const T xi = x[i];
                    const T *ai = &A_(0, i);
                    for (ptrdiff_t k = 0; k < i; ++k) x[k] -= xi * ai[k];
                }
            }
        } else {  /* TRANS = 'T': solve Aᵀ x = b. */
            if (UPLO == 'L') {
                /* Lower-stored A; Aᵀ is upper. Iterate i backward.
                 * x[i] = (b[i] - sum_{k>i} A(k,i) x[k]) / A(i,i).
                 *
                 * Inner walk is *backward* (k = N-1 .. i+1) to mirror the
                 * Fortran reference. With the outer loop also descending,
                 * x[i+1..N-1] is read in the same direction as the previous
                 * outer iter wrote x[i+1], so the bottom of x stays hot in
                 * L1. Forward inner under descending outer ends each iter
                 * at x[N-1] — the next outer's first read x[i] sits at the
                 * opposite end, so under cache pressure x gets evicted and
                 * has to be re-streamed every iter. At N=1024 the forward
                 * variant collapses to ~0.43× of migrated; backward closes
                 * the gap (Addendum 18).
                 *
                 * K-unroll-by-2 with split accumulators (t0, t1) breaks the
                 * single-acc fmul→fadd dep chain (same x87-latency fix as
                 * etrmv TRANS='T' and ytrsv U-T; Addendum 19 / Rule 22). */
                for (ptrdiff_t i = N - 1; i >= 0; --i) {
                    T t0 = x[i], t1 = zero;
                    const T *ai = &A_(0, i);
                    ptrdiff_t k = N - 1;
                    for (; k - 1 > i; k -= 2) {
                        t0 -= ai[k]     * x[k];
                        t1 -= ai[k - 1] * x[k - 1];
                    }
                    for (; k > i; --k) t0 -= ai[k] * x[k];
                    T t = t0 + t1;
                    if (nounit) t /= ai[i];
                    x[i] = t;
                }
            } else {
                /* UPLO='U': iterate i forward.
                 * x[i] = (b[i] - sum_{k<i} A(k,i) x[k]) / A(i,i).
                 *
                 * K-unroll-by-2 with split accumulators — see LT branch
                 * note above. */
                for (ptrdiff_t i = 0; i < N; ++i) {
                    T t0 = x[i], t1 = zero;
                    const T *ai = &A_(0, i);
                    ptrdiff_t k = 0;
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
    } else {
        /* General-stride fallback — hoist matrix column to ai[k] and
         * walk the strided vector with a running index (Class-B fix). */
        const ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'L') {
                ptrdiff_t ix = kx;
                for (ptrdiff_t i = 0; i < N; ++i) {
                    const T *ai = &A_(0, i);
                    if (x[ix] != zero) {
                        if (nounit) x[ix] /= ai[i];
                        const T xi = x[ix];
                        ptrdiff_t kk = ix + incx;
                        for (ptrdiff_t k = i + 1; k < N; ++k) {
                            x[kk] -= xi * ai[k];
                            kk += incx;
                        }
                    }
                    ix += incx;
                }
            } else {
                ptrdiff_t ix = kx + (N - 1) * incx;
                for (ptrdiff_t i = N - 1; i >= 0; --i) {
                    const T *ai = &A_(0, i);
                    if (x[ix] != zero) {
                        if (nounit) x[ix] /= ai[i];
                        const T xi = x[ix];
                        ptrdiff_t kk = kx;
                        for (ptrdiff_t k = 0; k < i; ++k) {
                            x[kk] -= xi * ai[k];
                            kk += incx;
                        }
                    }
                    ix -= incx;
                }
            }
        } else {
            if (UPLO == 'L') {
                /* Inner walks backward to match Fortran reference; same
                 * cache-direction reasoning as the incx=1 LT path above
                 * (Addendum 18 / Rule 21). K-unroll-by-2 with split
                 * accumulators (Addendum 19 / Rule 22). */
                ptrdiff_t ix = kx + (N - 1) * incx;
                for (ptrdiff_t i = N - 1; i >= 0; --i) {
                    const T *ai = &A_(0, i);
                    T t0 = x[ix], t1 = zero;
                    ptrdiff_t k = N - 1;
                    ptrdiff_t xk = kx + (N - 1) * incx;
                    for (; k - 1 > i; k -= 2) {
                        t0 -= ai[k]     * x[xk];
                        t1 -= ai[k - 1] * x[xk - incx];
                        xk -= 2 * incx;
                    }
                    for (; k > i; --k) { t0 -= ai[k] * x[xk]; xk -= incx; }
                    T t = t0 + t1;
                    if (nounit) t /= ai[i];
                    x[ix] = t;
                    ix -= incx;
                }
            } else {
                /* K-unroll-by-2 with split accumulators. */
                ptrdiff_t ix = kx;
                for (ptrdiff_t i = 0; i < N; ++i) {
                    const T *ai = &A_(0, i);
                    T t0 = x[ix], t1 = zero;
                    ptrdiff_t k = 0;
                    ptrdiff_t xk = kx;
                    for (; k + 1 < i; k += 2) {
                        t0 -= ai[k]     * x[xk];
                        t1 -= ai[k + 1] * x[xk + incx];
                        xk += 2 * incx;
                    }
                    if (k < i) t0 -= ai[k] * x[xk];
                    T t = t0 + t1;
                    if (nounit) t /= ai[i];
                    x[ix] = t;
                    ix += incx;
                }
            }
        }
    }
}

/* ── Block-parallel variant: single parallel region ─────────────────
 *
 * Mirrors qtrsv_blocked_ (Addendum 29). One `#pragma omp parallel`
 * wraps the entire diagonal walk:
 *
 *   - Thread 0 calls etrsv_serial_ on each diagonal sub-block.
 *   - All threads partition the trailing egemv across its long axis
 *     and call egemv_ on their slice. egemv's own OMP fork is gated
 *     off by omp_in_parallel(), so the inner gemv runs serially on
 *     each thread's slice.
 *   - Two `#pragma omp barrier`s per step.
 */

extern void egemv_core(
    char trans,
    ptrdiff_t m, ptrdiff_t n,
    const T *alpha,
    const T *a, ptrdiff_t lda,
    const T *x, ptrdiff_t incx,
    const T *beta,
    T *y, ptrdiff_t incy);

void etrsv_blocked_(
    const char *uplo, const char *trans, const char *diag,
    const ptrdiff_t *n_,
    const T *restrict a, const ptrdiff_t *lda_,
    T *restrict x, const ptrdiff_t *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    const ptrdiff_t N = *n_;
    const ptrdiff_t lda = *lda_, incx = *incx_;
    const ptrdiff_t nb = etrsv_blocked_nb();
    const char UPLO = blas_up(*uplo);
    char TR = blas_up(*trans);
    if (TR == 'C') TR = 'T';

    if (N == 0) return;
    if (incx != 1 || N < 2 * nb) {
        const ptrdiff_t n_pt = *n_, lda_pt = *lda_, incx_pt = *incx_;
        etrsv_serial_(uplo, trans, diag, &n_pt, a, &lda_pt, x, &incx_pt,
                      uplo_len, trans_len, diag_len);
        return;
    }

    const T neg_one = -1.0L;
    const T one_v   =  1.0L;
    const char NN[1] = {'N'};
    const char TT[1] = {'T'};
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
            for (ptrdiff_t j = 0; j < N; j += nb) {
                ptrdiff_t jb = (N - j < nb) ? (N - j) : nb;
                if (tid == 0) {
                    const ptrdiff_t lda_pt = *lda_;
                    etrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), &lda_pt,
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
                    long long lo = blas_part_bound(mt, tid, nt);
                    long long hi = blas_part_bound(mt, tid + 1, nt);
                    ptrdiff_t m_slice = (ptrdiff_t)(hi - lo);
                    if (m_slice > 0) {
                        const ptrdiff_t i_off = j2 + (ptrdiff_t)lo;
                        egemv_core(NN[0], m_slice, jb, &neg_one,
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
                    etrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), &lda_pt,
                                  &x[j], &one_i, uplo_len, trans_len, diag_len);
                }
#ifdef _OPENMP
                if (use_omp) {
                    #pragma omp barrier
                }
#endif
                if (j > 0) {
                    long long lo = blas_part_bound(j, tid, nt);
                    long long hi = blas_part_bound(j, tid + 1, nt);
                    ptrdiff_t m_slice = (ptrdiff_t)(hi - lo);
                    if (m_slice > 0) {
                        const ptrdiff_t i_off = (ptrdiff_t)lo;
                        egemv_core(NN[0], m_slice, jb, &neg_one,
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
                    etrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), &lda_pt,
                                  &x[j], &one_i, uplo_len, trans_len, diag_len);
                }
#ifdef _OPENMP
                if (use_omp) {
                    #pragma omp barrier
                }
#endif
                if (j > 0) {
                    long long lo = blas_part_bound(j, tid, nt);
                    long long hi = blas_part_bound(j, tid + 1, nt);
                    ptrdiff_t n_slice = (ptrdiff_t)(hi - lo);
                    if (n_slice > 0) {
                        const ptrdiff_t n_off = (ptrdiff_t)lo;
                        egemv_core(TT[0], jb, n_slice, &neg_one,
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
                    etrsv_serial_(uplo, trans, diag, &jb, &A_(j, j), &lda_pt,
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
                    long long lo = blas_part_bound(mt, tid, nt);
                    long long hi = blas_part_bound(mt, tid + 1, nt);
                    ptrdiff_t n_slice = (ptrdiff_t)(hi - lo);
                    if (n_slice > 0) {
                        const ptrdiff_t n_off = j2 + (ptrdiff_t)lo;
                        egemv_core(TT[0], jb, n_slice, &neg_one,
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

EPBLAS_FACADE_TRMV(etrsv, T)

#undef A_
