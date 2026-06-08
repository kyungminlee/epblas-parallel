/*
 * etrmv — kind10 (REAL(KIND=10)) triangular matrix-vector.
 *   x := A · x         (TRANS='N')
 *   x := Aᵀ · x        (TRANS='T'/'C')
 * A is N×N triangular (UPLO, DIAG). x updated in-place.
 *
 * Netlib reference + restrict + stride-1 column access. OMP path
 * uses an external output buffer so the in-place data dependency
 * dissolves (TR='T' simple, TR='N' needs per-thread y_priv+reduce
 * — same pattern as esymv per Addendum 36).
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define ETRMV_OMP_MIN 128

typedef long double T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#ifdef _OPENMP
/* OMP core on a CONTIGUOUS x. Shared by the incx==1 fast path and the
 * strided path (via gather/scatter), so the threading lives in one place.
 * TR='T' writes each x[j] once (disjoint dot → bit-exact across threads);
 * TR='N' uses per-thread y_priv + reduction (esymv pattern, Add-36 — matches
 * ref within tolerance, not bit-exact). Returns 1 on success, 0 if the
 * scratch alloc failed (caller falls back to serial). */
static int etrmv_omp_contig(char UPLO, char TR, ptrdiff_t nounit,
                            ptrdiff_t N, const T *restrict a, ptrdiff_t lda,
                            T *restrict x, ptrdiff_t nt)
{
    const T zero = 0.0L;
    if (TR == 'T') {
        T *y_buf = (T *)aligned_alloc(64,
            (((size_t)N * sizeof(T)) + 63) & ~(size_t)63);
        if (!y_buf) return 0;
        #pragma omp parallel
        {
            if (UPLO == 'L') {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T temp = nounit ? (x[j] * A_(j, j)) : x[j];
                    const T *aj = &A_(0, j);
                    T s0 = zero, s1 = zero;
                    ptrdiff_t i = j + 1;
                    for (; i + 1 < N; i += 2) {
                        s0 += aj[i]     * x[i];
                        s1 += aj[i + 1] * x[i + 1];
                    }
                    T s = s0 + s1;
                    for (; i < N; ++i) s += aj[i] * x[i];
                    y_buf[j] = temp + s;
                }
            } else {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T temp = nounit ? (x[j] * A_(j, j)) : x[j];
                    const T *aj = &A_(0, j);
                    T s0 = zero, s1 = zero;
                    ptrdiff_t i = j - 1;
                    for (; i - 1 >= 0; i -= 2) {
                        s0 += aj[i]     * x[i];
                        s1 += aj[i - 1] * x[i - 1];
                    }
                    T s = s0 + s1;
                    for (; i >= 0; --i) s += aj[i] * x[i];
                    y_buf[j] = temp + s;
                }
            }
            #pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < N; ++i) x[i] = y_buf[i];
        }
        free(y_buf);
        return 1;
    } else {
        /* TR='N' — per-thread y_priv + reduction. */
        T *y_priv_all = (T *)calloc((size_t)nt * (size_t)N, sizeof(T));
        if (!y_priv_all) return 0;
        #pragma omp parallel num_threads(nt)
        {
            const ptrdiff_t tid = omp_get_thread_num();
            T *y_priv = &y_priv_all[(size_t)tid * N];  /* calloc-zeroed */

            if (UPLO == 'L') {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < N; ++j) {
                    const T xj = x[j];
                    const T *aj = &A_(0, j);
                    y_priv[j] += xj * (nounit ? aj[j] : (T)1.0L);
                    for (ptrdiff_t i = j + 1; i < N; ++i)
                        y_priv[i] += xj * aj[i];
                }
            } else {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < N; ++j) {
                    const T xj = x[j];
                    const T *aj = &A_(0, j);
                    for (ptrdiff_t i = 0; i < j; ++i)
                        y_priv[i] += xj * aj[i];
                    y_priv[j] += xj * (nounit ? aj[j] : (T)1.0L);
                }
            }
            #pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < N; ++i) {
                T s = zero;
                for (ptrdiff_t t = 0; t < nt; ++t)
                    s += y_priv_all[(size_t)t * N + i];
                x[i] = s;
            }
        }
        free(y_priv_all);
        return 1;
    }
}
#endif

void etrmv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
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
    const T zero = 0.0L;

    if (incx == 1) {
#ifdef _OPENMP
        const ptrdiff_t nt = blas_omp_max_threads();
        if (N >= ETRMV_OMP_MIN && nt > 1 && !omp_in_parallel()
            && etrmv_omp_contig(UPLO, TR, nounit, N, a, lda, x, nt))
            return;
#endif
        if (TR == 'N') {
            if (UPLO == 'L') {
                /* j backward: x[i] for i>j updated by temp=x[j]; then scale x[j].
                 * Inner walks backward (i = N-1..j+1) to match Fortran
                 * etrmv.f (DO 50 I = N,J+1,-1). Sub-class C / Rule 21.
                 *
                 * J-unroll-by-2 (symmetric to the UNN path below): at iter
                 * j and j-1 both x[j] and x[j-1] are pristine on entry
                 * (iter j's inner only touches i>j). Save originals, fuse
                 * both column contributions into one i-pass over the
                 * trailing rows, then handle boundaries i=j and i=j-1
                 * separately. Halves x memory traffic on the AXPY inner.
                 * Previously LNN sat at 0.90-0.94x of migrated; unrolling
                 * closes the gap. */
                ptrdiff_t j = N - 1;
                for (; j - 1 >= 0; j -= 2) {
                    const T t0 = x[j];
                    const T t1 = x[j - 1];
                    const T *a0 = &A_(0, j);
                    const T *a1 = &A_(0, j - 1);
                    /* Inner i=N-1..j+1 — backward (Rule 21).
                     * Both columns contribute to each x[i]. */
                    for (ptrdiff_t i = N - 1; i > j; --i)
                        x[i] = (x[i] + t0 * a0[i]) + t1 * a1[i];
                    /* Boundary i=j: scale x[j] (was t0), then add t1*A(j,j-1). */
                    T xj = nounit ? t0 * A_(j, j) : t0;
                    x[j] = xj + t1 * a1[j];
                    /* Boundary i=j-1: scale x[j-1] (was t1). */
                    if (nounit) x[j - 1] = t1 * A_(j - 1, j - 1);
                }
                /* Odd-N tail. */
                for (; j >= 0; --j) {
                    const T temp = x[j];
                    if (temp != zero) {
                        const T *aj = &A_(0, j);
                        for (ptrdiff_t i = N - 1; i > j; --i) x[i] += temp * aj[i];
                    }
                    if (nounit) x[j] *= A_(j, j);
                }
            } else {
                /* UPLO='U', j forward: x[i] for i<j updated by t=x[j]; then
                 * scale x[j] by diag if nounit. J-unroll-by-2: at iter j and
                 * j+1 both x[j] and x[j+1] are pristine (iter j's inner only
                 * touches i<j); combine so each x[i] load+store services
                 * two column contributions. Halves x memory traffic on the
                 * AXPY-like inner. Without this, UNN at N=1024 sat at
                 * 0.58x of migrated. */
                ptrdiff_t j = 0;
                for (; j + 1 < N; j += 2) {
                    const T t0 = x[j];
                    const T t1 = x[j + 1];
                    const T *a0 = &A_(0, j);
                    const T *a1 = &A_(0, j + 1);
                    for (ptrdiff_t i = 0; i < j; ++i)
                        x[i] = (x[i] + t0 * a0[i]) + t1 * a1[i];
                    /* At i=j: x[j] += t1*A(j,j+1), with prior diag scale. */
                    T xj = nounit ? t0 * A_(j, j) : t0;
                    x[j] = xj + t1 * a1[j];
                    if (nounit) x[j + 1] = t1 * A_(j + 1, j + 1);
                }
                for (; j < N; ++j) {
                    const T temp = x[j];
                    if (temp != zero) {
                        const T *aj = &A_(0, j);
                        for (ptrdiff_t i = 0; i < j; ++i) x[i] += temp * aj[i];
                    }
                    if (nounit) x[j] *= A_(j, j);
                }
            }
        } else {  /* TRANS = 'T' */
            if (UPLO == 'L') {
                /* j forward: dot product over i>j into x[j]. */
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T temp = x[j];
                    if (nounit) temp *= A_(j, j);
                    const T *aj = &A_(0, j);
                    /* 2-chain dot product (x87 latency-hiding). */
                    T s0 = zero, s1 = zero;
                    ptrdiff_t i = j + 1;
                    for (; i + 1 < N; i += 2) {
                        s0 += aj[i]     * x[i];
                        s1 += aj[i + 1] * x[i + 1];
                    }
                    T s = s0 + s1;
                    for (; i < N; ++i) s += aj[i] * x[i];
                    x[j] = temp + s;
                }
            } else {
                /* UPLO='U', j backward: dot over i<j into x[j].
                 * Inner walks backward (i = j-1..0) to match the
                 * Fortran reference (DO 90 I = J-1,1,-1). 2-chain
                 * unroll preserved: descend in pairs. Sub-class D /
                 * Rule 21 — even though the current forward-2-chain
                 * already beats migrated at measured N because of
                 * x87 latency hiding, the backward walk keeps the
                 * direction consistent with the Fortran reference. */
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T temp = x[j];
                    if (nounit) temp *= A_(j, j);
                    const T *aj = &A_(0, j);
                    T s0 = zero, s1 = zero;
                    ptrdiff_t i = j - 1;
                    for (; i - 1 >= 0; i -= 2) {
                        s0 += aj[i]     * x[i];
                        s1 += aj[i - 1] * x[i - 1];
                    }
                    T s = s0 + s1;
                    for (; i >= 0; --i) s += aj[i] * x[i];
                    x[j] = temp + s;
                }
            }
        }
    } else {
        /* General-stride fallback. */
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
#ifdef _OPENMP
        /* Thread the strided path by gathering x into a contiguous buffer,
         * driving the shared OMP core, and scattering back — so the
         * threading lives in one place (etrmv_omp_contig) and the tuned
         * serial strided code below stays byte-for-byte unchanged. */
        const ptrdiff_t ntS = blas_omp_max_threads();
        if (N >= ETRMV_OMP_MIN && ntS > 1 && !omp_in_parallel()) {
            T *xc = (T *)malloc((size_t)N * sizeof(T));
            if (xc) {
                for (ptrdiff_t i = 0; i < N; ++i) xc[i] = x[kx + i * incx];
                if (etrmv_omp_contig(UPLO, TR, nounit, N, a, lda, xc, ntS)) {
                    for (ptrdiff_t i = 0; i < N; ++i) x[kx + i * incx] = xc[i];
                    free(xc);
                    return;
                }
                free(xc);
            }
        }
#endif
        if (TR == 'N') {
            if (UPLO == 'L') {
                /* Inner walks backward to match Fortran etrmv.f (DO 70
                 * I=N,J+1,-1). J-unroll-by-2 (mirrors the incx==1 path):
                 * fuse columns j and j-1 into one strided-x read-modify-
                 * write per row i, halving the strided-x traffic that
                 * dominates this light real scatter — the residual int->
                 * ptrdiff_t regression on the un-unrolled single-column
                 * form (the running-index hoist made it WORSE; halving x
                 * traffic is the real lever). Sub-class C / Rule 21. */
                ptrdiff_t j = N - 1;
                for (; j - 1 >= 0; j -= 2) {
                    const T t0 = x[kx + j * incx];
                    const T t1 = x[kx + (j - 1) * incx];
                    const T *a0 = &A_(0, j);
                    const T *a1 = &A_(0, j - 1);
                    for (ptrdiff_t i = N - 1; i > j; --i)
                        x[kx + i * incx] = (x[kx + i * incx] + t0 * a0[i]) + t1 * a1[i];
                    T xj = nounit ? t0 * a0[j] : t0;
                    x[kx + j * incx] = xj + t1 * a1[j];
                    if (nounit) x[kx + (j - 1) * incx] = t1 * a1[j - 1];
                }
                for (; j >= 0; --j) {
                    const T temp = x[kx + j * incx];
                    if (temp != zero) {
                        const T *aj = &A_(0, j);
                        for (ptrdiff_t i = N - 1; i > j; --i) x[kx + i * incx] += temp * aj[i];
                    }
                    if (nounit) x[kx + j * incx] *= A_(j, j);
                }
            } else {
                /* UNN: symmetric j-unroll-by-2 (fuse columns j and j+1). */
                ptrdiff_t j = 0;
                for (; j + 1 < N; j += 2) {
                    const T t0 = x[kx + j * incx];
                    const T t1 = x[kx + (j + 1) * incx];
                    const T *a0 = &A_(0, j);
                    const T *a1 = &A_(0, j + 1);
                    for (ptrdiff_t i = 0; i < j; ++i)
                        x[kx + i * incx] = (x[kx + i * incx] + t0 * a0[i]) + t1 * a1[i];
                    T xj = nounit ? t0 * a0[j] : t0;
                    x[kx + j * incx] = xj + t1 * a1[j];
                    if (nounit) x[kx + (j + 1) * incx] = t1 * a1[j + 1];
                }
                for (; j < N; ++j) {
                    const T temp = x[kx + j * incx];
                    if (temp != zero) {
                        const T *aj = &A_(0, j);
                        for (ptrdiff_t i = 0; i < j; ++i) x[kx + i * incx] += temp * aj[i];
                    }
                    if (nounit) x[kx + j * incx] *= A_(j, j);
                }
            }
        } else {
            if (UPLO == 'L') {
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp *= A_(j, j);
                    for (ptrdiff_t i = j + 1; i < N; ++i) temp += A_(i, j) * x[kx + i * incx];
                    x[kx + j * incx] = temp;
                }
            } else {
                /* Inner walks backward to match Fortran reference
                 * (DO 110 I = J-1,1,-1). Sub-class D / Rule 21. */
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp *= A_(j, j);
                    for (ptrdiff_t i = j - 1; i >= 0; --i) temp += A_(i, j) * x[kx + i * incx];
                    x[kx + j * incx] = temp;
                }
            }
        }
    }
}

#undef A_
