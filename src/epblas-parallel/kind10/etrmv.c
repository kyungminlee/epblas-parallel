/*
 * etrmv — kind10 (REAL(KIND=10)) triangular matrix-vector.
 *   x := A · x         (TRANS='N')
 *   x := Aᵀ · x        (TRANS='T'/'C')
 * A is N×N triangular (UPLO, DIAG). x updated in-place.
 *
 * Netlib reference + restrict + stride-1 column access. OMP path
 * uses an external output buffer so the in-place data dependency
 * dissolves (TRANS='T' simple, TRANS='N' needs per-thread y_priv+reduce
 * — same pattern as esymv per Addendum 36).
 */

#include <stddef.h>
#include <stdbool.h>
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

#define ETRMV_OMP_MIN 128

typedef long double TR;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#ifdef _OPENMP
/* OMP core on a CONTIGUOUS x. Shared by the incx==1 fast path and the
 * strided path (via gather/scatter), so the threading lives in one place.
 * TRANS='T' writes each x[j] once (disjoint dot → bit-exact across threads);
 * TRANS='N' uses per-thread y_priv + reduction (esymv pattern, Add-36 — matches
 * ref within tolerance, not bit-exact). Returns 1 on success, 0 if the
 * scratch alloc failed (caller falls back to serial). */
static bool etrmv_omp_contig(char UPLO, char TRANS, bool nounit,
                            ptrdiff_t n, const TR *restrict a, ptrdiff_t lda,
                            TR *restrict x, ptrdiff_t nthreads)
{
    const TR zero = 0.0L;
    if (TRANS == 'T') {
        TR *y_buf = (TR *)aligned_alloc(64,
            (((size_t)n * sizeof(TR)) + 63) & ~(size_t)63);
        if (!y_buf) return 0;
        #pragma omp parallel
        {
            if (UPLO == 'L') {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TR temp = nounit ? (x[j] * A_(j, j)) : x[j];
                    const TR *aj = &A_(0, j);
                    TR s0 = zero, s1 = zero;
                    ptrdiff_t i = j + 1;
                    for (; i + 1 < n; i += 2) {
                        s0 += aj[i]     * x[i];
                        s1 += aj[i + 1] * x[i + 1];
                    }
                    TR s = s0 + s1;
                    for (; i < n; ++i) s += aj[i] * x[i];
                    y_buf[j] = temp + s;
                }
            } else {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TR temp = nounit ? (x[j] * A_(j, j)) : x[j];
                    const TR *aj = &A_(0, j);
                    TR s0 = zero, s1 = zero;
                    ptrdiff_t i = j - 1;
                    for (; i - 1 >= 0; i -= 2) {
                        s0 += aj[i]     * x[i];
                        s1 += aj[i - 1] * x[i - 1];
                    }
                    TR s = s0 + s1;
                    for (; i >= 0; --i) s += aj[i] * x[i];
                    y_buf[j] = temp + s;
                }
            }
            #pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < n; ++i) x[i] = y_buf[i];
        }
        free(y_buf);
        return 1;
    } else {
        /* TRANS='N' — per-thread y_priv + reduction. */
        TR *y_priv_all = (TR *)calloc((size_t)nthreads * (size_t)n, sizeof(TR));
        if (!y_priv_all) return 0;
        #pragma omp parallel num_threads(nthreads)
        {
            const ptrdiff_t tid = omp_get_thread_num();
            TR *y_priv = &y_priv_all[(size_t)tid * n];  /* calloc-zeroed */

            if (UPLO == 'L') {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    const TR xj = x[j];
                    const TR *aj = &A_(0, j);
                    y_priv[j] += xj * (nounit ? aj[j] : (TR)1.0L);
                    for (ptrdiff_t i = j + 1; i < n; ++i)
                        y_priv[i] += xj * aj[i];
                }
            } else {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    const TR xj = x[j];
                    const TR *aj = &A_(0, j);
                    for (ptrdiff_t i = 0; i < j; ++i)
                        y_priv[i] += xj * aj[i];
                    y_priv[j] += xj * (nounit ? aj[j] : (TR)1.0L);
                }
            }
            #pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < n; ++i) {
                TR s = zero;
                for (ptrdiff_t t = 0; t < nthreads; ++t)
                    s += y_priv_all[(size_t)t * n + i];
                x[i] = s;
            }
        }
        free(y_priv_all);
        return 1;
    }
}
#endif

static void etrmv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n,
    const TR *restrict a, ptrdiff_t lda,
    TR *restrict x, ptrdiff_t incx)
{
    const char UPLO = blas_up(uplo);
    char TRANS = blas_up(trans);
    if (TRANS == 'C') TRANS = 'T';
    const char DIAG = blas_up(diag);
    const bool nounit = (DIAG != 'U');

    if (n == 0) return;
    const TR zero = 0.0L;

    if (incx == 1) {
#ifdef _OPENMP
        const ptrdiff_t nthreads = blas_omp_max_threads();
        if (n >= ETRMV_OMP_MIN && blas_omp_should_thread()
            && etrmv_omp_contig(UPLO, TRANS, nounit, n, a, lda, x, nthreads))
            return;
#endif
        if (TRANS == 'N') {
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
                ptrdiff_t j = n - 1;
                for (; j - 1 >= 0; j -= 2) {
                    const TR t0 = x[j];
                    const TR t1 = x[j - 1];
                    const TR *a0 = &A_(0, j);
                    const TR *a1 = &A_(0, j - 1);
                    /* Inner i=N-1..j+1 — backward (Rule 21).
                     * Both columns contribute to each x[i]. */
                    for (ptrdiff_t i = n - 1; i > j; --i)
                        x[i] = (x[i] + t0 * a0[i]) + t1 * a1[i];
                    /* Boundary i=j: scale x[j] (was t0), then add t1*A(j,j-1). */
                    TR xj = nounit ? t0 * A_(j, j) : t0;
                    x[j] = xj + t1 * a1[j];
                    /* Boundary i=j-1: scale x[j-1] (was t1). */
                    if (nounit) x[j - 1] = t1 * A_(j - 1, j - 1);
                }
                /* Odd-N tail. */
                for (; j >= 0; --j) {
                    const TR temp = x[j];
                    if (temp != zero) {
                        const TR *aj = &A_(0, j);
                        for (ptrdiff_t i = n - 1; i > j; --i) x[i] += temp * aj[i];
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
                for (; j + 1 < n; j += 2) {
                    const TR t0 = x[j];
                    const TR t1 = x[j + 1];
                    const TR *a0 = &A_(0, j);
                    const TR *a1 = &A_(0, j + 1);
                    for (ptrdiff_t i = 0; i < j; ++i)
                        x[i] = (x[i] + t0 * a0[i]) + t1 * a1[i];
                    /* At i=j: x[j] += t1*A(j,j+1), with prior diag scale. */
                    TR xj = nounit ? t0 * A_(j, j) : t0;
                    x[j] = xj + t1 * a1[j];
                    if (nounit) x[j + 1] = t1 * A_(j + 1, j + 1);
                }
                for (; j < n; ++j) {
                    const TR temp = x[j];
                    if (temp != zero) {
                        const TR *aj = &A_(0, j);
                        for (ptrdiff_t i = 0; i < j; ++i) x[i] += temp * aj[i];
                    }
                    if (nounit) x[j] *= A_(j, j);
                }
            }
        } else {  /* TRANS = 'T' */
            if (UPLO == 'L') {
                /* j forward: dot product over i>j into x[j]. */
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TR temp = x[j];
                    if (nounit) temp *= A_(j, j);
                    const TR *aj = &A_(0, j);
                    /* 2-chain dot product (x87 latency-hiding). */
                    TR s0 = zero, s1 = zero;
                    ptrdiff_t i = j + 1;
                    for (; i + 1 < n; i += 2) {
                        s0 += aj[i]     * x[i];
                        s1 += aj[i + 1] * x[i + 1];
                    }
                    TR s = s0 + s1;
                    for (; i < n; ++i) s += aj[i] * x[i];
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
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TR temp = x[j];
                    if (nounit) temp *= A_(j, j);
                    const TR *aj = &A_(0, j);
                    TR s0 = zero, s1 = zero;
                    ptrdiff_t i = j - 1;
                    for (; i - 1 >= 0; i -= 2) {
                        s0 += aj[i]     * x[i];
                        s1 += aj[i - 1] * x[i - 1];
                    }
                    TR s = s0 + s1;
                    for (; i >= 0; --i) s += aj[i] * x[i];
                    x[j] = temp + s;
                }
            }
        }
    } else {
        /* General-stride fallback. */
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
#ifdef _OPENMP
        /* Thread the strided path by gathering x into a contiguous buffer,
         * driving the shared OMP core, and scattering back — so the
         * threading lives in one place (etrmv_omp_contig) and the tuned
         * serial strided code below stays byte-for-byte unchanged. */
        const ptrdiff_t nthreads = blas_omp_max_threads();
        if (n >= ETRMV_OMP_MIN && blas_omp_should_thread()) {
            TR *xc = (TR *)malloc((size_t)n * sizeof(TR));
            if (xc) {
                for (ptrdiff_t i = 0; i < n; ++i) xc[i] = x[kx + i * incx];
                if (etrmv_omp_contig(UPLO, TRANS, nounit, n, a, lda, xc, nthreads)) {
                    for (ptrdiff_t i = 0; i < n; ++i) x[kx + i * incx] = xc[i];
                    free(xc);
                    return;
                }
                free(xc);
            }
        }
#endif
        if (TRANS == 'N') {
            if (UPLO == 'L') {
                /* Inner walks backward to match Fortran etrmv.f (DO 70
                 * I=N,J+1,-1). J-unroll-by-2 (mirrors the incx==1 path):
                 * fuse columns j and j-1 into one strided-x read-modify-
                 * write per row i, halving the strided-x traffic that
                 * dominates this light real scatter — the residual int->
                 * ptrdiff_t regression on the un-unrolled single-column
                 * form (the running-index hoist made it WORSE; halving x
                 * traffic is the real lever). Sub-class C / Rule 21. */
                ptrdiff_t j = n - 1;
                for (; j - 1 >= 0; j -= 2) {
                    const TR t0 = x[kx + j * incx];
                    const TR t1 = x[kx + (j - 1) * incx];
                    const TR *a0 = &A_(0, j);
                    const TR *a1 = &A_(0, j - 1);
                    for (ptrdiff_t i = n - 1; i > j; --i)
                        x[kx + i * incx] = (x[kx + i * incx] + t0 * a0[i]) + t1 * a1[i];
                    TR xj = nounit ? t0 * a0[j] : t0;
                    x[kx + j * incx] = xj + t1 * a1[j];
                    if (nounit) x[kx + (j - 1) * incx] = t1 * a1[j - 1];
                }
                for (; j >= 0; --j) {
                    const TR temp = x[kx + j * incx];
                    if (temp != zero) {
                        const TR *aj = &A_(0, j);
                        for (ptrdiff_t i = n - 1; i > j; --i) x[kx + i * incx] += temp * aj[i];
                    }
                    if (nounit) x[kx + j * incx] *= A_(j, j);
                }
            } else {
                /* UNN: symmetric j-unroll-by-2 (fuse columns j and j+1). */
                ptrdiff_t j = 0;
                for (; j + 1 < n; j += 2) {
                    const TR t0 = x[kx + j * incx];
                    const TR t1 = x[kx + (j + 1) * incx];
                    const TR *a0 = &A_(0, j);
                    const TR *a1 = &A_(0, j + 1);
                    for (ptrdiff_t i = 0; i < j; ++i)
                        x[kx + i * incx] = (x[kx + i * incx] + t0 * a0[i]) + t1 * a1[i];
                    TR xj = nounit ? t0 * a0[j] : t0;
                    x[kx + j * incx] = xj + t1 * a1[j];
                    if (nounit) x[kx + (j + 1) * incx] = t1 * a1[j + 1];
                }
                for (; j < n; ++j) {
                    const TR temp = x[kx + j * incx];
                    if (temp != zero) {
                        const TR *aj = &A_(0, j);
                        for (ptrdiff_t i = 0; i < j; ++i) x[kx + i * incx] += temp * aj[i];
                    }
                    if (nounit) x[kx + j * incx] *= A_(j, j);
                }
            }
        } else {
            if (UPLO == 'L') {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TR temp = x[kx + j * incx];
                    if (nounit) temp *= A_(j, j);
                    for (ptrdiff_t i = j + 1; i < n; ++i) temp += A_(i, j) * x[kx + i * incx];
                    x[kx + j * incx] = temp;
                }
            } else {
                /* Inner walks backward to match Fortran reference
                 * (DO 110 I = J-1,1,-1). Sub-class D / Rule 21. */
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TR temp = x[kx + j * incx];
                    if (nounit) temp *= A_(j, j);
                    for (ptrdiff_t i = j - 1; i >= 0; --i) temp += A_(i, j) * x[kx + i * incx];
                    x[kx + j * incx] = temp;
                }
            }
        }
    }
}

EPBLAS_FACADE_TRMV(etrmv, TR)

#undef A_
