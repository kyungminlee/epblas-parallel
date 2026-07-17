/*
 * qtrmv — kind16 (__float128) triangular matrix-vector.
 *   x := A · x  (TRANS='N') or Aᵀ · x  (TRANS='T'/'C')
 *
 * Serial reference is the in-place Netlib column sweep (sequential). The
 * threaded path (incx==1, large N) dissolves the in-place dependency with an
 * external output buffer, mirroring kind10 etrmv / esymv (Addendum 36):
 *   - TRANS='T': y[j] = dot(column j, x), disjoint per thread -> shared buffer, no
 *     reduce, copy back.
 *   - TRANS='N': column j scatters into x[i] (i>j for L, i<j for U), so cross-
 *     thread j-ranges write overlapping i ranges -> per-thread y_priv + reduce.
 * Quad is compute-bound under libquadmath, so the per-column work amortizes the
 * fork/buffer almost immediately. Serial reference stays byte-for-byte. */

#include <stddef.h>
#include <stdbool.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

typedef __float128 TR;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#ifndef QTRMV_OMP_MIN
#define QTRMV_OMP_MIN 128
#endif

/* Upper-NoTrans large-N switches from cyclic schedule(static,1) to a contiguous
 * sqrt-balanced block partition (see qtrmv_blkrange). Below this size the matrix
 * is L2-resident/compute-bound and cyclic's finer granularity wins; above it the
 * matrix exceeds L2 and a contiguous column slab per thread (one sequential
 * prefetch stream instead of 4 interleaved scattered ones) closes the ~13% gap
 * to the ob leg. Crossover measured between N=256 (cyclic faster) and N=320. */
#ifndef QTRMV_BLOCK_MIN
#define QTRMV_BLOCK_MIN 320
#endif

#ifdef _OPENMP
/* Contiguous triangular block partition for Upper NoTrans (work per column j is
 * proportional to j, so high columns are heavy). Mirrors the ob leg's
 * trmv_thread.c sqrt-partition: balance by triangular area, narrow range at the
 * heavy high-column end. range[0..nthreads] are column boundaries over [0,n). */
static void qtrmv_blkrange(ptrdiff_t n, ptrdiff_t nthreads, ptrdiff_t *range)
{
    const ptrdiff_t mask = 7;
    const double dnum = (double)n * (double)n / (double)nthreads;
    range[nthreads] = n;
    ptrdiff_t i = 0;
    ptrdiff_t nc = 0;
    while (i < n && nc < nthreads) {
        ptrdiff_t width;
        if (nthreads - nc > 1) {
            const double di = (double)(n - i);
            if (di * di - dnum > 0.0)
                width = ((ptrdiff_t)(-sqrt(di * di - dnum) + di) + mask) & ~mask;
            else
                width = n - i;
            if (width < 16) width = 16;
            if (width > n - i) width = n - i;
        } else {
            width = n - i;
        }
        range[nthreads - nc - 1] = range[nthreads - nc] - width;
        nc++;
        i += width;
    }
    for (ptrdiff_t t = 0; t < nthreads - nc; ++t) range[t] = range[nthreads - nc];
}

/* Threaded out-of-place path (incx==1). Returns 1 if handled, 0 to fall back to
 * the serial reference. noinline so the serial loops compile in a clean register
 * context. */
__attribute__((noinline))
static bool qtrmv_omp(bool upper, bool trans_t, bool nounit, ptrdiff_t n, ptrdiff_t lda,
                     const TR *restrict a, TR *restrict x)
{
    const ptrdiff_t nthreads = blas_omp_max_threads();
    if (n < QTRMV_OMP_MIN || !blas_omp_should_thread()) return 0;
    const TR zero = 0.0Q;

    if (trans_t) {
        /* TRANS='T': each j writes a single x[j] (dot of column j with x). All
         * threads read x then write disjoint y_buf[j] — own j, no overlap. */
        TR *y_buf = (TR *)malloc((size_t)n * sizeof(TR));
        if (!y_buf) return 0;
        #pragma omp parallel num_threads(nthreads)
        {
            if (!upper) {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    /* Accumulate onto temp directly, matching the serial
                     * arm and netlib DTRMV (keeps OMP bit-consistent). */
                    TR temp = nounit ? (x[j] * A_(j, j)) : x[j];
                    const TR *aj = &A_(0, j);
                    for (ptrdiff_t i = j + 1; i < n; ++i) temp += aj[i] * x[i];
                    y_buf[j] = temp;
                }
            } else {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    /* Descending accumulation onto temp, matching the serial
                     * arm and netlib DTRMV (keeps OMP bit-consistent). */
                    TR temp = nounit ? (x[j] * A_(j, j)) : x[j];
                    const TR *aj = &A_(0, j);
                    for (ptrdiff_t i = j - 1; i >= 0; --i) temp += aj[i] * x[i];
                    y_buf[j] = temp;
                }
            }
            #pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < n; ++i) x[i] = y_buf[i];
        }
        free(y_buf);
        return 1;
    } else {
        /* TRANS='N': per-thread y_priv + reduction (cross-thread overlapping writes).
         * Upper at large N uses a contiguous sqrt-balanced column-slab partition
         * (qtrmv_blkrange) so each thread streams one sequential block of the
         * matrix instead of cyclically scattered columns; Lower NoTrans and small
         * N keep the cyclic schedule(static,1) (already beats ob there). */
        TR *y_priv_all = (TR *)calloc((size_t)nthreads * (size_t)n, sizeof(TR));
        if (!y_priv_all) return 0;
        ptrdiff_t *rng = NULL;
        if (upper && n >= QTRMV_BLOCK_MIN) {
            rng = (ptrdiff_t *)malloc(((size_t)nthreads + 1) * sizeof(ptrdiff_t));
            if (rng) qtrmv_blkrange(n, nthreads, rng);
        }
        const bool blk = (rng != NULL);
        #pragma omp parallel num_threads(nthreads)
        {
            const ptrdiff_t tid = omp_get_thread_num();
            TR *y_priv = &y_priv_all[(size_t)tid * n];  /* calloc-zeroed */
            if (!upper) {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    const TR xj = x[j];
                    const TR *aj = &A_(0, j);
                    y_priv[j] += xj * (nounit ? aj[j] : (TR)1.0Q);
                    for (ptrdiff_t i = j + 1; i < n; ++i) y_priv[i] += xj * aj[i];
                }
            } else if (blk) {
                for (ptrdiff_t j = rng[tid]; j < rng[tid + 1]; ++j) {
                    const TR xj = x[j];
                    const TR *aj = &A_(0, j);
                    for (ptrdiff_t i = 0; i < j; ++i) y_priv[i] += xj * aj[i];
                    y_priv[j] += xj * (nounit ? aj[j] : (TR)1.0Q);
                }
            } else {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    const TR xj = x[j];
                    const TR *aj = &A_(0, j);
                    for (ptrdiff_t i = 0; i < j; ++i) y_priv[i] += xj * aj[i];
                    y_priv[j] += xj * (nounit ? aj[j] : (TR)1.0Q);
                }
            }
            /* blk path has no implicit barrier from an omp-for; sync before reduce. */
            #pragma omp barrier
            #pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < n; ++i) {
                TR s = zero;
                for (ptrdiff_t t = 0; t < nthreads; ++t) s += y_priv_all[(size_t)t * n + i];
                x[i] = s;
            }
        }
        free(rng);
        free(y_priv_all);
        return 1;
    }
}
#endif /* _OPENMP */

void qtrmv_core(
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
    const TR zero = 0.0Q;

    if (incx == 1) {
#ifdef _OPENMP
        if (qtrmv_omp(UPLO == 'U', TRANS == 'T', nounit, n, lda, a, x)) return;
#endif
        if (TRANS == 'N') {
            if (UPLO == 'L') {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    const TR temp = x[j];
                    if (temp != zero) {
                        const TR *aj = &A_(0, j);
                        for (ptrdiff_t i = j + 1; i < n; ++i) x[i] += temp * aj[i];
                    }
                    if (nounit) x[j] *= A_(j, j);
                }
            } else {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    const TR temp = x[j];
                    if (temp != zero) {
                        const TR *aj = &A_(0, j);
                        for (ptrdiff_t i = 0; i < j; ++i) x[i] += temp * aj[i];
                    }
                    if (nounit) x[j] *= A_(j, j);
                }
            }
        } else {
            if (UPLO == 'L') {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TR temp = x[j];
                    if (nounit) temp *= A_(j, j);
                    const TR *aj = &A_(0, j);
                    for (ptrdiff_t i = j + 1; i < n; ++i) temp += aj[i] * x[i];
                    x[j] = temp;
                }
            } else {
                /* Descending dot = netlib DTRMV's order: fewer soft-float
                 * branch misses (same mechanism as qtrsv) and bit-exact
                 * vs netlib. */
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TR temp = x[j];
                    if (nounit) temp *= A_(j, j);
                    const TR *aj = &A_(0, j);
                    for (ptrdiff_t i = j - 1; i >= 0; --i) temp += aj[i] * x[i];
                    x[j] = temp;
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
#ifdef _OPENMP
        /* Thread the strided path by gathering x into a contiguous buffer,
         * driving the shared OMP core, and scattering back — the threading
         * lives in one place (qtrmv_omp) and the serial strided code below
         * stays byte-for-byte unchanged. */
        if (n >= QTRMV_OMP_MIN && blas_omp_should_thread()) {
            TR *xc = (TR *)malloc((size_t)n * sizeof(TR));
            if (xc) {
                for (ptrdiff_t i = 0; i < n; ++i) xc[i] = x[kx + i * incx];
                if (qtrmv_omp(UPLO == 'U', TRANS == 'T', nounit, n, lda, a, xc)) {
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
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    const TR temp = x[kx + j * incx];
                    if (temp != zero)
                        for (ptrdiff_t i = j + 1; i < n; ++i) x[kx + i * incx] += temp * A_(i, j);
                    if (nounit) x[kx + j * incx] *= A_(j, j);
                }
            } else {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    const TR temp = x[kx + j * incx];
                    if (temp != zero)
                        for (ptrdiff_t i = 0; i < j; ++i) x[kx + i * incx] += temp * A_(i, j);
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
                /* Netlib DTRMV's descending jx/ix walk (see the contiguous
                 * arm above). */
                ptrdiff_t jx = kx + (n - 1) * incx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TR temp = x[jx];
                    if (nounit) temp *= A_(j, j);
                    ptrdiff_t ix = jx;
                    for (ptrdiff_t i = j - 1; i >= 0; --i) {
                        ix -= incx;
                        temp += A_(i, j) * x[ix];
                    }
                    x[jx] = temp;
                    jx -= incx;
                }
            }
        }
    }
}

EPBLAS_FACADE_TRMV(qtrmv, TR)

#undef A_
