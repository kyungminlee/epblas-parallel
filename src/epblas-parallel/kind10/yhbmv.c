/*
 * yhbmv — kind10 complex Hermitian band matrix-vector multiply.
 *   y := alpha*A*x + beta*y, A Hermitian with K super-(sub-)diagonals.
 *
 * Hermitian twin of esbmv. The serial sweep is this overlay's faster
 * reference and is defined FIRST so its hot-loop placement is unperturbed by
 * the threaded scaffolding (function order shifts alignment, ~2%). The
 * threaded path (large N) lives below in yhbmv_omp and mirrors OpenBLAS
 * zhbmv_thread: the column-oriented reference has cross-column writes, so
 * each thread owns a column range and accumulates the bare A*x (reflected
 * entry conjugated, diagonal real) into a private size-N slot, then a
 * band-aware reduction folds the slots into y — summing only the at-most-one
 * neighbour band that overlaps each row, beating ob's full O(n*nthreads)
 * controller reduction (see yhbmv_omp). The orchestration is a noinline
 * helper so its bookkeeping does not crowd the serial sweep's x87 allocator.
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef _Complex long double T;
typedef long double TR;
static inline T cconj(T z) { return ~z; }

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

/* Thread the column sweep once n reaches OpenBLAS zhbmv_thread's n>=200
 * crossover: below it both par and ob stay serial and par's faster sweep
 * wins; above it ob threads, so par must too. */
#define YHBMV_OMP_MIN 200
#define YHBMV_MAX_CPUS 256

#ifdef _OPENMP
static int yhbmv_omp(int upper, ptrdiff_t n, ptrdiff_t k,
                     const T *restrict a, ptrdiff_t lda,
                     const T *restrict x, ptrdiff_t incx,
                     T alpha, T *restrict y, ptrdiff_t incy);
#endif

void yhbmv_(
    const char *uplo,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict x, const int *incx_,
    const T *beta_,
    T *restrict y, const int *incy_,
    size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const T zero = 0.0L + 0.0Li, one = 1.0L + 0.0Li;
    const char UPLO = up(uplo);

    if (N == 0 || (alpha == zero && beta == one)) return;

    if (beta != one) {
        int iy = (incy < 0) ? -(N - 1) * incy : 0;
        if (beta == zero) for (int i = 0; i < N; ++i) { y[iy] = zero; iy += incy; }
        else              for (int i = 0; i < N; ++i) { y[iy] = beta * y[iy]; iy += incy; }
    }
    if (alpha == zero) return;

#ifdef _OPENMP
    /* Cheap inline gate first: at OMP=1 (or below threshold) short-circuit
     * before the noinline call's argument marshalling, keeping the serial
     * sweep's register allocation unperturbed (outlining tax). */
    if (N >= YHBMV_OMP_MIN && blas_omp_max_threads() > 1
        && yhbmv_omp(UPLO == 'U', N, K, a, lda, x, incx, alpha, y, incy))
        return;
#endif

    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero;
                const int L = K - j;
                const int i_lo = (j - K > 0) ? (j - K) : 0;
                for (int i = i_lo; i < j; ++i) {
                    const T aij = A_(L + i, j);
                    y[i] += t1 * aij;
                    t2 += cconj(aij) * x[i];
                }
                y[j] += t1 * (TR)__real__ A_(K, j) + alpha * t2;
            }
        } else {
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero;
                y[j] += t1 * (TR)__real__ A_(0, j);
                const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                for (int i = j + 1; i < i_hi; ++i) {
                    const T aij = A_(i - j, j);
                    y[i] += t1 * aij;
                    t2 += cconj(aij) * x[i];
                }
                y[j] += alpha * t2;
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        if (UPLO == 'U') {
            int jx = kx, jy = ky;
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero;
                int ix = kx, iy = ky;
                const int L = K - j;
                const int i_lo = (j - K > 0) ? (j - K) : 0;
                for (int i = i_lo; i < j; ++i) {
                    y[iy] += t1 * A_(L + i, j);
                    t2 += cconj(A_(L + i, j)) * x[ix];
                    ix += incx; iy += incy;
                }
                y[jy] += t1 * (TR)__real__ A_(K, j) + alpha * t2;
                jx += incx; jy += incy;
                if (j >= K) { kx += incx; ky += incy; }
            }
        } else {
            int jx = kx, jy = ky;
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero;
                y[jy] += t1 * (TR)__real__ A_(0, j);
                int ix = jx, iy = jy;
                const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                for (int i = j + 1; i < i_hi; ++i) {
                    ix += incx; iy += incy;
                    y[iy] += t1 * A_(i - j, j);
                    t2 += cconj(A_(i - j, j)) * x[ix];
                }
                y[jy] += alpha * t2;
                jx += incx; jy += incy;
            }
        }
    }
}

#ifdef _OPENMP
/* Per-thread kernel: walks column range [n_from, n_to), accumulating the bare
 * A*x for those columns into yp (size n, zero-init by caller). The reflected
 * band entry is conjugated and the diagonal is real, exactly as in the serial
 * sweep; alpha is applied by the reduction, not here. */
static void hbmv_kernel_U(ptrdiff_t n_from, ptrdiff_t n_to, ptrdiff_t k,
                          const T *restrict a, ptrdiff_t lda,
                          const T *restrict x, T *restrict yp)
{
    const T *acol = &A_(0, n_from);
    for (ptrdiff_t i = n_from; i < n_to; ++i) {
        ptrdiff_t length = (i < k) ? i : k;
        const T xi = x[i];
        const T *aoff = acol + (k - length);
        for (ptrdiff_t j = 0; j < length; ++j)
            yp[i - length + j] += xi * aoff[j];
        T s = 0.0L;
        const T *xoff = &x[i - length];
        for (ptrdiff_t j = 0; j < length; ++j)
            s += cconj(aoff[j]) * xoff[j];
        yp[i] += s + (TR)__real__ acol[k] * xi;   /* real diagonal */
        acol += lda;
    }
}

static void hbmv_kernel_L(ptrdiff_t n_from, ptrdiff_t n_to, ptrdiff_t n,
                          ptrdiff_t k, const T *restrict a, ptrdiff_t lda,
                          const T *restrict x, T *restrict yp)
{
    const T *acol = &A_(0, n_from);
    for (ptrdiff_t i = n_from; i < n_to; ++i) {
        ptrdiff_t length = (n - i - 1 < k) ? n - i - 1 : k;
        const T xi = x[i];
        for (ptrdiff_t j = 0; j < length; ++j)
            yp[i + 1 + j] += xi * acol[1 + j];
        T s = 0.0L;
        const T *xoff = &x[i + 1];
        for (ptrdiff_t j = 0; j < length; ++j)
            s += cconj(acol[1 + j]) * xoff[j];
        yp[i] += s + (TR)__real__ acol[0] * xi;    /* real diagonal */
        acol += lda;
    }
}

/* Width partition mirroring zhbmv_thread.c: load-balanced sqrt formula when
 * n < 2k (per-column work ~ min(i,k)), even split when n >= 2k. */
static void partition_hbmv_diag(int upper, ptrdiff_t n, int nthreads,
                                ptrdiff_t *range_m)
{
    const ptrdiff_t mask = 7;
    const double dnum = (double)n * (double)n / (double)nthreads;
    int num_cpu = 0;
    ptrdiff_t i = 0;
    if (!upper) {
        range_m[0] = 0;
        while (i < n) {
            ptrdiff_t width;
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                if (di * di - dnum > 0.0)
                    width = ((ptrdiff_t)(-sqrt(di * di - dnum) + di) + mask) & ~mask;
                else
                    width = n - i;
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else {
                width = n - i;
            }
            range_m[num_cpu + 1] = range_m[num_cpu] + width;
            num_cpu++;
            i += width;
        }
        for (int t = num_cpu + 1; t <= nthreads; ++t) range_m[t] = range_m[num_cpu];
    } else {
        range_m[nthreads] = n;
        while (i < n) {
            ptrdiff_t width;
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                if (di * di - dnum > 0.0)
                    width = ((ptrdiff_t)(-sqrt(di * di - dnum) + di) + mask) & ~mask;
                else
                    width = n - i;
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else {
                width = n - i;
            }
            range_m[nthreads - num_cpu - 1] = range_m[nthreads - num_cpu] - width;
            num_cpu++;
            i += width;
        }
        for (int t = num_cpu + 1; t <= nthreads; ++t) range_m[nthreads - t] = range_m[nthreads - num_cpu];
    }
}

static void partition_hbmv_even(ptrdiff_t n, int nthreads, ptrdiff_t *range_m)
{
    range_m[0] = 0;
    ptrdiff_t i = n;
    int num_cpu = 0;
    while (i > 0) {
        ptrdiff_t width = (i + nthreads - num_cpu - 1) / (nthreads - num_cpu);
        if (width < 4) width = 4;
        if (i < width) width = i;
        range_m[num_cpu + 1] = range_m[num_cpu] + width;
        num_cpu++;
        i -= width;
    }
    for (int t = num_cpu + 1; t <= nthreads; ++t) range_m[t] = range_m[num_cpu];
}

/* Threaded Hermitian band matvec, ported from OpenBLAS zhbmv_thread. Returns
 * 1 if it handled the call, 0 to fall back to the serial sweep. */
__attribute__((noinline)) static int yhbmv_omp(
    int upper, ptrdiff_t n, ptrdiff_t k,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    T alpha, T *restrict y, ptrdiff_t incy)
{
    if (n < YHBMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > YHBMV_MAX_CPUS) nthreads = YHBMV_MAX_CPUS;

    if (incx < 0) x -= (n - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;

    const T *xptr = x;
    T *xbuf = NULL;
    if (incx != 1) {
        xbuf = (T *)malloc((size_t)n * sizeof(T));
        if (!xbuf) return 0;
        for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[i * incx];
        xptr = xbuf;
    }
    T *y_priv = (T *)calloc((size_t)nthreads * (size_t)n, sizeof(T));
    ptrdiff_t *range_m = (ptrdiff_t *)malloc((size_t)(nthreads + 1) * sizeof(ptrdiff_t));
    if (!y_priv || !range_m) {
        free(y_priv); free(range_m); free(xbuf);
        return 0;
    }

    if (n < 2 * k) partition_hbmv_diag(upper, n, nthreads, range_m);
    else           partition_hbmv_even(n, nthreads, range_m);

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        T *restrict yp = &y_priv[(size_t)tid * (size_t)n];
        ptrdiff_t n_from = range_m[tid], n_to = range_m[tid + 1];
        if (n_from < n_to) {
            if (upper) hbmv_kernel_U(n_from, n_to, k, a, lda, xptr, yp);
            else       hbmv_kernel_L(n_from, n_to, n, k, a, lda, xptr, yp);
        }
    }

    /* Band-aware reduction (see esbmv for the derivation): a thread owning
     * columns [a,b) only writes rows within k of that range, so for widths
     * exceeding k each row overlaps at most one neighbour's band — slot s+1 on
     * the UPPER top border [b-k,b), slot s-1 on the LOWER bottom border
     * [a,a+k). O(n)+O(nthreads*k) vs ob's O(n*nthreads), no extra barrier.
     * Fall back to the full sum for narrow widths or strided y. */
    ptrdiff_t min_width = n;
    for (int t = 0; t < nthreads; ++t) {
        ptrdiff_t w = range_m[t + 1] - range_m[t];
        if (w > 0 && w < min_width) min_width = w;
    }
    if (incy == 1 && min_width > k) {
        for (int s = 0; s < nthreads; ++s) {
            ptrdiff_t a0 = range_m[s], b0 = range_m[s + 1];
            if (a0 >= b0) continue;
            const T *restrict ys = &y_priv[(size_t)s * (size_t)n];
            if (upper) {
                ptrdiff_t border = (b0 - k > a0) ? b0 - k : a0;
                for (ptrdiff_t r = a0; r < border; ++r) y[r] += alpha * ys[r];
                if (s + 1 < nthreads) {
                    const T *restrict ys1 = &y_priv[(size_t)(s + 1) * (size_t)n];
                    for (ptrdiff_t r = border; r < b0; ++r) y[r] += alpha * (ys[r] + ys1[r]);
                } else {
                    for (ptrdiff_t r = border; r < b0; ++r) y[r] += alpha * ys[r];
                }
            } else {
                ptrdiff_t border = (a0 + k < b0) ? a0 + k : b0;
                if (s - 1 >= 0) {
                    const T *restrict ysm1 = &y_priv[(size_t)(s - 1) * (size_t)n];
                    for (ptrdiff_t r = a0; r < border; ++r) y[r] += alpha * (ys[r] + ysm1[r]);
                } else {
                    for (ptrdiff_t r = a0; r < border; ++r) y[r] += alpha * ys[r];
                }
                for (ptrdiff_t r = border; r < b0; ++r) y[r] += alpha * ys[r];
            }
        }
    } else {
        ptrdiff_t iy = 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            T s = 0.0L;
            for (int t = 0; t < nthreads; ++t)
                s += y_priv[(size_t)t * (size_t)n + (size_t)i];
            y[iy] += alpha * s;
            iy += incy;
        }
    }
    free(range_m); free(y_priv); free(xbuf);
    return 1;
}
#endif /* _OPENMP */

#undef A_
