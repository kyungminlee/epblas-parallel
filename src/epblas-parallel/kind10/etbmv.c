/*
 * etbmv — kind10 (long double) triangular band matrix-vector.
 *   x := A*x or A^T*x, A triangular band with K+1 diagonals.
 *
 * The serial sweep is this overlay's faster reference and is defined FIRST so
 * its hot-loop placement is unperturbed by the threaded scaffolding (function
 * order shifts alignment). x := op(A)*x is a pure matvec (no solve
 * dependency), so although the in-place column sweep serializes it, every
 * OUTPUT element is independent and the threaded path (large N) computes
 * out-of-place into per-thread slots, mirroring OpenBLAS tbmv_thread, then
 * folds the slots into x. The fold is band-aware (see etbmv_omp): for NoTrans
 * each output row overlaps at most one neighbour's band; for Trans each thread
 * writes only its own rows (a dot per row) so the slots are disjoint and no
 * reduction is needed at all. Either way we beat ob's full O(n*nthreads)
 * controller reduce. The orchestration is a noinline helper so its bookkeeping
 * does not crowd the serial sweep's x87 allocator.
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef long double T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

#define ETBMV_OMP_MIN 200
#define ETBMV_MAX_CPUS 256

#ifdef _OPENMP
static int etbmv_omp(int upper, int trans, int nounit, ptrdiff_t n, ptrdiff_t k,
                     const T *restrict a, ptrdiff_t lda, T *restrict x, ptrdiff_t incx);
#endif

void etbmv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_, const int *k_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, incx = *incx_;
    const T zero = 0.0L;
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
    const int nounit = (up(diag) != 'U');

    if (N == 0) return;

#ifdef _OPENMP
    /* Cheap inline gate first: at OMP=1 (or below threshold) short-circuit
     * before the noinline call's argument marshalling (outlining tax). */
    if (N >= ETBMV_OMP_MIN && blas_omp_max_threads() > 1
        && etbmv_omp(UPLO == 'U', TR == 'T', nounit, N, K, a, lda, x, incx))
        return;
#endif

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'U') {
                for (int j = 0; j < N; ++j) {
                    if (x[j] != zero) {
                        const T tmp = x[j];
                        const int L = K - j;
                        const int i_lo = (j - K > 0) ? (j - K) : 0;
                        for (int i = i_lo; i < j; ++i) x[i] += tmp * A_(L + i, j);
                        if (nounit) x[j] *= A_(K, j);
                    }
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        const T tmp = x[j];
                        const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                        for (int i = i_hi - 1; i > j; --i) x[i] += tmp * A_(i - j, j);
                        if (nounit) x[j] *= A_(0, j);
                    }
                }
            }
        } else {
            if (UPLO == 'U') {
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    const int L = K - j;
                    if (nounit) tmp *= A_(K, j);
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    for (int i = j - 1; i >= i_lo; --i) tmp += A_(L + i, j) * x[i];
                    x[j] = tmp;
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    T tmp = x[j];
                    if (nounit) tmp *= A_(0, j);
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (int i = j + 1; i < i_hi; ++i) tmp += A_(i - j, j) * x[i];
                    x[j] = tmp;
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'U') {
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    if (x[jx] != zero) {
                        const T tmp = x[jx];
                        int ix = kx;
                        const int L = K - j;
                        const int i_lo = (j - K > 0) ? (j - K) : 0;
                        for (int i = i_lo; i < j; ++i) {
                            x[ix] += tmp * A_(L + i, j);
                            ix += incx;
                        }
                        if (nounit) x[jx] *= A_(K, j);
                    }
                    jx += incx;
                    if (j >= K) kx += incx;
                }
            } else {
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    if (x[jx] != zero) {
                        const T tmp = x[jx];
                        int ix = kx;
                        const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                        for (int i = i_hi - 1; i > j; --i) {
                            x[ix] += tmp * A_(i - j, j);
                            ix -= incx;
                        }
                        if (nounit) x[jx] *= A_(0, j);
                    }
                    jx -= incx;
                    if ((N - 1 - j) >= K) kx -= incx;
                }
            }
        } else {
            if (UPLO == 'U') {
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    kx -= incx;
                    int ix = kx;
                    const int L = K - j;
                    if (nounit) tmp *= A_(K, j);
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    for (int i = j - 1; i >= i_lo; --i) {
                        tmp += A_(L + i, j) * x[ix];
                        ix -= incx;
                    }
                    x[jx] = tmp;
                    jx -= incx;
                }
            } else {
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    T tmp = x[jx];
                    kx += incx;
                    int ix = kx;
                    if (nounit) tmp *= A_(0, j);
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (int i = j + 1; i < i_hi; ++i) {
                        tmp += A_(i - j, j) * x[ix];
                        ix += incx;
                    }
                    x[jx] = tmp;
                    jx += incx;
                }
            }
        }
    }
}

#ifdef _OPENMP
/* Per-thread kernel: for each column i in [n_from, n_to) accumulate that
 * column's contribution to the output into y (size n, zero-init by caller).
 * NoTrans writes a band of rows around i (cross-row); Trans writes only y[i]
 * (a dot, disjoint). Branches are hoisted out of the inner loop. */
static void tbmv_kernel(int upper, int trans, int nounit,
                        ptrdiff_t n, ptrdiff_t k,
                        ptrdiff_t n_from, ptrdiff_t n_to,
                        const T *restrict a, ptrdiff_t lda,
                        const T *restrict x, T *restrict y)
{
    for (ptrdiff_t i = n_from; i < n_to; ++i) {
        ptrdiff_t length = upper ? ((i < k) ? i : k)
                                 : ((n - i - 1 < k) ? n - i - 1 : k);
        const T *col = &A_(0, i);
        if (upper) {
            if (length > 0) {
                if (!trans) {
                    const T xi = x[i];
                    const T *coff = col + (k - length);
                    for (ptrdiff_t j = 0; j < length; ++j)
                        y[i - length + j] += xi * coff[j];
                } else {
                    T s = 0.0L;
                    const T *coff = col + (k - length);
                    const T *xoff = &x[i - length];
                    for (ptrdiff_t j = 0; j < length; ++j) s += coff[j] * xoff[j];
                    y[i] += s;
                }
            }
            y[i] += nounit ? col[k] * x[i] : x[i];
        } else {
            y[i] += nounit ? col[0] * x[i] : x[i];
            if (length > 0) {
                if (!trans) {
                    const T xi = x[i];
                    for (ptrdiff_t j = 0; j < length; ++j)
                        y[i + 1 + j] += xi * col[1 + j];
                } else {
                    T s = 0.0L;
                    const T *xoff = &x[i + 1];
                    for (ptrdiff_t j = 0; j < length; ++j) s += col[1 + j] * xoff[j];
                    y[i] += s;
                }
            }
        }
    }
}

/* Sqrt partition when n < 2k, even split when n >= 2k (mirrors tbmv_thread). */
static void tbmv_partition_diag(int upper, ptrdiff_t n, int nthreads, ptrdiff_t *range)
{
    const ptrdiff_t mask = 7;
    const double dnum = (double)n * (double)n / (double)nthreads;
    if (!upper) {
        range[0] = 0;
        ptrdiff_t i = 0; int num_cpu = 0;
        while (i < n && num_cpu < nthreads) {
            ptrdiff_t width;
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                width = (di * di - dnum > 0.0)
                    ? (((ptrdiff_t)(-sqrt(di * di - dnum) + di) + mask) & ~mask) : n - i;
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else width = n - i;
            range[num_cpu + 1] = range[num_cpu] + width; num_cpu++; i += width;
        }
        for (int t = num_cpu + 1; t <= nthreads; ++t) range[t] = range[num_cpu];
    } else {
        range[nthreads] = n;
        ptrdiff_t i = 0; int num_cpu = 0;
        while (i < n && num_cpu < nthreads) {
            ptrdiff_t width;
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                width = (di * di - dnum > 0.0)
                    ? (((ptrdiff_t)(-sqrt(di * di - dnum) + di) + mask) & ~mask) : n - i;
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else width = n - i;
            range[nthreads - num_cpu - 1] = range[nthreads - num_cpu] - width; num_cpu++; i += width;
        }
        for (int t = 0; t < nthreads - num_cpu; ++t) range[t] = range[nthreads - num_cpu];
    }
}

static void tbmv_partition_even(ptrdiff_t n, int nthreads, ptrdiff_t *range)
{
    range[0] = 0;
    ptrdiff_t i = n; int num_cpu = 0;
    while (i > 0 && num_cpu < nthreads) {
        ptrdiff_t width = (i + nthreads - num_cpu - 1) / (nthreads - num_cpu);
        if (width < 4) width = 4;
        if (i < width) width = i;
        range[num_cpu + 1] = range[num_cpu] + width; num_cpu++; i -= width;
    }
    for (int t = num_cpu + 1; t <= nthreads; ++t) range[t] = range[num_cpu];
}

/* Threaded triangular band matvec, ported from OpenBLAS tbmv_thread. Returns
 * 1 if it handled the call, 0 to fall back to the serial sweep. */
__attribute__((noinline)) static int etbmv_omp(
    int upper, int trans, int nounit, ptrdiff_t n, ptrdiff_t k,
    const T *restrict a, ptrdiff_t lda, T *restrict x, ptrdiff_t incx)
{
    if (n < ETBMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > ETBMV_MAX_CPUS) nthreads = ETBMV_MAX_CPUS;

    if (incx < 0) x -= (n - 1) * incx;

    const T *xptr = x;
    T *xbuf = NULL;
    if (incx != 1) {
        xbuf = (T *)malloc((size_t)n * sizeof(T));
        if (!xbuf) return 0;
        for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[i * incx];
        xptr = xbuf;
    }
    T *buf_all = (T *)calloc((size_t)nthreads * (size_t)n, sizeof(T));
    ptrdiff_t *range = (ptrdiff_t *)malloc((size_t)(nthreads + 1) * sizeof(ptrdiff_t));
    if (!buf_all || !range) {
        free(buf_all); free(range); free(xbuf);
        return 0;
    }

    const int diag_part = (n < 2 * k);
    if (diag_part) tbmv_partition_diag(upper, n, nthreads, range);
    else           tbmv_partition_even(n, nthreads, range);

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        T *restrict y = &buf_all[(size_t)tid * (size_t)n];
        ptrdiff_t n_from, n_to;
        if (diag_part && upper) { n_from = range[nthreads - tid - 1]; n_to = range[nthreads - tid]; }
        else                    { n_from = range[tid];               n_to = range[tid + 1]; }
        if (n_from < n_to)
            tbmv_kernel(upper, trans, nounit, n, k, n_from, n_to, a, lda, xptr, y);
    }

    /* Band-aware fold. The kernel slot for the thread owning columns [a,b)
     * is nonzero only within k of [a,b):
     *  - Trans: it writes ONLY y[i] for i in [a,b) (a dot per row), so the
     *    slots are disjoint — copy each thread's own range straight to x.
     *  - NoTrans: row r overlaps at most one neighbour's band when widths
     *    exceed k — UPPER picks up slot s+1 on the top border [b-k,b), LOWER
     *    picks up slot s-1 on the bottom border [a,a+k).
     * O(n)+O(nthreads*k) vs ob's O(n*nthreads) reduce. The result REPLACES x
     * (no alpha). Fall back to the full reduce for narrow widths / strided x.
     * (n>=200 here always takes the even partition, so ranges tile [0,n).) */
    ptrdiff_t min_width = n;
    for (int t = 0; t < nthreads; ++t) {
        ptrdiff_t w = range[t + 1] - range[t];
        if (w > 0 && w < min_width) min_width = w;
    }
    if (incx == 1 && !diag_part && (trans || min_width > k)) {
        for (int s = 0; s < nthreads; ++s) {
            ptrdiff_t a0 = range[s], b0 = range[s + 1];
            if (a0 >= b0) continue;
            const T *restrict ys = &buf_all[(size_t)s * (size_t)n];
            if (trans) {
                for (ptrdiff_t r = a0; r < b0; ++r) x[r] = ys[r];
            } else if (upper) {
                ptrdiff_t border = (b0 - k > a0) ? b0 - k : a0;
                for (ptrdiff_t r = a0; r < border; ++r) x[r] = ys[r];
                if (s + 1 < nthreads) {
                    const T *restrict ys1 = &buf_all[(size_t)(s + 1) * (size_t)n];
                    for (ptrdiff_t r = border; r < b0; ++r) x[r] = ys[r] + ys1[r];
                } else {
                    for (ptrdiff_t r = border; r < b0; ++r) x[r] = ys[r];
                }
            } else {
                ptrdiff_t border = (a0 + k < b0) ? a0 + k : b0;
                if (s - 1 >= 0) {
                    const T *restrict ysm1 = &buf_all[(size_t)(s - 1) * (size_t)n];
                    for (ptrdiff_t r = a0; r < border; ++r) x[r] = ys[r] + ysm1[r];
                } else {
                    for (ptrdiff_t r = a0; r < border; ++r) x[r] = ys[r];
                }
                for (ptrdiff_t r = border; r < b0; ++r) x[r] = ys[r];
            }
        }
    } else {
        for (int t = 1; t < nthreads; ++t) {
            const T *restrict slot = &buf_all[(size_t)t * (size_t)n];
            for (ptrdiff_t i = 0; i < n; ++i) buf_all[i] += slot[i];
        }
        if (incx == 1) for (ptrdiff_t i = 0; i < n; ++i) x[i] = buf_all[i];
        else           for (ptrdiff_t i = 0; i < n; ++i) x[i * incx] = buf_all[i];
    }
    free(buf_all); free(range); free(xbuf);
    return 1;
}
#endif /* _OPENMP */

#undef A_
