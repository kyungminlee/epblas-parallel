/* mspmv — multifloats real DD symmetric packed matrix-vector multiply.
 *   y := alpha*A*x + beta*y
 *
 * Serial — same overlapping-y-writes problem as msbmv.
 */

#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <multifloats.h>
#include "mf_tri_simd.h"   /* mf_tri::axpy_add / dot — shared SoA loop kernels */
#ifdef _OPENMP
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include "../common/blas_omp.h"
#define MSPMV_OMP_MIN 256
#define MSPMV_MAX_CPUS 256
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
const T zero_dd{0.0, 0.0};
inline bool dd_iszero(const T &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_isone (const T &x) { return x.limbs[0] == 1.0 && x.limbs[1] == 0.0; }
}

#define AP_(idx) ap[static_cast<std::size_t>(idx)]

namespace {
/* Contiguous (incx==incy==1) symmetric packed matvec, y += alpha*A*x (beta
 * already applied by the caller). Per column j the packed slice &aj[..] is a
 * contiguous run shared by the reflected AXPY (y[i] += t1*col[i], order-free ->
 * bit-exact) and the symmetric dot (t2 += col[i]*x[i], vector accumulate +
 * hreduce -> within tolerance); y and x are distinct (spmv forbids aliasing).
 * The strided entry gathers x/y to scratch and reuses this. */
void mspmv_contig(bool upper, int n, const T *ap, const T *x, T alpha, T *y)
{
    std::size_t kk = 0;
    if (upper) {
        for (int j = 0; j < n; ++j) {
            const T *aj = &AP_(kk);            /* column j: rows 0..j-1, diag at aj[j] */
            const T t1 = alpha * x[j];
            T t2 = zero_dd;
            if (j > 0) {
                mf_tri::axpy_add(j, &y[0], aj, t1);
                t2 = mf_tri::dot(j, aj, &x[0]);
            }
            y[j] = y[j] + t1 * aj[j] + alpha * t2;
            kk += static_cast<std::size_t>(j) + 1;
        }
    } else {
        for (int j = 0; j < n; ++j) {
            const T *aj = &AP_(kk);            /* column j: diag at aj[0], rows j+1.. at aj[1.. ] */
            const T t1 = alpha * x[j];
            y[j] = y[j] + t1 * aj[0];
            const int len = n - 1 - j;
            if (len > 0) {
                mf_tri::axpy_add(len, &y[j + 1], &aj[1], t1);
                y[j] = y[j] + alpha * mf_tri::dot(len, &aj[1], &x[j + 1]);
            }
            kk += static_cast<std::size_t>(n - j);
        }
    }
}
}

#ifdef _OPENMP
/* Sqrt-balanced contiguous column partition (port of ob spmv_thread.c
 * symv_partition): triangular work per column j is ~j (upper) / ~(n-j) (lower),
 * so equal-width blocks would skew load ~2x. Widths solve a quadratic so each
 * block carries ~n*n/nthreads packed entries. mask=3 rounds widths to multiples
 * of 4 (min_width 4) to keep slot writes off shared cache lines. */
static int mspmv_partition(bool upper, std::ptrdiff_t n, int nthreads,
                           int mask, int min_width, std::ptrdiff_t *range)
{
    int num_cpu = 0;
    double dnum = (double)n * (double)n / (double)nthreads;
    range[0] = 0;
    std::ptrdiff_t i = 0;
    while (i < n) {
        std::ptrdiff_t width;
        if (nthreads - num_cpu > 1) {
            if (upper) {
                double di = (double)i;
                width = (std::ptrdiff_t)(std::sqrt(di * di + dnum) - di);
            } else {
                double di = (double)(n - i);
                double rad = di * di - dnum;
                if (rad > 0.0) width = (std::ptrdiff_t)(-std::sqrt(rad) + di);
                else           width = n - i;
            }
            width = (width + mask) & ~(std::ptrdiff_t)mask;
            if (width < min_width) width = min_width;
            if (width > n - i)     width = n - i;
        } else {
            width = n - i;
        }
        range[num_cpu + 1] = range[num_cpu] + width;
        num_cpu++;
        i += width;
        if (num_cpu >= MSPMV_MAX_CPUS) break;
    }
    return num_cpu;
}

/* Unit-stride threaded path (port of ob mspmv axpydot). Threads own disjoint
 * COLUMN ranges and accumulate into a private slot[n]; a single contiguous pass
 * over column j does both the scatter (slot[i]+=x[j]*aj[i]) and the symmetric
 * dot (temp2+=aj[i]*x[i]), reading column j of the packed triangle contiguously.
 * Per-thread slots are then AXPY-reduced with alpha factored into the final fold.
 * Contiguous column read (vs the row-gather's anti-diagonal col-jump that spans
 * the whole packed array) is what lets this scale ~3.3x. Reorders the per-row
 * sum vs serial -> within DD fuzz tol (strided path keeps the row-gather). */
static bool mspmv_axpydot(bool upper, int n, const T *ap,
                          const T *x, T alpha, T *y, int nthreads)
{
    std::ptrdiff_t range[MSPMV_MAX_CPUS + 1];
    int num_cpu = mspmv_partition(upper, n, nthreads, 3, 4, range);
    if (num_cpu <= 1) return false;

    T *buf = static_cast<T *>(std::calloc((std::size_t)num_cpu * n, sizeof(T)));
    if (!buf) return false;

    #pragma omp parallel num_threads(num_cpu)
    {
        int t = omp_get_thread_num();
        std::ptrdiff_t m_from = range[t];
        std::ptrdiff_t m_to   = range[t + 1];
        T *slot = buf + (std::size_t)t * n;
        if (upper) {
            for (std::ptrdiff_t j = m_from; j < m_to; ++j) {
                T temp1 = x[j];
                T temp2 = zero_dd;
                const T *aj = &AP_((std::size_t)j * (j + 1) / 2);
                for (std::ptrdiff_t i = 0; i < j; ++i) {
                    slot[i] = slot[i] + temp1 * aj[i];
                    temp2   = temp2   + aj[i] * x[i];
                }
                slot[j] = slot[j] + (temp1 * aj[j] + temp2);
            }
        } else {
            for (std::ptrdiff_t j = m_from; j < m_to; ++j) {
                T temp1 = x[j];
                T temp2 = zero_dd;
                const T *aj = &AP_((std::size_t)j * (2 * (std::size_t)n - j - 1) / 2);
                slot[j] = slot[j] + temp1 * aj[j];
                for (std::ptrdiff_t i = j + 1; i < n; ++i) {
                    slot[i] = slot[i] + temp1 * aj[i];
                    temp2   = temp2   + aj[i] * x[i];
                }
                slot[j] = slot[j] + temp2;
            }
        }
    }

    /* AXPY-reduce the private slots, then alpha-scale into y. Each slot is only
     * populated over the rows its column range touches (upper: [0,range[t+1]);
     * lower: [range[t],n)), so the fold sums just those spans. */
    if (upper) {
        T *target = buf + (std::size_t)(num_cpu - 1) * n;
        for (int i = 0; i < num_cpu - 1; ++i) {
            const T *src = buf + (std::size_t)i * n;
            std::ptrdiff_t len = range[i + 1];
            for (std::ptrdiff_t k = 0; k < len; ++k) target[k] = target[k] + src[k];
        }
        for (std::ptrdiff_t k = 0; k < n; ++k) y[k] = y[k] + alpha * target[k];
    } else {
        T *target = buf;
        for (int i = 1; i < num_cpu; ++i) {
            const T *src = buf + (std::size_t)i * n;
            std::ptrdiff_t m_from = range[i];
            for (std::ptrdiff_t k = m_from; k < n; ++k) target[k] = target[k] + src[k];
        }
        for (std::ptrdiff_t k = 0; k < n; ++k) y[k] = y[k] + alpha * target[k];
    }
    std::free(buf);
    return true;
}

/* Row-gather: y[i] (i in [lo,hi)) is an independent dot over the full symmetric
 * row, reconstructed from the packed triangle. For UPLO='U' the diagonal of
 * column j sits at ap[j(j+1)/2 + j]; for UPLO='L' at ap[j*N - j(j-1)/2]. Each
 * row walks one contiguous run inside its own column plus a column-jumping run
 * (offset incremented in O(1) per step). x and y are distinct (spmv forbids
 * aliasing) → output rows partition with no cross-thread dependence. Reorders
 * the per-row summation vs the serial column scatter → within DD fuzz tol; the
 * serial path stays bit-exact. */
static void mspmv_rowgather(bool upper, int n, int lo, int hi,
                            const T *ap, const T *x, T alpha, T *y, int incy)
{
    if (upper) {
        for (int i = lo; i < hi; ++i) {
            const std::size_t kk_i = static_cast<std::size_t>(i) * (i + 1) / 2;
            T s = ap[kk_i + i] * x[i];                       /* diagonal */
            for (int j = 0; j < i; ++j) s = s + ap[kk_i + j] * x[j];   /* left (col i) */
            std::size_t kk_j = kk_i + (i + 1);               /* start of column i+1 */
            for (int j = i + 1; j < n; ++j) {                /* right (col-jump) */
                s = s + ap[kk_j + i] * x[j];
                kk_j += static_cast<std::size_t>(j + 1);
            }
            y[(std::ptrdiff_t)i * incy] = y[(std::ptrdiff_t)i * incy] + alpha * s;
        }
    } else {
        for (int i = lo; i < hi; ++i) {
            const std::size_t kk_i =
                static_cast<std::size_t>(i) * n - static_cast<std::size_t>(i) * (i - 1) / 2;
            T s = ap[kk_i] * x[i];                           /* diagonal */
            for (int j = i + 1; j < n; ++j) s = s + ap[kk_i + (j - i)] * x[j]; /* right (col i) */
            std::size_t kk_j = 0;                            /* start of column 0 */
            for (int j = 0; j < i; ++j) {                    /* left (col-jump) */
                s = s + ap[kk_j + (i - j)] * x[j];
                kk_j += static_cast<std::size_t>(n - j);
            }
            y[(std::ptrdiff_t)i * incy] = y[(std::ptrdiff_t)i * incy] + alpha * s;
        }
    }
}

/* Threaded symmetric packed matvec. Disjoint output-row ranges; x gathered to
 * contiguous when strided. Returns true if handled. Beta already applied. */
__attribute__((noinline)) static bool mspmv_omp(
    bool upper, int n, const T *ap,
    const T *x, int incx, T alpha, T *y, int incy)
{
    if (n < MSPMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MSPMV_MAX_CPUS) nthreads = MSPMV_MAX_CPUS;

    if (incx < 0) x -= (std::ptrdiff_t)(n - 1) * incx;
    if (incy < 0) y -= (std::ptrdiff_t)(n - 1) * incy;

    /* Unit-stride: ob-style column-partition axpydot (contiguous column read,
     * ~3.3x). Strided: row-gather below (ob serializes strided; par's
     * disjoint-output-row gather already wins there, so keep it). */
    if (incx == 1 && incy == 1)
        return mspmv_axpydot(upper, n, ap, x, alpha, y, nthreads);

    const T *xptr = x;
    T *xbuf = nullptr;
    if (incx != 1) {
        xbuf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
        if (!xbuf) return false;
        for (int i = 0; i < n; ++i) xbuf[i] = x[(std::ptrdiff_t)i * incx];
        xptr = xbuf;
    }

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int lo = (int)((long long)n * tid / nthreads);
        int hi = (int)((long long)n * (tid + 1) / nthreads);
        mspmv_rowgather(upper, n, lo, hi, ap, xptr, alpha, y, incy);
    }
    std::free(xbuf);
    return true;
}
#endif

extern "C" void mspmv_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *ap,
    const T *x, const int *incx_,
    const T *beta_,
    T *y, const int *incy_,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);

    if (N == 0 || (dd_iszero(alpha) && dd_isone(beta))) return;

    if (!dd_isone(beta)) {
        int iy = (incy < 0) ? -(N - 1) * incy : 0;
        if (dd_iszero(beta)) {
            for (int i = 0; i < N; ++i) { y[iy] = zero_dd; iy += incy; }
        } else {
            for (int i = 0; i < N; ++i) { y[iy] = beta * y[iy]; iy += incy; }
        }
    }
    if (dd_iszero(alpha)) return;

#ifdef _OPENMP
    if (N >= MSPMV_OMP_MIN && blas_omp_max_threads() > 1
        && mspmv_omp(UPLO == 'U', N, ap, x, incx, alpha, y, incy))
        return;
#endif

    if (incx == 1 && incy == 1) {
        mspmv_contig(UPLO == 'U', N, ap, x, alpha, y);
        return;
    }

    /* Strided: gather x and the beta-scaled y into contiguous scratch, run the
     * SIMD contiguous core (y += alpha*A*x), scatter y back. O(N) gather/scatter
     * vs O(N*N) packed work; the in-place strided walk is the alloc-fail fallback. */
    {
        const std::ptrdiff_t bx = (incx < 0) ? -(std::ptrdiff_t)(N - 1) * incx : 0;
        const std::ptrdiff_t by = (incy < 0) ? -(std::ptrdiff_t)(N - 1) * incy : 0;
        T *xs = static_cast<T *>(std::malloc((std::size_t)N * sizeof(T)));
        T *ys = static_cast<T *>(std::malloc((std::size_t)N * sizeof(T)));
        if (xs && ys) {
            for (int i = 0; i < N; ++i) {
                xs[i] = x[bx + (std::ptrdiff_t)i * incx];
                ys[i] = y[by + (std::ptrdiff_t)i * incy];
            }
            mspmv_contig(UPLO == 'U', N, ap, xs, alpha, ys);
            for (int i = 0; i < N; ++i) y[by + (std::ptrdiff_t)i * incy] = ys[i];
            std::free(xs); std::free(ys);
            return;
        }
        std::free(xs); std::free(ys);
    }

    {
        int kk = 0;
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        if (UPLO == 'U') {
            int jx = kx, jy = ky;
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero_dd;
                int ix = kx, iy = ky;
                for (int k = kk; k < kk + j; ++k) {
                    y[iy] = y[iy] + t1 * ap[k];
                    t2 = t2 + ap[k] * x[ix];
                    ix += incx; iy += incy;
                }
                y[jy] = y[jy] + t1 * ap[kk + j] + alpha * t2;
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            int jx = kx, jy = ky;
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero_dd;
                y[jy] = y[jy] + t1 * ap[kk];
                int ix = jx, iy = jy;
                for (int k = kk + 1; k < kk + N - j; ++k) {
                    ix += incx; iy += incy;
                    y[iy] = y[iy] + t1 * ap[k];
                    t2 = t2 + ap[k] * x[ix];
                }
                y[jy] = y[jy] + alpha * t2;
                jx += incx; jy += incy;
                kk += N - j;
            }
        }
    }
}
