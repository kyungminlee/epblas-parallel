/* mspmv — multifloats real DD symmetric packed matrix-vector multiply.
 *   y := alpha*A*x + beta*y
 *
 * Serial — same overlapping-y-writes problem as msbmv.
 */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <cstdlib>
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
const T one_dd{1.0, 0.0};
inline bool dd_iszero(const T &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_isone (const T &x) { return x.limbs[0] == 1.0 && x.limbs[1] == 0.0; }
}

#ifdef _OPENMP
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

    int kk = 0;
    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero_dd;
                int k = kk;
                for (int i = 0; i < j; ++i) {
                    y[i] = y[i] + t1 * ap[k];
                    t2 = t2 + ap[k] * x[i];
                    ++k;
                }
                y[j] = y[j] + t1 * ap[kk + j] + alpha * t2;
                kk += j + 1;
            }
        } else {
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero_dd;
                y[j] = y[j] + t1 * ap[kk];
                int k = kk + 1;
                for (int i = j + 1; i < N; ++i) {
                    y[i] = y[i] + t1 * ap[k];
                    t2 = t2 + ap[k] * x[i];
                    ++k;
                }
                y[j] = y[j] + alpha * t2;
                kk += N - j;
            }
        }
    } else {
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
