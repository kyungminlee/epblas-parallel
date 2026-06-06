/* msbmv — multifloats real DD symmetric band matrix-vector multiply.
 *   y := alpha*A*x + beta*y, A symmetric with K super-(or sub-)diagonals.
 *
 * Serial: each j-iteration touches both halves of A and updates y[j]
 * plus y[i] in two disjoint stripes, so column-parallel writes overlap.
 */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <cstdlib>
#include <omp.h>
#include "../common/blas_omp.h"
#define MSBMV_OMP_MIN 256
#define MSBMV_MAX_CPUS 256
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

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

#ifdef _OPENMP
/* Row-gather: y[i] (i in [lo,hi)) is an independent dot over the full 2K+1
 * symmetric band. A = upperTri + (strictUpper)^T, so each row reconstructs its
 * band as a same-triangle anti-diagonal walk (stride lda-1) plus the reflected
 * neighbours read contiguously in column i. x and y are distinct (sbmv forbids
 * aliasing) → output rows partition with no cross-thread dependence: no scratch,
 * no reduction, no barrier. Reorders the per-row summation vs the serial column
 * scatter → matches within DD fuzz tol; the serial path stays bit-exact. */
static void msbmv_rowgather(bool upper, int n, int k, int lo, int hi,
                            const T *a, std::size_t lda,
                            const T *x, T alpha, T *y, int incy)
{
    const std::ptrdiff_t s1 = static_cast<std::ptrdiff_t>(lda) - 1;
    if (upper) {
        for (int i = lo; i < hi; ++i) {
            const T *base = &A_(0, i);
            T s = base[k] * x[i];
            const int rlen = (n - 1 - i < k) ? (n - 1 - i) : k;
            for (int d = 1; d <= rlen; ++d) s = s + base[k + (std::ptrdiff_t)d * s1] * x[i + d];
            const int llen = (i < k) ? i : k;
            for (int d = 1; d <= llen; ++d) s = s + base[k - d] * x[i - d];
            y[(std::ptrdiff_t)i * incy] = y[(std::ptrdiff_t)i * incy] + alpha * s;
        }
    } else {
        for (int i = lo; i < hi; ++i) {
            const T *base = &A_(0, i);
            T s = base[0] * x[i];
            const int llen = (i < k) ? i : k;
            for (int d = 1; d <= llen; ++d) s = s + base[-(std::ptrdiff_t)d * s1] * x[i - d];
            const int rlen = (n - 1 - i < k) ? (n - 1 - i) : k;
            for (int d = 1; d <= rlen; ++d) s = s + base[d] * x[i + d];
            y[(std::ptrdiff_t)i * incy] = y[(std::ptrdiff_t)i * incy] + alpha * s;
        }
    }
}

/* Threaded symmetric band matvec. Each thread owns a disjoint output-row range
 * and writes y in place while reading x globally (gathered to contiguous when
 * strided). Returns true if handled. Beta-scaling already applied by caller. */
__attribute__((noinline)) static bool msbmv_omp(
    bool upper, int n, int k, const T *a, std::size_t lda,
    const T *x, int incx, T alpha, T *y, int incy)
{
    if (n < MSBMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MSBMV_MAX_CPUS) nthreads = MSBMV_MAX_CPUS;

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
        msbmv_rowgather(upper, n, k, lo, hi, a, lda, xptr, alpha, y, incy);
    }
    std::free(xbuf);
    return true;
}
#endif

extern "C" void msbmv_(
    const char *uplo,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *x, const int *incx_,
    const T *beta_,
    T *y, const int *incy_,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, incx = *incx_, incy = *incy_;
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
    if (N >= MSBMV_OMP_MIN && blas_omp_max_threads() > 1
        && msbmv_omp(UPLO == 'U', N, K, a, lda, x, incx, alpha, y, incy))
        return;
#endif

    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero_dd;
                const int L = K - j;
                const int i_lo = (j - K > 0) ? (j - K) : 0;
                for (int i = i_lo; i < j; ++i) {
                    y[i] = y[i] + t1 * A_(L + i, j);
                    t2 = t2 + A_(L + i, j) * x[i];
                }
                y[j] = y[j] + t1 * A_(K, j) + alpha * t2;
            }
        } else {
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero_dd;
                y[j] = y[j] + t1 * A_(0, j);
                const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                for (int i = j + 1; i < i_hi; ++i) {
                    y[i] = y[i] + t1 * A_(i - j, j);
                    t2 = t2 + A_(i - j, j) * x[i];
                }
                y[j] = y[j] + alpha * t2;
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
                const int L = K - j;
                const int i_lo = (j - K > 0) ? (j - K) : 0;
                for (int i = i_lo; i < j; ++i) {
                    y[iy] = y[iy] + t1 * A_(L + i, j);
                    t2 = t2 + A_(L + i, j) * x[ix];
                    ix += incx; iy += incy;
                }
                y[jy] = y[jy] + t1 * A_(K, j) + alpha * t2;
                jx += incx; jy += incy;
                if (j >= K) { kx += incx; ky += incy; }
            }
        } else {
            int jx = kx, jy = ky;
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero_dd;
                y[jy] = y[jy] + t1 * A_(0, j);
                int ix = jx, iy = jy;
                const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                for (int i = j + 1; i < i_hi; ++i) {
                    ix += incx; iy += incy;
                    y[iy] = y[iy] + t1 * A_(i - j, j);
                    t2 = t2 + A_(i - j, j) * x[ix];
                }
                y[jy] = y[jy] + alpha * t2;
                jx += incx; jy += incy;
            }
        }
    }
}

#undef A_
