/* msbmv — multifloats real DD symmetric band matrix-vector multiply.
 *   y := alpha*A*x + beta*y, A symmetric with K super-(or sub-)diagonals.
 *
 * Serial: each j-iteration touches both halves of A and updates y[j]
 * plus y[i] in two disjoint stripes, so column-parallel writes overlap.
 */

#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <multifloats.h>
#include "mf_util.h"
#include "mf_pred.h"
#include "mf_kernels.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#define MSBMV_OMP_MIN 256
#define MSBMV_MAX_CPUS 256
#endif

namespace mf = multifloats;
using T = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
using mf_pred::eq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
const T zero_dd{0.0, 0.0};
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

namespace {
/* Contiguous (incx==incy==1) symmetric band matvec, y += alpha*A*x (beta already
 * applied by the caller). Per column j the band slice &aj[..] is a contiguous run
 * shared by the reflected AXPY (y[i] += t1*col[i], order-free -> bit-exact) and
 * the dot (t2 += col[i]*x[i], vector accumulate + hreduce -> within tolerance);
 * y and x are distinct (sbmv forbids aliasing) so the two passes are independent.
 * The strided entry gathers x/y to scratch and reuses this. */
void msbmv_contig(bool upper, int n, int k, const T *a, std::size_t lda,
                  const T *x, T alpha, T *y)
{
    if (upper) {
        for (int j = 0; j < n; ++j) {
            const T *aj = &A_(0, j);
            const T t1 = alpha * x[j];
            const int i_lo = (j - k > 0) ? (j - k) : 0;
            const int len = j - i_lo;
            const T *col = &aj[k - j + i_lo];   /* A_(K-j+i_lo, j), contiguous */
            T t2 = zero_dd;
            if (len > 0) {
                mf_kernels::axpy_add(len, &y[i_lo], col, t1);
                t2 = mf_kernels::dot(len, col, &x[i_lo]);
            }
            y[j] = y[j] + t1 * aj[k] + alpha * t2;
        }
    } else {
        for (int j = 0; j < n; ++j) {
            const T *aj = &A_(0, j);
            const T t1 = alpha * x[j];
            y[j] = y[j] + t1 * aj[0];
            const int i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
            const int len = i_hi - (j + 1);
            if (len > 0) {
                mf_kernels::axpy_add(len, &y[j + 1], &aj[1], t1);
                y[j] = y[j] + alpha * mf_kernels::dot(len, &aj[1], &x[j + 1]);
            }
        }
    }
}
}

#ifdef _OPENMP
/* Unit-stride threaded path. Threads own disjoint COLUMN ranges and accumulate
 * into a private slot[n]; one contiguous pass over each band column j does both
 * the reflected scatter (slot[i]+=x[j]*col[i]) and the symmetric dot
 * (t2+=col[i]*x[i]) via the same SIMD SoA kernels as the serial msbmv_contig.
 * The scalar per-row row-gather this replaces left the threaded path un-SIMD'd
 * while serial was SIMD, so 4 scalar threads barely beat 1 SIMD thread
 * (par4/par1 ~0.83). Each slot is only touched over a band-width window around
 * its column range, so the fold sums just those windows into y (alpha applied at
 * the fold); adjacent windows overlap by K and each column's contribution lives
 * in exactly one slot, so summing the overlaps is correct. Reorders the per-row
 * sum vs serial -> within DD fuzz tol; serial stays bit-exact. */
static bool msbmv_axpydot(bool upper, int n, int k, const T *a, std::size_t lda,
                          const T *x, T alpha, T *y, int nthreads)
{
    std::ptrdiff_t range[MSBMV_MAX_CPUS + 1];
    /* equal-width column split: band work per column is uniform (mask3/min4 keep
     * slot writes cache-line aligned). */
    int num_cpu = mf_omp::band_bounds(n, nthreads, 3, 4, MSBMV_MAX_CPUS, range);
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
                const T *aj = &A_(0, j);
                const T t1 = x[j];
                const std::ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                const std::ptrdiff_t len = j - i_lo;
                const T *col = &aj[k - j + i_lo];   /* A_(K-j+i_lo, j), contiguous */
                T t2 = zero_dd;
                if (len > 0) {
                    mf_kernels::axpy_add(len, &slot[i_lo], col, t1);
                    t2 = mf_kernels::dot(len, col, &x[i_lo]);
                }
                slot[j] = slot[j] + (t1 * aj[k] + t2);
            }
        } else {
            for (std::ptrdiff_t j = m_from; j < m_to; ++j) {
                const T *aj = &A_(0, j);
                const T t1 = x[j];
                slot[j] = slot[j] + t1 * aj[0];
                const std::ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                const std::ptrdiff_t len = i_hi - (j + 1);
                if (len > 0) {
                    mf_kernels::axpy_add(len, &slot[j + 1], &aj[1], t1);
                    slot[j] = slot[j] + mf_kernels::dot(len, &aj[1], &x[j + 1]);
                }
            }
        }
    }

    for (int t = 0; t < num_cpu; ++t) {
        const T *slot = buf + (std::size_t)t * n;
        std::ptrdiff_t lo, hi;
        mf_omp::band_row_window(t, upper, range, n, k, lo, hi);
        for (std::ptrdiff_t i = lo; i < hi; ++i) y[i] = y[i] + alpha * slot[i];
    }
    std::free(buf);
    return true;
}

/* Threaded symmetric band matvec. Returns true if handled. Beta-scaling already
 * applied by caller. */
__attribute__((noinline)) static bool msbmv_omp(
    bool upper, int n, int k, const T *a, std::size_t lda,
    const T *x, int incx, T alpha, T *y, int incy)
{
    if (n < MSBMV_OMP_MIN || !blas_omp_available() || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MSBMV_MAX_CPUS) nthreads = MSBMV_MAX_CPUS;

    if (incx < 0) x -= (std::ptrdiff_t)(n - 1) * incx;
    if (incy < 0) y -= (std::ptrdiff_t)(n - 1) * incy;

    if (incx == 1 && incy == 1)
        return msbmv_axpydot(upper, n, k, a, lda, x, alpha, y, nthreads);

    /* Strided: gather x AND y to contiguous scratch, run the SIMD'd threaded
     * core, scatter y back — O(N) gather vs the scalar per-row row-gather, and
     * the core is SIMD. Mirrors the serial strided path but threaded. */
    T *xs = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
    T *ys = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
    if (!xs || !ys) { std::free(xs); std::free(ys); return false; }
    for (int i = 0; i < n; ++i) {
        xs[i] = x[(std::ptrdiff_t)i * incx];
        ys[i] = y[(std::ptrdiff_t)i * incy];
    }
    bool ok = msbmv_axpydot(upper, n, k, a, lda, xs, alpha, ys, nthreads);
    if (ok)
        for (int i = 0; i < n; ++i) y[(std::ptrdiff_t)i * incy] = ys[i];
    std::free(xs);
    std::free(ys);
    return ok;
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

    if (N == 0 || (eq0(alpha) && eq1(beta))) return;

    if (!eq1(beta)) {
        int iy = (incy < 0) ? -(N - 1) * incy : 0;
        if (eq0(beta)) {
            for (int i = 0; i < N; ++i) { y[iy] = zero_dd; iy += incy; }
        } else {
            for (int i = 0; i < N; ++i) { y[iy] = beta * y[iy]; iy += incy; }
        }
    }
    if (eq0(alpha)) return;

#ifdef _OPENMP
    if (N >= MSBMV_OMP_MIN && blas_omp_available()
        && msbmv_omp(UPLO == 'U', N, K, a, lda, x, incx, alpha, y, incy))
        return;
#endif

    if (incx == 1 && incy == 1) {
        msbmv_contig(UPLO == 'U', N, K, a, (std::size_t)lda, x, alpha, y);
        return;
    }

    /* Strided: gather x and the beta-scaled y into contiguous scratch, run the
     * SIMD contiguous core (y += alpha*A*x), scatter y back. O(N) gather/scatter
     * vs O(N*K) band work; the in-place strided walk is the alloc-fail fallback. */
    const std::ptrdiff_t bx = (incx < 0) ? -(std::ptrdiff_t)(N - 1) * incx : 0;
    const std::ptrdiff_t by = (incy < 0) ? -(std::ptrdiff_t)(N - 1) * incy : 0;
    T *xs = static_cast<T *>(std::malloc((std::size_t)N * sizeof(T)));
    T *ys = static_cast<T *>(std::malloc((std::size_t)N * sizeof(T)));
    if (xs && ys) {
        for (int i = 0; i < N; ++i) {
            xs[i] = x[bx + (std::ptrdiff_t)i * incx];
            ys[i] = y[by + (std::ptrdiff_t)i * incy];
        }
        msbmv_contig(UPLO == 'U', N, K, a, (std::size_t)lda, xs, alpha, ys);
        for (int i = 0; i < N; ++i) y[by + (std::ptrdiff_t)i * incy] = ys[i];
        std::free(xs); std::free(ys);
        return;
    }
    std::free(xs); std::free(ys);

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

#undef A_
