/* whbmv — multifloats complex DD Hermitian band matrix-vector multiply.
 *   y := alpha*A*x + beta*y, A Hermitian with K super-(sub-)diagonals.
 */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <cstdlib>
#include <omp.h>
#include "../common/blas_omp.h"
#define WHBMV_OMP_MIN 256
#define WHBMV_MAX_CPUS 256
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
const R rzero{0.0, 0.0};
const T czero{ rzero, rzero };
inline bool dd_iszero(const R &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool cdd_iszero(const T &x) { return dd_iszero(x.re) && dd_iszero(x.im); }
inline bool cdd_isone(const T &x) {
    return x.re.limbs[0] == 1.0 && x.re.limbs[1] == 0.0 && dd_iszero(x.im);
}
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }
inline T cmul_r(T const &a, R const &r) { return T{ a.re * r, a.im * r }; }
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

#ifdef _OPENMP
/* Row-gather: y[i] (i in [lo,hi)) is an independent Hermitian-band dot. Each row
 * reconstructs A[i][·]: the stored-triangle neighbours are used directly, the
 * reflected neighbours conjugated (Hermitian), and the diagonal takes only the
 * real part (matching the serial). x and y are distinct (hbmv forbids aliasing)
 * → output rows partition with no cross-thread dependence. Reorders the per-row
 * summation vs the serial column scatter → within DD fuzz tol; serial bit-exact. */
static void whbmv_rowgather(bool upper, int n, int k, int lo, int hi,
                            const T *a, std::size_t lda,
                            const T *x, T alpha, T *y, int incy)
{
    const std::ptrdiff_t s1 = static_cast<std::ptrdiff_t>(lda) - 1;
    if (upper) {
        for (int i = lo; i < hi; ++i) {
            const T *base = &A_(0, i);
            T s = cmul_r(x[i], base[k].re);                  /* real diagonal */
            const int rlen = (n - 1 - i < k) ? (n - 1 - i) : k;
            for (int d = 1; d <= rlen; ++d)                  /* right: upper elem direct */
                s = cadd(s, cmul(base[k + (std::ptrdiff_t)d * s1], x[i + d]));
            const int llen = (i < k) ? i : k;
            for (int d = 1; d <= llen; ++d)                  /* left: conj of upper */
                s = cadd(s, cmul(cconj(base[k - d]), x[i - d]));
            y[(std::ptrdiff_t)i * incy] = cadd(y[(std::ptrdiff_t)i * incy], cmul(alpha, s));
        }
    } else {
        for (int i = lo; i < hi; ++i) {
            const T *base = &A_(0, i);
            T s = cmul_r(x[i], base[0].re);                  /* real diagonal */
            const int llen = (i < k) ? i : k;
            for (int d = 1; d <= llen; ++d)                  /* left: lower elem direct */
                s = cadd(s, cmul(base[-(std::ptrdiff_t)d * s1], x[i - d]));
            const int rlen = (n - 1 - i < k) ? (n - 1 - i) : k;
            for (int d = 1; d <= rlen; ++d)                  /* right: conj of lower */
                s = cadd(s, cmul(cconj(base[d]), x[i + d]));
            y[(std::ptrdiff_t)i * incy] = cadd(y[(std::ptrdiff_t)i * incy], cmul(alpha, s));
        }
    }
}

/* Threaded Hermitian band matvec. Disjoint output-row ranges; x gathered to
 * contiguous when strided. Returns true if handled. Beta already applied. */
__attribute__((noinline)) static bool whbmv_omp(
    bool upper, int n, int k, const T *a, std::size_t lda,
    const T *x, int incx, T alpha, T *y, int incy)
{
    if (n < WHBMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > WHBMV_MAX_CPUS) nthreads = WHBMV_MAX_CPUS;

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
        whbmv_rowgather(upper, n, k, lo, hi, a, lda, xptr, alpha, y, incy);
    }
    std::free(xbuf);
    return true;
}
#endif

extern "C" void whbmv_(
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

    if (N == 0 || (cdd_iszero(alpha) && cdd_isone(beta))) return;

    if (!cdd_isone(beta)) {
        int iy = (incy < 0) ? -(N - 1) * incy : 0;
        if (cdd_iszero(beta)) for (int i = 0; i < N; ++i) { y[iy] = czero; iy += incy; }
        else                  for (int i = 0; i < N; ++i) { y[iy] = cmul(beta, y[iy]); iy += incy; }
    }
    if (cdd_iszero(alpha)) return;

#ifdef _OPENMP
    if (N >= WHBMV_OMP_MIN && blas_omp_max_threads() > 1
        && whbmv_omp(UPLO == 'U', N, K, a, lda, x, incx, alpha, y, incy))
        return;
#endif

    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
            for (int j = 0; j < N; ++j) {
                const T t1 = cmul(alpha, x[j]);
                T t2 = czero;
                const int L = K - j;
                const int i_lo = (j - K > 0) ? (j - K) : 0;
                for (int i = i_lo; i < j; ++i) {
                    y[i] = cadd(y[i], cmul(t1, A_(L + i, j)));
                    t2 = cadd(t2, cmul(cconj(A_(L + i, j)), x[i]));
                }
                y[j] = cadd(y[j], cadd(cmul_r(t1, A_(K, j).re), cmul(alpha, t2)));
            }
        } else {
            for (int j = 0; j < N; ++j) {
                const T t1 = cmul(alpha, x[j]);
                T t2 = czero;
                y[j] = cadd(y[j], cmul_r(t1, A_(0, j).re));
                const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                for (int i = j + 1; i < i_hi; ++i) {
                    y[i] = cadd(y[i], cmul(t1, A_(i - j, j)));
                    t2 = cadd(t2, cmul(cconj(A_(i - j, j)), x[i]));
                }
                y[j] = cadd(y[j], cmul(alpha, t2));
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        if (UPLO == 'U') {
            int jx = kx, jy = ky;
            for (int j = 0; j < N; ++j) {
                const T t1 = cmul(alpha, x[jx]);
                T t2 = czero;
                int ix = kx, iy = ky;
                const int L = K - j;
                const int i_lo = (j - K > 0) ? (j - K) : 0;
                for (int i = i_lo; i < j; ++i) {
                    y[iy] = cadd(y[iy], cmul(t1, A_(L + i, j)));
                    t2 = cadd(t2, cmul(cconj(A_(L + i, j)), x[ix]));
                    ix += incx; iy += incy;
                }
                y[jy] = cadd(y[jy], cadd(cmul_r(t1, A_(K, j).re), cmul(alpha, t2)));
                jx += incx; jy += incy;
                if (j >= K) { kx += incx; ky += incy; }
            }
        } else {
            int jx = kx, jy = ky;
            for (int j = 0; j < N; ++j) {
                const T t1 = cmul(alpha, x[jx]);
                T t2 = czero;
                y[jy] = cadd(y[jy], cmul_r(t1, A_(0, j).re));
                int ix = jx, iy = jy;
                const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                for (int i = j + 1; i < i_hi; ++i) {
                    ix += incx; iy += incy;
                    y[iy] = cadd(y[iy], cmul(t1, A_(i - j, j)));
                    t2 = cadd(t2, cmul(cconj(A_(i - j, j)), x[ix]));
                }
                y[jy] = cadd(y[jy], cmul(alpha, t2));
                jx += incx; jy += incy;
            }
        }
    }
}

#undef A_
