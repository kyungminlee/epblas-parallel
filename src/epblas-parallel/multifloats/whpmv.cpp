/* whpmv — multifloats complex DD Hermitian packed matrix-vector multiply. */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <cstdlib>
#include <omp.h>
#include "../common/blas_omp.h"
#define WHPMV_OMP_MIN 256
#define WHPMV_MAX_CPUS 256
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

#ifdef _OPENMP
/* Row-gather: y[i] (i in [lo,hi)) is an independent Hermitian-packed dot. The
 * stored-triangle neighbours are used directly, the reflected ones conjugated,
 * the diagonal takes only the real part (matching the serial). Per-row offsets:
 * UPLO='U' column j diagonal at ap[j(j+1)/2 + j]; UPLO='L' at ap[j*N - j(j-1)/2].
 * x and y are distinct (hpmv forbids aliasing) → output rows partition with no
 * cross-thread dependence. Reorders summation → within DD fuzz tol; serial exact. */
static void whpmv_rowgather(bool upper, int n, int lo, int hi,
                            const T *ap, const T *x, T alpha, T *y, int incy)
{
    if (upper) {
        for (int i = lo; i < hi; ++i) {
            const std::size_t kk_i = static_cast<std::size_t>(i) * (i + 1) / 2;
            T s = cmul_r(x[i], ap[kk_i + i].re);             /* real diagonal */
            for (int j = 0; j < i; ++j)                      /* left: conj of upper (col i) */
                s = cadd(s, cmul(cconj(ap[kk_i + j]), x[j]));
            std::size_t kk_j = kk_i + (i + 1);               /* start of column i+1 */
            for (int j = i + 1; j < n; ++j) {                /* right: upper direct (col-jump) */
                s = cadd(s, cmul(ap[kk_j + i], x[j]));
                kk_j += static_cast<std::size_t>(j + 1);
            }
            y[(std::ptrdiff_t)i * incy] = cadd(y[(std::ptrdiff_t)i * incy], cmul(alpha, s));
        }
    } else {
        for (int i = lo; i < hi; ++i) {
            const std::size_t kk_i =
                static_cast<std::size_t>(i) * n - static_cast<std::size_t>(i) * (i - 1) / 2;
            T s = cmul_r(x[i], ap[kk_i].re);                 /* real diagonal */
            for (int j = i + 1; j < n; ++j)                  /* right: conj of lower (col i) */
                s = cadd(s, cmul(cconj(ap[kk_i + (j - i)]), x[j]));
            std::size_t kk_j = 0;                            /* start of column 0 */
            for (int j = 0; j < i; ++j) {                    /* left: lower direct (col-jump) */
                s = cadd(s, cmul(ap[kk_j + (i - j)], x[j]));
                kk_j += static_cast<std::size_t>(n - j);
            }
            y[(std::ptrdiff_t)i * incy] = cadd(y[(std::ptrdiff_t)i * incy], cmul(alpha, s));
        }
    }
}

/* Threaded Hermitian packed matvec. Disjoint output-row ranges; x gathered to
 * contiguous when strided. Returns true if handled. Beta already applied. */
__attribute__((noinline)) static bool whpmv_omp(
    bool upper, int n, const T *ap,
    const T *x, int incx, T alpha, T *y, int incy)
{
    if (n < WHPMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > WHPMV_MAX_CPUS) nthreads = WHPMV_MAX_CPUS;

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
        whpmv_rowgather(upper, n, lo, hi, ap, xptr, alpha, y, incy);
    }
    std::free(xbuf);
    return true;
}
#endif

extern "C" void whpmv_(
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

    if (N == 0 || (cdd_iszero(alpha) && cdd_isone(beta))) return;

    if (!cdd_isone(beta)) {
        int iy = (incy < 0) ? -(N - 1) * incy : 0;
        if (cdd_iszero(beta)) for (int i = 0; i < N; ++i) { y[iy] = czero; iy += incy; }
        else                  for (int i = 0; i < N; ++i) { y[iy] = cmul(beta, y[iy]); iy += incy; }
    }
    if (cdd_iszero(alpha)) return;

#ifdef _OPENMP
    if (N >= WHPMV_OMP_MIN && blas_omp_max_threads() > 1
        && whpmv_omp(UPLO == 'U', N, ap, x, incx, alpha, y, incy))
        return;
#endif

    int kk = 0;
    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
            for (int j = 0; j < N; ++j) {
                const T t1 = cmul(alpha, x[j]);
                T t2 = czero;
                int k = kk;
                for (int i = 0; i < j; ++i) {
                    y[i] = cadd(y[i], cmul(t1, ap[k]));
                    t2 = cadd(t2, cmul(cconj(ap[k]), x[i]));
                    ++k;
                }
                y[j] = cadd(y[j], cadd(cmul_r(t1, ap[kk + j].re), cmul(alpha, t2)));
                kk += j + 1;
            }
        } else {
            for (int j = 0; j < N; ++j) {
                const T t1 = cmul(alpha, x[j]);
                T t2 = czero;
                y[j] = cadd(y[j], cmul_r(t1, ap[kk].re));
                int k = kk + 1;
                for (int i = j + 1; i < N; ++i) {
                    y[i] = cadd(y[i], cmul(t1, ap[k]));
                    t2 = cadd(t2, cmul(cconj(ap[k]), x[i]));
                    ++k;
                }
                y[j] = cadd(y[j], cmul(alpha, t2));
                kk += N - j;
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
                for (int k = kk; k < kk + j; ++k) {
                    y[iy] = cadd(y[iy], cmul(t1, ap[k]));
                    t2 = cadd(t2, cmul(cconj(ap[k]), x[ix]));
                    ix += incx; iy += incy;
                }
                y[jy] = cadd(y[jy], cadd(cmul_r(t1, ap[kk + j].re), cmul(alpha, t2)));
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            int jx = kx, jy = ky;
            for (int j = 0; j < N; ++j) {
                const T t1 = cmul(alpha, x[jx]);
                T t2 = czero;
                y[jy] = cadd(y[jy], cmul_r(t1, ap[kk].re));
                int ix = jx, iy = jy;
                for (int k = kk + 1; k < kk + N - j; ++k) {
                    ix += incx; iy += incy;
                    y[iy] = cadd(y[iy], cmul(t1, ap[k]));
                    t2 = cadd(t2, cmul(cconj(ap[k]), x[ix]));
                }
                y[jy] = cadd(y[jy], cmul(alpha, t2));
                jx += incx; jy += incy;
                kk += N - j;
            }
        }
    }
}
