/* whbmv — multifloats complex DD Hermitian band matrix-vector multiply.
 *   y := alpha*A*x + beta*y, A Hermitian with K super-(sub-)diagonals.
 */

#include <cstddef>
#include <cctype>
#include <multifloats.h>

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
