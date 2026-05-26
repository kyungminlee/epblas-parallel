/* whpmv — multifloats complex DD Hermitian packed matrix-vector multiply. */

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
