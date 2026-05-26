/* wgbmv — multifloats complex DD general band matrix-vector multiply.
 *
 * Reference algorithm + OMP over j on T/C-path only.
 */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
#define WGBMV_OMP_MIN 64
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
const R rzero{0.0, 0.0};
const T czero{ rzero, rzero };
const T cone { R{1.0, 0.0}, rzero };
inline bool dd_iszero(const R &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool cdd_iszero(const T &x) { return dd_iszero(x.re) && dd_iszero(x.im); }
inline bool cdd_isone(const T &x) {
    return x.re.limbs[0] == 1.0 && x.re.limbs[1] == 0.0
        && dd_iszero(x.im);
}
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

extern "C" void wgbmv_(
    const char *trans,
    const int *m_, const int *n_,
    const int *kl_, const int *ku_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *x, const int *incx_,
    const T *beta_,
    T *y, const int *incy_,
    std::size_t trans_len)
{
    (void)trans_len;
    const int M = *m_, N = *n_;
    const int KL = *kl_, KU = *ku_;
    const int lda = *lda_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const char TR = up(trans);
    const int noconj = (TR == 'T');

    if (M == 0 || N == 0 || (cdd_iszero(alpha) && cdd_isone(beta))) return;

    const int leny = (TR == 'N') ? M : N;
    const int lenx = (TR == 'N') ? N : M;

    if (!cdd_isone(beta)) {
        int iy = (incy < 0) ? -(leny - 1) * incy : 0;
        if (cdd_iszero(beta)) for (int i = 0; i < leny; ++i) { y[iy] = czero; iy += incy; }
        else                  for (int i = 0; i < leny; ++i) { y[iy] = cmul(beta, y[iy]); iy += incy; }
    }
    if (cdd_iszero(alpha)) return;

    if (TR == 'N' && incx == 1 && incy == 1) {
        for (int j = 0; j < N; ++j) {
            const T tmp = cmul(alpha, x[j]);
            const int i_lo = (j - KU > 0) ? (j - KU) : 0;
            const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
            const int k = KU - j;
            for (int i = i_lo; i < i_hi; ++i) y[i] = cadd(y[i], cmul(tmp, A_(k + i, j)));
        }
    } else if (TR != 'N' && incx == 1 && incy == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= WGBMV_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            T s = czero;
            const int i_lo = (j - KU > 0) ? (j - KU) : 0;
            const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
            const int k = KU - j;
            if (noconj) for (int i = i_lo; i < i_hi; ++i) s = cadd(s, cmul(A_(k + i, j), x[i]));
            else        for (int i = i_lo; i < i_hi; ++i) s = cadd(s, cmul(cconj(A_(k + i, j)), x[i]));
            y[j] = cadd(y[j], cmul(alpha, s));
        }
    } else {
        int kx = (incx < 0) ? -(lenx - 1) * incx : 0;
        int ky = (incy < 0) ? -(leny - 1) * incy : 0;
        if (TR == 'N') {
            int jx = kx;
            for (int j = 0; j < N; ++j) {
                const T tmp = cmul(alpha, x[jx]);
                int iy = ky;
                const int i_lo = (j - KU > 0) ? (j - KU) : 0;
                const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
                const int k = KU - j;
                for (int i = i_lo; i < i_hi; ++i) {
                    y[iy] = cadd(y[iy], cmul(tmp, A_(k + i, j)));
                    iy += incy;
                }
                jx += incx;
                if (j >= KU) ky += incy;
            }
        } else {
            int jy = ky;
            for (int j = 0; j < N; ++j) {
                T s = czero;
                int ix = kx;
                const int i_lo = (j - KU > 0) ? (j - KU) : 0;
                const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
                const int k = KU - j;
                if (noconj) {
                    for (int i = i_lo; i < i_hi; ++i) {
                        s = cadd(s, cmul(A_(k + i, j), x[ix]));
                        ix += incx;
                    }
                } else {
                    for (int i = i_lo; i < i_hi; ++i) {
                        s = cadd(s, cmul(cconj(A_(k + i, j)), x[ix]));
                        ix += incx;
                    }
                }
                y[jy] = cadd(y[jy], cmul(alpha, s));
                jy += incy;
                if (j >= KU) kx += incx;
            }
        }
    }
}

#undef A_
