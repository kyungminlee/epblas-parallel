/* whpr2 — multifloats complex DD Hermitian packed rank-2 update.
 *   A := alpha*x*y^H + conj(alpha)*y*x^H + A
 *
 * Columns independent → OMP over j.
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
#define WHPR2_OMP_MIN 64
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
const R rzero{0.0, 0.0};
inline bool dd_iszero(const R &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool cdd_iszero(const T &x) { return dd_iszero(x.re) && dd_iszero(x.im); }
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }
}

extern "C" void whpr2_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *x, const int *incx_,
    const T *y, const int *incy_,
    T *ap,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, incy = *incy_;
    const T alpha = *alpha_;
    const char UPLO = up(uplo);

    if (N == 0 || cdd_iszero(alpha)) return;

    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
#ifdef _OPENMP
            const int use_omp = (N >= WHPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (int j = 0; j < N; ++j) {
                const int kk = (j * (j + 1)) / 2;
                if (!cdd_iszero(x[j]) || !cdd_iszero(y[j])) {
                    const T t1 = cmul(alpha, cconj(y[j]));
                    const T t2 = cconj(cmul(alpha, x[j]));
                    for (int i = 0; i < j; ++i)
                        ap[kk + i] = cadd(ap[kk + i], cadd(cmul(x[i], t1), cmul(y[i], t2)));
                    const T prod = cadd(cmul(x[j], t1), cmul(y[j], t2));
                    ap[kk + j] = T{ ap[kk + j].re + prod.re, rzero };
                } else {
                    ap[kk + j] = T{ ap[kk + j].re, rzero };
                }
            }
        } else {
#ifdef _OPENMP
            const int use_omp = (N >= WHPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (int j = 0; j < N; ++j) {
                const int kk = j * N - (j * (j - 1)) / 2;
                if (!cdd_iszero(x[j]) || !cdd_iszero(y[j])) {
                    const T t1 = cmul(alpha, cconj(y[j]));
                    const T t2 = cconj(cmul(alpha, x[j]));
                    const T prod = cadd(cmul(x[j], t1), cmul(y[j], t2));
                    ap[kk] = T{ ap[kk].re + prod.re, rzero };
                    for (int i = j + 1; i < N; ++i)
                        ap[kk + (i - j)] = cadd(ap[kk + (i - j)], cadd(cmul(x[i], t1), cmul(y[i], t2)));
                } else {
                    ap[kk] = T{ ap[kk].re, rzero };
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        int kk = 0;
        int jx = kx, jy = ky;
        if (UPLO == 'U') {
            for (int j = 0; j < N; ++j) {
                if (!cdd_iszero(x[jx]) || !cdd_iszero(y[jy])) {
                    const T t1 = cmul(alpha, cconj(y[jy]));
                    const T t2 = cconj(cmul(alpha, x[jx]));
                    int ix = kx, iy = ky;
                    for (int k = kk; k < kk + j; ++k) {
                        ap[k] = cadd(ap[k], cadd(cmul(x[ix], t1), cmul(y[iy], t2)));
                        ix += incx; iy += incy;
                    }
                    const T prod = cadd(cmul(x[jx], t1), cmul(y[jy], t2));
                    ap[kk + j] = T{ ap[kk + j].re + prod.re, rzero };
                } else {
                    ap[kk + j] = T{ ap[kk + j].re, rzero };
                }
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            for (int j = 0; j < N; ++j) {
                if (!cdd_iszero(x[jx]) || !cdd_iszero(y[jy])) {
                    const T t1 = cmul(alpha, cconj(y[jy]));
                    const T t2 = cconj(cmul(alpha, x[jx]));
                    const T prod = cadd(cmul(x[jx], t1), cmul(y[jy], t2));
                    ap[kk] = T{ ap[kk].re + prod.re, rzero };
                    int ix = jx, iy = jy;
                    for (int k = kk + 1; k < kk + N - j; ++k) {
                        ix += incx; iy += incy;
                        ap[k] = cadd(ap[k], cadd(cmul(x[ix], t1), cmul(y[iy], t2)));
                    }
                } else {
                    ap[kk] = T{ ap[kk].re, rzero };
                }
                jx += incx; jy += incy;
                kk += N - j;
            }
        }
    }
}
