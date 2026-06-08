/* whpr — multifloats complex DD Hermitian packed rank-1 update.
 *   A := alpha*x*x^H + A, alpha real.
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
#define WHPR_OMP_MIN 64
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
inline T scale_r(R const &alpha, T const &b) { return T{ alpha * b.re, alpha * b.im }; }
}

extern "C" void whpr_(
    const char *uplo,
    const int *n_,
    const R *alpha_,
    const T *x, const int *incx_,
    T *ap,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_;
    const R alpha = *alpha_;
    const char UPLO = up(uplo);

    if (N == 0 || dd_iszero(alpha)) return;

    if (incx == 1) {
        if (UPLO == 'U') {
#ifdef _OPENMP
            const int use_omp = (N >= WHPR_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (int j = 0; j < N; ++j) {
                const int kk = (j * (j + 1)) / 2;
                if (!cdd_iszero(x[j])) {
                    const T tmp = scale_r(alpha, cconj(x[j]));
                    for (int i = 0; i < j; ++i) ap[kk + i] = cadd(ap[kk + i], cmul(x[i], tmp));
                    const R new_re = ap[kk + j].re + cmul(x[j], tmp).re;
                    ap[kk + j] = T{ new_re, rzero };
                } else {
                    ap[kk + j] = T{ ap[kk + j].re, rzero };
                }
            }
        } else {
#ifdef _OPENMP
            const int use_omp = (N >= WHPR_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (int j = 0; j < N; ++j) {
                const int kk = j * N - (j * (j - 1)) / 2;
                if (!cdd_iszero(x[j])) {
                    const T tmp = scale_r(alpha, cconj(x[j]));
                    const R new_re = ap[kk].re + cmul(tmp, x[j]).re;
                    ap[kk] = T{ new_re, rzero };
                    for (int i = j + 1; i < N; ++i) ap[kk + (i - j)] = cadd(ap[kk + (i - j)], cmul(x[i], tmp));
                } else {
                    ap[kk] = T{ ap[kk].re, rzero };
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int kk = 0;
        if (UPLO == 'U') {
            int jx = kx;
            for (int j = 0; j < N; ++j) {
                if (!cdd_iszero(x[jx])) {
                    const T tmp = scale_r(alpha, cconj(x[jx]));
                    int ix = kx;
                    for (int k = kk; k < kk + j; ++k) {
                        ap[k] = cadd(ap[k], cmul(x[ix], tmp));
                        ix += incx;
                    }
                    const R new_re = ap[kk + j].re + cmul(x[jx], tmp).re;
                    ap[kk + j] = T{ new_re, rzero };
                } else {
                    ap[kk + j] = T{ ap[kk + j].re, rzero };
                }
                jx += incx;
                kk += j + 1;
            }
        } else {
            int jx = kx;
            for (int j = 0; j < N; ++j) {
                if (!cdd_iszero(x[jx])) {
                    const T tmp = scale_r(alpha, cconj(x[jx]));
                    const R new_re = ap[kk].re + cmul(tmp, x[jx]).re;
                    ap[kk] = T{ new_re, rzero };
                    int ix = jx;
                    for (int k = kk + 1; k < kk + N - j; ++k) {
                        ix += incx;
                        ap[k] = cadd(ap[k], cmul(x[ix], tmp));
                    }
                } else {
                    ap[kk] = T{ ap[kk].re, rzero };
                }
                jx += incx;
                kk += N - j;
            }
        }
    }
}
