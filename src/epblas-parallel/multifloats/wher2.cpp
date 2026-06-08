/* wher2 — multifloats Hermitian rank-2 update (alpha complex, diag real). */

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
#define WHER2_OMP_MIN 64
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
const R rzero{0.0, 0.0};
inline bool dd_iszero(R x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool cdd_iszero(const T &x) { return dd_iszero(x.re) && dd_iszero(x.im); }
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

extern "C" void wher2_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *x, const int *incx_,
    const T *y, const int *incy_,
    T *a, const int *lda_,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, incy = *incy_, lda = *lda_;
    const T alpha = *alpha_;
    const char UPLO = up(uplo);

    if (N == 0 || cdd_iszero(alpha)) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= WHER2_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const T xj = x[j], yj = y[j];
            if (!cdd_iszero(xj) || !cdd_iszero(yj)) {
                const T temp1 = cmul(alpha, cconj(yj));
                const T temp2 = cconj(cmul(alpha, xj));
                T *aj = &A_(0, j);
                if (UPLO == 'L') {
                    for (int i = j + 1; i < N; ++i)
                        aj[i] = cadd(aj[i], cadd(cmul(x[i], temp1), cmul(y[i], temp2)));
                    T prod = cadd(cmul(x[j], temp1), cmul(y[j], temp2));
                    aj[j] = T{ aj[j].re + prod.re, rzero };
                } else {
                    for (int i = 0; i < j; ++i)
                        aj[i] = cadd(aj[i], cadd(cmul(x[i], temp1), cmul(y[i], temp2)));
                    T prod = cadd(cmul(x[j], temp1), cmul(y[j], temp2));
                    aj[j] = T{ aj[j].re + prod.re, rzero };
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        for (int j = 0; j < N; ++j) {
            const T xj = x[kx + j * incx];
            const T yj = y[ky + j * incy];
            if (!cdd_iszero(xj) || !cdd_iszero(yj)) {
                const T temp1 = cmul(alpha, cconj(yj));
                const T temp2 = cconj(cmul(alpha, xj));
                if (UPLO == 'L') {
                    for (int i = j + 1; i < N; ++i)
                        A_(i, j) = cadd(A_(i, j), cadd(cmul(x[kx + i * incx], temp1), cmul(y[ky + i * incy], temp2)));
                    T prod = cadd(cmul(xj, temp1), cmul(yj, temp2));
                    A_(j, j) = T{ A_(j, j).re + prod.re, rzero };
                } else {
                    for (int i = 0; i < j; ++i)
                        A_(i, j) = cadd(A_(i, j), cadd(cmul(x[kx + i * incx], temp1), cmul(y[ky + i * incy], temp2)));
                    T prod = cadd(cmul(xj, temp1), cmul(yj, temp2));
                    A_(j, j) = T{ A_(j, j).re + prod.re, rzero };
                }
            }
        }
    }
}

#undef A_
