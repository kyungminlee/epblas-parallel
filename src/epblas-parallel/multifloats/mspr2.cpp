/* mspr2 — multifloats real DD symmetric packed rank-2 update.
 *   A := alpha*x*y^T + alpha*y*x^T + A
 */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
#define MSPR2_OMP_MIN 64
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
inline bool dd_iszero(const T &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
}

extern "C" void mspr2_(
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

    if (N == 0 || dd_iszero(alpha)) return;

    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
#ifdef _OPENMP
            const int use_omp = (N >= MSPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (int j = 0; j < N; ++j) {
                if (!dd_iszero(x[j]) || !dd_iszero(y[j])) {
                    const T t1 = alpha * y[j];
                    const T t2 = alpha * x[j];
                    const int kk = (j * (j + 1)) / 2;
                    for (int i = 0; i <= j; ++i) ap[kk + i] = ap[kk + i] + x[i] * t1 + y[i] * t2;
                }
            }
        } else {
#ifdef _OPENMP
            const int use_omp = (N >= MSPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (int j = 0; j < N; ++j) {
                if (!dd_iszero(x[j]) || !dd_iszero(y[j])) {
                    const T t1 = alpha * y[j];
                    const T t2 = alpha * x[j];
                    const int kk = j * N - (j * (j - 1)) / 2;
                    for (int i = j; i < N; ++i) ap[kk + (i - j)] = ap[kk + (i - j)] + x[i] * t1 + y[i] * t2;
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
                if (!dd_iszero(x[jx]) || !dd_iszero(y[jy])) {
                    const T t1 = alpha * y[jy];
                    const T t2 = alpha * x[jx];
                    int ix = kx, iy = ky;
                    for (int k = kk; k < kk + j + 1; ++k) {
                        ap[k] = ap[k] + x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                }
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            for (int j = 0; j < N; ++j) {
                if (!dd_iszero(x[jx]) || !dd_iszero(y[jy])) {
                    const T t1 = alpha * y[jy];
                    const T t2 = alpha * x[jx];
                    int ix = jx, iy = jy;
                    for (int k = kk; k < kk + N - j; ++k) {
                        ap[k] = ap[k] + x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                }
                jx += incx; jy += incy;
                kk += N - j;
            }
        }
    }
}
