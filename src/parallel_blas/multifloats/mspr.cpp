/* mspr — multifloats real DD symmetric packed rank-1 update.
 *   A := alpha*x*x^T + A
 *
 * OMP over j (columns are independent in packed storage when accessed via kk(j)).
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
#define MSPR_OMP_MIN 64
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
inline bool dd_iszero(const T &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
}

extern "C" void mspr_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *x, const int *incx_,
    T *ap,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_;
    const T alpha = *alpha_;
    const char UPLO = up(uplo);

    if (N == 0 || dd_iszero(alpha)) return;

    if (incx == 1) {
        if (UPLO == 'U') {
#ifdef _OPENMP
            const int use_omp = (N >= MSPR_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (int j = 0; j < N; ++j) {
                if (!dd_iszero(x[j])) {
                    const T tmp = alpha * x[j];
                    const int kk = (j * (j + 1)) / 2;
                    for (int i = 0; i <= j; ++i) ap[kk + i] = ap[kk + i] + x[i] * tmp;
                }
            }
        } else {
#ifdef _OPENMP
            const int use_omp = (N >= MSPR_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (int j = 0; j < N; ++j) {
                if (!dd_iszero(x[j])) {
                    const T tmp = alpha * x[j];
                    const int kk = j * N - (j * (j - 1)) / 2;
                    for (int i = j; i < N; ++i) ap[kk + (i - j)] = ap[kk + (i - j)] + x[i] * tmp;
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int kk = 0;
        if (UPLO == 'U') {
            int jx = kx;
            for (int j = 0; j < N; ++j) {
                if (!dd_iszero(x[jx])) {
                    const T tmp = alpha * x[jx];
                    int ix = kx;
                    for (int k = kk; k < kk + j + 1; ++k) {
                        ap[k] = ap[k] + x[ix] * tmp;
                        ix += incx;
                    }
                }
                jx += incx;
                kk += j + 1;
            }
        } else {
            int jx = kx;
            for (int j = 0; j < N; ++j) {
                if (!dd_iszero(x[jx])) {
                    const T tmp = alpha * x[jx];
                    int ix = jx;
                    for (int k = kk; k < kk + N - j; ++k) {
                        ap[k] = ap[k] + x[ix] * tmp;
                        ix += incx;
                    }
                }
                jx += incx;
                kk += N - j;
            }
        }
    }
}
