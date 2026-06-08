/* msyr — multifloats real DD symmetric rank-1 update. */

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
#define MSYR_OMP_MIN 64
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
inline bool dd_iszero(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

extern "C" void msyr_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *x, const int *incx_,
    T *a, const int *lda_,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, lda = *lda_;
    const T alpha = *alpha_;
    const char UPLO = up(uplo);

    if (N == 0 || dd_iszero(alpha)) return;

    if (incx == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= MSYR_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const T xj = x[j];
            if (!dd_iszero(xj)) {
                const T t = alpha * xj;
                T *aj = &A_(0, j);
                if (UPLO == 'L') for (int i = j; i < N; ++i) aj[i] = aj[i] + t * x[i];
                else             for (int i = 0; i <= j; ++i) aj[i] = aj[i] + t * x[i];
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        for (int j = 0; j < N; ++j) {
            const T xj = x[kx + j * incx];
            if (!dd_iszero(xj)) {
                const T t = alpha * xj;
                if (UPLO == 'L') {
                    for (int i = j; i < N; ++i) A_(i, j) = A_(i, j) + t * x[kx + i * incx];
                } else {
                    for (int i = 0; i <= j; ++i) A_(i, j) = A_(i, j) + t * x[kx + i * incx];
                }
            }
        }
    }
}

#undef A_
