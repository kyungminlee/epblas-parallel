/*
 * qger — kind16 (__float128) rank-1 update.
 *   A := alpha · x · yᵀ + A
 */

#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define QGER_OMP_MIN 64

typedef __float128 T;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void qger_(
    const int *m_, const int *n_,
    const T *alpha_,
    const T *restrict x, const int *incx_,
    const T *restrict y, const int *incy_,
    T *restrict a, const int *lda_)
{
    const int M = *m_, N = *n_;
    const int incx = *incx_, incy = *incy_, lda = *lda_;
    const T alpha = *alpha_;
    const T zero = 0.0Q;

    if (M == 0 || N == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= QGER_OMP_MIN && blas_omp_max_threads() > 1
                             && !omp_in_parallel());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const T yj = y[j];
            if (yj != zero) {
                const T t = alpha * yj;
                T *aj = &A_(0, j);
                for (int i = 0; i < M; ++i) aj[i] += t * x[i];
            }
        }
    } else {
        int jy = (incy < 0) ? -(N - 1) * incy : 0;
        for (int j = 0; j < N; ++j) {
            const T yj = y[jy];
            if (yj != zero) {
                const T t = alpha * yj;
                int ix = (incx < 0) ? -(M - 1) * incx : 0;
                T *aj = &A_(0, j);
                for (int i = 0; i < M; ++i) {
                    aj[i] += t * x[ix];
                    ix += incx;
                }
            }
            jy += incy;
        }
    }
}

#undef A_
