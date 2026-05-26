/*
 * ygeru — kind10 complex unconjugated rank-1 update.
 *   A := alpha · x · yᵀ + A    where A is M×N
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define YGERU_OMP_MIN 64

typedef _Complex long double T;
static const T ZERO = 0.0L + 0.0Li;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void ygeru_(
    const int *m_, const int *n_,
    const T *alpha_,
    const T *restrict x, const int *incx_,
    const T *restrict y, const int *incy_,
    T *restrict a, const int *lda_)
{
    const int M = *m_, N = *n_;
    const int incx = *incx_, incy = *incy_, lda = *lda_;
    const T alpha = *alpha_;

    if (M == 0 || N == 0 || alpha == ZERO) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= YGERU_OMP_MIN && blas_omp_max_threads() > 1
                             && !omp_in_parallel());
#else
        const int use_omp = 0;
#endif
        /* C-source branch on use_omp (Add-16). */
#define YGERU_BODY                                                           \
        for (int j = 0; j < N; ++j) {                                        \
            const T yj = y[j];                                               \
            if (yj != ZERO) {                                                \
                const T t = alpha * yj;                                      \
                T *aj = &A_(0, j);                                           \
                for (int i = 0; i < M; ++i) aj[i] += t * x[i];               \
            }                                                                \
        }
        if (use_omp) {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            YGERU_BODY
        } else {
            YGERU_BODY
        }
#undef YGERU_BODY
    } else {
        int jy = (incy < 0) ? -(N - 1) * incy : 0;
        for (int j = 0; j < N; ++j) {
            const T yj = y[jy];
            if (yj != ZERO) {
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
