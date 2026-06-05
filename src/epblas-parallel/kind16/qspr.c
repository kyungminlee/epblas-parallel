/*
 * qspr — kind16 (__float128) symmetric packed rank-1 update.
 *   A := alpha*x*x^T + A
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define QSPR_OMP_MIN 64

typedef __float128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

void qspr_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *restrict x, const int *incx_,
    T *restrict ap,
    size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_;
    const T alpha = *alpha_;
    const T zero = 0.0Q;
    const char UPLO = up(uplo);

    if (N == 0 || alpha == zero) return;

    if (incx == 1) {
        /* Columns are independent in packed storage when accessed via kk(j). */
        if (UPLO == 'U') {
#ifdef _OPENMP
            const int use_omp = (N >= QSPR_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static, 8)
#endif
            for (int j = 0; j < N; ++j) {
                if (x[j] != zero) {
                    const T tmp = alpha * x[j];
                    const int kk = (j * (j + 1)) / 2;
                    for (int i = 0; i <= j; ++i) ap[kk + i] += x[i] * tmp;
                }
            }
        } else {
#ifdef _OPENMP
            const int use_omp = (N >= QSPR_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static, 8)
#endif
            for (int j = 0; j < N; ++j) {
                if (x[j] != zero) {
                    const T tmp = alpha * x[j];
                    /* kk0 = sum_{l=0}^{j-1}(N-l) = j*N - j*(j-1)/2 */
                    const int kk = j * N - (j * (j - 1)) / 2;
                    for (int i = j; i < N; ++i) ap[kk + (i - j)] += x[i] * tmp;
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int kk = 0;
        if (UPLO == 'U') {
            int jx = kx;
            for (int j = 0; j < N; ++j) {
                if (x[jx] != zero) {
                    const T tmp = alpha * x[jx];
                    int ix = kx;
                    for (int k = kk; k < kk + j + 1; ++k) {
                        ap[k] += x[ix] * tmp;
                        ix += incx;
                    }
                }
                jx += incx;
                kk += j + 1;
            }
        } else {
            int jx = kx;
            for (int j = 0; j < N; ++j) {
                if (x[jx] != zero) {
                    const T tmp = alpha * x[jx];
                    int ix = jx;
                    for (int k = kk; k < kk + N - j; ++k) {
                        ap[k] += x[ix] * tmp;
                        ix += incx;
                    }
                }
                jx += incx;
                kk += N - j;
            }
        }
    }
}
