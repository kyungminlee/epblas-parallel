/*
 * yhpr — kind10 complex Hermitian packed rank-1 update.
 *   A := alpha*x*x^H + A, alpha real.
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define YHPR_OMP_MIN 64

typedef _Complex long double T;
typedef long double TR;
static inline T cconj(T z) { return ~z; }

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

void yhpr_(
    const char *uplo,
    const int *n_,
    const TR *alpha_,
    const T *restrict x, const int *incx_,
    T *restrict ap,
    size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_;
    const TR alpha = *alpha_;
    const TR rzero = 0.0L;
    const T czero = 0.0L + 0.0Li;
    const char UPLO = up(uplo);

    if (N == 0 || alpha == rzero) return;

    if (incx == 1) {
        if (UPLO == 'U') {
#ifdef _OPENMP
            const int use_omp = (N >= YHPR_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (int j = 0; j < N; ++j) {
                const int kk = (j * (j + 1)) / 2;
                if (x[j] != czero) {
                    const T tmp = alpha * cconj(x[j]);
                    for (int i = 0; i < j; ++i) ap[kk + i] += x[i] * tmp;
                    ap[kk + j] = (TR)__real__ ap[kk + j] + (TR)__real__ (x[j] * tmp);
                } else {
                    ap[kk + j] = (TR)__real__ ap[kk + j];
                }
            }
        } else {
#ifdef _OPENMP
            const int use_omp = (N >= YHPR_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (int j = 0; j < N; ++j) {
                const int kk = j * N - (j * (j - 1)) / 2;
                if (x[j] != czero) {
                    const T tmp = alpha * cconj(x[j]);
                    ap[kk] = (TR)__real__ ap[kk] + (TR)__real__ (tmp * x[j]);
                    for (int i = j + 1; i < N; ++i) ap[kk + (i - j)] += x[i] * tmp;
                } else {
                    ap[kk] = (TR)__real__ ap[kk];
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int kk = 0;
        if (UPLO == 'U') {
            int jx = kx;
            for (int j = 0; j < N; ++j) {
                if (x[jx] != czero) {
                    const T tmp = alpha * cconj(x[jx]);
                    int ix = kx;
                    for (int k = kk; k < kk + j; ++k) {
                        ap[k] += x[ix] * tmp;
                        ix += incx;
                    }
                    ap[kk + j] = (TR)__real__ ap[kk + j] + (TR)__real__ (x[jx] * tmp);
                } else {
                    ap[kk + j] = (TR)__real__ ap[kk + j];
                }
                jx += incx;
                kk += j + 1;
            }
        } else {
            int jx = kx;
            for (int j = 0; j < N; ++j) {
                if (x[jx] != czero) {
                    const T tmp = alpha * cconj(x[jx]);
                    ap[kk] = (TR)__real__ ap[kk] + (TR)__real__ (tmp * x[jx]);
                    int ix = jx;
                    for (int k = kk + 1; k < kk + N - j; ++k) {
                        ix += incx;
                        ap[k] += x[ix] * tmp;
                    }
                } else {
                    ap[kk] = (TR)__real__ ap[kk];
                }
                jx += incx;
                kk += N - j;
            }
        }
    }
}
