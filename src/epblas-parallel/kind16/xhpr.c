/*
 * xhpr — kind16 complex Hermitian packed rank-1 update.
 *   A := alpha*x*x^H + A, alpha real.
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

#define XHPR_OMP_MIN 64

typedef __complex128 TC;
typedef __float128 TR;


void xhpr_core(
    char uplo,
    ptrdiff_t n,
    const TR *alpha_,
    const TC *restrict x, ptrdiff_t incx,
    TC *restrict ap)
{
    const TR alpha = *alpha_;
    const TR rzero = 0.0Q;
    const TC czero = 0.0Q + 0.0Qi;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == rzero) return;

    if (incx == 1) {
        if (UPLO == 'U') {
#ifdef _OPENMP
            const bool use_omp = (n >= XHPR_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
            for (ptrdiff_t j = 0; j < n; ++j) {
                const ptrdiff_t kk = (j * (j + 1)) / 2;
                if (x[j] != czero) {
                    const TC tmp = alpha * conjq(x[j]);
                    for (ptrdiff_t i = 0; i < j; ++i) ap[kk + i] += x[i] * tmp;
                    ap[kk + j] = (TR)crealq(ap[kk + j]) + (TR)crealq(x[j] * tmp);
                } else {
                    ap[kk + j] = (TR)crealq(ap[kk + j]);
                }
            }
        } else {
#ifdef _OPENMP
            const bool use_omp = (n >= XHPR_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
            for (ptrdiff_t j = 0; j < n; ++j) {
                const ptrdiff_t kk = j * n - (j * (j - 1)) / 2;
                if (x[j] != czero) {
                    const TC tmp = alpha * conjq(x[j]);
                    ap[kk] = (TR)crealq(ap[kk]) + (TR)crealq(tmp * x[j]);
                    for (ptrdiff_t i = j + 1; i < n; ++i) ap[kk + (i - j)] += x[i] * tmp;
                } else {
                    ap[kk] = (TR)crealq(ap[kk]);
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t kk = 0;
        if (UPLO == 'U') {
            ptrdiff_t jx = kx;
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != czero) {
                    const TC tmp = alpha * conjq(x[jx]);
                    ptrdiff_t ix = kx;
                    for (ptrdiff_t k = kk; k < kk + j; ++k) {
                        ap[k] += x[ix] * tmp;
                        ix += incx;
                    }
                    ap[kk + j] = (TR)crealq(ap[kk + j]) + (TR)crealq(x[jx] * tmp);
                } else {
                    ap[kk + j] = (TR)crealq(ap[kk + j]);
                }
                jx += incx;
                kk += j + 1;
            }
        } else {
            ptrdiff_t jx = kx;
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != czero) {
                    const TC tmp = alpha * conjq(x[jx]);
                    ap[kk] = (TR)crealq(ap[kk]) + (TR)crealq(tmp * x[jx]);
                    ptrdiff_t ix = jx;
                    for (ptrdiff_t k = kk + 1; k < kk + n - j; ++k) {
                        ix += incx;
                        ap[k] += x[ix] * tmp;
                    }
                } else {
                    ap[kk] = (TR)crealq(ap[kk]);
                }
                jx += incx;
                kk += n - j;
            }
        }
    }
}

EPBLAS_FACADE_SPR(xhpr, TR, TC)
