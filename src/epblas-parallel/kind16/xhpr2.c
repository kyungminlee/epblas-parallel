/*
 * xhpr2 — kind16 complex Hermitian packed rank-2 update.
 *   A := alpha*x*y^H + conj(alpha)*y*x^H + A
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

#define XHPR2_OMP_MIN 64

typedef __complex128 TC;
typedef __float128 TR;


void xhpr2_core(
    char uplo,
    ptrdiff_t n,
    const TC *alpha_,
    const TC *restrict x, ptrdiff_t incx,
    const TC *restrict y, ptrdiff_t incy,
    TC *restrict ap)
{
    const TC alpha = *alpha_;
    const TC zero = 0.0Q + 0.0Qi;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
#ifdef _OPENMP
            const bool use_omp = (n >= XHPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (ptrdiff_t j = 0; j < n; ++j) {
                const ptrdiff_t kk = (j * (j + 1)) / 2;
                if (x[j] != zero || y[j] != zero) {
                    const TC t1 = alpha * conjq(y[j]);
                    const TC t2 = conjq(alpha * x[j]);
                    for (ptrdiff_t i = 0; i < j; ++i) ap[kk + i] += x[i] * t1 + y[i] * t2;
                    ap[kk + j] = (TR)crealq(ap[kk + j]) + (TR)crealq(x[j] * t1 + y[j] * t2);
                } else {
                    ap[kk + j] = (TR)crealq(ap[kk + j]);
                }
            }
        } else {
#ifdef _OPENMP
            const bool use_omp = (n >= XHPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (ptrdiff_t j = 0; j < n; ++j) {
                const ptrdiff_t kk = j * n - (j * (j - 1)) / 2;
                if (x[j] != zero || y[j] != zero) {
                    const TC t1 = alpha * conjq(y[j]);
                    const TC t2 = conjq(alpha * x[j]);
                    ap[kk] = (TR)crealq(ap[kk]) + (TR)crealq(x[j] * t1 + y[j] * t2);
                    for (ptrdiff_t i = j + 1; i < n; ++i) ap[kk + (i - j)] += x[i] * t1 + y[i] * t2;
                } else {
                    ap[kk] = (TR)crealq(ap[kk]);
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        ptrdiff_t kk = 0;
        ptrdiff_t jx = kx, jy = ky;
        if (UPLO == 'U') {
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const TC t1 = alpha * conjq(y[jy]);
                    const TC t2 = conjq(alpha * x[jx]);
                    ptrdiff_t ix = kx, iy = ky;
                    for (ptrdiff_t k = kk; k < kk + j; ++k) {
                        ap[k] += x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                    ap[kk + j] = (TR)crealq(ap[kk + j]) + (TR)crealq(x[jx] * t1 + y[jy] * t2);
                } else {
                    ap[kk + j] = (TR)crealq(ap[kk + j]);
                }
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const TC t1 = alpha * conjq(y[jy]);
                    const TC t2 = conjq(alpha * x[jx]);
                    ap[kk] = (TR)crealq(ap[kk]) + (TR)crealq(x[jx] * t1 + y[jy] * t2);
                    ptrdiff_t ix = jx, iy = jy;
                    for (ptrdiff_t k = kk + 1; k < kk + n - j; ++k) {
                        ix += incx; iy += incy;
                        ap[k] += x[ix] * t1 + y[iy] * t2;
                    }
                } else {
                    ap[kk] = (TR)crealq(ap[kk]);
                }
                jx += incx; jy += incy;
                kk += n - j;
            }
        }
    }
}

EPBLAS_FACADE_SPR2(xhpr2, TC)
