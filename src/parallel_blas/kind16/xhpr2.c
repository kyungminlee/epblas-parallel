/*
 * xhpr2 — kind16 complex Hermitian packed rank-2 update.
 *   A := alpha*x*y^H + conj(alpha)*y*x^H + A
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define XHPR2_OMP_MIN 64

typedef __complex128 T;
typedef __float128 TR;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

void xhpr2_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *restrict x, const int *incx_,
    const T *restrict y, const int *incy_,
    T *restrict ap,
    size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, incy = *incy_;
    const T alpha = *alpha_;
    const T zero = 0.0Q + 0.0Qi;
    const char UPLO = up(uplo);

    if (N == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
#ifdef _OPENMP
            const int use_omp = (N >= XHPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (int j = 0; j < N; ++j) {
                const int kk = (j * (j + 1)) / 2;
                if (x[j] != zero || y[j] != zero) {
                    const T t1 = alpha * conjq(y[j]);
                    const T t2 = conjq(alpha * x[j]);
                    for (int i = 0; i < j; ++i) ap[kk + i] += x[i] * t1 + y[i] * t2;
                    ap[kk + j] = (TR)crealq(ap[kk + j]) + (TR)crealq(x[j] * t1 + y[j] * t2);
                } else {
                    ap[kk + j] = (TR)crealq(ap[kk + j]);
                }
            }
        } else {
#ifdef _OPENMP
            const int use_omp = (N >= XHPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (int j = 0; j < N; ++j) {
                const int kk = j * N - (j * (j - 1)) / 2;
                if (x[j] != zero || y[j] != zero) {
                    const T t1 = alpha * conjq(y[j]);
                    const T t2 = conjq(alpha * x[j]);
                    ap[kk] = (TR)crealq(ap[kk]) + (TR)crealq(x[j] * t1 + y[j] * t2);
                    for (int i = j + 1; i < N; ++i) ap[kk + (i - j)] += x[i] * t1 + y[i] * t2;
                } else {
                    ap[kk] = (TR)crealq(ap[kk]);
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
                if (x[jx] != zero || y[jy] != zero) {
                    const T t1 = alpha * conjq(y[jy]);
                    const T t2 = conjq(alpha * x[jx]);
                    int ix = kx, iy = ky;
                    for (int k = kk; k < kk + j; ++k) {
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
            for (int j = 0; j < N; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const T t1 = alpha * conjq(y[jy]);
                    const T t2 = conjq(alpha * x[jx]);
                    ap[kk] = (TR)crealq(ap[kk]) + (TR)crealq(x[jx] * t1 + y[jy] * t2);
                    int ix = jx, iy = jy;
                    for (int k = kk + 1; k < kk + N - j; ++k) {
                        ix += incx; iy += incy;
                        ap[k] += x[ix] * t1 + y[iy] * t2;
                    }
                } else {
                    ap[kk] = (TR)crealq(ap[kk]);
                }
                jx += incx; jy += incy;
                kk += N - j;
            }
        }
    }
}
