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

static inline TC cconj(TC z) { return conjq(z); }
typedef __float128 TR;

/* Off-diagonal contiguous run: a[i] += x[i]*t1 + y[i]*t2 for i in [0,cnt).
 * Decomposing the complex constants into scalar real/imag locals keeps them
 * register-resident across the loop (gcc otherwise reloads the __complex128
 * temps each iteration), and reinterpreting x/y/a as 2n reals lets the eight
 * loop-invariant __float128 operands stay put — ~2-3% fewer cycles in the
 * compute-bound (L2-resident) regime, neutral when memory-bound. Product-then-
 * sum-then-accumulate order is byte-identical to gcc's inlined complex MAC, so
 * the result is bit-exact. (TFmode analogue of the kind10 yerot decompose.) */
__attribute__((noinline)) static void
xhpr2_run(ptrdiff_t cnt, TR t1r, TR t1i, TR t2r, TR t2i,
          const TC *restrict x, const TC *restrict y, TC *restrict a)
{
    const TR *restrict xr = (const TR *)x;
    const TR *restrict yr = (const TR *)y;
    TR *restrict ar = (TR *)a;
    for (ptrdiff_t i = 0; i < cnt; ++i) {
        const TR xR = xr[2*i], xI = xr[2*i+1], yR = yr[2*i], yI = yr[2*i+1];
        const TR pxr = xR*t1r - xI*t1i, pyr = yR*t2r - yI*t2i;
        const TR pxi = xR*t1i + xI*t1r, pyi = yR*t2i + yI*t2r;
        ar[2*i]   += pxr + pyr;
        ar[2*i+1] += pxi + pyi;
    }
}


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
        /* schedule(static,1): column j touches j (upper) or N-1-j (lower)
         * off-diagonal packed elements, so a contiguous static block hands one
         * thread the heavy triangle end and starves the rest (par caps at ~2x
         * on 4 cores). Cyclic static,1 interleaves short and long columns across
         * the team, balancing the skew symmetrically for both UPLO. Mirrors the
         * kind10 yhpr2 twin. */
        if (UPLO == 'U') {
#ifdef _OPENMP
            const bool use_omp = (n >= XHPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
            for (ptrdiff_t j = 0; j < n; ++j) {
                const ptrdiff_t kk = (j * (j + 1)) / 2;
                if (x[j] != zero || y[j] != zero) {
                    const TC t1 = alpha * cconj(y[j]);
                    const TC t2 = cconj(alpha * x[j]);
                    xhpr2_run(j, crealq(t1), cimagq(t1), crealq(t2), cimagq(t2), x, y, ap + kk);
                    ap[kk + j] = (TR)crealq(ap[kk + j]) + (TR)crealq(x[j] * t1 + y[j] * t2);
                } else {
                    ap[kk + j] = (TR)crealq(ap[kk + j]);
                }
            }
        } else {
#ifdef _OPENMP
            const bool use_omp = (n >= XHPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
            for (ptrdiff_t j = 0; j < n; ++j) {
                const ptrdiff_t kk = j * n - (j * (j - 1)) / 2;
                if (x[j] != zero || y[j] != zero) {
                    const TC t1 = alpha * cconj(y[j]);
                    const TC t2 = cconj(alpha * x[j]);
                    ap[kk] = (TR)crealq(ap[kk]) + (TR)crealq(x[j] * t1 + y[j] * t2);
                    xhpr2_run(n - (j + 1), crealq(t1), cimagq(t1), crealq(t2), cimagq(t2),
                              x + j + 1, y + j + 1, ap + kk + 1);
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
                    const TC t1 = alpha * cconj(y[jy]);
                    const TC t2 = cconj(alpha * x[jx]);
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
                    const TC t1 = alpha * cconj(y[jy]);
                    const TC t2 = cconj(alpha * x[jx]);
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
