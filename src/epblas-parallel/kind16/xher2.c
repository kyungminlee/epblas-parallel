/*
 * xher2 — kind16 complex Hermitian rank-2 update.
 *   A := alpha · x · yᴴ + conj(alpha) · y · xᴴ + A
 * alpha complex. Diagonal stays real.
 */

#include <stddef.h>
#include <stdlib.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

#define XHER2_OMP_MIN 64

typedef __complex128 TC;
typedef __float128 TR;

static inline TC cconj(TC z) { return conjq(z); }
#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* Off-diagonal contiguous run: a[i] += x[i]*t1 + y[i]*t2 for i in [0,cnt).
 * Decomposing the complex constants into scalar real/imag locals keeps them
 * register-resident across the loop (gcc otherwise reloads the __complex128
 * temps each iteration), and reinterpreting x/y/a as 2n reals lets the eight
 * loop-invariant __float128 operands stay put — ~2-3% fewer cycles in the
 * compute-bound (L2-resident) regime, neutral when memory-bound. The product-
 * then-sum-then-accumulate order is byte-identical to gcc's inlined complex
 * `a[i] += x[i]*t1 + y[i]*t2`, so the result is bit-exact. (TFmode analogue of
 * the kind10 yerot/ygemv decompose trick.) */
__attribute__((noinline)) static void
xher2_run(ptrdiff_t cnt, TR t1r, TR t1i, TR t2r, TR t2i,
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

/* Contiguous (incx=incy=1) Hermitian rank-2 core over all columns.
 * schedule(static,1): column j touches (N-1-j) (lower) or j (upper)
 * off-diagonal elements, so a contiguous static block hands one thread
 * the heavy triangle end and starves the rest (par caps at ~2x on 4
 * cores). Cyclic static,1 interleaves short and long columns across the
 * team, balancing the skew for both UPLO. Mirrors the kind10 yher2 twin. */
static void xher2_contig(char UPLO, ptrdiff_t n, TC alpha,
                         const TC *restrict x, const TC *restrict y,
                         TC *restrict a, ptrdiff_t lda)
{
    const TC zero = 0.0Q + 0.0Qi;
#ifdef _OPENMP
    const bool use_omp = (n >= XHER2_OMP_MIN && blas_omp_should_thread());
    #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
    for (ptrdiff_t j = 0; j < n; ++j) {
        const TC xj = x[j], yj = y[j];
        if (xj != zero || yj != zero) {
            const TC temp1 = alpha * cconj(yj);
            const TC temp2 = cconj(alpha * xj);
            const TR t1r = crealq(temp1), t1i = cimagq(temp1);
            const TR t2r = crealq(temp2), t2i = cimagq(temp2);
            TC *aj = &A_(0, j);
            if (UPLO == 'L') {
                xher2_run(n - (j + 1), t1r, t1i, t2r, t2i, x + j + 1, y + j + 1, aj + j + 1);
                aj[j] = crealq(aj[j]) + crealq(x[j] * temp1 + y[j] * temp2);
            } else {
                xher2_run(j, t1r, t1i, t2r, t2i, x, y, aj);
                aj[j] = crealq(aj[j]) + crealq(x[j] * temp1 + y[j] * temp2);
            }
        }
    }
}

void xher2_core(
    char uplo,
    ptrdiff_t n,
    const TC *alpha_,
    const TC *restrict x, ptrdiff_t incx,
    const TC *restrict y, ptrdiff_t incy,
    TC *restrict a, ptrdiff_t lda)
{
    const TC alpha = *alpha_;
    const TC zero = 0.0Q + 0.0Qi;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
        xher2_contig(UPLO, n, alpha, x, y, a, lda);
    } else {
        /* Threaded strided only: gather x/y into contiguous scratch and run
         * the stride-1 core (which threads).  x/y read-only (only A written)
         * so no scatter-back; the O(N) gather is dwarfed by the O(N^2)
         * threaded work and buys the contig core's omp scaling (the direct
         * strided walk never threaded).  Serial strided falls through to the
         * direct in-place loop (mirrors the kind10 yher2 / kind16 qsyr2
         * gates).  See project_l2_strided_gather. */
#ifdef _OPENMP
        const bool would_thread = (n >= XHER2_OMP_MIN && blas_omp_should_thread());
#else
        const bool would_thread = false;
#endif
        if (would_thread) {
            /* Exact fit: the gather writes xc[0..n-1] and yc[0..n-1] with
             * yc = stackbuf + n (max offset 2n-1), so the threshold and the
             * array length must move together. */
            enum { XHER2_STACK_N = 256 };
            TC stackbuf[2 * XHER2_STACK_N];
            _Static_assert(2 * XHER2_STACK_N * sizeof(TC) <= sizeof(stackbuf),
                           "xher2 stack-gather threshold exceeds stackbuf");
            TC *heap = NULL;
            TC *xc, *yc;
            if (n <= XHER2_STACK_N) {
                xc = stackbuf; yc = stackbuf + n;
            } else {
                heap = (TC *)malloc((size_t)2 * n * sizeof(TC));
                xc = heap; yc = heap ? heap + n : NULL;
            }
            if (xc && yc) {
                ptrdiff_t ix = (incx < 0) ? -(n - 1) * incx : 0;
                ptrdiff_t iy = (incy < 0) ? -(n - 1) * incy : 0;
                for (ptrdiff_t k = 0; k < n; ++k) {
                    xc[k] = x[ix]; yc[k] = y[iy];
                    ix += incx; iy += incy;
                }
                xher2_contig(UPLO, n, alpha, xc, yc, a, lda);
                free(heap);
                return;
            }
            free(heap);
        }
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        for (ptrdiff_t j = 0; j < n; ++j) {
            const TC xj = x[kx + (ptrdiff_t)j * incx];
            const TC yj = y[ky + (ptrdiff_t)j * incy];
            if (xj != zero || yj != zero) {
                const TC temp1 = alpha * cconj(yj);
                const TC temp2 = cconj(alpha * xj);
                if (UPLO == 'L') {
                    for (ptrdiff_t i = j + 1; i < n; ++i)
                        A_(i, j) += x[kx + (ptrdiff_t)i * incx] * temp1 + y[ky + (ptrdiff_t)i * incy] * temp2;
                    A_(j, j) = crealq(A_(j, j)) + crealq(xj * temp1 + yj * temp2);
                } else {
                    for (ptrdiff_t i = 0; i < j; ++i)
                        A_(i, j) += x[kx + (ptrdiff_t)i * incx] * temp1 + y[ky + (ptrdiff_t)i * incy] * temp2;
                    A_(j, j) = crealq(A_(j, j)) + crealq(xj * temp1 + yj * temp2);
                }
            }
        }
    }
}


EPBLAS_FACADE_SYR2(xher2, TC)

#undef A_
