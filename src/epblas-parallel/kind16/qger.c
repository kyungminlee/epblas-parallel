/*
 * qger — kind16 (__float128) rank-1 update.
 *   A := alpha · x · yᵀ + A
 */

#include <stddef.h>
#include <stdlib.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

#define QGER_OMP_MIN 64

typedef __float128 TR;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* Contiguous (incx=incy=1) rank-1 core.  Columns of A are disjoint, work is
 * uniform m per column → plain schedule(static) over j balances. */
static void qger_contig(ptrdiff_t m, ptrdiff_t n, TR alpha,
                        const TR *restrict x, const TR *restrict y,
                        TR *restrict a, ptrdiff_t lda)
{
    const TR zero = 0.0Q;
#ifdef _OPENMP
    const bool use_omp = (n >= QGER_OMP_MIN && blas_omp_should_thread());
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (ptrdiff_t j = 0; j < n; ++j) {
        const TR yj = y[j];
        if (yj != zero) {
            const TR t = alpha * yj;
            TR *aj = &A_(0, j);
            for (ptrdiff_t i = 0; i < m; ++i) aj[i] += t * x[i];
        }
    }
}

void qger_core(
    ptrdiff_t m, ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict x, ptrdiff_t incx,
    const TR *restrict y, ptrdiff_t incy,
    TR *restrict a, ptrdiff_t lda)
{
    const TR alpha = *alpha_;
    const TR zero = 0.0Q;

    if (m == 0 || n == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
        qger_contig(m, n, alpha, x, y, a, lda);
        return;
    }

    /* General stride: gather x (length m) and y (length n) into contiguous
     * scratch and run the stride-1 core — removing the strided x reload from the
     * inner i-loop.  x/y read-only, no scatter-back; gather is O(m+n) vs O(m*n)
     * work.  Direct strided walk is the alloc-failure fallback.  The strided
     * path already threaded; gather only sharpens the inner loop.  See
     * project_l2_strided_gather. */
    {
        const ptrdiff_t ix0 = (incx < 0) ? -(m - 1) * incx : 0;
        const ptrdiff_t jy0 = (incy < 0) ? -(n - 1) * incy : 0;
        /* Exact fit: the gather writes xc[0..m-1] and yc[0..n-1] with
         * yc = stackbuf + m (max offset m+n-1), so the threshold and the
         * array length must move together. */
        enum { QGER_STACK_MN = 512 };
        TR stackbuf[QGER_STACK_MN];
        _Static_assert(QGER_STACK_MN * sizeof(TR) <= sizeof(stackbuf),
                       "qger stack-gather threshold exceeds stackbuf");
        TR *heap = NULL;
        TR *xc, *yc;
        if (m + n <= QGER_STACK_MN) {
            xc = stackbuf; yc = stackbuf + m;
        } else {
            heap = (TR *)malloc((size_t)(m + n) * sizeof(TR));
            xc = heap; yc = heap ? heap + m : NULL;
        }
        if (xc && yc) {
            ptrdiff_t ix = ix0;
            for (ptrdiff_t i = 0; i < m; ++i) { xc[i] = x[ix]; ix += incx; }
            ptrdiff_t iy = jy0;
            for (ptrdiff_t j = 0; j < n; ++j) { yc[j] = y[iy]; iy += incy; }
            qger_contig(m, n, alpha, xc, yc, a, lda);
            free(heap);
            return;
        }
        free(heap);
    }

    /* Direct strided fallback (scratch alloc failed). Columns of A are disjoint
     * → OMP-over-j race-free and bit-exact (jy recomputed as jy0 + j*incy). */
    {
        const ptrdiff_t jy0 = (incy < 0) ? -(n - 1) * incy : 0;
        const ptrdiff_t ix0 = (incx < 0) ? -(m - 1) * incx : 0;
#ifdef _OPENMP
        const bool use_omp = (n >= QGER_OMP_MIN && blas_omp_should_thread());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            const TR yj = y[jy0 + j * incy];
            if (yj != zero) {
                const TR t = alpha * yj;
                ptrdiff_t ix = ix0;
                TR *aj = &A_(0, j);
                for (ptrdiff_t i = 0; i < m; ++i) {
                    aj[i] += t * x[ix];
                    ix += incx;
                }
            }
        }
    }
}


EPBLAS_FACADE_GER(qger, TR)

#undef A_
