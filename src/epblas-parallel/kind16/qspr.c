/*
 * qspr — kind16 (__float128) symmetric packed rank-1 update.
 *   A := alpha*x*x^T + A
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

#define QSPR_OMP_MIN 64

typedef __float128 TR;


/* Contiguous (incx=1) symmetric packed rank-1 core over all columns.  Columns
 * are independent in packed storage via kk(j).  schedule(static, 8): plain
 * static dumps the heavy triangle end on one thread (~1.8x cap); a balanced
 * schedule fixes it, and a moderate chunk dodges the false sharing cyclic
 * chunk-1 would create on the contiguous packed columns (kind10 espr twin). */
static void qspr_contig(char UPLO, ptrdiff_t n, TR alpha,
                        const TR *restrict x, TR *restrict ap)
{
    const TR zero = 0.0Q;
    if (UPLO == 'U') {
#ifdef _OPENMP
        const bool use_omp = (n >= QSPR_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static, 8)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (x[j] != zero) {
                const TR tmp = alpha * x[j];
                const ptrdiff_t kk = (j * (j + 1)) / 2;
                for (ptrdiff_t i = 0; i <= j; ++i) ap[kk + i] += x[i] * tmp;
            }
        }
    } else {
#ifdef _OPENMP
        const bool use_omp = (n >= QSPR_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static, 8)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (x[j] != zero) {
                const TR tmp = alpha * x[j];
                /* kk0 = sum_{l=0}^{j-1}(N-l) = j*N - j*(j-1)/2 */
                const ptrdiff_t kk = j * n - (j * (j - 1)) / 2;
                for (ptrdiff_t i = j; i < n; ++i) ap[kk + (i - j)] += x[i] * tmp;
            }
        }
    }
}

void qspr_core(
    char uplo,
    ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict x, ptrdiff_t incx,
    TR *restrict ap)
{
    const TR alpha = *alpha_;
    const TR zero = 0.0Q;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == zero) return;

    if (incx == 1) {
        qspr_contig(UPLO, n, alpha, x, ap);
        return;
    }

    /* General stride: gather x into contiguous scratch and run the stride-1 core
     * (which threads).  x read-only (only ap written), no scatter-back; O(N)
     * gather vs O(N^2) work.  Direct strided walk is the alloc-failure fallback.
     * See project_l2_strided_gather. */
    {
        const ptrdiff_t kx0 = (incx < 0) ? -(n - 1) * incx : 0;
        /* Exact fit: the gather writes xc[0..n-1], so the threshold and
         * the array length must move together. */
        enum { QSPR_STACK_N = 256 };
        TR stackbuf[QSPR_STACK_N];
        _Static_assert(QSPR_STACK_N * sizeof(TR) <= sizeof(stackbuf),
                       "qspr stack-gather threshold exceeds stackbuf");
        TR *heap = NULL;
        TR *xc;
        if (n <= QSPR_STACK_N) {
            xc = stackbuf;
        } else {
            heap = (TR *)malloc((size_t)n * sizeof(TR));
            xc = heap;
        }
        if (xc) {
            ptrdiff_t ix = kx0;
            for (ptrdiff_t k = 0; k < n; ++k) { xc[k] = x[ix]; ix += incx; }
            qspr_contig(UPLO, n, alpha, xc, ap);
            free(heap);
            return;
        }
        free(heap);
    }

    {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t kk = 0;
        if (UPLO == 'U') {
            ptrdiff_t jx = kx;
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != zero) {
                    const TR tmp = alpha * x[jx];
                    ptrdiff_t ix = kx;
                    for (ptrdiff_t k = kk; k < kk + j + 1; ++k) {
                        ap[k] += x[ix] * tmp;
                        ix += incx;
                    }
                }
                jx += incx;
                kk += j + 1;
            }
        } else {
            ptrdiff_t jx = kx;
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != zero) {
                    const TR tmp = alpha * x[jx];
                    ptrdiff_t ix = jx;
                    for (ptrdiff_t k = kk; k < kk + n - j; ++k) {
                        ap[k] += x[ix] * tmp;
                        ix += incx;
                    }
                }
                jx += incx;
                kk += n - j;
            }
        }
    }
}

EPBLAS_FACADE_SPR(qspr, TR, TR)
