/*
 * qspr2 — kind16 (__float128) symmetric packed rank-2 update.
 *   A := alpha*x*y^T + alpha*y*x^T + A
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

#define QSPR2_OMP_MIN 64

typedef __float128 TR;


/* Contiguous (incx=incy=1) symmetric packed rank-2 core over all columns.
 * schedule(static, 8): per-column work grows with j (U) / shrinks with j (L),
 * so plain static dumps the heavy triangle end on one thread (caps at ~2x); a
 * balanced schedule fixes that, and a MODERATE chunk beats cyclic chunk-1 for
 * this body because adjacent packed columns are contiguous in ap (chunk-1 puts
 * neighbour pairs on different threads → false sharing).  Mirrors the kind10
 * espr2 packed twin. */
static void qspr2_contig(char UPLO, ptrdiff_t n, TR alpha,
                         const TR *restrict x, const TR *restrict y,
                         TR *restrict ap)
{
    const TR zero = 0.0Q;
    if (UPLO == 'U') {
#ifdef _OPENMP
        const bool use_omp = (n >= QSPR2_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static, 8)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (x[j] != zero || y[j] != zero) {
                const TR t1 = alpha * y[j];
                const TR t2 = alpha * x[j];
                const ptrdiff_t kk = (j * (j + 1)) / 2;
                /* Left-to-right ((ap + x*t1) + y*t2) — netlib's order; NOT
                 * ap += (x*t1 + y*t2). Adding two products first drives the
                 * __addtf3 normalize branch into a worse-predicted path
                 * (~+95M mispredicts vs the gfortran leg). See qsyr2.c. */
                for (ptrdiff_t i = 0; i <= j; ++i) ap[kk + i] = ap[kk + i] + x[i] * t1 + y[i] * t2;
            }
        }
    } else {
#ifdef _OPENMP
        const bool use_omp = (n >= QSPR2_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static, 8)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (x[j] != zero || y[j] != zero) {
                const TR t1 = alpha * y[j];
                const TR t2 = alpha * x[j];
                const ptrdiff_t kk = j * n - (j * (j - 1)) / 2;
                for (ptrdiff_t i = j; i < n; ++i) ap[kk + (i - j)] = ap[kk + (i - j)] + x[i] * t1 + y[i] * t2;
            }
        }
    }
}

void qspr2_core(
    char uplo,
    ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict x, ptrdiff_t incx,
    const TR *restrict y, ptrdiff_t incy,
    TR *restrict ap)
{
    const TR alpha = *alpha_;
    const TR zero = 0.0Q;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
        qspr2_contig(UPLO, n, alpha, x, y, ap);
        return;
    }

    /* Threaded strided only: gather x/y into contiguous scratch and run the
     * stride-1 core (which threads).  x/y read-only (only ap written) so no
     * scatter-back; the O(N) gather is dwarfed by the O(N^2) threaded work and
     * buys the contig core's omp scaling (the direct walk never threaded).  At
     * SERIAL strided the contig core is itself at the __float128 codegen floor vs
     * gfortran, so gathering only adds overhead — serial falls through to the
     * direct in-place loop (mirrors the kind10 yher2 gate).  See
     * project_l2_strided_gather. */
#ifdef _OPENMP
    const bool would_thread = (n >= QSPR2_OMP_MIN && blas_omp_max_threads() > 1);
#else
    const bool would_thread = false;
#endif
    if (would_thread) {
        const ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        const ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        TR stackbuf[2 * 256];
        TR *heap = NULL;
        TR *xc, *yc;
        if (n <= 256) {
            xc = stackbuf; yc = stackbuf + n;
        } else {
            heap = (TR *)malloc((size_t)2 * n * sizeof(TR));
            xc = heap; yc = heap ? heap + n : NULL;
        }
        if (xc && yc) {
            ptrdiff_t ix = kx, iy = ky;
            for (ptrdiff_t k = 0; k < n; ++k) {
                xc[k] = x[ix]; yc[k] = y[iy];
                ix += incx; iy += incy;
            }
            qspr2_contig(UPLO, n, alpha, xc, yc, ap);
            free(heap);
            return;
        }
        free(heap);
    }

    {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        ptrdiff_t kk = 0;
        ptrdiff_t jx = kx, jy = ky;
        if (UPLO == 'U') {
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const TR t1 = alpha * y[jy];
                    const TR t2 = alpha * x[jx];
                    ptrdiff_t ix = kx, iy = ky;
                    for (ptrdiff_t k = kk; k < kk + j + 1; ++k) {
                        ap[k] = ap[k] + x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                }
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const TR t1 = alpha * y[jy];
                    const TR t2 = alpha * x[jx];
                    ptrdiff_t ix = jx, iy = jy;
                    for (ptrdiff_t k = kk; k < kk + n - j; ++k) {
                        ap[k] = ap[k] + x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                }
                jx += incx; jy += incy;
                kk += n - j;
            }
        }
    }
}

EPBLAS_FACADE_SPR2(qspr2, TR)
