/*
 * qsyr2 — kind16 (__float128) symmetric rank-2 update.
 *   A := alpha · x · yᵀ + alpha · y · xᵀ + A
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

#define QSYR2_OMP_MIN 64

/* __float128 associativity gate (smaller N → product-first).  The two valid
 * orderings of the rank-2 element update trade off with N on libquadmath:
 *   accumulator-first  aj = aj + x*tx + y*ty  ==  ((aj + x*tx) + y*ty)
 *   product-first      aj += x*tx + y*ty      ==  (aj + (x*tx + y*ty))
 * Product-first first adds two PRODUCTS, whose exponent pattern drives
 * __addtf3's data-dependent normalize branch into a worse-predicted path
 * (+95M mispredicts vs the gfortran leg on the byte-identical libgcc __addtf3)
 * — a net loss once the triangle outgrows cache (N≥256: par/mig 1.00→1.05-1.07).
 * But for a small working set (N=128, triangle resident) the mispredict cost is
 * hidden and product-first's one-fewer dependent __addtf3 per element wins
 * (N=128 par/ob 1.06→1.00, par/mig 1.007→0.957).  So gate on N: product-first
 * below ~192, accumulator-first at/after 256.  (The kind16-serial=netlib
 * bit-exactness rule was removed 2026-06-10, so the small-N path may diverge
 * from mig — it stays bit-identical at N≥192 where accumulator-first runs.) */
#define QSYR2_PRODFIRST_MAX 192

typedef __float128 TR;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* Contiguous (incx=incy=1) symmetric rank-2 core over all columns.
 * schedule(static, 1): per-column work is linear in (N-j) for L / (j+1) for U,
 * so plain schedule(static) hands one thread the heavy triangle end and caps
 * threading at ~2x; cyclic chunk-1 balances it (mirrors the kind10 esyr2 twin —
 * the full-storage body is heavy enough that the finest balance wins over the
 * static,8 the lighter packed espr2 uses). */
static void qsyr2_contig(char UPLO, ptrdiff_t n, TR alpha,
                         const TR *restrict x, const TR *restrict y,
                         TR *restrict a, ptrdiff_t lda)
{
    const TR zero = 0.0Q;
    const bool prodfirst = (n < QSYR2_PRODFIRST_MAX);  /* see QSYR2_PRODFIRST_MAX */
#ifdef _OPENMP
    const bool use_omp = (n >= QSYR2_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
    for (ptrdiff_t j = 0; j < n; ++j) {
        const TR xj = x[j], yj = y[j];
        if (xj != zero || yj != zero) {
            const TR tx = alpha * yj;
            const TR ty = alpha * xj;
            TR *aj = &A_(0, j);
            const ptrdiff_t lo = (UPLO == 'L') ? j : 0;
            const ptrdiff_t hi = (UPLO == 'L') ? n : j + 1;
            if (prodfirst) for (ptrdiff_t i = lo; i < hi; ++i) aj[i] += x[i] * tx + y[i] * ty;
            else           for (ptrdiff_t i = lo; i < hi; ++i) aj[i] = aj[i] + x[i] * tx + y[i] * ty;
        }
    }
}

void qsyr2_core(
    char uplo,
    ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict x, ptrdiff_t incx,
    const TR *restrict y, ptrdiff_t incy,
    TR *restrict a, ptrdiff_t lda)
{
    const TR alpha = *alpha_;
    const TR zero = 0.0Q;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
        qsyr2_contig(UPLO, n, alpha, x, y, a, lda);
        return;
    }

    /* Threaded strided only: gather x/y into contiguous scratch and run the
     * stride-1 core (which threads).  x/y read-only (only A written) so no
     * scatter-back; the O(N) gather is dwarfed by the O(N^2) threaded work and
     * buys the contig core's omp scaling (the direct strided walk never
     * threaded).  At SERIAL strided the gather buys nothing — the contig core is
     * itself at the __float128 codegen floor vs gfortran, so a gathered serial
     * run only adds overhead — so serial falls through to the direct in-place
     * loop (mirrors the kind10 yher2 gate).  See project_l2_strided_gather. */
#ifdef _OPENMP
    const bool would_thread = (n >= QSYR2_OMP_MIN && blas_omp_max_threads() > 1);
#else
    const bool would_thread = false;
#endif
    if (would_thread) {
        const ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        const ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        /* Exact fit: the gather writes xc[0..n-1] and yc[0..n-1] with
         * yc = stackbuf + n (max offset 2n-1), so the threshold and the
         * array length must move together. */
        enum { QSYR2_STACK_N = 256 };
        TR stackbuf[2 * QSYR2_STACK_N];
        _Static_assert(2 * QSYR2_STACK_N * sizeof(TR) <= sizeof(stackbuf),
                       "qsyr2 stack-gather threshold exceeds stackbuf");
        TR *heap = NULL;
        TR *xc, *yc;
        if (n <= QSYR2_STACK_N) {
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
            qsyr2_contig(UPLO, n, alpha, xc, yc, a, lda);
            free(heap);
            return;
        }
        free(heap);
    }

    /* Direct strided walk: the serial path (gather not worth it) and the
     * alloc-failure fallback for the threaded path.  Small-N associativity gate
     * is DIRECTION-dependent here (unlike the contig core's unconditional
     * product-first): at N=128 the fastest reference flips with the x/y walk
     * direction, and __addtf3's data-dependent normalize branch tracks it —
     *   forward stride (incx>0): mig (gfortran) is fastest and accumulator-first
     *     matches it (ob, product-first, trails mig ~2.5% — ob/mig~1.03);
     *   backward stride (incx<0): ob is fastest and product-first matches it
     *     (accumulator-first matches the slower mig — ob/mig~0.96).
     * So product-first only for the small-N backward walk; accumulator-first
     * otherwise.  At N>=192 accumulator-first wins both directions.  (Gathering
     * into the contig core buys nothing serial — see project_l2_strided_gather.) */
    {
        const ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        const ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        const bool prodfirst = (n < QSYR2_PRODFIRST_MAX && incx < 0);
        for (ptrdiff_t j = 0; j < n; ++j) {
            const TR xj = x[kx + j * incx];
            const TR yj = y[ky + j * incy];
            if (xj != zero || yj != zero) {
                const TR tx = alpha * yj;
                const TR ty = alpha * xj;
                const ptrdiff_t lo = (UPLO == 'L') ? j : 0;
                const ptrdiff_t hi = (UPLO == 'L') ? n : j + 1;
                if (prodfirst)
                    for (ptrdiff_t i = lo; i < hi; ++i)
                        A_(i, j) += x[kx + i * incx] * tx + y[ky + i * incy] * ty;
                else
                    for (ptrdiff_t i = lo; i < hi; ++i)
                        A_(i, j) = A_(i, j) + x[kx + i * incx] * tx + y[ky + i * incy] * ty;
            }
        }
    }
}


EPBLAS_FACADE_SYR2(qsyr2, TR)

#undef A_
