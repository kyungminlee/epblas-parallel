/*
 * espmv — kind10 (long double) symmetric packed matrix-vector multiply.
 *   y := alpha*A*x + beta*y
 *
 * The serial reference sweep is column-oriented with cross-column writes
 * (column j updates y[0..j] / y[j..n-1] *and* accumulates into y[j]), so it
 * cannot be parallelized by a bare `omp parallel for` over columns — threads
 * would race on y[i]. The threaded path (contiguous case, large N) mirrors
 * OpenBLAS spmv_thread: a sqrt-balanced contiguous column partition, each
 * thread accumulating the *bare* A*x for its columns into a private size-N
 * slot, then a range-limited controller reduction applies alpha into y. The
 * per-thread kernel keeps this overlay's faster *sequential* packed-index
 * carry (running `k`) rather than recomputing j*(j+1)/2 per column.
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#include "../common/epblas_facade.h"
#ifdef _OPENMP
#include <math.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef long double TR;

/* Thread the contiguous path once n*n exceeds this (matches OpenBLAS dspmv's
 * MULTI_THREAD_MINIMAL): below it the serial sweep — faster than ob's serial
 * here — wins outright, and ob also stays serial, so par keeps the lead. */
#define ESPMV_OMP_MIN 16384
#define ESPMV_MAX_CPUS 256


/* Contiguous (stride-1) two-pass run shared by the serial and threaded
 * paths: y[i] += t1·ap[i] for i in [0, cnt), returning sum ap[i]·x[i].
 * Carved out noinline so the fp80 hot loop compiles in a clean x87 register
 * context — inlined amid the routine's OMP/beta/strided scaffolding GCC
 * spills and RELOADS ap[i] every element (4 fldt/elt vs 3); in isolation it
 * keeps ap[i] resident, matching the migrated Fortran reference. Callers pass
 * pointers at the run start and carry the packed index. Bit-identical. */
__attribute__((noinline)) static
TR espmv_axpydot(ptrdiff_t cnt, TR t1,
                const TR *restrict ap, const TR *restrict x, TR *restrict y)
{
    TR t2 = 0.0L;
    for (ptrdiff_t i = 0; i < cnt; ++i) {
        const TR a = ap[i];
        y[i] += t1 * a;
        t2   += a * x[i];
    }
    return t2;
}

/* Strided twin of espmv_axpydot: ap walked contiguously, x/y by stride.
 * Same noinline rationale, applied to the general-stride fallback. ix/iy
 * carry the running strided indices; bit-identical to the inline form. */
__attribute__((noinline)) static
TR espmv_axpydot_strided(ptrdiff_t cnt, TR t1, const TR *restrict ap,
                        const TR *restrict x, ptrdiff_t incx, ptrdiff_t ix,
                        TR *restrict y, ptrdiff_t incy, ptrdiff_t iy)
{
    TR t2 = 0.0L;
    for (ptrdiff_t i = 0; i < cnt; ++i) {
        const TR a = ap[i];
        y[iy] += t1 * a;
        t2    += a * x[ix];
        ix += incx; iy += incy;
    }
    return t2;
}

/* Serial stride-1 packed two-pass core (shared by the contiguous path and the
 * strided gather path). Bit-identical column-order accumulation to the direct
 * strided form.
 *
 * The two-pass kernel is inlined directly here (NOT routed through
 * espmv_axpydot) so the whole column nest compiles as a SINGLE function with no
 * per-column call — matching the migrated gfortran reference. The noinline
 * carve-out escapes the dispatcher's OMP/beta/strided register pressure; this
 * dedicated core has none of it, so the inlined fp80 loop keeps ap[i]
 * x87-resident (3 fldt/elt) on its own. The per-column call cost a fixed
 * call/ret + fldz that at small N showed up as a ~1/N par/mig gap. espmv_axpydot
 * is kept for the OMP path, where the call amortises over longer ranges. */
__attribute__((noinline)) static
void espmv_serial_core(char UPLO, ptrdiff_t n, TR alpha,
                       const TR *restrict ap, const TR *restrict x, TR *restrict y)
{
    ptrdiff_t kk = 0;
    if (UPLO == 'U') {
        for (ptrdiff_t j = 0; j < n; ++j) {
            const TR t1 = alpha * x[j];
            const TR *restrict apc = &ap[kk];
            TR t2 = 0.0L;
            for (ptrdiff_t i = 0; i < j; ++i) {
                const TR a = apc[i];
                y[i] += t1 * a;
                t2   += a * x[i];
            }
            y[j] += t1 * ap[kk + j] + alpha * t2;
            kk += j + 1;
        }
    } else {
        for (ptrdiff_t j = 0; j < n; ++j) {
            const TR t1 = alpha * x[j];
            y[j] += t1 * ap[kk];
            const TR *restrict apc = &ap[kk + 1];
            const TR *restrict xc = &x[j + 1];
            TR *restrict yc = &y[j + 1];
            const ptrdiff_t cnt = n - 1 - j;
            TR t2 = 0.0L;
            for (ptrdiff_t i = 0; i < cnt; ++i) {
                const TR a = apc[i];
                yc[i] += t1 * a;
                t2    += a * xc[i];
            }
            y[j] += alpha * t2;
            kk += n - j;
        }
    }
}

#ifdef _OPENMP
/* Sqrt-balanced contiguous column partition (OpenBLAS symv_partition, mask=3,
 * min_width=4). Per-column work grows with j for UPPER (length j+1) and shrinks
 * for LOWER (length n-j), so widths shrink / grow to equalize triangle area. */
static ptrdiff_t espmv_partition(bool upper, ptrdiff_t n, ptrdiff_t nthreads, ptrdiff_t *range)
{
    const ptrdiff_t mask = 3, min_width = 4;
    const double dnum = (double)n * (double)n / (double)nthreads;
    ptrdiff_t num_cpu = 0;
    range[0] = 0;
    ptrdiff_t i = 0;
    while (i < n) {
        ptrdiff_t width;
        if (nthreads - num_cpu > 1) {
            if (upper) {
                double di = (double)i;
                width = (ptrdiff_t)(sqrt(di * di + dnum) - di);
            } else {
                double di = (double)(n - i);
                double rad = di * di - dnum;
                width = (rad > 0.0) ? (ptrdiff_t)(-sqrt(rad) + di) : (n - i);
            }
            width = (width + mask) & ~(ptrdiff_t)mask;
            if (width < min_width) width = min_width;
            if (width > n - i)     width = n - i;
        } else {
            width = n - i;
        }
        range[num_cpu + 1] = range[num_cpu] + width;
        num_cpu++;
        i += width;
        if (num_cpu >= ESPMV_MAX_CPUS) break;
    }
    return num_cpu;
}
#endif

static void espmv_core(
    char uplo,
    ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict ap,
    const TR *restrict x, ptrdiff_t incx,
    const TR *beta_,
    TR *restrict y, ptrdiff_t incy)
{
    const TR alpha = *alpha_, beta = *beta_;
    const TR zero = 0.0L, one = 1.0L;
    const char UPLO = blas_up(uplo);

    if (n == 0 || (alpha == zero && beta == one)) return;

    const bool contiguous = (incx == 1 && incy == 1);
    /* Scale y := beta*y up front for the contiguous path and for the alpha==0
     * early-out. The strided path (below) instead folds beta into the y-gather,
     * which saves a whole extra strided pass over y — the beta pre-scale and the
     * gather-read otherwise touch y strided twice. Bit-identical: same beta*y
     * rounding, just applied while copying into the contiguous scratch. */
    if (beta != one && (contiguous || alpha == zero)) {
        ptrdiff_t iy = (incy < 0) ? -(n - 1) * incy : 0;
        if (beta == zero) {
            for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = zero; iy += incy; }
        } else {
            for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = beta * y[iy]; iy += incy; }
        }
    }
    if (alpha == zero) return;

    ptrdiff_t kk = 0;
    if (contiguous) {
#ifdef _OPENMP
        if ((size_t)n * (size_t)n > ESPMV_OMP_MIN
            && blas_omp_should_thread()) {
            ptrdiff_t nthreads = blas_omp_max_threads();
            if (nthreads > ESPMV_MAX_CPUS) nthreads = ESPMV_MAX_CPUS;
            ptrdiff_t range[ESPMV_MAX_CPUS + 1];
            ptrdiff_t num_cpu = espmv_partition(UPLO == 'U', n, nthreads, range);
            TR *buf = (num_cpu > 1)
                ? (TR *)calloc((size_t)num_cpu * (size_t)n, sizeof(TR)) : NULL;
            if (buf) {
                #pragma omp parallel for schedule(static, 1) num_threads(num_cpu)
                for (ptrdiff_t t = 0; t < num_cpu; ++t)
                {
                    ptrdiff_t m_from = range[t], m_to = range[t + 1];
                    TR *restrict slot = buf + (size_t)t * (size_t)n;
                    if (UPLO == 'U') {
                        size_t k = (size_t)m_from * (size_t)(m_from + 1) / 2;
                        for (ptrdiff_t j = m_from; j < m_to; ++j) {
                            const TR t1 = x[j];
                            const TR t2 = espmv_axpydot(j, t1, &ap[k], x, slot);
                            k += (size_t)j;
                            slot[j] += t1 * ap[k] + t2;   /* diagonal */
                            ++k;
                        }
                    } else {
                        size_t k = (size_t)m_from * (size_t)(2 * n - m_from + 1) / 2;
                        for (ptrdiff_t j = m_from; j < m_to; ++j) {
                            const TR t1 = x[j];
                            slot[j] += t1 * ap[k];        /* diagonal */
                            ++k;
                            const TR t2 = espmv_axpydot(n - 1 - j, t1, &ap[k], &x[j + 1], &slot[j + 1]);
                            k += (size_t)(n - 1 - j);
                            slot[j] += t2;
                        }
                    }
                }
                /* Range-limited reduction: each UPPER thread touched [0,range[t+1]),
                 * each LOWER thread [range[t],n). Fold into one slot, then alpha-AXPY. */
                if (UPLO == 'U') {
                    TR *restrict target = buf + (size_t)(num_cpu - 1) * (size_t)n;
                    for (ptrdiff_t t = 0; t < num_cpu - 1; ++t) {
                        const TR *restrict src = buf + (size_t)t * (size_t)n;
                        ptrdiff_t len = range[t + 1];
                        for (ptrdiff_t k = 0; k < len; ++k) target[k] += src[k];
                    }
                    for (ptrdiff_t k = 0; k < n; ++k) y[k] += alpha * target[k];
                } else {
                    TR *restrict target = buf;
                    for (ptrdiff_t t = 1; t < num_cpu; ++t) {
                        const TR *restrict src = buf + (size_t)t * (size_t)n;
                        for (ptrdiff_t k = range[t]; k < n; ++k) target[k] += src[k];
                    }
                    for (ptrdiff_t k = 0; k < n; ++k) y[k] += alpha * target[k];
                }
                free(buf);
                return;
            }
            free(buf);
        }
#endif
        espmv_serial_core(UPLO, n, alpha, ap, x, y);
    } else {
        /* General-stride: gather x and y into contiguous scratch, run the
         * stride-1 packed core — which beats both the OpenBLAS clone and the
         * gfortran reference, neither of which gathers — then scatter y back.
         * O(N) gather/scatter against O(N^2) work, so free past tiny N. The
         * y := beta*y scale is FOLDED into the gather (y not pre-scaled for the
         * strided path), avoiding a second strided pass over y. Same column-order
         * accumulation and same beta*y rounding as the direct strided walk, so
         * bit-identical. Falls back to the direct strided helper if the scratch
         * allocation fails. */
        const ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        const ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        /* Stack scratch for the common small-N case avoids malloc latency;
         * spill to the heap for large N. */
        /* Exact fit: the gather writes xc[0..n-1] and yc[0..n-1] with
         * yc = stackbuf + n (max offset 2n-1), so the threshold and the
         * array length must move together. */
        enum { ESPMV_STACK_N = 512 };
        TR stackbuf[2 * ESPMV_STACK_N];
        _Static_assert(2 * ESPMV_STACK_N * sizeof(TR) <= sizeof(stackbuf),
                       "espmv stack-gather threshold exceeds stackbuf");
        TR *heap = NULL;
        TR *xc, *yc;
        if (n <= ESPMV_STACK_N) {
            xc = stackbuf; yc = stackbuf + n;
        } else {
            heap = (TR *)malloc((size_t)2 * n * sizeof(TR));
            xc = heap; yc = heap ? heap + n : NULL;
        }
        if (xc && yc) {
            ptrdiff_t ix = kx, iy = ky;
            if (beta == one) {
                for (ptrdiff_t k = 0; k < n; ++k) {
                    xc[k] = x[ix]; yc[k] = y[iy];
                    ix += incx; iy += incy;
                }
            } else if (beta == zero) {
                for (ptrdiff_t k = 0; k < n; ++k) {
                    xc[k] = x[ix]; yc[k] = zero;
                    ix += incx; iy += incy;
                }
            } else {
                for (ptrdiff_t k = 0; k < n; ++k) {
                    xc[k] = x[ix]; yc[k] = beta * y[iy];
                    ix += incx; iy += incy;
                }
            }
            espmv_serial_core(UPLO, n, alpha, ap, xc, yc);
            iy = ky;
            for (ptrdiff_t k = 0; k < n; ++k) { y[iy] = yc[k]; iy += incy; }
            free(heap);
            return;
        }
        free(heap);
        /* Fallback (alloc failed): the strided path skipped the beta pre-scale,
         * so apply it now before the direct in-place strided RMW. */
        if (beta != one) {
            ptrdiff_t iy = ky;
            if (beta == zero)
                for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = zero; iy += incy; }
            else
                for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = beta * y[iy]; iy += incy; }
        }
        if (UPLO == 'U') {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t j = 0; j < n; ++j) {
                const TR t1 = alpha * x[jx];
                const TR t2 = espmv_axpydot_strided(
                    j, t1, &ap[kk], x, incx, kx, y, incy, ky);
                y[jy] += t1 * ap[kk + j] + alpha * t2;
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t j = 0; j < n; ++j) {
                const TR t1 = alpha * x[jx];
                y[jy] += t1 * ap[kk];
                const TR t2 = espmv_axpydot_strided(
                    n - j - 1, t1, &ap[kk + 1],
                    x, incx, jx + incx, y, incy, jy + incy);
                y[jy] += alpha * t2;
                jx += incx; jy += incy;
                kk += n - j;
            }
        }
    }
}

EPBLAS_FACADE_SPMV(espmv, TR)
