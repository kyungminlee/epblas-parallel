/*
 * esbmv — kind10 (long double) symmetric band matrix-vector multiply.
 *   y := alpha*A*x + beta*y, A symmetric with K super-(or sub-)diagonals,
 *   stored as one triangle (upper or lower) in band format.
 *
 * y := A*x is a pure matvec: every OUTPUT row is an independent dot over the
 * full 2K+1 band, so we GATHER -- each y[i] is accumulated in a register-
 * resident x87 scalar and stored once (y[i] += alpha*s), never the column
 * SCATTER (y[i] += t1*A per element = read-modify-write to memory) that the
 * reference uses. Keeping the accumulator off memory runs ~2x faster per
 * element (project_x87_accumulator_spill).
 *
 * Only one triangle is stored, but the symmetric reflection is still a pure
 * read: with base = &A_(0,i) and s1 = lda-1, row i reconstructs its full band
 * from the upper triangle as  A(i,i)=base[K],  right A(i,i+d)=base[K+d*s1]
 * (anti-diagonal walk), left A(i,i-d)=A(i-d,i)=base[K-d] (contiguous in column
 * i) -- i.e. symmetric = NoTrans-gather + Trans-gather over the same row,
 * because A = upperTri + (strictUpper)^T. Lower storage is the mirror (left
 * anti-diagonal, right contiguous). Each stored off-diagonal element is read
 * twice (once per row it serves), vs once for the scatter; on fp80 (x87, no
 * SIMD) that second read stays L2/L3-resident and is far cheaper than the
 * per-element y read-modify-write it replaces.
 *
 * Because x and y are DISTINCT arrays (BLAS forbids sbmv aliasing), the gather
 * needs NO buffer and NO ordering trick, and the threaded path needs NO barrier
 * and NO copy-back (unlike the in-place triangular etbmv): each thread owns a
 * disjoint output-row range [lo,hi) and writes y[lo,hi) directly while reading x
 * globally. This replaces the old OpenBLAS dsbmv_thread port (per-thread
 * size-n private slots + a band-aware reduction fold whose O(n*nthreads) serial
 * cost floored the speedup) -- the only serial cost is now one malloc(n) for the
 * strided-x gather, so speedup climbs toward nthreads.
 *
 * The serial reference is defined FIRST so its hot-loop placement is unperturbed
 * by the threaded scaffolding (function order shifts alignment, ~2% otherwise);
 * the orchestration is a noinline helper so its bookkeeping does not crowd the
 * serial gather's x87 allocator.
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include "../common/epblas_facade.h"
#ifdef _OPENMP
#include <stdlib.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef long double TR;


#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

/* Thread-entry threshold = the measured break-even where forking beats our own
 * serial gather. The register-resident serial gather is fast, so threading only
 * pays once N amortizes the parallel region's fork/join (+ the strided-x malloc).
 * esbmv does the full 2K+1 band per row (vs etbmv's triangular K+1), so its
 * break-even is lower than etbmv's. Only upper/lower exist (no trans/diag) and
 * share the gather body, so a single threshold suffices. Calibrated at K=16;
 * #ifndef so the calibration probe can force it.
 *
 * Recalibrated 2026-06-10 under iomp5 (hot-team reuse → cheap fork): the old 320
 * was a stale libgomp-era value that kept N=256 serial, but ob threads at 256 and
 * its threaded path beat our serial gather there (par4/ob4 ~1.3-1.5). Under iomp5
 * N=256 threads to par4/par1 ~0.38 (~2.6×) → par4/ob4 ~0.57. Drop to 256 so the
 * N=256 band cells win; N=128 stays serial and the serial gather still beats ob's
 * (threaded or not) at 128. (Same stale-threshold pattern as the y* audit.) */
#ifndef ESBMV_OMP_MIN
#define ESBMV_OMP_MIN 256
#endif
#define ESBMV_MAX_CPUS 256

#ifdef _OPENMP
static ptrdiff_t esbmv_omp(bool upper, ptrdiff_t n, ptrdiff_t k,
                     const TR *restrict a, ptrdiff_t lda,
                     const TR *restrict x, ptrdiff_t incx,
                     TR alpha, TR *restrict y, ptrdiff_t incy);
#endif

static void esbmv_core(
    char uplo,
    ptrdiff_t n, ptrdiff_t k,
    const TR *alpha_,
    const TR *restrict a, ptrdiff_t lda,
    const TR *restrict x, ptrdiff_t incx,
    const TR *beta_,
    TR *restrict y, ptrdiff_t incy)
{
    const TR alpha = *alpha_, beta = *beta_;
    const TR zero = 0.0L, one = 1.0L;
    const char UPLO = blas_up(uplo);

    if (n == 0 || (alpha == zero && beta == one)) return;

    if (beta != one) {
        ptrdiff_t iy = (incy < 0) ? -(n - 1) * incy : 0;
        if (beta == zero) {
            for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = zero; iy += incy; }
        } else {
            for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = beta * y[iy]; iy += incy; }
        }
    }
    if (alpha == zero) return;

#ifdef _OPENMP
    /* Cheap inline gate first: at OMP=1 (or below threshold) this short-circuits
     * before the noinline call's argument marshalling, so the serial gather that
     * follows keeps its unperturbed register allocation (outlining tax). */
    if (n >= ESBMV_OMP_MIN && blas_omp_max_threads() > 1
        && esbmv_omp(UPLO == 'U', n, k, a, lda, x, incx, alpha, y, incy))
        return;
#endif

    /* Serial row-gather: each y[i] is a register-resident band dot over the full
     * 2K+1 band, stored once. base[K +- d*s1] walks the lda-1 anti-diagonal for
     * the same-triangle neighbours; base[K-/+d] reads the reflected neighbours
     * contiguously in column i. x and y distinct -> no buffer, no ordering. */
    const ptrdiff_t s1 = lda - 1;
    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TR *base = &A_(0, i);
                TR s = base[k] * x[i];
                const ptrdiff_t rlen = (n - 1 - i < k) ? (n - 1 - i) : k;
                for (ptrdiff_t d = 1; d <= rlen; ++d) s += base[k + (ptrdiff_t)d * s1] * x[i + d];
                const ptrdiff_t llen = (i < k) ? i : k;
                for (ptrdiff_t d = 1; d <= llen; ++d) s += base[k - d] * x[i - d];
                y[i] += alpha * s;
            }
        } else {
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TR *base = &A_(0, i);
                TR s = base[0] * x[i];
                const ptrdiff_t llen = (i < k) ? i : k;
                for (ptrdiff_t d = 1; d <= llen; ++d) s += base[-(ptrdiff_t)d * s1] * x[i - d];
                const ptrdiff_t rlen = (n - 1 - i < k) ? (n - 1 - i) : k;
                for (ptrdiff_t d = 1; d <= rlen; ++d) s += base[d] * x[i + d];
                y[i] += alpha * s;
            }
        }
    } else {
        const ptrdiff_t ix0 = (incx < 0) ? -(ptrdiff_t)(n - 1) * incx : 0;
        const ptrdiff_t iy0 = (incy < 0) ? -(ptrdiff_t)(n - 1) * incy : 0;
        if (UPLO == 'U') {
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TR *base = &A_(0, i);
                const ptrdiff_t xi = ix0 + (ptrdiff_t)i * incx;
                TR s = base[k] * x[xi];
                const ptrdiff_t rlen = (n - 1 - i < k) ? (n - 1 - i) : k;
                ptrdiff_t xx = xi + incx;
                for (ptrdiff_t d = 1; d <= rlen; ++d) { s += base[k + (ptrdiff_t)d * s1] * x[xx]; xx += incx; }
                const ptrdiff_t llen = (i < k) ? i : k;
                xx = xi - incx;
                for (ptrdiff_t d = 1; d <= llen; ++d) { s += base[k - d] * x[xx]; xx -= incx; }
                y[iy0 + (ptrdiff_t)i * incy] += alpha * s;
            }
        } else {
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TR *base = &A_(0, i);
                const ptrdiff_t xi = ix0 + (ptrdiff_t)i * incx;
                TR s = base[0] * x[xi];
                const ptrdiff_t llen = (i < k) ? i : k;
                ptrdiff_t xx = xi - incx;
                for (ptrdiff_t d = 1; d <= llen; ++d) { s += base[-(ptrdiff_t)d * s1] * x[xx]; xx -= incx; }
                const ptrdiff_t rlen = (n - 1 - i < k) ? (n - 1 - i) : k;
                xx = xi + incx;
                for (ptrdiff_t d = 1; d <= rlen; ++d) { s += base[d] * x[xx]; xx += incx; }
                y[iy0 + (ptrdiff_t)i * incy] += alpha * s;
            }
        }
    }
}

#ifdef _OPENMP
/* Row-partitioned gather kernel: compute y[i] for i in [lo,hi) as a register-
 * resident band dot reading x (contiguous logical order) and adding alpha*s into
 * the already-beta-scaled y[i*incy]. Same gather as the serial path. Output rows
 * partition across threads with no cross-thread write dependence -- no scratch,
 * no zero-fill, no reduction, no barrier (x and y distinct). Branch hoisted out
 * of the i-loop; lda-1 anti-diagonal stride for same-triangle neighbours,
 * contiguous for the reflected ones. */
static void sbmv_rowgather(bool upper, ptrdiff_t n, ptrdiff_t k,
                           ptrdiff_t lo, ptrdiff_t hi,
                           const TR *restrict a, ptrdiff_t lda,
                           const TR *restrict x, TR alpha,
                           TR *restrict y, ptrdiff_t incy)
{
    const ptrdiff_t s1 = lda - 1;
    if (upper) {
        for (ptrdiff_t i = lo; i < hi; ++i) {
            const TR *base = &A_(0, i);
            TR s = base[k] * x[i];
            ptrdiff_t rlen = (n - 1 - i < k) ? n - 1 - i : k;
            for (ptrdiff_t d = 1; d <= rlen; ++d) s += base[k + d * s1] * x[i + d];
            ptrdiff_t llen = (i < k) ? i : k;
            for (ptrdiff_t d = 1; d <= llen; ++d) s += base[k - d] * x[i - d];
            y[i * incy] += alpha * s;
        }
    } else {
        for (ptrdiff_t i = lo; i < hi; ++i) {
            const TR *base = &A_(0, i);
            TR s = base[0] * x[i];
            ptrdiff_t llen = (i < k) ? i : k;
            for (ptrdiff_t d = 1; d <= llen; ++d) s += base[-d * s1] * x[i - d];
            ptrdiff_t rlen = (n - 1 - i < k) ? n - 1 - i : k;
            for (ptrdiff_t d = 1; d <= rlen; ++d) s += base[d] * x[i + d];
            y[i * incy] += alpha * s;
        }
    }
}

/* Threaded symmetric band matvec. Each thread owns a disjoint output-row range
 * [lo,hi): it gathers y[lo,hi) reading x. No cross-thread data dependence (x and
 * y distinct) -- no barrier, no scratch beyond the strided-x gather buffer.
 * Returns 1 if handled, 0 to fall back. Carved out (noinline) so its bookkeeping
 * does not pressure the serial gather's x87 allocation. */
__attribute__((noinline)) static ptrdiff_t esbmv_omp(
    bool upper, ptrdiff_t n, ptrdiff_t k,
    const TR *restrict a, ptrdiff_t lda,
    const TR *restrict x, ptrdiff_t incx,
    TR alpha, TR *restrict y, ptrdiff_t incy)
{
    if (n < ESBMV_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > ESBMV_MAX_CPUS) nthreads = ESBMV_MAX_CPUS;

    /* Negative-increment base adjustment so x[i*incx], y[i*incy] walk logically. */
    if (incx < 0) x -= (n - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;

    /* Gather strided x to contiguous (logical order) so the inner dot is unit
     * stride; y is written disjointly per thread in place. */
    const TR *xptr = x;
    TR *xbuf = NULL;
    if (incx != 1) {
        xbuf = (TR *)malloc((size_t)n * sizeof(TR));
        if (!xbuf) return 0;
        for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[i * incx];
        xptr = xbuf;
    }

    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t lo = (n * (ptrdiff_t)tid) / nthreads;
        ptrdiff_t hi = (n * (ptrdiff_t)(tid + 1)) / nthreads;
        sbmv_rowgather(upper, n, k, lo, hi, a, lda, xptr, alpha, y, incy);
    }

    free(xbuf);
    return 1;
}
#endif /* _OPENMP */

EPBLAS_FACADE_SBMV(esbmv, TR)

#undef A_
