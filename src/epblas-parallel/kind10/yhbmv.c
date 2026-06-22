/*
 * yhbmv — kind10 complex Hermitian band matrix-vector multiply.
 *   y := alpha*A*x + beta*y, A Hermitian with K super-(or sub-)diagonals,
 *   stored as one triangle (upper or lower) in band format.
 *
 * Hermitian twin of esbmv (real symmetric) and ytbmv (complex triangular).
 * y := A*x is a pure matvec: every OUTPUT row is an independent dot over the
 * full 2K+1 band, so we GATHER -- each y[i] is accumulated in a register-
 * resident x87 complex scalar and stored once (y[i] += alpha*s), never the
 * column SCATTER (y[i] += t1*A per element = read-modify-write to memory) that
 * reference hbmv uses. Keeping the accumulator off memory runs ~2x faster per
 * element (project_x87_accumulator_spill). The complex accumulator stays
 * resident across BOTH the direct and the conjugated inner loop (verified:
 * 4 fldt/element/loop, no accumulator reload between loops).
 *
 * A row is exactly ytbmv's NoTrans-gather loop + ytbmv's ConjTrans-gather loop
 * summed into one s, with a real diagonal seed: A = upperTri + (strictUpper)^H,
 * so with base = &A_(0,i) and s1 = lda-1, row i reconstructs its full band from
 * the upper triangle as  A(i,i)=Re(base[K]) (REAL),  right A(i,i+d)=base[K+d*s1]
 * (anti-diagonal walk, used DIRECTLY), left A(i,i-d)=conj(A(i-d,i))=conj(base[K-d])
 * (contiguous in column i, CONJUGATED). Lower storage is the mirror (left
 * anti-diagonal direct, right contiguous conjugated). The contiguous/reflected
 * neighbour is the conjugated one in both triangles; the diagonal is real. The
 * real diagonal is seeded as Re(base)*x[i] -- a real*complex (2 mults, not 4),
 * which both drops the spurious imag(diag) cross-terms (required: LAPACK leaves
 * imag(diag) unreferenced) and is cheaper.
 *
 * Because x and y are DISTINCT arrays (BLAS forbids hbmv aliasing), the gather
 * needs NO buffer and NO ordering trick, and the threaded path needs NO barrier
 * and NO copy-back (unlike the in-place triangular ytbmv): each thread owns a
 * disjoint output-row range [lo,hi) and writes y[lo,hi) directly while reading x
 * globally. This replaces the old OpenBLAS zhbmv_thread port (per-thread size-n
 * private slots + two partition functions + a band-aware reduction fold whose
 * O(n*nthreads) serial cost floored the speedup) -- the only serial cost is now
 * one malloc(n) for the strided-x gather, so speedup climbs toward nthreads.
 * Row-gather work is uniform 2K+1 per interior row, so an even row split needs
 * no load-balancer (the deleted sqrt partition existed for the column scatter's
 * triangular per-column ramp).
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

typedef _Complex long double T;
typedef long double TR;
static inline T cconj(T z) { return ~z; }


#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

/* Thread-entry threshold = the measured break-even where par4 < par1 (forking
 * actually beats our own serial gather). The register-resident serial gather is
 * fast, so threading only pays once N amortizes the parallel region's fork/join
 * (+ the strided-x malloc). yhbmv's region has LESS overhead than ytbmv's (no
 * barrier, no copy-back, no scratch) and HEAVIER per-row work than esbmv (complex
 * MAC), both pushing break-even WAY below their 320: measured break-even is
 * complex band dot amortizes the barrier-free fork almost immediately. Only
 * upper/lower exist (no trans/diag) sharing the gather body, so a single
 * threshold suffices. Calibrated at K=16; #ifndef so the calibration probe can
 * force it.
 * RECALIBRATED 2026-06-07 (was 96): the N~=88 break-even above predates iomp5
 * (it was set under libgomp's fork/join wakeup tax — "N=64 still 1.34"). With
 * iomp5 hot-team reuse the fork is nearly free, so threading now pays from N=32.
 * Measured par4/par1 (taskset 0-3, min-of-10): N=32 0.64/0.64, N=48 0.55,
 * N=128 0.37, N=256 0.32. N=28 wins too (0.73) but N=24 anomalously loses
 * (1.16, band nearly fills the matrix) so 32 is the robust floor. Bit-exact
 * across thread counts (disjoint-row gather; relerr 0). */
#ifndef YHBMV_OMP_MIN
#define YHBMV_OMP_MIN 32
#endif
#define YHBMV_MAX_CPUS 256

#ifdef _OPENMP
static ptrdiff_t yhbmv_omp(bool upper, ptrdiff_t n, ptrdiff_t k,
                     const T *restrict a, ptrdiff_t lda,
                     const T *restrict x, ptrdiff_t incx,
                     T alpha, T *restrict y, ptrdiff_t incy);
#endif

static void yhbmv_core(
    char uplo,
    ptrdiff_t n, ptrdiff_t k,
    const T *alpha_,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    const T *beta_,
    T *restrict y, ptrdiff_t incy)
{
    const T alpha = *alpha_, beta = *beta_;
    const T zero = 0.0L + 0.0Li, one = 1.0L + 0.0Li;
    const char UPLO = blas_up(uplo);

    if (n == 0 || (alpha == zero && beta == one)) return;

    if (beta != one) {
        ptrdiff_t iy = (incy < 0) ? -(n - 1) * incy : 0;
        if (beta == zero) for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = zero; iy += incy; }
        else              for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = beta * y[iy]; iy += incy; }
    }
    if (alpha == zero) return;

#ifdef _OPENMP
    /* Cheap inline gate first: at OMP=1 (or below threshold) this short-circuits
     * before the noinline call's argument marshalling, so the serial gather that
     * follows keeps its unperturbed register allocation (outlining tax). */
    if (n >= YHBMV_OMP_MIN && blas_omp_max_threads() > 1
        && yhbmv_omp(UPLO == 'U', n, k, a, lda, x, incx, alpha, y, incy))
        return;
#endif

    /* Serial row-gather: each y[i] is a register-resident band dot over the full
     * 2K+1 band, stored once. Diagonal real (Re*x = 2 mults). base[K +- d*s1]
     * walks the lda-1 anti-diagonal for the same-triangle (DIRECT) neighbours;
     * base[K-/+d] reads the reflected neighbours contiguously in column i and
     * CONJUGATES them. x and y distinct -> no buffer, no ordering. */
    const ptrdiff_t s1 = lda - 1;
    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
            for (ptrdiff_t i = 0; i < n; ++i) {
                const T *base = &A_(0, i);
                T s = (TR)__real__ base[k] * x[i];
                const ptrdiff_t rlen = (n - 1 - i < k) ? (n - 1 - i) : k;
                for (ptrdiff_t d = 1; d <= rlen; ++d) s += base[k + (ptrdiff_t)d * s1] * x[i + d];
                const ptrdiff_t llen = (i < k) ? i : k;
                for (ptrdiff_t d = 1; d <= llen; ++d) s += cconj(base[k - d]) * x[i - d];
                y[i] += alpha * s;
            }
        } else {
            for (ptrdiff_t i = 0; i < n; ++i) {
                const T *base = &A_(0, i);
                T s = (TR)__real__ base[0] * x[i];
                const ptrdiff_t llen = (i < k) ? i : k;
                for (ptrdiff_t d = 1; d <= llen; ++d) s += base[-(ptrdiff_t)d * s1] * x[i - d];
                const ptrdiff_t rlen = (n - 1 - i < k) ? (n - 1 - i) : k;
                for (ptrdiff_t d = 1; d <= rlen; ++d) s += cconj(base[d]) * x[i + d];
                y[i] += alpha * s;
            }
        }
    } else {
        const ptrdiff_t ix0 = (incx < 0) ? -(ptrdiff_t)(n - 1) * incx : 0;
        const ptrdiff_t iy0 = (incy < 0) ? -(ptrdiff_t)(n - 1) * incy : 0;
        if (UPLO == 'U') {
            for (ptrdiff_t i = 0; i < n; ++i) {
                const T *base = &A_(0, i);
                const ptrdiff_t xi = ix0 + (ptrdiff_t)i * incx;
                T s = (TR)__real__ base[k] * x[xi];
                const ptrdiff_t rlen = (n - 1 - i < k) ? (n - 1 - i) : k;
                ptrdiff_t xx = xi + incx;
                for (ptrdiff_t d = 1; d <= rlen; ++d) { s += base[k + (ptrdiff_t)d * s1] * x[xx]; xx += incx; }
                const ptrdiff_t llen = (i < k) ? i : k;
                xx = xi - incx;
                for (ptrdiff_t d = 1; d <= llen; ++d) { s += cconj(base[k - d]) * x[xx]; xx -= incx; }
                y[iy0 + (ptrdiff_t)i * incy] += alpha * s;
            }
        } else {
            for (ptrdiff_t i = 0; i < n; ++i) {
                const T *base = &A_(0, i);
                const ptrdiff_t xi = ix0 + (ptrdiff_t)i * incx;
                T s = (TR)__real__ base[0] * x[xi];
                const ptrdiff_t llen = (i < k) ? i : k;
                ptrdiff_t xx = xi - incx;
                for (ptrdiff_t d = 1; d <= llen; ++d) { s += base[-(ptrdiff_t)d * s1] * x[xx]; xx -= incx; }
                const ptrdiff_t rlen = (n - 1 - i < k) ? (n - 1 - i) : k;
                xx = xi + incx;
                for (ptrdiff_t d = 1; d <= rlen; ++d) { s += cconj(base[d]) * x[xx]; xx += incx; }
                y[iy0 + (ptrdiff_t)i * incy] += alpha * s;
            }
        }
    }
}

#ifdef _OPENMP
/* Row-partitioned gather kernel: compute y[i] for i in [lo,hi) as a register-
 * resident complex band dot reading x (contiguous logical order) and adding
 * alpha*s into the already-beta-scaled y[i*incy]. Same gather as the serial
 * path: real diagonal, DIRECT same-triangle neighbour (lda-1 anti-diagonal),
 * CONJUGATED reflected neighbour (contiguous). Output rows partition across
 * threads with no cross-thread write dependence -- no scratch, no zero-fill,
 * no reduction, no barrier (x and y distinct). Branch hoisted out of i-loop. */
static void hbmv_rowgather(bool upper, ptrdiff_t n, ptrdiff_t k,
                           ptrdiff_t lo, ptrdiff_t hi,
                           const T *restrict a, ptrdiff_t lda,
                           const T *restrict x, T alpha,
                           T *restrict y, ptrdiff_t incy)
{
    const ptrdiff_t s1 = lda - 1;
    if (upper) {
        for (ptrdiff_t i = lo; i < hi; ++i) {
            const T *base = &A_(0, i);
            T s = (TR)__real__ base[k] * x[i];
            ptrdiff_t rlen = (n - 1 - i < k) ? n - 1 - i : k;
            for (ptrdiff_t d = 1; d <= rlen; ++d) s += base[k + d * s1] * x[i + d];
            ptrdiff_t llen = (i < k) ? i : k;
            for (ptrdiff_t d = 1; d <= llen; ++d) s += cconj(base[k - d]) * x[i - d];
            y[i * incy] += alpha * s;
        }
    } else {
        for (ptrdiff_t i = lo; i < hi; ++i) {
            const T *base = &A_(0, i);
            T s = (TR)__real__ base[0] * x[i];
            ptrdiff_t llen = (i < k) ? i : k;
            for (ptrdiff_t d = 1; d <= llen; ++d) s += base[-d * s1] * x[i - d];
            ptrdiff_t rlen = (n - 1 - i < k) ? n - 1 - i : k;
            for (ptrdiff_t d = 1; d <= rlen; ++d) s += cconj(base[d]) * x[i + d];
            y[i * incy] += alpha * s;
        }
    }
}

/* Threaded Hermitian band matvec. Each thread owns a disjoint output-row range
 * [lo,hi): it gathers y[lo,hi) reading x. No cross-thread data dependence (x and
 * y distinct) -- no barrier, no scratch beyond the strided-x gather buffer.
 * Returns 1 if handled, 0 to fall back. Carved out (noinline) so its bookkeeping
 * does not pressure the serial gather's x87 allocation. */
__attribute__((noinline)) static ptrdiff_t yhbmv_omp(
    bool upper, ptrdiff_t n, ptrdiff_t k,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    T alpha, T *restrict y, ptrdiff_t incy)
{
    if (n < YHBMV_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > YHBMV_MAX_CPUS) nthreads = YHBMV_MAX_CPUS;

    /* Negative-increment base adjustment so x[i*incx], y[i*incy] walk logically. */
    if (incx < 0) x -= (n - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;

    /* Gather strided x to contiguous (logical order) so the inner dot is unit
     * stride; y is written disjointly per thread in place. */
    const T *xptr = x;
    T *xbuf = NULL;
    if (incx != 1) {
        xbuf = (T *)malloc((size_t)n * sizeof(T));
        if (!xbuf) return 0;
        for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[i * incx];
        xptr = xbuf;
    }

    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t lo = (n * (ptrdiff_t)tid) / nthreads;
        ptrdiff_t hi = (n * (ptrdiff_t)(tid + 1)) / nthreads;
        hbmv_rowgather(upper, n, k, lo, hi, a, lda, xptr, alpha, y, incy);
    }

    free(xbuf);
    return 1;
}
#endif /* _OPENMP */

EPBLAS_FACADE_SBMV(yhbmv, T)

#undef A_
