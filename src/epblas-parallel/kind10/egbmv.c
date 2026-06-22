/*
 * egbmv — kind10 (long double) general band matrix-vector multiply.
 *   y := alpha*A*x + beta*y  or  y := alpha*A^T*x + beta*y
 * Band storage: A(i,j) at AB[(ku + i - j) + j*lda], for max(0,j-ku) <= i <= min(M,j+kl).
 *
 * NoTrans (y := A*x): every OUTPUT row is an independent dot over the row's band,
 * so we GATHER -- each y[i] is accumulated in a register-resident x87 scalar and
 * stored once (y[i] += alpha*s), never the column SCATTER (y[i] += tmp*A per
 * element = read-modify-write to memory) that reference gbmv uses. Keeping the
 * accumulator off memory runs ~2x faster per element on fp80 -- there is no SIMD,
 * so the x87 dependency chain dominates and the per-element RMW store to y is the
 * real cost, not the strided A read (project_x87_accumulator_spill,
 * project_l2_rowgather_scaling). For row i, A(i,j) = a[(KU+i) + j*(lda-1)], so
 * with base = a + (KU+i) and s1 = lda-1 the row is a unit-j dot base[j*s1]*x[j]
 * over j in [max(0,i-KL), min(N,i+KU+1)) -- an anti-diagonal walk of A (the same
 * stride the triangular/symmetric/Hermitian band gathers use). Because x and y
 * are DISTINCT arrays (BLAS forbids gbmv aliasing), the threaded path partitions
 * output rows [lo,hi) with NO scratch, NO reduction, and NO barrier: each thread
 * writes its own y[lo,hi) while reading x globally.
 *
 * Per-output summation order is preserved (ascending j == the column order in
 * which the old scatter accumulated each y[i]), so the serial gather is
 * bit-identical to the old scatter, not merely close.
 *
 * Trans (y := A^T*x) was already a gather over disjoint y[j] and is left as-is
 * (it threads cleanly via the source-level if(use_omp) macro duplication).
 *
 * The serial NoTrans reference is defined inline FIRST; the threaded
 * orchestration is a noinline helper so its bookkeeping does not crowd the serial
 * gather's x87 register allocation (outlining tax).
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

/* Thread-entry threshold = measured break-even where par4 < par1 (forking beats
 * our own serial gather). The register-resident serial NoTrans gather is fast, so
 * threading only pays once the output-row count amortizes the parallel region's
 * fork/join (+ any strided-x malloc). Calibrated at KL=KU=16: NoTrans first clean
 * all-win at N=256 (p4/p1: 224->1.017 loss, 256->0.980, 320->0.811). The Trans
 * gather shares this threshold (its break-even is ~256-512; the old 64 made it
 * thread far too early, a 1.7-3.2x loss at N=64-128). #ifndef so the calibration
 * probe can force it. */
#ifndef EGBMV_OMP_MIN
#define EGBMV_OMP_MIN 256
#endif
#define EGBMV_MAX_CPUS 256

typedef long double T;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#ifdef _OPENMP
static ptrdiff_t egbmv_n_omp(ptrdiff_t m, ptrdiff_t n, ptrdiff_t kl, ptrdiff_t ku,
                       const T *restrict a, ptrdiff_t lda,
                       const T *restrict x, ptrdiff_t incx,
                       T alpha, T *restrict y, ptrdiff_t incy);
static ptrdiff_t egbmv_t_omp(ptrdiff_t m, ptrdiff_t n, ptrdiff_t kl, ptrdiff_t ku,
                       const T *restrict a, ptrdiff_t lda,
                       const T *restrict x, ptrdiff_t incx,
                       T alpha, T *restrict y, ptrdiff_t incy);
#endif

static void egbmv_core(
    char trans,
    ptrdiff_t M, ptrdiff_t N,
    ptrdiff_t KL, ptrdiff_t KU,
    const T *alpha_,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    const T *beta_,
    T *restrict y, ptrdiff_t incy)
{
    const T alpha = *alpha_, beta = *beta_;
    const T zero = 0.0L, one = 1.0L;
    char TR = blas_up(trans);
    if (TR == 'C') TR = 'T';

    if (M == 0 || N == 0 || (alpha == zero && beta == one)) return;

    const ptrdiff_t leny = (TR == 'N') ? M : N;
    const ptrdiff_t lenx = (TR == 'N') ? N : M;

    if (beta != one) {
        ptrdiff_t iy = (incy < 0) ? -(leny - 1) * incy : 0;
        if (beta == zero) {
            for (ptrdiff_t i = 0; i < leny; ++i) { y[iy] = zero; iy += incy; }
        } else {
            for (ptrdiff_t i = 0; i < leny; ++i) { y[iy] = beta * y[iy]; iy += incy; }
        }
    }
    if (alpha == zero) return;

    const ptrdiff_t s1 = (ptrdiff_t)lda - 1;

    if (TR == 'N') {
#ifdef _OPENMP
        /* Cheap inline gate before the noinline call's marshalling, so the serial
         * gather below keeps its unperturbed register allocation (outlining tax). */
        if (M >= EGBMV_OMP_MIN && blas_omp_max_threads() > 1
            && egbmv_n_omp(M, N, KL, KU, a, lda, x, incx, alpha, y, incy))
            return;
#endif
        /* Serial NoTrans row-gather: y[i] = alpha * Σ_j A(i,j)*x[j], accumulated
         * in a register-resident x87 scalar. A(i,j) = base[j*s1], base = a+(KU+i),
         * an lda-1 anti-diagonal walk; j ranges over row i's band. */
        if (incx == 1 && incy == 1) {
            for (ptrdiff_t i = 0; i < M; ++i) {
                const ptrdiff_t j_lo = (i - KL > 0) ? (i - KL) : 0;
                const ptrdiff_t j_hi = (i + KU + 1 < N) ? (i + KU + 1) : N;
                const T *base = a + (KU + i);
                T s = zero;
                for (ptrdiff_t j = j_lo; j < j_hi; ++j) s += base[(ptrdiff_t)j * s1] * x[j];
                y[i] += alpha * s;
            }
        } else {
            const ptrdiff_t ix0 = (incx < 0) ? -(ptrdiff_t)(lenx - 1) * incx : 0;
            const ptrdiff_t iy0 = (incy < 0) ? -(ptrdiff_t)(leny - 1) * incy : 0;
            for (ptrdiff_t i = 0; i < M; ++i) {
                const ptrdiff_t j_lo = (i - KL > 0) ? (i - KL) : 0;
                const ptrdiff_t j_hi = (i + KU + 1 < N) ? (i + KU + 1) : N;
                const T *base = a + (KU + i);
                T s = zero;
                ptrdiff_t xx = ix0 + (ptrdiff_t)j_lo * incx;
                for (ptrdiff_t j = j_lo; j < j_hi; ++j) { s += base[(ptrdiff_t)j * s1] * x[xx]; xx += incx; }
                y[iy0 + (ptrdiff_t)i * incy] += alpha * s;
            }
        }
    } else if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const bool use_omp = (N >= EGBMV_OMP_MIN && blas_omp_max_threads() > 1);
#else
        const bool use_omp = 0;
#endif
        /* Branch on use_omp in C source — `if(use_omp)` on the pragma
         * still outlines (see Addendum 16). */
#define EGBMV_T_BODY                                                         \
        for (ptrdiff_t j = 0; j < N; ++j) {                                        \
            T s = zero;                                                      \
            const ptrdiff_t i_lo = (j - KU > 0) ? (j - KU) : 0;                    \
            const ptrdiff_t i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;            \
            const ptrdiff_t k = KU - j;                                            \
            for (ptrdiff_t i = i_lo; i < i_hi; ++i) s += A_(k + i, j) * x[i];      \
            y[j] += alpha * s;                                               \
        }
        if (use_omp) {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            EGBMV_T_BODY
        } else {
            EGBMV_T_BODY
        }
#undef EGBMV_T_BODY
    } else {
#ifdef _OPENMP
        /* Thread the strided Trans gather too (the contiguous Trans path above
         * already threads). Each y[j] is a disjoint dot over column j, so it
         * partitions over j with no race; strided x is gathered to contiguous. */
        if (N >= EGBMV_OMP_MIN && blas_omp_max_threads() > 1
            && egbmv_t_omp(M, N, KL, KU, a, lda, x, incx, alpha, y, incy))
            return;
#endif
        /* Strided Trans gather (serial). */
        ptrdiff_t kx = (incx < 0) ? -(lenx - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(leny - 1) * incy : 0;
        ptrdiff_t jy = ky;
        for (ptrdiff_t j = 0; j < N; ++j) {
            T s = zero;
            ptrdiff_t ix = kx;
            const ptrdiff_t i_lo = (j - KU > 0) ? (j - KU) : 0;
            const ptrdiff_t i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
            const ptrdiff_t k = KU - j;
            /* Hoist column to a running pointer so the contiguous A read is one
             * `*col++` (single IV) instead of recomputing A_(k+i,j) each row,
             * which ptrdiff_t indexing strength-reduces into a redundant IV
             * (project_ptrdiff_conversion_regressors, Class-B). */
            const T *col = &A_(k + i_lo, j);
            for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
                s += *col++ * x[ix];
                ix += incx;
            }
            y[jy] += alpha * s;
            jy += incy;
            if (j >= KU) kx += incx;
        }
    }
}

#ifdef _OPENMP
/* Row-partitioned NoTrans gather kernel: compute y[i] for i in [lo,hi) as a
 * register-resident band dot reading x (contiguous logical order) and adding
 * alpha*s into the already-beta-scaled y[i*incy]. Same gather as the serial path.
 * Output rows partition across threads with no cross-thread write dependence --
 * no scratch, no zero-fill, no reduction, no barrier (x and y distinct). */
static void gbmv_n_rowgather(ptrdiff_t m, ptrdiff_t n, ptrdiff_t kl, ptrdiff_t ku,
                             ptrdiff_t lo, ptrdiff_t hi,
                             const T *restrict a, ptrdiff_t lda,
                             const T *restrict x, T alpha,
                             T *restrict y, ptrdiff_t incy)
{
    const ptrdiff_t s1 = lda - 1;
    for (ptrdiff_t i = lo; i < hi; ++i) {
        ptrdiff_t j_lo = (i - kl > 0) ? (i - kl) : 0;
        ptrdiff_t j_hi = (i + ku + 1 < n) ? (i + ku + 1) : n;
        const T *base = a + (ku + i);
        T s = 0.0L;
        for (ptrdiff_t j = j_lo; j < j_hi; ++j) s += base[j * s1] * x[j];
        y[i * incy] += alpha * s;
    }
}

/* Threaded NoTrans general band matvec. Each thread owns a disjoint output-row
 * range [lo,hi): it gathers y[lo,hi) reading x. No cross-thread data dependence
 * (x and y distinct) -- no barrier, no scratch beyond the strided-x gather
 * buffer. Returns 1 if handled, 0 to fall back. Carved out (noinline) so its
 * bookkeeping does not pressure the serial gather's x87 allocation. */
__attribute__((noinline)) static ptrdiff_t egbmv_n_omp(
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t kl, ptrdiff_t ku,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    T alpha, T *restrict y, ptrdiff_t incy)
{
    if (m < EGBMV_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > EGBMV_MAX_CPUS) nthreads = EGBMV_MAX_CPUS;

    /* Negative-increment base adjustment so x[i*incx], y[i*incy] walk logically.
     * lenx = n (NoTrans reads N elements of x), leny = m (writes M of y). */
    if (incx < 0) x -= (ptrdiff_t)(n - 1) * incx;
    if (incy < 0) y -= (ptrdiff_t)(m - 1) * incy;

    /* Gather strided x to contiguous (logical order) so the inner dot is unit
     * stride; y is written disjointly per thread in place. */
    const T *xptr = x;
    T *xbuf = NULL;
    if (incx != 1) {
        xbuf = (T *)malloc((size_t)n * sizeof(T));
        if (!xbuf) return 0;
        for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[(ptrdiff_t)i * incx];
        xptr = xbuf;
    }

    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t lo = ((ptrdiff_t)m * tid) / nthreads;
        ptrdiff_t hi = ((ptrdiff_t)m * (tid + 1)) / nthreads;
        gbmv_n_rowgather(m, n, kl, ku, lo, hi, a, lda, xptr, alpha, y, (ptrdiff_t)incy);
    }

    free(xbuf);
    return 1;
}

/* Column-partitioned Trans gather kernel: y[j] = alpha * Σ_i A(i,j)*x[i] for j
 * in [lo,hi), the column read as one running pointer. Output columns partition
 * across threads with no cross-thread write dependence (x and y distinct). */
static void gbmv_t_colgather(ptrdiff_t m, ptrdiff_t n, ptrdiff_t kl, ptrdiff_t ku,
                             ptrdiff_t lo, ptrdiff_t hi,
                             const T *restrict a, ptrdiff_t lda,
                             const T *restrict x, T alpha,
                             T *restrict y, ptrdiff_t incy)
{
    for (ptrdiff_t j = lo; j < hi; ++j) {
        const ptrdiff_t i_lo = (j - ku > 0) ? (j - ku) : 0;
        const ptrdiff_t i_hi = (j + kl + 1 < m) ? (j + kl + 1) : m;
        const ptrdiff_t k = ku - j;
        const T *col = &A_(k + i_lo, j);
        T s = 0.0L;
        for (ptrdiff_t i = i_lo; i < i_hi; ++i) s += *col++ * x[i];
        y[j * incy] += alpha * s;
    }
}

/* Threaded Trans general band matvec (strided x and/or y). Each thread owns a
 * disjoint output-column range [lo,hi): it writes y[lo,hi) reading x. No
 * cross-thread dependence (x and y distinct) -- no barrier, no scratch beyond
 * the strided-x gather buffer. Returns 1 if handled, 0 to fall back. Trans reads
 * M elements of x and writes N of y. */
__attribute__((noinline)) static ptrdiff_t egbmv_t_omp(
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t kl, ptrdiff_t ku,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    T alpha, T *restrict y, ptrdiff_t incy)
{
    if (n < EGBMV_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > EGBMV_MAX_CPUS) nthreads = EGBMV_MAX_CPUS;

    if (incx < 0) x -= (ptrdiff_t)(m - 1) * incx;
    if (incy < 0) y -= (ptrdiff_t)(n - 1) * incy;

    const T *xptr = x;
    T *xbuf = NULL;
    if (incx != 1) {
        xbuf = (T *)malloc((size_t)m * sizeof(T));
        if (!xbuf) return 0;
        for (ptrdiff_t i = 0; i < m; ++i) xbuf[i] = x[(ptrdiff_t)i * incx];
        xptr = xbuf;
    }

    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t lo = ((ptrdiff_t)n * tid) / nthreads;
        ptrdiff_t hi = ((ptrdiff_t)n * (tid + 1)) / nthreads;
        gbmv_t_colgather(m, n, kl, ku, lo, hi, a, lda, xptr, alpha, y, (ptrdiff_t)incy);
    }

    free(xbuf);
    return 1;
}
#endif /* _OPENMP */

EPBLAS_FACADE_GBMV(egbmv, T)

#undef A_
