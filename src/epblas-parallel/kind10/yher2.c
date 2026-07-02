/*
 * yher2 — kind10 complex Hermitian rank-2 update.
 *   A := alpha · x · yᴴ + conj(alpha) · y · xᴴ + A
 * alpha is complex. Diagonal of A stays real on output.
 */

#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include "rank2c_x87_core.h"
#include "../common/blas_char.h"
#include <ctype.h>
#include "../common/epblas_facade.h"
#include "../common/blas_omp.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/* RECALIBRATED 2026-06-07 (was 64): stale libgomp-era break-even; iomp5 hot-team
 * reuse lets this O(N^2) complex Hermitian rank-2 update thread from N=24.
 * Measured par4/par1 (taskset 0-3, min-of-10): N=24 0.62/0.56, N=32 0.47/0.44,
 * N=64 0.30, N=128 0.27. N=20 marginal (0.69-0.88), so 24 is the robust floor.
 * Bit-exact (relerr 0). Uniform across the y* rank-update family. */
#define YHER2_OMP_MIN 24

typedef _Complex long double TC;
typedef long double R;
static const TC ZERO = 0.0L + 0.0Li;
static inline TC cconj(TC z) { return ~z; }


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* Contiguous (incx=incy=1) Hermitian rank-2 update over the column range
 * [j0,j1).  Per Netlib ZHER2: TEMP1 = ALPHA·conj(Y(J)), TEMP2 = conj(ALPHA·X(J)),
 * A(i,j) += X(i)·TEMP1 + Y(i)·TEMP2; the diagonal entry stays real.
 *
 * UPPER and LOWER are SEPARATE (non-inlined) functions — not one branchy body
 * — for two reasons:
 *   1. Isolation/regalloc — each is compiled away from the other and from the
 *      strided/OMP scaffolding, so gcc keeps the complex temporaries on the
 *      8-deep x87 register stack.  A shared body forces a compromise schedule
 *      that suits one triangle and costs the other ~4%; epblas-openblas emits
 *      two distinct loops for exactly this reason.
 *   2. The serial path calls one ONCE over [0,N) — its hot loop is inlined with
 *      zero per-column overhead; the OMP path calls it per column, where that
 *      overhead is hidden by parallelism.
 * Each column streams sequentially — UPPER writes 0..j-1 then j, LOWER writes
 * j then j+1..N-1.  The diagonal's real contribution is reduced to a SCALAR
 * (dadd) BEFORE the off-diagonal loop, consuming x[j]/y[j] up front; only that
 * one long double is carried across the loop (instead of the two complex
 * endpoints), which keeps t1/t2 register-resident (fld=9/iter).  Diagonal and
 * off-diagonal entries are disjoint, so the order is bit-identical. */
/* Off-diagonal run via the shared hand-tuned x87 asm core (yr2c_run), then the
 * Hermitian diagonal forced real. Upper run is [0,j) at the array bases; lower
 * run is [j+1,N) at the j+1 offsets. The matrix column aj = &A_(0,j) is
 * contiguous in i, so the core's contiguous c-advance applies unchanged. Same
 * kernel as the packed yhpr2 — only the column base and diagonal index differ.
 * always_inline so the serial single-call-over-[0,N) path inlines with no
 * per-column overhead; the OMP path calls per column where that is hidden. */
__attribute__((always_inline))
static inline void yher2_contig_U(ptrdiff_t j0, ptrdiff_t j1, TC alpha,
                           const TC *restrict x, const TC *restrict y,
                           TC *restrict a, ptrdiff_t lda)
{
    for (ptrdiff_t j = j0; j < j1; ++j) {
        TC *aj = &A_(0, j);
        if (x[j] != ZERO || y[j] != ZERO) {
            const TC t1 = alpha * cconj(y[j]);
            const TC t2 = cconj(alpha * x[j]);
            const R dadd = __real__ (x[j] * t1 + y[j] * t2);
            yr2c_run(j, t1, t2, x, y, aj);
            aj[j] = __real__ aj[j] + dadd;
        }
    }
}

__attribute__((always_inline))
static inline void yher2_contig_L(ptrdiff_t j0, ptrdiff_t j1, ptrdiff_t n, TC alpha,
                           const TC *restrict x, const TC *restrict y,
                           TC *restrict a, ptrdiff_t lda)
{
    for (ptrdiff_t j = j0; j < j1; ++j) {
        TC *aj = &A_(0, j);
        if (x[j] != ZERO || y[j] != ZERO) {
            const TC t1 = alpha * cconj(y[j]);
            const TC t2 = cconj(alpha * x[j]);
            aj[j] = __real__ aj[j] + __real__ (x[j] * t1 + y[j] * t2);
            yr2c_run(n - j - 1, t1, t2, x + j + 1, y + j + 1, aj + j + 1);
        }
    }
}

/* Strided column twins: x/y walked in place via yr2c_run_strided (no gather),
 * matching gfortran's strided .L40. The matrix column stays contiguous. */
__attribute__((always_inline))
static inline void yher2_strided_U(ptrdiff_t j, TC t1, TC t2, TC *restrict aj,
                            const TC *restrict x, ptrdiff_t incx, ptrdiff_t kx,
                            const TC *restrict y, ptrdiff_t incy, ptrdiff_t ky) {
    const ptrdiff_t es = (ptrdiff_t)sizeof(TC);
    yr2c_run_strided(j, t1, t2, x + kx, y + ky, aj, incx * es, incy * es);
}

__attribute__((always_inline))
static inline void yher2_strided_L(ptrdiff_t mo, TC t1, TC t2, TC *restrict aj1,
                            const TC *restrict x, ptrdiff_t incx, ptrdiff_t jx,
                            const TC *restrict y, ptrdiff_t incy, ptrdiff_t jy) {
    const ptrdiff_t es = (ptrdiff_t)sizeof(TC);
    yr2c_run_strided(mo, t1, t2, x + jx + incx, y + jy + incy, aj1,
                     incx * es, incy * es);
}

/* Unit-stride dispatch, shared by the native-contiguous path and the gathered
 * strided path.  The serial path issues ONE call over [0,N) — the hot loop is
 * inlined there with no per-column overhead.  The OMP path round-robins single
 * columns: schedule(static, 1) balances the triangular work (linear in (N-j)
 * for L, j for U) — Rule 49; plain schedule(static) hands thread 0 the heaviest
 * block.  Branching on use_omp at C source level (Add-16) avoids `if(use_omp)`
 * outlining the region and paying GOMP_parallel at OMP=1. */
static void yher2_contig(char UPLO, ptrdiff_t n, TC alpha,
                         const TC *restrict x, const TC *restrict y,
                         TC *restrict a, ptrdiff_t lda)
{
    const bool use_omp = (n >= YHER2_OMP_MIN && blas_omp_should_thread());
    if (use_omp) {
#ifdef _OPENMP
        if (UPLO == 'L') {
            #pragma omp parallel for schedule(static, 1)
            for (ptrdiff_t j = 0; j < n; ++j)
                yher2_contig_L(j, j + 1, n, alpha, x, y, a, lda);
        } else {
            #pragma omp parallel for schedule(static, 1)
            for (ptrdiff_t j = 0; j < n; ++j)
                yher2_contig_U(j, j + 1, alpha, x, y, a, lda);
        }
#endif
    } else if (UPLO == 'L') {
        yher2_contig_L(0, n, n, alpha, x, y, a, lda);
    } else {
        yher2_contig_U(0, n, alpha, x, y, a, lda);
    }
}

/* Serial strided dispatch: walk columns in place via the strided asm core (no
 * gather). Used only when the run would NOT thread; the threaded strided path
 * gathers into the contiguous core (preserving omp scaling). Column order
 * matches yher2_contig_{U,L} (U: off-diag then diagonal; L: diagonal then
 * off-diag), so bit-identical. */
static void yher2_strided(char UPLO, ptrdiff_t n, TC alpha,
                          const TC *restrict x, ptrdiff_t incx,
                          const TC *restrict y, ptrdiff_t incy,
                          TC *restrict a, ptrdiff_t lda)
{
    const ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
    const ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
    ptrdiff_t jx = kx, jy = ky;
    for (ptrdiff_t j = 0; j < n; ++j) {
        const TC xj = x[jx];
        const TC yj = y[jy];
        if (xj != ZERO || yj != ZERO) {
            const TC t1 = alpha * cconj(yj);
            const TC t2 = cconj(alpha * xj);
            TC *aj = &A_(0, j);
            if (UPLO == 'L') {
                aj[j] = __real__ aj[j] + __real__ (xj * t1 + yj * t2);
                yher2_strided_L(n - j - 1, t1, t2, aj + j + 1,
                                x, incx, jx, y, incy, jy);
            } else {
                yher2_strided_U(j, t1, t2, aj, x, incx, kx, y, incy, ky);
                aj[j] = __real__ aj[j] + __real__ (xj * t1 + yj * t2);
            }
        }
        jx += incx; jy += incy;
    }
}

static void yher2_core(
    char uplo,
    ptrdiff_t n,
    const TC *alpha_,
    const TC *restrict x, ptrdiff_t incx,
    const TC *restrict y, ptrdiff_t incy,
    TC *restrict a, ptrdiff_t lda)
{
    const TC alpha = *alpha_;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == ZERO) return;

    if (incx == 1 && incy == 1) {
        yher2_contig(UPLO, n, alpha, x, y, a, lda);
        return;
    }

    /* Serial strided: walk the inputs in place via the strided asm core (no
     * gather), exactly matching gfortran's strided .L40. Only the threaded path
     * below gathers — there the O(N) gather is dwarfed by the O(N^2) threaded
     * work and buys the contiguous core's omp scaling (the in-place walk never
     * threaded); at serial small N the gather is pure overhead. The predicate
     * mirrors yher2_contig's use_omp so a would-thread run is never sent down
     * the serial path. See project_l2_strided_gather. */
#ifdef _OPENMP
    const bool would_thread = (n >= YHER2_OMP_MIN && blas_omp_should_thread());
#else
    const bool would_thread = false;
#endif
    if (!would_thread) {
        yher2_strided(UPLO, n, alpha, x, incx, y, incy, a, lda);
        return;
    }

    /* Threaded strided: gather x/y into contiguous scratch and run the threaded
     * stride-1 core. x/y read-only (only A written) so no scatter-back. Direct
     * strided walk is the alloc-failure fallback. */
    {
        const ptrdiff_t kx0 = (incx < 0) ? -(n - 1) * incx : 0;
        const ptrdiff_t ky0 = (incy < 0) ? -(n - 1) * incy : 0;
        /* Exact fit: the gather writes xc[0..n-1] and yc[0..n-1] with
         * yc = stackbuf + n (max offset 2n-1), so the threshold and the
         * array length must move together. */
        enum { YHER2_STACK_N = 256 };
        TC stackbuf[2 * YHER2_STACK_N];
        _Static_assert(2 * YHER2_STACK_N * sizeof(TC) <= sizeof(stackbuf),
                       "yher2 stack-gather threshold exceeds stackbuf");
        TC *heap = NULL;
        TC *xc, *yc;
        if (n <= YHER2_STACK_N) {
            xc = stackbuf; yc = stackbuf + n;
        } else {
            heap = (TC *)malloc((size_t)2 * n * sizeof(TC));
            xc = heap; yc = heap ? heap + n : NULL;
        }
        if (xc && yc) {
            ptrdiff_t ix = kx0, iy = ky0;
            for (ptrdiff_t k = 0; k < n; ++k) {
                xc[k] = x[ix]; yc[k] = y[iy];
                ix += incx; iy += incy;
            }
            yher2_contig(UPLO, n, alpha, xc, yc, a, lda);
            free(heap);
            return;
        }
        free(heap);
        yher2_strided(UPLO, n, alpha, x, incx, y, incy, a, lda);
    }
}

EPBLAS_FACADE_SYR2(yher2, TC)

#undef A_
