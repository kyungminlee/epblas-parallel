/*
 * yher2 — kind10 complex Hermitian rank-2 update.
 *   A := alpha · x · yᴴ + conj(alpha) · y · xᴴ + A
 * alpha is complex. Diagonal of A stays real on output.
 */

#include <stddef.h>
#include <ctype.h>
#include "../common/epblas_facade.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

/* RECALIBRATED 2026-06-07 (was 64): stale libgomp-era break-even; iomp5 hot-team
 * reuse lets this O(N^2) complex Hermitian rank-2 update thread from N=24.
 * Measured par4/par1 (taskset 0-3, min-of-10): N=24 0.62/0.56, N=32 0.47/0.44,
 * N=64 0.30, N=128 0.27. N=20 marginal (0.69-0.88), so 24 is the robust floor.
 * Bit-exact (relerr 0). Uniform across the y* rank-update family. */
#define YHER2_OMP_MIN 24

typedef _Complex long double T;
typedef long double R;
static const T ZERO = 0.0L + 0.0Li;
static inline T cconj(T z) { return ~z; }

static inline char up(char c) {
    return (char)toupper((unsigned char)c);
}

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
__attribute__((noinline))
static void yher2_contig_U(ptrdiff_t j0, ptrdiff_t j1, T alpha,
                           const T *restrict x, const T *restrict y,
                           T *restrict a, ptrdiff_t lda)
{
    for (ptrdiff_t j = j0; j < j1; ++j) {
        T *aj = &A_(0, j);
        if (x[j] != ZERO || y[j] != ZERO) {
            const T t1 = alpha * cconj(y[j]);
            const T t2 = cconj(alpha * x[j]);
            const R dadd = __real__ (x[j] * t1 + y[j] * t2);
            for (ptrdiff_t i = 0; i < j; ++i) aj[i] += x[i] * t1 + y[i] * t2;
            aj[j] = __real__ aj[j] + dadd;
        }
    }
}

__attribute__((noinline))
static void yher2_contig_L(ptrdiff_t j0, ptrdiff_t j1, ptrdiff_t N, T alpha,
                           const T *restrict x, const T *restrict y,
                           T *restrict a, ptrdiff_t lda)
{
    for (ptrdiff_t j = j0; j < j1; ++j) {
        T *aj = &A_(0, j);
        if (x[j] != ZERO || y[j] != ZERO) {
            const T t1 = alpha * cconj(y[j]);
            const T t2 = cconj(alpha * x[j]);
            aj[j] = __real__ aj[j] + __real__ (x[j] * t1 + y[j] * t2);
            for (ptrdiff_t i = j + 1; i < N; ++i) aj[i] += x[i] * t1 + y[i] * t2;
        }
    }
}

/* Strided off-diagonal run: aj[i] += x[ix]·t1 + y[iy]·t2 for i in [0,cnt),
 * x/y walked by stride. Carved noinline so the complex-MAC loop keeps t1/t2
 * register-resident (see the strided-fallback comment below). */
__attribute__((noinline))
static void yher2_strided_run(ptrdiff_t cnt, T t1, T t2, T *restrict aj,
                              const T *restrict x, ptrdiff_t incx, ptrdiff_t ix,
                              const T *restrict y, ptrdiff_t incy, ptrdiff_t iy)
{
    for (ptrdiff_t i = 0; i < cnt; ++i) {
        aj[i] += x[ix] * t1 + y[iy] * t2;
        ix += incx; iy += incy;
    }
}

static void yher2_core(
    char uplo,
    ptrdiff_t N,
    const T *alpha_,
    const T *restrict x, ptrdiff_t incx,
    const T *restrict y, ptrdiff_t incy,
    T *restrict a, ptrdiff_t lda)
{
    const T alpha = *alpha_;
    const char UPLO = up(uplo);

    if (N == 0 || alpha == ZERO) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const ptrdiff_t use_omp = (N >= YHER2_OMP_MIN && blas_omp_max_threads() > 1
                             && !omp_in_parallel());
#else
        const ptrdiff_t use_omp = 0;
#endif
        /* All column work runs through yher2_contig_{U,L}().  The serial path
         * issues ONE call over [0,N) — the hot loop is inlined there with no
         * per-column overhead.  The OMP path round-robins single columns:
         * schedule(static, 1) balances the triangular work (linear in (N-j) for
         * L, j for U) — Rule 49; plain schedule(static) hands thread 0 the
         * heaviest block.  Branching on use_omp at C source level (Add-16)
         * avoids `if(use_omp)` outlining the region and paying GOMP_parallel at
         * OMP=1; the per-column call cost is hidden by parallelism. */
        if (use_omp) {
#ifdef _OPENMP
            if (UPLO == 'L') {
                #pragma omp parallel for schedule(static, 1)
                for (ptrdiff_t j = 0; j < N; ++j)
                    yher2_contig_L(j, j + 1, N, alpha, x, y, a, lda);
            } else {
                #pragma omp parallel for schedule(static, 1)
                for (ptrdiff_t j = 0; j < N; ++j)
                    yher2_contig_U(j, j + 1, alpha, x, y, a, lda);
            }
#endif
        } else if (UPLO == 'L') {
            yher2_contig_L(0, N, N, alpha, x, y, a, lda);
        } else {
            yher2_contig_U(0, N, alpha, x, y, a, lda);
        }
    } else {
        /* General-stride fallback — hoist the matrix column to aj[i] and
         * walk the strided vectors with running indices (Class-B fix). The off-diagonal run
         * is carved into a noinline helper for the same reason the contiguous
         * path is (project_x87_accumulator_spill variant 3): inlined amid the
         * strided/uplo scaffolding gcc spills the complex temporaries t1/t2,
         * costing ~5-10% on UPPER (and LOWER); in isolation they stay on the
         * x87 stack. Bit-identical to the inline form. */
        const ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        const ptrdiff_t ky = (incy < 0) ? -(N - 1) * incy : 0;
        ptrdiff_t jx = kx, jy = ky;
        for (ptrdiff_t j = 0; j < N; ++j) {
            const T xj = x[jx];
            const T yj = y[jy];
            if (xj != ZERO || yj != ZERO) {
                const T temp1 = alpha * cconj(yj);
                const T temp2 = cconj(alpha * xj);
                T *aj = &A_(0, j);
                if (UPLO == 'L') {
                    yher2_strided_run(N - j - 1, temp1, temp2, aj + j + 1,
                                      x, incx, jx + incx, y, incy, jy + incy);
                    aj[j] = __real__ aj[j] + __real__ (xj * temp1 + yj * temp2);
                } else {
                    yher2_strided_run(j, temp1, temp2, aj,
                                      x, incx, kx, y, incy, ky);
                    aj[j] = __real__ aj[j] + __real__ (xj * temp1 + yj * temp2);
                }
            }
            jx += incx; jy += incy;
        }
    }
}

EPBLAS_FACADE_SYR2(yher2, T)

#undef A_
