/*
 * yhpr — kind10 complex Hermitian packed rank-1 update.
 *   A := alpha*x*x^H + A, alpha real.
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include "../common/epblas_facade.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

/* RECALIBRATED 2026-06-07 (was 64): stale libgomp-era break-even; iomp5 hot-team
 * reuse lets this O(N^2) complex Hermitian packed rank-1 update thread from N=24.
 * Measured par4/par1 (taskset 0-3, min-of-10): N=24 0.70/0.68, N=32 0.54/0.52,
 * N=64 0.35, N=128 0.29. N=20 marginal (0.94) and N=16 loses (1.15), so 24 is
 * the robust floor. Bit-exact (relerr 0). Uniform across the y* rank family. */
#define YHPR_OMP_MIN 24

typedef _Complex long double TC;
typedef long double TR;
static inline TC cconj(TC z) { return ~z; }


static void yhpr_core(
    char uplo,
    ptrdiff_t n,
    const TR *alpha_,
    const TC *restrict x, ptrdiff_t incx,
    TC *restrict ap)
{
    const TR alpha = *alpha_;
    const TR rzero = 0.0L;
    const TC czero = 0.0L + 0.0Li;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == rzero) return;

    if (incx == 1) {
        /* schedule(static, 1): plain static dumps the heavy triangle end on one
         * thread (caps threading ~1.8x); cyclic static,1 interleaves short and
         * long columns, balancing both uplo. Chunk-1 is the right grain here —
         * this complex rank-1 body does enough fp80 work per written element to
         * hide the adjacent-column false sharing, so the finest balance wins
         * (static,1 ~= or beats static,8; measured 2026-06-02, vs the lighter
         * real rank-1 espr which prefers static,8). The serial path (OMP=1, or
         * if(use_omp) false) is already at parity with the reference — the
         * rank-1 loop body is small enough not to spill when outlined — so no
         * noinline carve-out is needed (unlike yhpr2). */
        /* The `#pragma omp parallel for if(use_omp)` outlines the loop body even
         * when use_omp is false, so the serial path pays the outlining tax AND
         * recomputes the packed column base kk from j each column (Lower's
         * j*n-j*(j-1)/2 is heavier than Upper's j*(j+1)/2 — Lower lost more at
         * small N). Split source-level: OMP keeps kk-from-j (threads need it
         * per-column); serial runs an un-outlined loop with a RUNNING kk
         * (kk += j+1 / kk += n-j). Bit-identical (running kk == the closed form). */
#ifdef _OPENMP
        const bool use_omp = (n >= YHPR_OMP_MIN && blas_omp_max_threads() > 1);
#else
        const bool use_omp = false;
#endif
        if (UPLO == 'U') {
#ifdef _OPENMP
            if (use_omp) {
                #pragma omp parallel for schedule(static, 1)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    const ptrdiff_t kk = (j * (j + 1)) / 2;
                    if (x[j] != czero) {
                        const TC tmp = alpha * cconj(x[j]);
                        for (ptrdiff_t i = 0; i < j; ++i) ap[kk + i] += x[i] * tmp;
                        ap[kk + j] = (TR)__real__ ap[kk + j] + (TR)__real__ (x[j] * tmp);
                    } else {
                        ap[kk + j] = (TR)__real__ ap[kk + j];
                    }
                }
            } else
#endif
            {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != czero) {
                        const TC tmp = alpha * cconj(x[j]);
                        for (ptrdiff_t i = 0; i < j; ++i) ap[kk + i] += x[i] * tmp;
                        ap[kk + j] = (TR)__real__ ap[kk + j] + (TR)__real__ (x[j] * tmp);
                    } else {
                        ap[kk + j] = (TR)__real__ ap[kk + j];
                    }
                    kk += j + 1;
                }
            }
        } else {
#ifdef _OPENMP
            if (use_omp) {
                #pragma omp parallel for schedule(static, 1)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    const ptrdiff_t kk = j * n - (j * (j - 1)) / 2;
                    if (x[j] != czero) {
                        const TC tmp = alpha * cconj(x[j]);
                        ap[kk] = (TR)__real__ ap[kk] + (TR)__real__ (tmp * x[j]);
                        for (ptrdiff_t i = j + 1; i < n; ++i) ap[kk + (i - j)] += x[i] * tmp;
                    } else {
                        ap[kk] = (TR)__real__ ap[kk];
                    }
                }
            } else
#endif
            {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != czero) {
                        const TC tmp = alpha * cconj(x[j]);
                        ap[kk] = (TR)__real__ ap[kk] + (TR)__real__ (tmp * x[j]);
                        for (ptrdiff_t i = j + 1; i < n; ++i) ap[kk + (i - j)] += x[i] * tmp;
                    } else {
                        ap[kk] = (TR)__real__ ap[kk];
                    }
                    kk += n - j;
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t kk = 0;
        if (UPLO == 'U') {
            ptrdiff_t jx = kx;
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != czero) {
                    const TC tmp = alpha * cconj(x[jx]);
                    ptrdiff_t ix = kx;
                    for (ptrdiff_t k = kk; k < kk + j; ++k) {
                        ap[k] += x[ix] * tmp;
                        ix += incx;
                    }
                    ap[kk + j] = (TR)__real__ ap[kk + j] + (TR)__real__ (x[jx] * tmp);
                } else {
                    ap[kk + j] = (TR)__real__ ap[kk + j];
                }
                jx += incx;
                kk += j + 1;
            }
        } else {
            ptrdiff_t jx = kx;
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != czero) {
                    const TC tmp = alpha * cconj(x[jx]);
                    ap[kk] = (TR)__real__ ap[kk] + (TR)__real__ (tmp * x[jx]);
                    ptrdiff_t ix = jx;
                    for (ptrdiff_t k = kk + 1; k < kk + n - j; ++k) {
                        ix += incx;
                        ap[k] += x[ix] * tmp;
                    }
                } else {
                    ap[kk] = (TR)__real__ ap[kk];
                }
                jx += incx;
                kk += n - j;
            }
        }
    }
}

EPBLAS_FACADE_SPR(yhpr, TR, TC)
