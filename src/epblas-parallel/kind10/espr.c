/*
 * espr — kind10 (long double) symmetric packed rank-1 update.
 *   A := alpha*x*x^T + A
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include "../common/epblas_facade.h"
#include "../common/blas_omp.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define ESPR_OMP_MIN 64

typedef long double TR;


static void espr_core(
    char uplo,
    ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict x, ptrdiff_t incx,
    TR *restrict ap)
{
    const TR alpha = *alpha_;
    const TR zero = 0.0L;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == zero) return;

    if (incx == 1) {
        const bool use_omp = (n >= ESPR_OMP_MIN && blas_omp_max_threads() > 1);
        /* Branching on use_omp at the outer level — gcc with -fopenmp
         * still outlines the loop body into a `._omp_fn.0` function
         * even with `if(use_omp)` clause on the pragma, and the runtime
         * pays GOMP_parallel setup + omp_get_{num,thread}_num overhead
         * per call (~µs scale). At OMP=1 that's a measurable fraction
         * of the call for small-N. The L path happens to amortize this
         * cost better; the U path's per-outer-j work is smaller, so
         * the same fixed dispatch cost shows up as a bigger ratio gap.
         * Two separate loop bodies, one with pragma, one without.
         *
         * schedule(static, 8): plain static dumps the heavy triangle end on
         * one thread (column j touches j+1 (U) / N-j (L) packed elems), capping
         * threading at ~1.8x; a balanced schedule fixes that. Among balanced
         * options a MODERATE chunk beats cyclic chunk-1 here: adjacent packed
         * columns are contiguous in ap, so chunk-1 puts every neighbour pair on
         * different threads (a cross-thread cache-line boundary at every
         * column). This light real rank-1 body exposes the false sharing —
         * static,8 is ~2-8% faster than static,1 across both uplo and all N
         * (measured 2026-06-02). Heavier bodies (complex rank-1, any rank-2)
         * hide it and prefer static,1; see yhpr/espr2/yhpr2. */
        if (UPLO == 'U') {
            if (use_omp) {
#ifdef _OPENMP
                #pragma omp parallel for schedule(static, 8)
#endif
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != zero) {
                        const TR tmp = alpha * x[j];
                        TR *restrict apk  = &ap[(size_t)j * (j + 1) / 2];
                        TR *restrict aend = apk + j + 1;
                        const TR *restrict xp = x;
                        for (; apk < aend; ++apk, ++xp) *apk += *xp * tmp;
                    }
                }
            } else {
                /* Inner loop uses Addendum-7 char* byte-offset shared-
                 * index walk so gcc emits one `add` per iter (8 insns)
                 * instead of two pointer increments (9 insns). Migrated
                 * gfortran picks shared-index for the L path naturally
                 * but not for U — the U path's two pointers start at
                 * different bases (apk = &ap[kk], xp = &x[0]) so gcc
                 * doesn't fold them; the explicit char* form forces it. */
                TR *restrict apk_base = ap;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != zero) {
                        const TR tmp = alpha * x[j];
                        char *restrict apkb = (char *)apk_base;
                        const char *restrict xpb = (const char *)x;
                        const size_t end = (size_t)(j + 1) * sizeof(TR);
                        for (size_t k = 0; k < end; k += sizeof(TR)) {
                            TR *apk  = (TR *)(apkb + k);
                            const TR *xp = (const TR *)(xpb + k);
                            *apk += *xp * tmp;
                        }
                    }
                    apk_base += (j + 1);
                }
            }
        } else {
            if (use_omp) {
#ifdef _OPENMP
                #pragma omp parallel for schedule(static, 8)
#endif
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != zero) {
                        const TR tmp = alpha * x[j];
                        TR *restrict apk  = &ap[(size_t)j * n - (size_t)j * (j - 1) / 2];
                        TR *restrict aend = apk + (n - j);
                        const TR *restrict xp = &x[j];
                        for (; apk < aend; ++apk, ++xp) *apk += *xp * tmp;
                    }
                }
            } else {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != zero) {
                        const TR tmp = alpha * x[j];
                        TR *restrict apk  = &ap[(size_t)j * n - (size_t)j * (j - 1) / 2];
                        TR *restrict aend = apk + (n - j);
                        const TR *restrict xp = &x[j];
                        for (; apk < aend; ++apk, ++xp) *apk += *xp * tmp;
                    }
                }
            }
        }
    } else {
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

EPBLAS_FACADE_SPR(espr, TR, TR)
