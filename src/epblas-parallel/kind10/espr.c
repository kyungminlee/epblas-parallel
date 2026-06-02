/*
 * espr — kind10 (long double) symmetric packed rank-1 update.
 *   A := alpha*x*x^T + A
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define ESPR_OMP_MIN 64

typedef long double T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

void espr_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *restrict x, const int *incx_,
    T *restrict ap,
    size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_;
    const T alpha = *alpha_;
    const T zero = 0.0L;
    const char UPLO = up(uplo);

    if (N == 0 || alpha == zero) return;

    if (incx == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= ESPR_OMP_MIN && blas_omp_max_threads() > 1);
#else
        const int use_omp = 0;
#endif
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
                for (int j = 0; j < N; ++j) {
                    if (x[j] != zero) {
                        const T tmp = alpha * x[j];
                        T *restrict apk  = &ap[(size_t)j * (j + 1) / 2];
                        T *restrict aend = apk + j + 1;
                        const T *restrict xp = x;
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
                T *restrict apk_base = ap;
                for (int j = 0; j < N; ++j) {
                    if (x[j] != zero) {
                        const T tmp = alpha * x[j];
                        char *restrict apkb = (char *)apk_base;
                        const char *restrict xpb = (const char *)x;
                        const size_t end = (size_t)(j + 1) * sizeof(T);
                        for (size_t k = 0; k < end; k += sizeof(T)) {
                            T *apk  = (T *)(apkb + k);
                            const T *xp = (const T *)(xpb + k);
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
                for (int j = 0; j < N; ++j) {
                    if (x[j] != zero) {
                        const T tmp = alpha * x[j];
                        T *restrict apk  = &ap[(size_t)j * N - (size_t)j * (j - 1) / 2];
                        T *restrict aend = apk + (N - j);
                        const T *restrict xp = &x[j];
                        for (; apk < aend; ++apk, ++xp) *apk += *xp * tmp;
                    }
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    if (x[j] != zero) {
                        const T tmp = alpha * x[j];
                        T *restrict apk  = &ap[(size_t)j * N - (size_t)j * (j - 1) / 2];
                        T *restrict aend = apk + (N - j);
                        const T *restrict xp = &x[j];
                        for (; apk < aend; ++apk, ++xp) *apk += *xp * tmp;
                    }
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int kk = 0;
        if (UPLO == 'U') {
            int jx = kx;
            for (int j = 0; j < N; ++j) {
                if (x[jx] != zero) {
                    const T tmp = alpha * x[jx];
                    int ix = kx;
                    for (int k = kk; k < kk + j + 1; ++k) {
                        ap[k] += x[ix] * tmp;
                        ix += incx;
                    }
                }
                jx += incx;
                kk += j + 1;
            }
        } else {
            int jx = kx;
            for (int j = 0; j < N; ++j) {
                if (x[jx] != zero) {
                    const T tmp = alpha * x[jx];
                    int ix = jx;
                    for (int k = kk; k < kk + N - j; ++k) {
                        ap[k] += x[ix] * tmp;
                        ix += incx;
                    }
                }
                jx += incx;
                kk += N - j;
            }
        }
    }
}
