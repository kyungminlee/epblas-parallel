/*
 * espr2 — kind10 (long double) symmetric packed rank-2 update.
 *   A := alpha*x*y^T + alpha*y*x^T + A
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define ESPR2_OMP_MIN 64

typedef long double T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

/* Per-column rank-2 updates, carved out as their own functions so the inner
 * loop compiles with clean x87 register allocation. Inlined into the
 * `omp parallel for` body, the upper-triangle loop loses ~10% (the outlined
 * region spills the kept-resident operands); keeping it a separate noinline
 * function restores parity with the reference and lets both the serial and
 * threaded paths share one tight loop. */
__attribute__((noinline))
static void espr2_col_upper(int j, T t1, T t2,
                            const T *restrict x, const T *restrict y, T *restrict ap) {
    T *restrict c = ap + (size_t)j * (j + 1) / 2;
    for (int i = 0; i <= j; ++i) c[i] += x[i] * t1 + y[i] * t2;
}

__attribute__((noinline))
static void espr2_col_lower(int j, int N, T t1, T t2,
                            const T *restrict x, const T *restrict y, T *restrict ap) {
    T *restrict c = ap + ((size_t)j * N - (size_t)j * (j - 1) / 2) - j;
    for (int i = j; i < N; ++i) c[i] += x[i] * t1 + y[i] * t2;
}

void espr2_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *restrict x, const int *incx_,
    const T *restrict y, const int *incy_,
    T *restrict ap,
    size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, incy = *incy_;
    const T alpha = *alpha_;
    const T zero = 0.0L;
    const char UPLO = up(uplo);

    if (N == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
        /* schedule(static, 8): column j touches j+1 (upper) or N-j (lower)
         * packed elements, so a contiguous static block hands one thread the
         * heavy triangle end and starves the rest (par caps at ~2x on 4 cores).
         * A balanced schedule fixes that. Among balanced options a MODERATE
         * chunk beats cyclic chunk-1 for this real rank-2 body: adjacent packed
         * columns are contiguous in ap, so chunk-1 puts every neighbour pair on
         * different threads (false sharing). static,8 is ~1-2% faster than
         * static,1 here (measured 2026-06-02). NOTE the complex rank-2 twin
         * yhpr2 keeps static,1 — its heavier body fully hides the false sharing
         * and the finest balance wins there; the lighter real rank-1 espr also
         * uses static,8. */
        if (UPLO == 'U') {
#ifdef _OPENMP
            const int use_omp = (N >= ESPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static, 8)
#endif
            for (int j = 0; j < N; ++j) {
                if (x[j] != zero || y[j] != zero)
                    espr2_col_upper(j, alpha * y[j], alpha * x[j], x, y, ap);
            }
        } else {
#ifdef _OPENMP
            const int use_omp = (N >= ESPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static, 8)
#endif
            for (int j = 0; j < N; ++j) {
                if (x[j] != zero || y[j] != zero)
                    espr2_col_lower(j, N, alpha * y[j], alpha * x[j], x, y, ap);
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        int kk = 0;
        int jx = kx, jy = ky;
        if (UPLO == 'U') {
            for (int j = 0; j < N; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const T t1 = alpha * y[jy];
                    const T t2 = alpha * x[jx];
                    int ix = kx, iy = ky;
                    for (int k = kk; k < kk + j + 1; ++k) {
                        ap[k] += x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                }
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            for (int j = 0; j < N; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const T t1 = alpha * y[jy];
                    const T t2 = alpha * x[jx];
                    int ix = jx, iy = jy;
                    for (int k = kk; k < kk + N - j; ++k) {
                        ap[k] += x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                }
                jx += incx; jy += incy;
                kk += N - j;
            }
        }
    }
}
