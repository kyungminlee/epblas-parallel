/*
 * yhpmv — kind10 complex Hermitian packed matrix-vector multiply.
 *   y := alpha*A*x + beta*y
 *
 * Hermitian twin of espmv: the reference sweep is column-oriented with
 * cross-column writes (column j updates y and accumulates y[j]) so a bare
 * `omp parallel for` over columns would race on y[i]. The threaded path
 * (contiguous, large N) mirrors OpenBLAS hpmv_thread / spmv_thread: a
 * sqrt-balanced contiguous column partition, each thread accumulating the
 * bare A*x for its columns into a private size-N slot, then a range-limited
 * controller reduction applies alpha into y. The reflected (sub-)triangle
 * entry is the conjugate of the stored one and the diagonal is real, exactly
 * as in the serial sweep. The per-thread kernel keeps this overlay's faster
 * sequential packed-index carry (running k).
 */

#include <stddef.h>
#include <ctype.h>
#include "../common/epblas_facade.h"
#ifdef _OPENMP
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef _Complex long double T;
typedef long double TR;
static inline T cconj(T z) { return ~z; }

/* RECALIBRATED 2026-06-07 (was 16384, i.e. N>128): the old break-even matched
 * OpenBLAS zhpmv's MULTI_THREAD_MINIMAL, calibrated under libgomp's fork/join
 * wakeup tax. The build now links Intel libiomp5 (hot-team reuse removes that
 * tax), so this compute-heavy O(N^2) complex packed Hermitian matvec threads
 * from far lower N. Measured par4/par1 (taskset 0-3, min-of-10): N=24 0.87/0.87,
 * N=32 0.74/0.71, N=64 0.46, N=128 0.33; N=20 loses (1.13). Gate is n*n > MIN,
 * so 575 engages at N>=24 (24^2=576 > 575, 23^2=529 <= 575). omp4-vs-omp1
 * relerr ~1e-18 (fp80 floor; the partial-sum fold reorders within tolerance). */
#define YHPMV_OMP_MIN 575  /* N>=24 */
#define YHPMV_MAX_CPUS 256

static inline char up(char c) {
    return (char)toupper((unsigned char)c);
}

#ifdef _OPENMP
/* Sqrt-balanced contiguous column partition (OpenBLAS symv_partition, mask=3,
 * min_width=4): per-column work grows with j for UPPER, shrinks for LOWER. */
static ptrdiff_t yhpmv_partition(ptrdiff_t upper, ptrdiff_t n, ptrdiff_t nthreads, ptrdiff_t *range)
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
        if (num_cpu >= YHPMV_MAX_CPUS) break;
    }
    return num_cpu;
}
#endif

static void yhpmv_core(
    char uplo,
    ptrdiff_t N,
    const T *alpha_,
    const T *restrict ap,
    const T *restrict x, ptrdiff_t incx,
    const T *beta_,
    T *restrict y, ptrdiff_t incy)
{
    const T alpha = *alpha_, beta = *beta_;
    const T zero = 0.0L + 0.0Li, one = 1.0L + 0.0Li;
    const char UPLO = up(uplo);

    if (N == 0 || (alpha == zero && beta == one)) return;

    if (beta != one) {
        ptrdiff_t iy = (incy < 0) ? -(N - 1) * incy : 0;
        if (beta == zero) for (ptrdiff_t i = 0; i < N; ++i) { y[iy] = zero; iy += incy; }
        else              for (ptrdiff_t i = 0; i < N; ++i) { y[iy] = beta * y[iy]; iy += incy; }
    }
    if (alpha == zero) return;

    ptrdiff_t kk = 0;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if ((size_t)N * (size_t)N > YHPMV_OMP_MIN
            && blas_omp_max_threads() > 1 && !omp_in_parallel()) {
            ptrdiff_t nthreads = blas_omp_max_threads();
            if (nthreads > YHPMV_MAX_CPUS) nthreads = YHPMV_MAX_CPUS;
            ptrdiff_t range[YHPMV_MAX_CPUS + 1];
            ptrdiff_t num_cpu = yhpmv_partition(UPLO == 'U', N, nthreads, range);
            T *buf = (num_cpu > 1)
                ? (T *)calloc((size_t)num_cpu * (size_t)N, sizeof(T)) : NULL;
            if (buf) {
                const ptrdiff_t n = N;
                #pragma omp parallel num_threads(num_cpu)
                {
                    ptrdiff_t t = omp_get_thread_num();
                    ptrdiff_t m_from = range[t], m_to = range[t + 1];
                    T *restrict slot = buf + (size_t)t * (size_t)n;
                    if (UPLO == 'U') {
                        for (ptrdiff_t j = m_from; j < m_to; ++j) {
                            const T *restrict aj = &ap[(size_t)j * (size_t)(j + 1) / 2];
                            const T t1 = x[j];
                            T t2 = zero;
                            for (ptrdiff_t i = 0; i < j; ++i) {
                                slot[i] += t1 * aj[i];
                                t2      += cconj(aj[i]) * x[i];
                            }
                            slot[j] += t1 * (TR)__real__ aj[j] + t2;  /* real diag */
                        }
                    } else {
                        for (ptrdiff_t j = m_from; j < m_to; ++j) {
                            const T *restrict aj = &ap[(size_t)j * (size_t)(2 * n - j - 1) / 2];
                            const T t1 = x[j];
                            T t2 = zero;
                            slot[j] += t1 * (TR)__real__ aj[j];       /* real diag */
                            for (ptrdiff_t i = j + 1; i < n; ++i) {
                                slot[i] += t1 * aj[i];
                                t2      += cconj(aj[i]) * x[i];
                            }
                            slot[j] += t2;
                        }
                    }
                }
                /* Range-limited reduction: UPPER thread touched [0,range[t+1]),
                 * LOWER thread [range[t],n). Fold into one slot, then alpha-AXPY. */
                if (UPLO == 'U') {
                    T *restrict target = buf + (size_t)(num_cpu - 1) * (size_t)n;
                    for (ptrdiff_t t = 0; t < num_cpu - 1; ++t) {
                        const T *restrict src = buf + (size_t)t * (size_t)n;
                        ptrdiff_t len = range[t + 1];
                        for (ptrdiff_t k = 0; k < len; ++k) target[k] += src[k];
                    }
                    for (ptrdiff_t k = 0; k < n; ++k) y[k] += alpha * target[k];
                } else {
                    T *restrict target = buf;
                    for (ptrdiff_t t = 1; t < num_cpu; ++t) {
                        const T *restrict src = buf + (size_t)t * (size_t)n;
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
        if (UPLO == 'U') {
            for (ptrdiff_t j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero;
                ptrdiff_t k = kk;
                for (ptrdiff_t i = 0; i < j; ++i) {
                    y[i] += t1 * ap[k];
                    t2 += cconj(ap[k]) * x[i];
                    ++k;
                }
                y[j] += t1 * (TR)__real__ ap[kk + j] + alpha * t2;
                kk += j + 1;
            }
        } else {
            for (ptrdiff_t j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero;
                y[j] += t1 * (TR)__real__ ap[kk];
                ptrdiff_t k = kk + 1;
                for (ptrdiff_t i = j + 1; i < N; ++i) {
                    y[i] += t1 * ap[k];
                    t2 += cconj(ap[k]) * x[i];
                    ++k;
                }
                y[j] += alpha * t2;
                kk += N - j;
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(N - 1) * incy : 0;
        if (UPLO == 'U') {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero;
                ptrdiff_t ix = kx, iy = ky;
                for (ptrdiff_t k = kk; k < kk + j; ++k) {
                    y[iy] += t1 * ap[k];
                    t2 += cconj(ap[k]) * x[ix];
                    ix += incx; iy += incy;
                }
                y[jy] += t1 * (TR)__real__ ap[kk + j] + alpha * t2;
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero;
                y[jy] += t1 * (TR)__real__ ap[kk];
                ptrdiff_t ix = jx, iy = jy;
                for (ptrdiff_t k = kk + 1; k < kk + N - j; ++k) {
                    ix += incx; iy += incy;
                    y[iy] += t1 * ap[k];
                    t2 += cconj(ap[k]) * x[ix];
                }
                y[jy] += alpha * t2;
                jx += incx; jy += incy;
                kk += N - j;
            }
        }
    }
}

EPBLAS_FACADE_SPMV(yhpmv, T)
