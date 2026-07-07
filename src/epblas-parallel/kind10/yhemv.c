/*
 * yhemv — kind10 complex Hermitian matrix-vector multiply.
 *   y := alpha · A · x + beta · y    where A is N×N Hermitian
 *
 * Same two-pass pattern as esymv. Hermitian twist: the cross
 * reflection conjugates A. Diagonal of A is treated as real (matches
 * Netlib ZHEMV — uses DBLE(A(I,I))).
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#include "../common/epblas_facade.h"
#include "../common/blas_omp.h"
#ifdef _OPENMP
#include <omp.h>
#include <math.h>
#endif

/* RECALIBRATED 2026-06-07 (was 128): old break-even predates iomp5 hot-team reuse
 * (libgomp fork/join wakeup tax). This compute-heavy complex Hermitian matvec
 * threads from very low N. Measured par4/par1 (taskset 0-3): N=32 ~0.72-0.78,
 * N=64 ~0.42, N=128 ~0.32; clear win at 32. omp4-vs-omp1 relerr ~1e-19 (fp80
 * floor; the Hermitian two-sided fold reorders at ULP level, within tolerance). */
#define YHEMV_OMP_MIN 32
#define YHEMV_MAX_CPUS 256

typedef _Complex long double TC;
static const TC ZERO = 0.0L + 0.0Li;
static const TC ONE  = 1.0L + 0.0Li;
static inline TC cconj(TC z) { return ~z; }


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#ifdef _OPENMP
/* Sqrt-balanced contiguous column partition (OpenBLAS symv_partition,
 * mask=3, min_width=4): per-column work grows with j for UPPER, shrinks
 * for LOWER. Mirrors the espmv/yhpmv packed twins and this routine's own
 * OpenBLAS reference (epblas-openblas/kind10/yhemv.c). */
static ptrdiff_t yhemv_partition(bool upper, ptrdiff_t n, ptrdiff_t nthreads, ptrdiff_t *range)
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
        if (num_cpu >= YHEMV_MAX_CPUS) break;
    }
    return num_cpu;
}
#endif

static void yhemv_core(
    char uplo,
    ptrdiff_t n,
    const TC *alpha_,
    const TC *restrict a, ptrdiff_t lda,
    const TC *restrict x, ptrdiff_t incx,
    const TC *beta_,
    TC *restrict y, ptrdiff_t incy)
{
    const TC alpha = *alpha_, beta = *beta_;
    const char UPLO = blas_up(uplo);

    if (n == 0) return;

    if (beta != ONE) {
        if (incy == 1) {
            if (beta == ZERO) for (ptrdiff_t i = 0; i < n; ++i) y[i] = ZERO;
            else              for (ptrdiff_t i = 0; i < n; ++i) y[i] *= beta;
        } else {
            ptrdiff_t iy = (incy < 0) ? -(n - 1) * incy : 0;
            for (ptrdiff_t i = 0; i < n; ++i) {
                if (beta == ZERO) y[iy] = ZERO;
                else              y[iy] *= beta;
                iy += incy;
            }
        }
    }

    if (alpha == ZERO) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const ptrdiff_t nthreads = blas_omp_max_threads();
        if (n >= YHEMV_OMP_MIN && blas_omp_should_thread() && nthreads > 1) {
            /* Faithful port of this routine's OpenBLAS reference (and the
             * espmv/yhpmv packed twins): sqrt-balanced contiguous column
             * partition, each thread accumulating bare A*x into a private
             * range-limited slot, then a serial fold + single alpha-AXPY.
             * This replaces the old cyclic schedule(static,1) + rectangular
             * parallel reduction, whose nthreads*n re-read cost the complex
             * (2x data) variant ~3-6% at omp4 on N>=256 (task 15). */
            ptrdiff_t nt = (nthreads > YHEMV_MAX_CPUS) ? YHEMV_MAX_CPUS : nthreads;
            ptrdiff_t range[YHEMV_MAX_CPUS + 1];
            ptrdiff_t num_cpu = yhemv_partition(UPLO == 'U', n, nt, range);
            TC *buf = (num_cpu > 1)
                ? (TC *)calloc((size_t)num_cpu * (size_t)n, sizeof(TC)) : NULL;
            if (buf) {
                #pragma omp parallel for schedule(static, 1) num_threads(num_cpu)
                for (ptrdiff_t t = 0; t < num_cpu; ++t)
                {
                    TC *restrict slot = buf + (size_t)t * (size_t)n;
                    if (UPLO == 'U') {
                        /* Reversed thread->column assignment (task 14): the
                         * master (thread 0) takes the HIGH columns so its own
                         * slot 0 gains full [0,n) coverage and the serial fold
                         * below RMWs core-0-local memory (mirroring LOWER). */
                        ptrdiff_t r = num_cpu - 1 - t;
                        for (ptrdiff_t j = range[r]; j < range[r + 1]; ++j) {
                            const TC *restrict aj = &A_(0, j);
                            const TC t1 = x[j];
                            TC t2 = ZERO;
                            for (ptrdiff_t i = 0; i < j; ++i) {
                                slot[i] += t1 * aj[i];
                                t2      += cconj(aj[i]) * x[i];
                            }
                            slot[j] += t1 * __real__ aj[j] + t2;   /* real diag */
                        }
                    } else {
                        for (ptrdiff_t j = range[t]; j < range[t + 1]; ++j) {
                            const TC *restrict aj = &A_(0, j);
                            const TC t1 = x[j];
                            TC t2 = ZERO;
                            slot[j] += t1 * __real__ aj[j];        /* real diag */
                            for (ptrdiff_t i = j + 1; i < n; ++i) {
                                slot[i] += t1 * aj[i];
                                t2      += cconj(aj[i]) * x[i];
                            }
                            slot[j] += t2;
                        }
                    }
                }
                /* Range-limited serial fold into the master's full-coverage
                 * slot 0 (local), then one alpha-AXPY into y. UPPER slot t
                 * covers [0,range[num_cpu-t]); LOWER slot t covers [range[t],n). */
                TC *restrict target = buf;
                if (UPLO == 'U') {
                    for (ptrdiff_t t = 1; t < num_cpu; ++t) {
                        const TC *restrict src = buf + (size_t)t * (size_t)n;
                        ptrdiff_t len = range[num_cpu - t];
                        for (ptrdiff_t k = 0; k < len; ++k) target[k] += src[k];
                    }
                } else {
                    for (ptrdiff_t t = 1; t < num_cpu; ++t) {
                        const TC *restrict src = buf + (size_t)t * (size_t)n;
                        for (ptrdiff_t k = range[t]; k < n; ++k) target[k] += src[k];
                    }
                }
                for (ptrdiff_t k = 0; k < n; ++k) y[k] += alpha * target[k];
                free(buf);
                return;
            }
            free(buf);
        }
#endif
        if (UPLO == 'L') {
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TC temp1 = alpha * x[i];
                TC temp2 = ZERO;
                const TC *ai = &A_(0, i);
                y[i] += temp1 * __real__ ai[i];
                for (ptrdiff_t k = i + 1; k < n; ++k) {
                    y[k]  += temp1 * ai[k];
                    temp2 += cconj(ai[k]) * x[k];
                }
                y[i] += alpha * temp2;
            }
        } else {
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TC temp1 = alpha * x[i];
                TC temp2 = ZERO;
                const TC *ai = &A_(0, i);
                for (ptrdiff_t k = 0; k < i; ++k) {
                    y[k]  += temp1 * ai[k];
                    temp2 += cconj(ai[k]) * x[k];
                }
                y[i] += temp1 * __real__ ai[i] + alpha * temp2;
            }
        }
    } else {
        /* General-stride fallback — hoist the matrix column to ai[k] and
         * walk the strided vectors with running indices (Class-B fix). */
        const ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        const ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        if (UPLO == 'L') {
            ptrdiff_t ix = kx, iy = ky;
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TC temp1 = alpha * x[ix];
                TC temp2 = ZERO;
                const TC *ai = &A_(0, i);
                y[iy] += temp1 * __real__ ai[i];
                ptrdiff_t jx = ix + incx, jy = iy + incy;
                for (ptrdiff_t k = i + 1; k < n; ++k) {
                    y[jy] += temp1 * ai[k];
                    temp2 += cconj(ai[k]) * x[jx];
                    jx += incx; jy += incy;
                }
                y[iy] += alpha * temp2;
                ix += incx; iy += incy;
            }
        } else {
            ptrdiff_t ix = kx, iy = ky;
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TC temp1 = alpha * x[ix];
                TC temp2 = ZERO;
                const TC *ai = &A_(0, i);
                ptrdiff_t jx = kx, jy = ky;
                for (ptrdiff_t k = 0; k < i; ++k) {
                    y[jy] += temp1 * ai[k];
                    temp2 += cconj(ai[k]) * x[jx];
                    jx += incx; jy += incy;
                }
                y[iy] += temp1 * __real__ ai[i] + alpha * temp2;
                ix += incx; iy += incy;
            }
        }
    }
}

EPBLAS_FACADE_SYMV(yhemv, TC)

#undef A_
