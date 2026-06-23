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
#endif

/* RECALIBRATED 2026-06-07 (was 128): old break-even predates iomp5 hot-team reuse
 * (libgomp fork/join wakeup tax). This compute-heavy complex Hermitian matvec
 * threads from very low N. Measured par4/par1 (taskset 0-3): N=32 ~0.72-0.78,
 * N=64 ~0.42, N=128 ~0.32; clear win at 32. omp4-vs-omp1 relerr ~1e-19 (fp80
 * floor; the Hermitian two-sided fold reorders at ULP level, within tolerance). */
#define YHEMV_OMP_MIN 32

typedef _Complex long double TC;
static const TC ZERO = 0.0L + 0.0Li;
static const TC ONE  = 1.0L + 0.0Li;
static inline TC cconj(TC z) { return ~z; }


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

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
        const ptrdiff_t nthreads = blas_omp_max_threads();
        const bool use_omp = (n >= YHEMV_OMP_MIN && blas_omp_should_thread());
        if (use_omp) {
            /* Parallel column-walk with per-thread private y, then reduce.
             * Same pattern as esymv (Addendum 36). Hermitian conjugation
             * stays inside the column loop unchanged. */
            TC *y_priv_all = (TC *)calloc((size_t)nthreads * (size_t)n, sizeof(TC));
            if (y_priv_all) {
#ifdef _OPENMP
                #pragma omp parallel num_threads(nthreads)
                {
                    const ptrdiff_t tid = omp_get_thread_num();
                    TC *y_priv = &y_priv_all[(size_t)tid * n];  /* calloc-zeroed */

                    if (UPLO == 'L') {
                        #pragma omp for schedule(static, 1)
                        for (ptrdiff_t j = 0; j < n; ++j) {
                            const TC temp1 = alpha * x[j];
                            TC temp2 = ZERO;
                            const TC *aj = &A_(0, j);
                            y_priv[j] += temp1 * __real__ aj[j];
                            for (ptrdiff_t k = j + 1; k < n; ++k) {
                                y_priv[k] += temp1 * aj[k];
                                temp2 += cconj(aj[k]) * x[k];
                            }
                            y_priv[j] += alpha * temp2;
                        }
                    } else {
                        #pragma omp for schedule(static, 1)
                        for (ptrdiff_t j = 0; j < n; ++j) {
                            const TC temp1 = alpha * x[j];
                            TC temp2 = ZERO;
                            const TC *aj = &A_(0, j);
                            for (ptrdiff_t k = 0; k < j; ++k) {
                                y_priv[k] += temp1 * aj[k];
                                temp2 += cconj(aj[k]) * x[k];
                            }
                            y_priv[j] += temp1 * __real__ aj[j] + alpha * temp2;
                        }
                    }

                    #pragma omp for schedule(static)
                    for (ptrdiff_t i = 0; i < n; ++i) {
                        TC s = ZERO;
                        for (ptrdiff_t t = 0; t < nthreads; ++t)
                            s += y_priv_all[(size_t)t * n + i];
                        y[i] += s;
                    }
                }
#endif
                free(y_priv_all);
                return;
            }
        }
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
