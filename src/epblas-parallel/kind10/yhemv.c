/*
 * yhemv — kind10 complex Hermitian matrix-vector multiply.
 *   y := alpha · A · x + beta · y    where A is N×N Hermitian
 *
 * Same two-pass pattern as esymv. Hermitian twist: the cross
 * reflection conjugates A. Diagonal of A is treated as real (matches
 * Netlib ZHEMV — uses DBLE(A(I,I))).
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

/* RECALIBRATED 2026-06-07 (was 128): old break-even predates iomp5 hot-team reuse
 * (libgomp fork/join wakeup tax). This compute-heavy complex Hermitian matvec
 * threads from very low N. Measured par4/par1 (taskset 0-3): N=32 ~0.72-0.78,
 * N=64 ~0.42, N=128 ~0.32; clear win at 32. omp4-vs-omp1 relerr ~1e-19 (fp80
 * floor; the Hermitian two-sided fold reorders at ULP level, within tolerance). */
#define YHEMV_OMP_MIN 32

typedef _Complex long double T;
static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;
static inline T cconj(T z) { return ~z; }

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void yhemv_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict x, const int *incx_,
    const T *beta_,
    T *restrict y, const int *incy_,
    size_t uplo_len)
{
    (void)uplo_len;
    const ptrdiff_t N = *n_;
    const ptrdiff_t lda = *lda_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);

    if (N == 0) return;

    if (beta != ONE) {
        if (incy == 1) {
            if (beta == ZERO) for (ptrdiff_t i = 0; i < N; ++i) y[i] = ZERO;
            else              for (ptrdiff_t i = 0; i < N; ++i) y[i] *= beta;
        } else {
            ptrdiff_t iy = (incy < 0) ? -(N - 1) * incy : 0;
            for (ptrdiff_t i = 0; i < N; ++i) {
                if (beta == ZERO) y[iy] = ZERO;
                else              y[iy] *= beta;
                iy += incy;
            }
        }
    }

    if (alpha == ZERO) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const ptrdiff_t nt = blas_omp_max_threads();
        const ptrdiff_t use_omp = (N >= YHEMV_OMP_MIN && nt > 1 && !omp_in_parallel());
#else
        const ptrdiff_t use_omp = 0;
        const ptrdiff_t nt = 1;
#endif
        if (use_omp) {
            /* Parallel column-walk with per-thread private y, then reduce.
             * Same pattern as esymv (Addendum 36). Hermitian conjugation
             * stays inside the column loop unchanged. */
            T *y_priv_all = (T *)calloc((size_t)nt * (size_t)N, sizeof(T));
            if (y_priv_all) {
#ifdef _OPENMP
                #pragma omp parallel num_threads(nt)
                {
                    const ptrdiff_t tid = omp_get_thread_num();
                    T *y_priv = &y_priv_all[(size_t)tid * N];  /* calloc-zeroed */

                    if (UPLO == 'L') {
                        #pragma omp for schedule(static, 1)
                        for (ptrdiff_t j = 0; j < N; ++j) {
                            const T temp1 = alpha * x[j];
                            T temp2 = ZERO;
                            const T *aj = &A_(0, j);
                            y_priv[j] += temp1 * __real__ aj[j];
                            for (ptrdiff_t k = j + 1; k < N; ++k) {
                                y_priv[k] += temp1 * aj[k];
                                temp2 += cconj(aj[k]) * x[k];
                            }
                            y_priv[j] += alpha * temp2;
                        }
                    } else {
                        #pragma omp for schedule(static, 1)
                        for (ptrdiff_t j = 0; j < N; ++j) {
                            const T temp1 = alpha * x[j];
                            T temp2 = ZERO;
                            const T *aj = &A_(0, j);
                            for (ptrdiff_t k = 0; k < j; ++k) {
                                y_priv[k] += temp1 * aj[k];
                                temp2 += cconj(aj[k]) * x[k];
                            }
                            y_priv[j] += temp1 * __real__ aj[j] + alpha * temp2;
                        }
                    }

                    #pragma omp for schedule(static)
                    for (ptrdiff_t i = 0; i < N; ++i) {
                        T s = ZERO;
                        for (ptrdiff_t t = 0; t < nt; ++t)
                            s += y_priv_all[(size_t)t * N + i];
                        y[i] += s;
                    }
                }
#endif
                free(y_priv_all);
                return;
            }
        }
        if (UPLO == 'L') {
            for (ptrdiff_t i = 0; i < N; ++i) {
                const T temp1 = alpha * x[i];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                y[i] += temp1 * __real__ ai[i];
                for (ptrdiff_t k = i + 1; k < N; ++k) {
                    y[k]  += temp1 * ai[k];
                    temp2 += cconj(ai[k]) * x[k];
                }
                y[i] += alpha * temp2;
            }
        } else {
            for (ptrdiff_t i = 0; i < N; ++i) {
                const T temp1 = alpha * x[i];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
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
        const ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        const ptrdiff_t ky = (incy < 0) ? -(N - 1) * incy : 0;
        if (UPLO == 'L') {
            ptrdiff_t ix = kx, iy = ky;
            for (ptrdiff_t i = 0; i < N; ++i) {
                const T temp1 = alpha * x[ix];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                y[iy] += temp1 * __real__ ai[i];
                ptrdiff_t jx = ix + incx, jy = iy + incy;
                for (ptrdiff_t k = i + 1; k < N; ++k) {
                    y[jy] += temp1 * ai[k];
                    temp2 += cconj(ai[k]) * x[jx];
                    jx += incx; jy += incy;
                }
                y[iy] += alpha * temp2;
                ix += incx; iy += incy;
            }
        } else {
            ptrdiff_t ix = kx, iy = ky;
            for (ptrdiff_t i = 0; i < N; ++i) {
                const T temp1 = alpha * x[ix];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
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

#undef A_
