/*
 * esymv — kind10 (REAL(KIND=10)) symmetric matrix-vector multiply.
 *   y := alpha · A · x + beta · y    where A is N×N symmetric
 *
 * Uses Netlib DSYMV's two-pass pattern: for each i,
 *   temp1 = alpha · x(i)   (contributes to y(k) for k!=i via A column reads)
 *   temp2 = sum_k A(k,i) · x(k)   (dot-product accumulator)
 *   y(i) += temp1 · A(i,i) + alpha · temp2
 * Stride-1 walks of A by columns; same direction-flip trick as the
 * symm/hemm diagonal kernel.
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#include "../common/epblas_facade.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define ESYMV_OMP_MIN 128

typedef long double T;

static inline char up(char c) {
    return (char)toupper((unsigned char)c);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* Contiguous (stride-1) two-pass column kernel shared by the serial and
 * threaded paths: y[k] += temp1·acol[k] for k in [k_lo, k_hi), returning
 * sum acol[k]·x[k] over the same range. Carved out noinline so the fp80 hot
 * loop compiles in a clean x87 register context — inlined amid the routine's
 * OMP/beta/strided scaffolding GCC spills and RELOADS acol[k] every element
 * (4 fldt/elt vs 3); in isolation it keeps acol[k] resident, matching the
 * migrated Fortran reference. Bit-identical to the inline two-pass form. */
__attribute__((noinline)) static
T esymv_axpydot(ptrdiff_t k_lo, ptrdiff_t k_hi, T temp1,
                const T *restrict acol, const T *restrict x, T *restrict y)
{
    T temp2 = 0.0L;
    for (ptrdiff_t k = k_lo; k < k_hi; ++k) {
        const T a = acol[k];
        y[k]  += temp1 * a;
        temp2 += a * x[k];
    }
    return temp2;
}

/* Strided twin of esymv_axpydot: A walked contiguously (ak[k]), x/y by stride.
 * Same noinline rationale — keeps ak[k] x87-resident in the general-stride
 * fallback. ix/iy carry the running strided indices; bit-identical to the
 * inline strided two-pass form. */
__attribute__((noinline)) static
T esymv_axpydot_strided(ptrdiff_t cnt, T temp1, const T *restrict ak,
                        const T *restrict x, ptrdiff_t incx, ptrdiff_t ix,
                        T *restrict y, ptrdiff_t incy, ptrdiff_t iy)
{
    T temp2 = 0.0L;
    for (ptrdiff_t k = 0; k < cnt; ++k) {
        const T a = ak[k];
        y[iy] += temp1 * a;
        temp2 += a * x[ix];
        ix += incx; iy += incy;
    }
    return temp2;
}

/* Serial stride-1 two-pass core (shared by the contiguous path and the
 * strided gather path). Bit-identical column-order accumulation to the
 * direct strided form. */
__attribute__((noinline)) static
void esymv_serial_core(char UPLO, ptrdiff_t N, ptrdiff_t lda, T alpha,
                       const T *restrict a, const T *restrict x, T *restrict y)
{
    if (UPLO == 'L') {
        for (ptrdiff_t i = 0; i < N; ++i) {
            const T temp1 = alpha * x[i];
            const T *ai = &A_(0, i);
            y[i] += temp1 * ai[i];
            const T temp2 = esymv_axpydot(i + 1, N, temp1, ai, x, y);
            y[i] += alpha * temp2;
        }
    } else {
        for (ptrdiff_t i = 0; i < N; ++i) {
            const T temp1 = alpha * x[i];
            const T *ai = &A_(0, i);
            const T temp2 = esymv_axpydot(0, i, temp1, ai, x, y);
            y[i] += temp1 * ai[i] + alpha * temp2;
        }
    }
}

static void esymv_core(
    char uplo,
    ptrdiff_t N,
    const T *alpha_,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    const T *beta_,
    T *restrict y, ptrdiff_t incy)
{
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);

    if (N == 0) return;

    const T zero = 0.0L, one = 1.0L;

    if (beta != one) {
        if (incy == 1) {
            if (beta == zero) for (ptrdiff_t i = 0; i < N; ++i) y[i] = zero;
            else              for (ptrdiff_t i = 0; i < N; ++i) y[i] *= beta;
        } else {
            ptrdiff_t iy = (incy < 0) ? -(N - 1) * incy : 0;
            for (ptrdiff_t i = 0; i < N; ++i) {
                if (beta == zero) y[iy] = zero;
                else              y[iy] *= beta;
                iy += incy;
            }
        }
    }

    if (alpha == zero) return;

    /* The unit-stride path: stride-1 column walks of A. */
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const ptrdiff_t nt = blas_omp_max_threads();
        const ptrdiff_t use_omp = (N >= ESYMV_OMP_MIN && nt > 1 && !omp_in_parallel());
#else
        const ptrdiff_t use_omp = 0;
        const ptrdiff_t nt = 1;
#endif
        if (use_omp) {
            /* Parallel two-pass with per-thread private y accumulator.
             *
             * The Netlib two-pass form walks A column-by-column (stride-1)
             * and on each column j writes y[k] for k > j (L) or k < j (U),
             * which races if multiple threads share column ranges. Fix:
             * each thread gets a private y_priv[N], accumulates its own
             * column contributions, then a final reduction sums all
             * y_priv[t] into y.
             *
             * schedule(static, 1) interleaves columns across threads to
             * balance the triangular work (per-column work is linear in
             * (N - j) for L, j for U). */
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
                            const T *aj = &A_(0, j);
                            y_priv[j] += temp1 * aj[j];
                            const T temp2 = esymv_axpydot(j + 1, N, temp1, aj, x, y_priv);
                            y_priv[j] += alpha * temp2;
                        }
                    } else {
                        #pragma omp for schedule(static, 1)
                        for (ptrdiff_t j = 0; j < N; ++j) {
                            const T temp1 = alpha * x[j];
                            const T *aj = &A_(0, j);
                            const T temp2 = esymv_axpydot(0, j, temp1, aj, x, y_priv);
                            y_priv[j] += temp1 * aj[j] + alpha * temp2;
                        }
                    }
                    /* Implicit barrier at end of the `omp for` ensures
                     * every thread's y_priv slice is fully written
                     * before the reduction begins reading. */

                    #pragma omp for schedule(static)
                    for (ptrdiff_t i = 0; i < N; ++i) {
                        T s = zero;
                        for (ptrdiff_t t = 0; t < nt; ++t)
                            s += y_priv_all[(size_t)t * N + i];
                        y[i] += s;
                    }
                }
#endif
                free(y_priv_all);
                return;
            }
            /* aligned_alloc failed — fall through to serial. */
        }
        esymv_serial_core(UPLO, N, lda, alpha, a, x, y);
    } else {
        /* General-stride: gather x and the (already beta-scaled) y into
         * contiguous scratch, run the stride-1 core — which beats both the
         * OpenBLAS clone and the gfortran reference, neither of which gathers
         * — then scatter y back. The gather/scatter is O(N) against O(N^2)
         * work, so it is free past tiny N. Same column-order accumulation as
         * the direct strided walk, so bit-identical. Falls back to the direct
         * strided helper if the scratch allocation fails. */
        const ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        const ptrdiff_t ky = (incy < 0) ? -(N - 1) * incy : 0;
        /* Stack scratch for the common small-N case avoids malloc latency;
         * spill to the heap for large N. */
        T stackbuf[2 * 512];
        T *heap = NULL;
        T *xc, *yc;
        if (N <= 512) {
            xc = stackbuf; yc = stackbuf + N;
        } else {
            heap = (T *)malloc((size_t)2 * N * sizeof(T));
            xc = heap; yc = heap ? heap + N : NULL;
        }
        if (xc && yc) {
            ptrdiff_t ix = kx, iy = ky;
            for (ptrdiff_t k = 0; k < N; ++k) {
                xc[k] = x[ix]; yc[k] = y[iy];
                ix += incx; iy += incy;
            }
            esymv_serial_core(UPLO, N, lda, alpha, a, xc, yc);
            iy = ky;
            for (ptrdiff_t k = 0; k < N; ++k) { y[iy] = yc[k]; iy += incy; }
            free(heap);
            return;
        }
        free(heap);
        if (UPLO == 'L') {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t i = 0; i < N; ++i) {
                const T temp1 = alpha * x[jx];
                y[jy] += temp1 * A_(i, i);
                const T temp2 = esymv_axpydot_strided(
                    N - (i + 1), temp1, &A_(i + 1, i),
                    x, incx, jx + incx, y, incy, jy + incy);
                y[jy] += alpha * temp2;
                jx += incx; jy += incy;
            }
        } else {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t i = 0; i < N; ++i) {
                const T temp1 = alpha * x[jx];
                const T temp2 = esymv_axpydot_strided(
                    i, temp1, &A_(0, i),
                    x, incx, kx, y, incy, ky);
                y[jy] += temp1 * A_(i, i) + alpha * temp2;
                jx += incx; jy += incy;
            }
        }
    }
}

EPBLAS_FACADE_SYMV(esymv, T)

#undef A_
