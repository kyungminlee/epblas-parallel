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
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#include "../common/epblas_facade.h"
#include "../common/blas_omp.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define ESYMV_OMP_MIN 128

typedef long double TR;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* Contiguous (stride-1) two-pass column kernel shared by the serial and
 * threaded paths: y[k] += temp1·acol[k] for k in [k_lo, k_hi), returning
 * sum acol[k]·x[k] over the same range. Carved out noinline so the fp80 hot
 * loop compiles in a clean x87 register context — inlined amid the routine's
 * OMP/beta/strided scaffolding GCC spills and RELOADS acol[k] every element
 * (4 fldt/elt vs 3); in isolation it keeps acol[k] resident, matching the
 * migrated Fortran reference. Bit-identical to the inline two-pass form. */
__attribute__((noinline)) static
TR esymv_axpydot(ptrdiff_t k_lo, ptrdiff_t k_hi, TR temp1,
                const TR *restrict acol, const TR *restrict x, TR *restrict y)
{
    TR temp2 = 0.0L;
    for (ptrdiff_t k = k_lo; k < k_hi; ++k) {
        const TR a = acol[k];
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
TR esymv_axpydot_strided(ptrdiff_t cnt, TR temp1, const TR *restrict ak,
                        const TR *restrict x, ptrdiff_t incx, ptrdiff_t ix,
                        TR *restrict y, ptrdiff_t incy, ptrdiff_t iy)
{
    TR temp2 = 0.0L;
    for (ptrdiff_t k = 0; k < cnt; ++k) {
        const TR a = ak[k];
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
void esymv_serial_core(char UPLO, ptrdiff_t n, ptrdiff_t lda, TR alpha,
                       const TR *restrict a, const TR *restrict x, TR *restrict y)
{
    if (UPLO == 'L') {
        for (ptrdiff_t i = 0; i < n; ++i) {
            const TR temp1 = alpha * x[i];
            const TR *ai = &A_(0, i);
            y[i] += temp1 * ai[i];
            const TR temp2 = esymv_axpydot(i + 1, n, temp1, ai, x, y);
            y[i] += alpha * temp2;
        }
    } else {
        for (ptrdiff_t i = 0; i < n; ++i) {
            const TR temp1 = alpha * x[i];
            const TR *ai = &A_(0, i);
            const TR temp2 = esymv_axpydot(0, i, temp1, ai, x, y);
            y[i] += temp1 * ai[i] + alpha * temp2;
        }
    }
}

static void esymv_core(
    char uplo,
    ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict a, ptrdiff_t lda,
    const TR *restrict x, ptrdiff_t incx,
    const TR *beta_,
    TR *restrict y, ptrdiff_t incy)
{
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO = blas_up(uplo);

    if (n == 0) return;

    const TR zero = 0.0L, one = 1.0L;

    if (beta != one) {
        if (incy == 1) {
            if (beta == zero) for (ptrdiff_t i = 0; i < n; ++i) y[i] = zero;
            else              for (ptrdiff_t i = 0; i < n; ++i) y[i] *= beta;
        } else {
            ptrdiff_t iy = (incy < 0) ? -(n - 1) * incy : 0;
            for (ptrdiff_t i = 0; i < n; ++i) {
                if (beta == zero) y[iy] = zero;
                else              y[iy] *= beta;
                iy += incy;
            }
        }
    }

    if (alpha == zero) return;

    /* The unit-stride path: stride-1 column walks of A. */
    if (incx == 1 && incy == 1) {
        const ptrdiff_t nthreads = blas_omp_max_threads();
        const bool use_omp = (n >= ESYMV_OMP_MIN && blas_omp_should_thread());
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
            TR *y_priv_all = (TR *)calloc((size_t)nthreads * (size_t)n, sizeof(TR));
            if (y_priv_all) {
#ifdef _OPENMP
                #pragma omp parallel num_threads(nthreads)
                {
                    const ptrdiff_t tid = omp_get_thread_num();
                    TR *y_priv = &y_priv_all[(size_t)tid * n];  /* calloc-zeroed */

                    if (UPLO == 'L') {
                        #pragma omp for schedule(static, 1)
                        for (ptrdiff_t j = 0; j < n; ++j) {
                            const TR temp1 = alpha * x[j];
                            const TR *aj = &A_(0, j);
                            y_priv[j] += temp1 * aj[j];
                            const TR temp2 = esymv_axpydot(j + 1, n, temp1, aj, x, y_priv);
                            y_priv[j] += alpha * temp2;
                        }
                    } else {
                        #pragma omp for schedule(static, 1)
                        for (ptrdiff_t j = 0; j < n; ++j) {
                            const TR temp1 = alpha * x[j];
                            const TR *aj = &A_(0, j);
                            const TR temp2 = esymv_axpydot(0, j, temp1, aj, x, y_priv);
                            y_priv[j] += temp1 * aj[j] + alpha * temp2;
                        }
                    }
                    /* Implicit barrier at end of the `omp for` ensures
                     * every thread's y_priv slice is fully written
                     * before the reduction begins reading. */

                    #pragma omp for schedule(static)
                    for (ptrdiff_t i = 0; i < n; ++i) {
                        TR s = zero;
                        for (ptrdiff_t t = 0; t < nthreads; ++t)
                            s += y_priv_all[(size_t)t * n + i];
                        y[i] += s;
                    }
                }
#endif
                free(y_priv_all);
                return;
            }
            /* aligned_alloc failed — fall through to serial. */
        }
        esymv_serial_core(UPLO, n, lda, alpha, a, x, y);
    } else {
        /* General-stride: gather x and the (already beta-scaled) y into
         * contiguous scratch, run the stride-1 core — which beats both the
         * OpenBLAS clone and the gfortran reference, neither of which gathers
         * — then scatter y back. The gather/scatter is O(N) against O(N^2)
         * work, so it is free past tiny N. Same column-order accumulation as
         * the direct strided walk, so bit-identical. Falls back to the direct
         * strided helper if the scratch allocation fails. */
        const ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        const ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        /* Stack scratch for the common small-N case avoids malloc latency;
         * spill to the heap for large N. */
        TR stackbuf[2 * 512];
        TR *heap = NULL;
        TR *xc, *yc;
        if (n <= 512) {
            xc = stackbuf; yc = stackbuf + n;
        } else {
            heap = (TR *)malloc((size_t)2 * n * sizeof(TR));
            xc = heap; yc = heap ? heap + n : NULL;
        }
        if (xc && yc) {
            ptrdiff_t ix = kx, iy = ky;
            for (ptrdiff_t k = 0; k < n; ++k) {
                xc[k] = x[ix]; yc[k] = y[iy];
                ix += incx; iy += incy;
            }
            esymv_serial_core(UPLO, n, lda, alpha, a, xc, yc);
            iy = ky;
            for (ptrdiff_t k = 0; k < n; ++k) { y[iy] = yc[k]; iy += incy; }
            free(heap);
            return;
        }
        free(heap);
        if (UPLO == 'L') {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TR temp1 = alpha * x[jx];
                y[jy] += temp1 * A_(i, i);
                const TR temp2 = esymv_axpydot_strided(
                    n - (i + 1), temp1, &A_(i + 1, i),
                    x, incx, jx + incx, y, incy, jy + incy);
                y[jy] += alpha * temp2;
                jx += incx; jy += incy;
            }
        } else {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TR temp1 = alpha * x[jx];
                const TR temp2 = esymv_axpydot_strided(
                    i, temp1, &A_(0, i),
                    x, incx, kx, y, incy, ky);
                y[jy] += temp1 * A_(i, i) + alpha * temp2;
                jx += incx; jy += incy;
            }
        }
    }
}

EPBLAS_FACADE_SYMV(esymv, TR)

#undef A_
