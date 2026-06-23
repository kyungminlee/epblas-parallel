/*
 * ytrmv — kind10 complex (`_Complex long double`) triangular matrix-vector.
 *   x := A · x         (TRANS='N')
 *   x := Aᵀ · x        (TRANS='T')
 *   x := Aᴴ · x        (TRANS='C')
 */

#include <stddef.h>
#include <stdbool.h>
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#include "../common/epblas_facade.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

/* RECALIBRATED 2026-06-07 (was 128): the old break-even was set under libgomp's
 * fork/join wakeup tax; iomp5 hot-team reuse removes it, so this compute-heavy
 * complex triangular matvec threads from far lower N. Measured par4/par1 (N=N,
 * K=full, taskset 0-3): N=48 ~0.61-0.73 (lowest all-shape win), N=64 ~0.46-0.58,
 * N=128 ~0.31-0.41. At N=32 Trans/ConjTrans still lose (~1.08-1.11), so 48 is the
 * floor. omp4-vs-omp1 relerr ~1e-19 (fp80 floor; Trans/Conj bytewise-exact). */
#define YTRMV_OMP_MIN 48

typedef _Complex long double TC;
static const TC ZERO = 0.0L + 0.0Li;
static const TC ONE_C = 1.0L + 0.0Li;
static inline TC cconj(TC z) { return ~z; }


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#ifdef _OPENMP
/* OMP core on a CONTIGUOUS x. Shared by the incx==1 fast path and the strided
 * path (via gather/scatter), so the threading lives in one place. TRANS='N' uses
 * per-thread y_priv + reduction (esymv pattern, Add-36 — within tolerance, not
 * bit-exact); TRANS='T'/'C' writes a single shared y_buf (each j owns its output
 * → bit-exact). Returns 1 on success, 0 if a scratch alloc failed (caller
 * falls back to serial). */
static bool ytrmv_omp_contig(char UPLO, char TRANS, bool nounit,
                            ptrdiff_t n, const TC *restrict a, ptrdiff_t lda,
                            TC *restrict x, ptrdiff_t nthreads)
{
    if (TRANS == 'N') {
        TC *y_priv_all = (TC *)calloc((size_t)nthreads * (size_t)n, sizeof(TC));
        if (!y_priv_all) return 0;
        #pragma omp parallel num_threads(nthreads)
        {
            const ptrdiff_t tid = omp_get_thread_num();
            TC *y_priv = &y_priv_all[(size_t)tid * n];  /* calloc-zeroed */

            if (UPLO == 'L') {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    const TC xj = x[j];
                    const TC *aj = &A_(0, j);
                    y_priv[j] += xj * (nounit ? aj[j] : ONE_C);
                    for (ptrdiff_t i = j + 1; i < n; ++i)
                        y_priv[i] += xj * aj[i];
                }
            } else {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    const TC xj = x[j];
                    const TC *aj = &A_(0, j);
                    for (ptrdiff_t i = 0; i < j; ++i)
                        y_priv[i] += xj * aj[i];
                    y_priv[j] += xj * (nounit ? aj[j] : ONE_C);
                }
            }
            #pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < n; ++i) {
                TC s = ZERO;
                for (ptrdiff_t t = 0; t < nthreads; ++t)
                    s += y_priv_all[(size_t)t * n + i];
                x[i] = s;
            }
        }
        free(y_priv_all);
        return 1;
    } else {
        const bool conj_a = (TRANS == 'C');
        TC *y_buf = (TC *)aligned_alloc(64,
            (((size_t)n * sizeof(TC)) + 63) & ~(size_t)63);
        if (!y_buf) return 0;
        #pragma omp parallel
        {
            if (UPLO == 'L') {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TC temp = x[j];
                    if (nounit) temp *= (conj_a ? cconj(A_(j, j)) : A_(j, j));
                    const TC *aj = &A_(0, j);
                    if (conj_a) {
                        for (ptrdiff_t i = j + 1; i < n; ++i) temp += cconj(aj[i]) * x[i];
                    } else {
                        for (ptrdiff_t i = j + 1; i < n; ++i) temp += aj[i] * x[i];
                    }
                    y_buf[j] = temp;
                }
            } else {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TC temp = x[j];
                    if (nounit) temp *= (conj_a ? cconj(A_(j, j)) : A_(j, j));
                    const TC *aj = &A_(0, j);
                    if (conj_a) {
                        for (ptrdiff_t i = j - 1; i >= 0; --i) temp += cconj(aj[i]) * x[i];
                    } else {
                        for (ptrdiff_t i = j - 1; i >= 0; --i) temp += aj[i] * x[i];
                    }
                    y_buf[j] = temp;
                }
            }
            #pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < n; ++i) x[i] = y_buf[i];
        }
        free(y_buf);
        return 1;
    }
}
#endif

static void ytrmv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n,
    const TC *restrict a, ptrdiff_t lda,
    TC *restrict x, ptrdiff_t incx)
{
    const char UPLO = blas_up(uplo);
    const char TRANS   = blas_up(trans);
    const char DIAG = blas_up(diag);
    const bool nounit = (DIAG != 'U');

    if (n == 0) return;

    if (incx == 1) {
#ifdef _OPENMP
        const ptrdiff_t nthreads = blas_omp_max_threads();
        if (n >= YTRMV_OMP_MIN && blas_omp_should_thread()
            && ytrmv_omp_contig(UPLO, TRANS, nounit, n, a, lda, x, nthreads))
            return;
#endif
        if (TRANS == 'N') {
            if (UPLO == 'L') {
                /* Inner walks backward to match Fortran ytrmv.f
                 * (DO 50 I = N,J+1,-1). Sub-class C / Rule 21. */
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    const TC temp = x[j];
                    if (temp != ZERO) {
                        const TC *aj = &A_(0, j);
                        for (ptrdiff_t i = n - 1; i > j; --i) x[i] += temp * aj[i];
                    }
                    if (nounit) x[j] *= A_(j, j);
                }
            } else {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    const TC temp = x[j];
                    if (temp != ZERO) {
                        const TC *aj = &A_(0, j);
                        for (ptrdiff_t i = 0; i < j; ++i) x[i] += temp * aj[i];
                    }
                    if (nounit) x[j] *= A_(j, j);
                }
            }
        } else {
            const bool conj_a = (TRANS == 'C');
            if (UPLO == 'L') {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TC temp = x[j];
                    if (nounit) temp *= (conj_a ? cconj(A_(j, j)) : A_(j, j));
                    const TC *aj = &A_(0, j);
                    if (conj_a) {
                        for (ptrdiff_t i = j + 1; i < n; ++i) temp += cconj(aj[i]) * x[i];
                    } else {
                        for (ptrdiff_t i = j + 1; i < n; ++i) temp += aj[i] * x[i];
                    }
                    x[j] = temp;
                }
            } else {
                /* Inner walks backward to match Fortran ytrmv.f
                 * (DO 90 I = J-1,1,-1). Sub-class D / Rule 21. */
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TC temp = x[j];
                    if (nounit) temp *= (conj_a ? cconj(A_(j, j)) : A_(j, j));
                    const TC *aj = &A_(0, j);
                    if (conj_a) {
                        for (ptrdiff_t i = j - 1; i >= 0; --i) temp += cconj(aj[i]) * x[i];
                    } else {
                        for (ptrdiff_t i = j - 1; i >= 0; --i) temp += aj[i] * x[i];
                    }
                    x[j] = temp;
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
#ifdef _OPENMP
        /* Thread the strided path by gathering x into a contiguous buffer,
         * driving the shared OMP core, and scattering back — so the threading
         * lives in one place and the serial strided code below stays
         * byte-for-byte unchanged. */
        const ptrdiff_t nthreads = blas_omp_max_threads();
        if (n >= YTRMV_OMP_MIN && blas_omp_should_thread()) {
            TC *xc = (TC *)malloc((size_t)n * sizeof(TC));
            if (xc) {
                for (ptrdiff_t i = 0; i < n; ++i) xc[i] = x[kx + i * incx];
                if (ytrmv_omp_contig(UPLO, TRANS, nounit, n, a, lda, xc, nthreads)) {
                    for (ptrdiff_t i = 0; i < n; ++i) x[kx + i * incx] = xc[i];
                    free(xc);
                    return;
                }
                free(xc);
            }
        }
#endif
        if (TRANS == 'N') {
            if (UPLO == 'L') {
                /* Inner walks backward to match Fortran ytrmv.f
                 * (DO 70 I = N,J+1,-1). Sub-class C / Rule 21. */
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    const TC temp = x[kx + j * incx];
                    if (temp != ZERO) {
                        for (ptrdiff_t i = n - 1; i > j; --i) x[kx + i * incx] += temp * A_(i, j);
                    }
                    if (nounit) x[kx + j * incx] *= A_(j, j);
                }
            } else {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    const TC temp = x[kx + j * incx];
                    if (temp != ZERO) {
                        for (ptrdiff_t i = 0; i < j; ++i) x[kx + i * incx] += temp * A_(i, j);
                    }
                    if (nounit) x[kx + j * incx] *= A_(j, j);
                }
            }
        } else {
            const bool conj_a = (TRANS == 'C');
            if (UPLO == 'L') {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TC temp = x[kx + j * incx];
                    if (nounit) temp *= (conj_a ? cconj(A_(j, j)) : A_(j, j));
                    for (ptrdiff_t i = j + 1; i < n; ++i) {
                        const TC aij = conj_a ? cconj(A_(i, j)) : A_(i, j);
                        temp += aij * x[kx + i * incx];
                    }
                    x[kx + j * incx] = temp;
                }
            } else {
                /* Inner walks backward to match Fortran ytrmv.f
                 * (DO 110 I = J-1,1,-1). Sub-class D / Rule 21. */
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TC temp = x[kx + j * incx];
                    if (nounit) temp *= (conj_a ? cconj(A_(j, j)) : A_(j, j));
                    for (ptrdiff_t i = j - 1; i >= 0; --i) {
                        const TC aij = conj_a ? cconj(A_(i, j)) : A_(i, j);
                        temp += aij * x[kx + i * incx];
                    }
                    x[kx + j * incx] = temp;
                }
            }
        }
    }
}

EPBLAS_FACADE_TRMV(ytrmv, TC)

#undef A_
