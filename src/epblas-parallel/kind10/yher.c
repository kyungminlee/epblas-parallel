/*
 * yher — kind10 complex Hermitian rank-1 update.
 *   A := alpha · x · xᴴ + A
 * alpha is REAL. Diagonal of A stays real on output.
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

/* RECALIBRATED 2026-06-07 (was 64): the old break-even predates iomp5 (libgomp
 * fork/join wakeup tax). With iomp5 hot-team reuse this O(N^2) complex Hermitian
 * rank-1 update threads from N=24. Measured par4/par1 (taskset 0-3, min-of-10):
 * N=24 0.68/0.65, N=32 0.49/0.53, N=64 0.31, N=128 0.27. N=20 is marginal (0.90)
 * and N=16 loses (1.18), so 24 is the robust floor. Bit-exact (relerr 0).
 * Uniform across the y* rank-update family (yher/yher2/ygerc/ygeru/yhpr/yhpr2). */
#define YHER_OMP_MIN 24

typedef _Complex long double TC;
typedef long double          TR;
static inline TC cconj(TC z) { return ~z; }

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void yher_(
    const char *uplo,
    const int *n_,
    const TR *alpha_,
    const TC *restrict x, const int *incx_,
    TC *restrict a, const int *lda_,
    size_t uplo_len)
{
    (void)uplo_len;
    const ptrdiff_t N = *n_;
    const ptrdiff_t incx = *incx_, lda = *lda_;
    const TR alpha = *alpha_;
    const TR rzero = 0.0L;
    const TC czero = 0.0L + 0.0Li;
    const char UPLO = up(uplo);

    if (N == 0 || alpha == rzero) return;

    if (incx == 1) {
#ifdef _OPENMP
        const ptrdiff_t use_omp = (N >= YHER_OMP_MIN && blas_omp_max_threads() > 1
                             && !omp_in_parallel());
#else
        const ptrdiff_t use_omp = 0;
#endif
        /* Branch on use_omp at C source level (Add-16). schedule(static, 1)
         * for triangular load balance (Rule 49). */
#define YHER_BODY                                                            \
        for (ptrdiff_t j = 0; j < N; ++j) {                                        \
            const TC xj = x[j];                                              \
            if (xj != czero) {                                               \
                /* t = alpha * conj(x[j]); A(i,j) += t * x[i].               \
                 * Diagonal i==j contribution is alpha*|x[j]|^2 (real);      \
                 * write real part only to keep imag zeroed. */              \
                const TC t = alpha * cconj(xj);                              \
                TC *aj = &A_(0, j);                                          \
                if (UPLO == 'L') {                                           \
                    for (ptrdiff_t i = j + 1; i < N; ++i) aj[i] += t * x[i];       \
                    aj[j] = __real__ aj[j] + __real__ (t * x[j]);            \
                } else {                                                     \
                    for (ptrdiff_t i = 0; i < j; ++i) aj[i] += t * x[i];           \
                    aj[j] = __real__ aj[j] + __real__ (t * x[j]);            \
                }                                                            \
            }                                                                \
        }
        if (use_omp) {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static, 1)
#endif
            YHER_BODY
        } else {
            YHER_BODY
        }
#undef YHER_BODY
    } else {
        /* General-stride fallback — hoist the matrix column to aj[i] and
         * walk the strided vector with a running index (Class-B fix,
         * memory project_ptrdiff_conversion_regressors). */
        const ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        ptrdiff_t jx = kx;
        for (ptrdiff_t j = 0; j < N; ++j) {
            const TC xj = x[jx];
            if (xj != czero) {
                const TC t = alpha * cconj(xj);
                TC *aj = &A_(0, j);
                if (UPLO == 'L') {
                    ptrdiff_t ix = jx + incx;
                    for (ptrdiff_t i = j + 1; i < N; ++i) { aj[i] += t * x[ix]; ix += incx; }
                    aj[j] = __real__ aj[j] + __real__ (t * xj);
                } else {
                    ptrdiff_t ix = kx;
                    for (ptrdiff_t i = 0; i < j; ++i) { aj[i] += t * x[ix]; ix += incx; }
                    aj[j] = __real__ aj[j] + __real__ (t * xj);
                }
            }
            jx += incx;
        }
    }
}

#undef A_
