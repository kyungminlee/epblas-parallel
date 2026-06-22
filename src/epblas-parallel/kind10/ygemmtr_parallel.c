/*
 * ygemmtr_ — kind10 complex (_Complex long double) triangular GEMM-update,
 * the public Fortran entry and threading-orchestration half of the ygemmtr
 * overlay (see ygemmtr_kernel.h; all the math lives in ygemmtr_serial.c).
 *
 * Parallel shape: one `omp parallel for schedule(static,1)` over the
 * columns j. Each thread owns disjoint columns of C, so the per-column
 * worker runs with no cross-thread races.
 *
 * Nesting guard: when ygemmtr_ is itself called from inside another
 * routine's parallel region, it delegates to ygemmtr_serial and opens no
 * region of its own (the libgomp barrier wedge guard,
 * project-etrsm-omp4-wedge).
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "ygemmtr_kernel.h"
#include "../common/epblas_facade.h"

typedef ygemmtr_T T;

#define YGEMMTR_OMP_MIN 32

static void ygemmtr_core(char uplo, char transa, char transb,
              ptrdiff_t N, ptrdiff_t K,
              const T *alpha_,
              const T *a, ptrdiff_t lda,
              const T *b, ptrdiff_t ldb,
              const T *beta_,
              T *c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        ygemmtr_serial(uplo, transa, transb, N, K, alpha_, a, lda, b, ldb, beta_, c, ldc);
        return;
    }
#endif
    const T alpha = *alpha_, beta = *beta_;
    const bool upper = (blas_up(uplo) == 'U');
    const char ta = blas_up(transa);
    const char tb = blas_up(transb);

    if (N <= 0) return;
    const T zero = 0.0L + 0.0iL;
    const T one  = 1.0L + 0.0iL;

    const bool conj_a = (ta == 'C');
    const bool conj_b = (tb == 'C');
    const bool trans_a = (ta != 'N');
    const bool trans_b = (tb != 'N');

    if (alpha == zero || K == 0) {
        if (beta == one) return;
#ifdef _OPENMP
        const bool use_omp0 = (N >= YGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp0) schedule(static, 1)
#endif
        for (ptrdiff_t j = 0; j < N; ++j)
            ygemmtr_beta_scale(j, j + 1, N, upper, beta, c, ldc);
        return;
    }

#ifdef _OPENMP
    const bool use_omp = (N >= YGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
    for (ptrdiff_t j = 0; j < N; ++j)
        ygemmtr_col(j, N, K, upper, alpha, beta, a, lda, b, ldb, c, ldc,
                    trans_a, conj_a, trans_b, conj_b);
}

EPBLAS_FACADE_GEMMTR(ygemmtr, T)
