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
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "ygemmtr_kernel.h"

typedef ygemmtr_T T;

#define YGEMMTR_OMP_MIN 32

void ygemmtr_(const char *uplo, const char *transa, const char *transb,
              const int *n_, const int *k_,
              const T *alpha_,
              const T *a, const int *lda_,
              const T *b, const int *ldb_,
              const T *beta_,
              T *c, const int *ldc_,
              size_t uplo_len, size_t ta_len, size_t tb_len)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        ygemmtr_serial(uplo, transa, transb, n_, k_, alpha_, a, lda_, b, ldb_,
                       beta_, c, ldc_, uplo_len, ta_len, tb_len);
        return;
    }
#endif
    (void)uplo_len; (void)ta_len; (void)tb_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const int upper = ((char)toupper((unsigned char)*uplo) == 'U');
    const int ta = (char)toupper((unsigned char)*transa);
    const int tb = (char)toupper((unsigned char)*transb);

    if (N <= 0) return;
    const T zero = 0.0L + 0.0iL;
    const T one  = 1.0L + 0.0iL;

    const int conj_a = (ta == 'C');
    const int conj_b = (tb == 'C');
    const int trans_a = (ta != 'N');
    const int trans_b = (tb != 'N');

    if (alpha == zero || K == 0) {
        if (beta == one) return;
#ifdef _OPENMP
        const int use_omp0 = (N >= YGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp0) schedule(static, 1)
#endif
        for (int j = 0; j < N; ++j)
            ygemmtr_beta_scale(j, j + 1, N, upper, beta, c, ldc);
        return;
    }

#ifdef _OPENMP
    const int use_omp = (N >= YGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
    for (int j = 0; j < N; ++j)
        ygemmtr_col(j, N, K, upper, alpha, beta, a, lda, b, ldb, c, ldc,
                    trans_a, conj_a, trans_b, conj_b);
}
