/*
 * ysyrk_ — kind10 complex (_Complex long double) symmetric rank-k, the
 * public Fortran entry and threading-orchestration half of the ysyrk
 * overlay (see ysyrk_kernel.h; all the math lives in ysyrk_serial.c).
 *
 * Parallel shape: one `omp parallel for schedule(dynamic,1)` over the
 * diagonal blocks (jc). Triangular work per block is uneven, so dynamic
 * scheduling balances better than static. The scalar diagonal and the
 * ygemm trailing update run single-thread inside each block worker.
 *
 * Nesting guard: when ysyrk_ is itself called from inside another
 * routine's parallel region, it delegates to ysyrk_serial and opens no
 * region of its own — opening a nested team here trips the libgomp barrier
 * wedge (memory project-etrsm-omp4-wedge).
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "ysyrk_kernel.h"

typedef ysyrk_T T;

#define YSYRK_OMP_MIN 32

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

void ysyrk_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *beta_,
    T *restrict c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        ysyrk_serial(uplo, trans, n_, k_, alpha_, a, lda_, beta_, c, ldc_,
                     uplo_len, trans_len);
        return;
    }
#endif
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = (char)toupper((unsigned char)*uplo);
    const char TR   = (char)toupper((unsigned char)*trans);

    if (N == 0) return;

    if (alpha == ZERO || K == 0) {
        if (beta == ONE) return;
#ifdef _OPENMP
        const int use_omp = (N >= YSYRK_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j)
            ysyrk_beta_scale(j, j + 1, N, beta, c, ldc, UPLO);
        return;
    }

    const int nb = ysyrk_nb();

#ifdef _OPENMP
    const int use_omp = (N >= YSYRK_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        ysyrk_block(jc, jb, N, K, alpha, beta, a, lda, c, ldc, UPLO, TR);
    }
}
