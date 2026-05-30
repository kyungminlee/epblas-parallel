/*
 * yherk_ — kind10 complex (_Complex long double) Hermitian rank-k, the
 * public Fortran entry and threading-orchestration half of the yherk
 * overlay (see yherk_kernel.h; all the math lives in yherk_serial.c).
 *
 * Parallel shape: one `omp parallel for schedule(dynamic,1)` over the
 * diagonal blocks (jc). Triangular work per block is uneven, so dynamic
 * scheduling balances better than static. The scalar Hermitian diagonal
 * and the ygemm trailing update run single-thread inside each block worker.
 *
 * Nesting guard: when yherk_ is itself called from inside another routine's
 * parallel region, it delegates to yherk_serial and opens no region of its
 * own — opening a nested team here trips the libgomp barrier wedge (memory
 * project-etrsm-omp4-wedge).
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "yherk_kernel.h"

typedef yherk_TC TC;
typedef yherk_TR TR;

#define YHERK_OMP_MIN 32

void yherk_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const TR *alpha_,
    const TC *restrict a, const int *lda_,
    const TR *beta_,
    TC *restrict c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        yherk_serial(uplo, trans, n_, k_, alpha_, a, lda_, beta_, c, ldc_,
                     uplo_len, trans_len);
        return;
    }
#endif
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const TR alpha = *alpha_, beta = *beta_;
    const int ldc = *ldc_;
    const char UPLO = (char)toupper((unsigned char)*uplo);
    const char TR_c = (char)toupper((unsigned char)*trans);

    const TR rzero = 0.0L, rone = 1.0L;

    if (N == 0) return;

    if (alpha == rzero || K == 0) {
        if (beta == rone) {
            for (int j = 0; j < N; ++j) c[(size_t)j * ldc + j] = __real__ c[(size_t)j * ldc + j];
            return;
        }
#ifdef _OPENMP
        const int use_omp = (N >= YHERK_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j)
            yherk_beta_scale(j, j + 1, N, beta, c, ldc, UPLO);
        return;
    }

    const int nb = yherk_nb();
    const int lda = *lda_;

#ifdef _OPENMP
    const int use_omp = (N >= YHERK_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        yherk_block(jc, jb, N, K, alpha, beta, a, lda, c, ldc, UPLO, TR_c);
    }
}
