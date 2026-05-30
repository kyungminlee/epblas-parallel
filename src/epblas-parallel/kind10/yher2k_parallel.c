/*
 * yher2k_ — kind10 complex (_Complex long double) Hermitian rank-2k, the
 * public Fortran entry and threading-orchestration half of the yher2k
 * overlay (see yher2k_kernel.h; all the math lives in yher2k_serial.c).
 *
 * Parallel shape: one `omp parallel for schedule(dynamic,1)` over the
 * diagonal blocks (jc). Triangular work per block is uneven, so dynamic
 * scheduling balances better than static. The scalar Hermitian diagonal and
 * the two ygemm trailing updates run single-thread inside each block worker.
 *
 * Nesting guard: when yher2k_ is itself called from inside another routine's
 * parallel region, it delegates to yher2k_serial and opens no region of its
 * own — opening a nested team here trips the libgomp barrier wedge (memory
 * project-etrsm-omp4-wedge).
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "yher2k_kernel.h"

typedef yher2k_TC TC;
typedef yher2k_TR TR;

#define YHER2K_OMP_MIN 32

void yher2k_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const TC *alpha_,
    const TC *restrict a, const int *lda_,
    const TC *restrict b, const int *ldb_,
    const TR *beta_,
    TC *restrict c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        yher2k_serial(uplo, trans, n_, k_, alpha_, a, lda_, b, ldb_, beta_,
                      c, ldc_, uplo_len, trans_len);
        return;
    }
#endif
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const TC alpha = *alpha_;
    const TR beta  = *beta_;
    const int ldc = *ldc_;
    const char UPLO = (char)toupper((unsigned char)*uplo);
    const char TR_c = (char)toupper((unsigned char)*trans);

    const TR rzero = 0.0L, rone = 1.0L;
    const TC czero = 0.0L + 0.0Li;

    if (N == 0) return;

    if ((alpha == czero) || K == 0) {
        if (beta == rone) {
            for (int j = 0; j < N; ++j) c[(size_t)j * ldc + j] = __real__ c[(size_t)j * ldc + j];
            return;
        }
#ifdef _OPENMP
        const int use_omp = (N >= YHER2K_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j)
            yher2k_beta_scale(j, j + 1, N, beta, c, ldc, UPLO);
        return;
    }

    const int nb = yher2k_nb();
    const int lda = *lda_, ldb = *ldb_;

#ifdef _OPENMP
    const int use_omp = (N >= YHER2K_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        yher2k_block(jc, jb, N, K, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, TR_c);
    }
}
