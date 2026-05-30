/*
 * esyr2k_ — kind10 real (long double) symmetric rank-2k, the public Fortran
 * entry and threading-orchestration half of the esyr2k overlay (see
 * esyr2k_kernel.h; all the math lives in esyr2k_serial.c).
 *
 * Parallel shape: one `omp parallel for schedule(dynamic,1)` over the diagonal
 * blocks (jc). Triangular work per block is uneven, so dynamic scheduling
 * balances better than static. The scalar diagonal and the two egemm trailing
 * updates run single-thread inside each block worker.
 *
 * Nesting guard: when esyr2k_ is itself called from inside another routine's
 * parallel region, it delegates to esyr2k_serial and opens no region of its
 * own — opening a nested team here trips the libgomp barrier wedge (memory
 * project-etrsm-omp4-wedge).
 */

#include "esyr2k_kernel.h"
#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef esyr2k_T T;

#define ESYR2K_OMP_MIN 32

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

void esyr2k_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict b, const int *ldb_,
    const T *beta_,
    T *restrict c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        esyr2k_serial(uplo, trans, n_, k_, alpha_, a, lda_, b, ldb_, beta_,
                      c, ldc_, uplo_len, trans_len);
        return;
    }
#endif
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';

    if (N == 0) return;

    const T zero = 0.0L, one = 1.0L;

    if (alpha == zero || K == 0) {
        if (beta == one) return;
#ifdef _OPENMP
        const int use_omp = (N >= ESYR2K_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j)
            esyr2k_beta_scale(j, j + 1, N, beta, c, ldc, UPLO);
        return;
    }

    /* TR-aware nb (see esyr2k_kernel.h). */
    const int nb = (TR == 'T') ? N : esyr2k_nb();

#ifdef _OPENMP
    const int use_omp = (N >= ESYR2K_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        esyr2k_block(jc, jb, N, K, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, TR);
    }
}
