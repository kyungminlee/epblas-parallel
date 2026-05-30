/*
 * ysymm_ — kind10 complex (_Complex long double) symmetric matrix-multiply,
 * the public Fortran entry and threading-orchestration half of the ysymm
 * overlay (see ysymm_kernel.h; all the math lives in ysymm_serial.c).
 *
 * Parallel shape: one `omp parallel for schedule(static)` over the outer
 * panel axis — J column panels of C for SIDE='L', I row panels for
 * SIDE='R'. Each thread owns a disjoint slice of C, so the inner I/K block
 * loops and the ygemm trailing updates run single-thread inside the worker.
 *
 * Nesting guard: when ysymm_ is itself called from inside another routine's
 * parallel region, it delegates to ysymm_serial and opens no region of its
 * own — opening a nested team here trips the libgomp barrier wedge (memory
 * project-etrsm-omp4-wedge).
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "ysymm_kernel.h"

typedef ysymm_T T;

#define YSYMM_OMP_MIN 32

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

void ysymm_(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict b, const int *ldb_,
    const T *beta_,
    T *restrict c, const int *ldc_,
    size_t side_len, size_t uplo_len)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        ysymm_serial(side, uplo, m_, n_, alpha_, a, lda_, b, ldb_, beta_,
                     c, ldc_, side_len, uplo_len);
        return;
    }
#endif
    (void)side_len; (void)uplo_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char SIDE = (char)toupper((unsigned char)*side);
    const char UPLO = (char)toupper((unsigned char)*uplo);

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        if (beta == ONE) return;
#ifdef _OPENMP
        const int axis = (SIDE == 'L') ? N : M;
        const int use_omp = (axis >= YSYMM_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j)
            ysymm_beta_only(j, j + 1, M, beta, c, ldc);
        return;
    }

    int nt = 1;
#ifdef _OPENMP
    nt = blas_omp_max_threads();
#endif
    const int nb = ysymm_nb();

    if (SIDE == 'L' && M <= nb) {
#ifdef _OPENMP
        const int use_omp = (N >= YSYMM_OMP_MIN && nt > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j)
            ysymm_L_singleblock(j, j + 1, M, alpha, beta, a, lda, b, ldb, c, ldc, UPLO);
        return;
    }

    if (SIDE == 'L') {
#ifdef _OPENMP
        const int use_omp = (N >= YSYMM_OMP_MIN && nt > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            ysymm_L_panel(jc, jb, M, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    } else {
#ifdef _OPENMP
        const int use_omp = (M >= YSYMM_OMP_MIN && nt > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            ysymm_R_panel(ic, ib, N, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    }
}
