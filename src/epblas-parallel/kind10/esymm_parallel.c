/*
 * esymm_ — kind10 real (long double) symmetric matrix-multiply, the public
 * Fortran entry and threading-orchestration half of the esymm overlay (see
 * esymm_kernel.h; all the math lives in esymm_serial.c).
 *
 * Parallel shape: one `omp parallel for schedule(static)` over the outer panel
 * axis — J column panels of C for SIDE='L', I row panels for SIDE='R'. Each
 * thread owns a disjoint slice of C, so the inner I/K block loops and the
 * egemm trailing updates run single-thread inside the worker.
 *
 * Nesting guard: when esymm_ is itself called from inside another routine's
 * parallel region, it delegates to esymm_serial and opens no region of its
 * own — opening a nested team here trips the libgomp barrier wedge (memory
 * project-etrsm-omp4-wedge).
 */

#include "esymm_kernel.h"
#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef esymm_T T;

#define ESYMM_OMP_MIN 32

void esymm_(
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
        esymm_serial(side, uplo, m_, n_, alpha_, a, lda_, b, ldb_, beta_,
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

    const T zero = 0.0L, one = 1.0L;

    if (alpha == zero) {
        if (beta == one) return;
#ifdef _OPENMP
        const int axis = (SIDE == 'L') ? N : M;
        const int use_omp = (axis >= ESYMM_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j)
            esymm_beta_only(j, j + 1, M, beta, c, ldc);
        return;
    }

    int nt = 1;
#ifdef _OPENMP
    nt = blas_omp_max_threads();
#endif
    const int nb = esymm_nb();

    if (SIDE == 'L' && M <= nb) {
#ifdef _OPENMP
        const int use_omp = (N >= ESYMM_OMP_MIN && nt > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j)
            esymm_L_singleblock(j, j + 1, M, alpha, beta, a, lda, b, ldb, c, ldc, UPLO);
        return;
    }

    if (SIDE == 'L') {
#ifdef _OPENMP
        const int use_omp = (N >= ESYMM_OMP_MIN && nt > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            esymm_L_panel(jc, jb, M, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    } else {
#ifdef _OPENMP
        const int use_omp = (M >= ESYMM_OMP_MIN && nt > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            esymm_R_panel(ic, ib, N, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    }
}
