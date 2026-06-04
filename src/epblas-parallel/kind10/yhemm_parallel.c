/*
 * yhemm_ — kind10 complex (_Complex long double) Hermitian matrix-multiply,
 * the public Fortran entry and threading-orchestration half of the yhemm
 * overlay (see yhemm_kernel.h; all the math lives in yhemm_serial.c).
 *
 * Parallel shape: one `omp parallel for schedule(static)` over the outer
 * panel axis — J column panels of C for SIDE='L', I row panels for
 * SIDE='R'. Each thread owns a disjoint slice of C, so the inner block
 * loops and the ygemm trailing updates run single-thread inside the worker.
 *
 * Nesting guard: when yhemm_ is itself called from inside another routine's
 * parallel region, it delegates to yhemm_serial and opens no region of its
 * own — opening a nested team here trips the libgomp barrier wedge (memory
 * project-etrsm-omp4-wedge).
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "yhemm_kernel.h"

typedef yhemm_T T;

#define YHEMM_OMP_MIN 32

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

void yhemm_(
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
        const ptrdiff_t m_pt = *m_, n_pt = *n_, lda_pt = *lda_, ldb_pt = *ldb_, ldc_pt = *ldc_;
        yhemm_serial(side, uplo, &m_pt, &n_pt, alpha_, a, &lda_pt, b, &ldb_pt, beta_,
                     c, &ldc_pt, side_len, uplo_len);
        return;
    }
#endif
    (void)side_len; (void)uplo_len;
    const ptrdiff_t M = *m_, N = *n_;
    const ptrdiff_t lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char SIDE = (char)toupper((unsigned char)*side);
    const char UPLO = (char)toupper((unsigned char)*uplo);

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        if (beta == ONE) return;
#ifdef _OPENMP
        const ptrdiff_t axis = (SIDE == 'L') ? N : M;
        const ptrdiff_t use_omp = (axis >= YHEMM_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < N; ++j)
            yhemm_beta_only(j, j + 1, M, beta, c, ldc);
        return;
    }

    ptrdiff_t nt = 1;
#ifdef _OPENMP
    nt = blas_omp_max_threads();
#endif
    const ptrdiff_t nb = yhemm_nb();

    if (SIDE == 'L' && M <= nb) {
#ifdef _OPENMP
        const ptrdiff_t use_omp = (N >= YHEMM_OMP_MIN && nt > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < N; ++j)
            yhemm_L_singleblock(j, j + 1, M, alpha, beta, a, lda, b, ldb, c, ldc, UPLO);
        return;
    }

    if (SIDE == 'L') {
#ifdef _OPENMP
        const ptrdiff_t use_omp = (N >= YHEMM_OMP_MIN && nt > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t jc = 0; jc < N; jc += nb) {
            const ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
            yhemm_L_panel(jc, jb, M, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    } else {
#ifdef _OPENMP
        const ptrdiff_t use_omp = (M >= YHEMM_OMP_MIN && nt > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t ic = 0; ic < M; ic += nb) {
            const ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
            yhemm_R_panel(ic, ib, N, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    }
}
