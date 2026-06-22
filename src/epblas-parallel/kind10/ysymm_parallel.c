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
#include "../common/epblas_facade.h"

typedef ysymm_T T;

#define YSYMM_OMP_MIN 32

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

static void ysymm_core(
    char side, char uplo,
    ptrdiff_t M, ptrdiff_t N,
    const T *alpha_,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict b, ptrdiff_t ldb,
    const T *beta_,
    T *restrict c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        ysymm_serial(side, uplo, M, N, alpha_, a, lda, b, ldb, beta_, c, ldc);
        return;
    }
#endif
    const T alpha = *alpha_, beta = *beta_;
    const char SIDE = (char)toupper((unsigned char)side);
    const char UPLO = (char)toupper((unsigned char)uplo);

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        if (beta == ONE) return;
#ifdef _OPENMP
        const ptrdiff_t axis = (SIDE == 'L') ? N : M;
        const ptrdiff_t use_omp = (axis >= YSYMM_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < N; ++j)
            ysymm_beta_only(j, j + 1, M, beta, c, ldc);
        return;
    }

    ptrdiff_t nt = 1;
#ifdef _OPENMP
    nt = blas_omp_max_threads();
#endif
    const ptrdiff_t nb = ysymm_nb();

    if (SIDE == 'L' && M <= nb) {
#ifdef _OPENMP
        const ptrdiff_t use_omp = (N >= YSYMM_OMP_MIN && nt > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < N; ++j)
            ysymm_L_singleblock(j, j + 1, M, alpha, beta, a, lda, b, ldb, c, ldc, UPLO);
        return;
    }

    if (SIDE == 'L') {
        ptrdiff_t pw = nb;
#ifdef _OPENMP
        const ptrdiff_t use_omp = (N >= YSYMM_OMP_MIN && nt > 1);
        /* Thin the column panels so the team has ~1 panel/thread at small N
         * (N=64, nb=32 -> 2 panels -> 2 idle threads); inner nb is preserved
         * for the trailing-GEMM blocking. Rectangular work -> ppt=1, static. */
        if (use_omp) pw = blas_omp_panel_width(N, (int)nt, nb, 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t jc = 0; jc < N; jc += pw) {
            const ptrdiff_t jb = (N - jc < pw) ? (N - jc) : pw;
            ysymm_L_panel(jc, jb, M, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    } else {
        ptrdiff_t pw = nb;
#ifdef _OPENMP
        const ptrdiff_t use_omp = (M >= YSYMM_OMP_MIN && nt > 1);
        if (use_omp) pw = blas_omp_panel_width(M, (int)nt, nb, 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t ic = 0; ic < M; ic += pw) {
            const ptrdiff_t ib = (M - ic < pw) ? (M - ic) : pw;
            ysymm_R_panel(ic, ib, N, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    }
}

EPBLAS_FACADE_SYMM(ysymm, T)
