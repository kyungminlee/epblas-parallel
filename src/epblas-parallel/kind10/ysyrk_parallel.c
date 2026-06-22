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
 * wedge.
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "ysyrk_kernel.h"
#include "../common/epblas_facade.h"

typedef ysyrk_T T;

#define YSYRK_OMP_MIN 32

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

static void ysyrk_core(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const T *alpha_,
    const T *restrict a, ptrdiff_t lda,
    const T *beta_,
    T *restrict c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        ysyrk_serial(uplo, trans, n, k, alpha_, a, lda, beta_, c, ldc);
        return;
    }
#endif
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = blas_up(uplo);
    const char TRANS   = blas_up(trans);

    if (n == 0) return;

    if (alpha == ZERO || k == 0) {
        if (beta == ONE) return;
#ifdef _OPENMP
        const bool use_omp = (n >= YSYRK_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < n; ++j)
            ysyrk_beta_scale(j, j + 1, n, beta, c, ldc, UPLO);
        return;
    }

    const ptrdiff_t nb = ysyrk_nb();

    ptrdiff_t pw = nb;
#ifdef _OPENMP
    const ptrdiff_t nthreads = blas_omp_max_threads();
    const bool use_omp = (n >= YSYRK_OMP_MIN && nthreads > 1);
    /* Thin the diagonal blocks so the team can balance the triangular
     * per-block load at small N (N=64, nb=32 -> 2 blocks -> at most 2x);
     * triangular + schedule(dynamic) -> ppt=2 for finer balance. */
    if (use_omp) pw = blas_omp_panel_width(n, nthreads, nb, 2);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (ptrdiff_t jc = 0; jc < n; jc += pw) {
        const ptrdiff_t jb = (n - jc < pw) ? (n - jc) : pw;
        ysyrk_block(jc, jb, n, k, alpha, beta, a, lda, c, ldc, UPLO, TRANS);
    }
}

EPBLAS_FACADE_SYRK(ysyrk, T, T)
