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
#include "../common/blas_char.h"
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "yherk_kernel.h"
#include "../common/epblas_facade.h"

typedef yherk_TC TC;
typedef yherk_TR TR;

#define YHERK_OMP_MIN 32

static void yherk_core(
    char uplo, char trans,
    ptrdiff_t N, ptrdiff_t K,
    const TR *alpha_,
    const TC *restrict a, ptrdiff_t lda,
    const TR *beta_,
    TC *restrict c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        yherk_serial(uplo, trans, N, K, alpha_, a, lda, beta_, c, ldc);
        return;
    }
#endif
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO = blas_up(uplo);
    const char TR_c = blas_up(trans);

    const TR rzero = 0.0L, rone = 1.0L;

    if (N == 0) return;

    if (alpha == rzero || K == 0) {
        if (beta == rone) {
            for (ptrdiff_t j = 0; j < N; ++j) c[(size_t)j * ldc + j] = __real__ c[(size_t)j * ldc + j];
            return;
        }
#ifdef _OPENMP
        const bool use_omp = (N >= YHERK_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < N; ++j)
            yherk_beta_scale(j, j + 1, N, beta, c, ldc, UPLO);
        return;
    }

    const ptrdiff_t nb = yherk_nb();

    ptrdiff_t pw = nb;
#ifdef _OPENMP
    const ptrdiff_t nt = blas_omp_max_threads();
    const bool use_omp = (N >= YHERK_OMP_MIN && nt > 1);
    /* Thin the diagonal blocks so the team has enough work units to balance
     * the triangular per-block load at small N (N=64, nb=32 -> 2 blocks -> at
     * most 2x); triangular + schedule(dynamic) -> ppt=2 for finer balance. */
    if (use_omp) pw = blas_omp_panel_width(N, nt, nb, 2);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (ptrdiff_t jc = 0; jc < N; jc += pw) {
        const ptrdiff_t jb = (N - jc < pw) ? (N - jc) : pw;
        yherk_block(jc, jb, N, K, alpha, beta, a, lda, c, ldc, UPLO, TR_c);
    }
}

EPBLAS_FACADE_SYRK(yherk, TR, TC)
