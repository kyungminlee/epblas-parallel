/*
 * ysyr2k_ — kind10 complex (_Complex long double) symmetric rank-2k, the
 * public Fortran entry and threading-orchestration half of the ysyr2k
 * overlay (see ysyr2k_kernel.h; all the math lives in ysyr2k_serial.c).
 *
 * Parallel shape: one `omp parallel for schedule(dynamic,1)` over the
 * diagonal blocks (jc). Triangular work per block is uneven, so dynamic
 * scheduling balances better than static. The scalar diagonal and the two
 * ygemm trailing updates run single-thread inside each block worker.
 *
 * Nesting guard: when ysyr2k_ is itself called from inside another routine's
 * parallel region, it delegates to ysyr2k_serial and opens no region of its
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

#include "ysyr2k_kernel.h"
#include "../common/epblas_facade.h"

typedef ysyr2k_T T;

#define YSYR2K_OMP_MIN 32

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

static void ysyr2k_core(
    char uplo, char trans,
    ptrdiff_t N, ptrdiff_t K,
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
        ysyr2k_serial(uplo, trans, N, K, alpha_, a, lda, b, ldb, beta_, c, ldc);
        return;
    }
#endif
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = blas_up(uplo);
    char TR = blas_up(trans);
    if (TR == 'C') TR = 'T';

    if (N == 0) return;

    if (alpha == ZERO || K == 0) {
        if (beta == ONE) return;
#ifdef _OPENMP
        const bool use_omp = (N >= YSYR2K_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < N; ++j)
            ysyr2k_beta_scale(j, j + 1, N, beta, c, ldc, UPLO);
        return;
    }

    const ptrdiff_t nb = ysyr2k_nb();

    ptrdiff_t pw = nb;
#ifdef _OPENMP
    const ptrdiff_t nthreads = blas_omp_max_threads();
    const bool use_omp = (N >= YSYR2K_OMP_MIN && nthreads > 1);
    /* Thin the diagonal blocks so the team can balance the triangular
     * per-block load at small N (N=64, nb=32 -> 2 blocks -> at most 2x);
     * triangular + schedule(dynamic) -> ppt=2 for finer balance. */
    if (use_omp) pw = blas_omp_panel_width(N, nthreads, nb, 2);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (ptrdiff_t jc = 0; jc < N; jc += pw) {
        const ptrdiff_t jb = (N - jc < pw) ? (N - jc) : pw;
        ysyr2k_block(jc, jb, N, K, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, TR);
    }
}

EPBLAS_FACADE_SYR2K(ysyr2k, T, T, T)
