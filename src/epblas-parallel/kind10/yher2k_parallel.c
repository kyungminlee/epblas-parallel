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
#include "../common/blas_char.h"
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "yher2k_kernel.h"
#include "../common/epblas_facade.h"

typedef yher2k_TC TC;
typedef yher2k_TR TR;

#define YHER2K_OMP_MIN 32

static void yher2k_core(
    char uplo, char trans,
    ptrdiff_t N, ptrdiff_t K,
    const TC *alpha_,
    const TC *restrict a, ptrdiff_t lda,
    const TC *restrict b, ptrdiff_t ldb,
    const TR *beta_,
    TC *restrict c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        yher2k_serial(uplo, trans, N, K, alpha_, a, lda, b, ldb, beta_, c, ldc);
        return;
    }
#endif
    const TC alpha = *alpha_;
    const TR beta  = *beta_;
    const char UPLO = blas_up(uplo);
    const char TR_c = blas_up(trans);

    const TR rone = 1.0L;
    const TC czero = 0.0L + 0.0Li;

    if (N == 0) return;

    if ((alpha == czero) || K == 0) {
        if (beta == rone) {
            for (ptrdiff_t j = 0; j < N; ++j) c[(size_t)j * ldc + j] = __real__ c[(size_t)j * ldc + j];
            return;
        }
#ifdef _OPENMP
        const bool use_omp = (N >= YHER2K_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < N; ++j)
            yher2k_beta_scale(j, j + 1, N, beta, c, ldc, UPLO);
        return;
    }

    const ptrdiff_t nb = yher2k_nb();

    ptrdiff_t pw = nb;
#ifdef _OPENMP
    const ptrdiff_t nthreads = blas_omp_max_threads();
    const bool use_omp = (N >= YHER2K_OMP_MIN && nthreads > 1);
    /* Thin the diagonal blocks so the team can balance the triangular
     * per-block load at small N (N=64, nb=32 -> 2 blocks -> at most 2x);
     * triangular + schedule(dynamic) -> ppt=2 for finer balance. */
    if (use_omp) pw = blas_omp_panel_width(N, nthreads, nb, 2);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (ptrdiff_t jc = 0; jc < N; jc += pw) {
        const ptrdiff_t jb = (N - jc < pw) ? (N - jc) : pw;
        yher2k_block(jc, jb, N, K, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, TR_c);
    }
}

EPBLAS_FACADE_SYR2K(yher2k, TC, TR, TC)
