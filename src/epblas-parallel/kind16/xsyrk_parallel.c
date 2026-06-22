/*
 * xsyrk_ — kind16 complex (__complex128) symmetric rank-k, the public
 * Fortran entry and threading-orchestration half of the xsyrk overlay (see
 * xsyrk_kernel.h; all the math lives in xsyrk_serial.c).
 *
 * Parallel shape: one `omp parallel for schedule(dynamic,1)` over the
 * diagonal blocks (jc). Triangular work per block is uneven, so dynamic
 * scheduling balances better than static. The scalar diagonal and the xgemm
 * trailing update run single-thread inside each block worker.
 *
 * Nesting guard: when xsyrk_ is itself called from inside another routine's
 * parallel region, it delegates to xsyrk_serial and opens no region of its
 * own. Mirrors the kind10 ysyrk overlay.
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "xsyrk_kernel.h"
#include "../common/epblas_facade.h"

typedef xsyrk_T T;

#define XSYRK_OMP_MIN 32
#define XSYRK_OMP_NB  8   /* fine panel for triangular dynamic balance */

static const T ZERO = 0.0Q + 0.0Qi;
static const T ONE  = 1.0Q + 0.0Qi;

static void xsyrk_core(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const T *alpha_,
    const T *restrict a, ptrdiff_t lda,
    const T *beta_,
    T *restrict c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own. */
    if (omp_in_parallel()) {
        xsyrk_serial(uplo, trans, n, k, alpha_, a, lda, beta_, c, ldc);
        return;
    }
#endif
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = blas_up(uplo);
    const char TR   = blas_up(trans);

    if (n == 0) return;

    if (alpha == ZERO || k == 0) {
        if (beta == ONE) return;
#ifdef _OPENMP
        const bool use_omp = (n >= XSYRK_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < n; ++j)
            xsyrk_beta_scale(j, j + 1, n, beta, c, ldc, UPLO);
        return;
    }

    ptrdiff_t nb = xsyrk_nb();

#ifdef _OPENMP
    const bool use_omp = (n >= XSYRK_OMP_MIN && blas_omp_max_threads() > 1);
    if (use_omp) {
        /* Use a fine OMP panel so dynamic scheduling can balance the
         * triangular per-block work: the trailing GEMM shrinks from N-jc
         * down to 0 across the diagonal, so few coarse panels (the serial
         * nb=32 gives only 2 at N=64) leave threads idle on the cheap tail.
         * Measured optimum is a small fixed block (~8) across N∈[64,256] —
         * the packed trailing GEMM amortizes packing even at jb=8, and the
         * balance win dominates at every size. The serial path keeps nb=32
         * (no balance concern, better amortization). Floor to >=1 panel per
         * thread for tiny N. */
        const ptrdiff_t nthr = blas_omp_max_threads();
        nb = XSYRK_OMP_NB;
        if (nb * nthr > n) {
            ptrdiff_t want = (n + nthr - 1) / nthr;
            nb = (want < 2) ? 2 : ((want + 1) / 2) * 2;   /* round up to MR */
        }
    }
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (ptrdiff_t jc = 0; jc < n; jc += nb) {
        const ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
        xsyrk_block(jc, jb, n, k, alpha, beta, a, lda, c, ldc, UPLO, TR);
    }
}

EPBLAS_FACADE_SYRK(xsyrk, T, T)
