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
 * parallel region, it delegates to xsyrk_serial_ and opens no region of its
 * own. xsyrk_serial_ shares the int Fortran ABI, so forward the pointers
 * unchanged. Mirrors the kind10 ysyrk overlay.
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "xsyrk_kernel.h"

typedef xsyrk_T T;

#define XSYRK_OMP_MIN 32
#define XSYRK_OMP_NB  8   /* fine panel for triangular dynamic balance */

static const T ZERO = 0.0Q + 0.0Qi;
static const T ONE  = 1.0Q + 0.0Qi;

void xsyrk_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own. xsyrk_serial_ shares the int
     * Fortran ABI, so forward the pointers unchanged. */
    if (omp_in_parallel()) {
        xsyrk_serial_(uplo, trans, n_, k_, alpha_, a, lda_, beta_, c, ldc_,
                      uplo_len, trans_len);
        return;
    }
#endif
    (void)uplo_len; (void)trans_len;
    const ptrdiff_t N = *n_, K = *k_;
    const ptrdiff_t lda = *lda_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = (char)toupper((unsigned char)*uplo);
    const char TR   = (char)toupper((unsigned char)*trans);

    if (N == 0) return;

    if (alpha == ZERO || K == 0) {
        if (beta == ONE) return;
#ifdef _OPENMP
        const ptrdiff_t use_omp = (N >= XSYRK_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < N; ++j)
            xsyrk_beta_scale(j, j + 1, N, beta, c, ldc, UPLO);
        return;
    }

    ptrdiff_t nb = xsyrk_nb();

#ifdef _OPENMP
    const ptrdiff_t use_omp = (N >= XSYRK_OMP_MIN && blas_omp_max_threads() > 1);
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
        if (nb * nthr > N) {
            ptrdiff_t want = (N + nthr - 1) / nthr;
            nb = (want < 2) ? 2 : ((want + 1) / 2) * 2;   /* round up to MR */
        }
    }
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (ptrdiff_t jc = 0; jc < N; jc += nb) {
        const ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
        xsyrk_block(jc, jb, N, K, alpha, beta, a, lda, c, ldc, UPLO, TR);
    }
}
