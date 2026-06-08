/*
 * xherk_ — kind16 complex (__complex128) Hermitian rank-k, the public
 * Fortran entry and threading-orchestration half of the xherk overlay (see
 * xherk_kernel.h; all the math lives in xherk_serial.c).
 *
 * Parallel shape: one `omp parallel for schedule(dynamic,1)` over the
 * diagonal blocks (jc). Triangular work per block is uneven, so dynamic
 * scheduling balances better than static. The scalar Hermitian diagonal and
 * the xgemm trailing update run single-thread inside each block worker.
 *
 * Nesting guard: when xherk_ is itself called from inside another routine's
 * parallel region, it delegates to xherk_serial_ and opens no region of its
 * own. xherk_serial_ shares the int Fortran ABI, so forward the pointers
 * unchanged. Mirrors the kind10 yherk overlay.
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "xherk_kernel.h"

typedef xherk_TC TC;
typedef xherk_TR TR;

#define XHERK_OMP_MIN 32
#define XHERK_OMP_NB  8   /* fine panel for triangular dynamic balance */

void xherk_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const TR *alpha_,
    const TC *a, const int *lda_,
    const TR *beta_,
    TC *c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own. xherk_serial_ shares the int
     * Fortran ABI, so forward the pointers unchanged. */
    if (omp_in_parallel()) {
        xherk_serial_(uplo, trans, n_, k_, alpha_, a, lda_, beta_, c, ldc_,
                      uplo_len, trans_len);
        return;
    }
#endif
    (void)uplo_len; (void)trans_len;
    const ptrdiff_t N = *n_, K = *k_;
    const TR alpha = *alpha_, beta = *beta_;
    const ptrdiff_t ldc = *ldc_;
    const char UPLO = (char)toupper((unsigned char)*uplo);
    const char TR_c = (char)toupper((unsigned char)*trans);

    const TR rzero = 0.0Q, rone = 1.0Q;

    if (N == 0) return;

    if (alpha == rzero || K == 0) {
        if (beta == rone) {
            for (ptrdiff_t j = 0; j < N; ++j) c[(size_t)j * ldc + j] = __real__ c[(size_t)j * ldc + j];
            return;
        }
#ifdef _OPENMP
        const ptrdiff_t use_omp = (N >= XHERK_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < N; ++j)
            xherk_beta_scale(j, j + 1, N, beta, c, ldc, UPLO);
        return;
    }

    ptrdiff_t nb = xherk_nb();
    const ptrdiff_t lda = *lda_;

#ifdef _OPENMP
    const ptrdiff_t use_omp = (N >= XHERK_OMP_MIN && blas_omp_max_threads() > 1);
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
        nb = XHERK_OMP_NB;
        if (nb * nthr > N) {
            ptrdiff_t want = (N + nthr - 1) / nthr;
            nb = (want < 2) ? 2 : ((want + 1) / 2) * 2;   /* round up to MR */
        }
    }
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (ptrdiff_t jc = 0; jc < N; jc += nb) {
        const ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
        xherk_block(jc, jb, N, K, alpha, beta, a, lda, c, ldc, UPLO, TR_c);
    }
}
