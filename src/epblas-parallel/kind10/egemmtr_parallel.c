/*
 * egemmtr_ — kind10 real (long double) triangular GEMM-update, public
 * Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * egemmtr_serial.c (packers, micro-kernels, macro kernels, beta-scale,
 * block policy), shared through egemmtr_kernel.h.
 *
 *   C := alpha · op(A) · op(B) + beta · C   (only the UPLO triangle of C)
 *
 * Threading: outer `omp parallel`, Bp shared and packed once per (jc, pc)
 * via `omp single` (implicit barrier), Ap private per thread, `omp for`
 * over the ic loop with `schedule(static, 1)` (interleaves ic chunks so the
 * triangular load — early-ic threads get fewer skipped tiles for LOWER,
 * later-ic threads for UPPER — balances).
 *
 * Nesting guard: when called from inside another routine's parallel region,
 * delegates to egemmtr_serial and opens no team of its own (the libgomp
 * barrier wedge guard, project-etrsm-omp4-wedge).
 */

#include "egemmtr_kernel.h"
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef egemmtr_T T;

#define MR EGEMMTR_MR
#define NR EGEMMTR_NR
#define EGEMMTR_OMP_MIN 32

static inline int imin(int a, int b) { return a < b ? a : b; }

#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

void egemmtr_(const char *uplo, const char *transa, const char *transb,
              const int *n_, const int *k_,
              const T *alpha_,
              const T *restrict a, const int *lda_,
              const T *restrict b, const int *ldb_,
              const T *beta_,
              T *restrict c, const int *ldc_,
              size_t uplo_len, size_t ta_len, size_t tb_len)
{
#ifdef _OPENMP
    /* Already inside a team → run serially in this thread, no nested
     * region (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        egemmtr_serial(uplo, transa, transb, n_, k_, alpha_, a, lda_, b, ldb_,
                       beta_, c, ldc_, uplo_len, ta_len, tb_len);
        return;
    }
#endif
    (void)uplo_len; (void)ta_len; (void)tb_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = (char)toupper((unsigned char)*uplo);
    const int ta = egemmtr_trans_code(transa);
    const int tb = egemmtr_trans_code(transb);

    if (N <= 0) return;
    const T zero = 0.0L, one = 1.0L;

    if (alpha == zero || K == 0) {
        if (beta == one) return;
#ifdef _OPENMP
        const int use_omp0 = (N >= EGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp0) schedule(static, 1)
#endif
        for (int j = 0; j < N; ++j)
            egemmtr_beta_scale(j, j + 1, N, UPLO, beta, c, ldc);
        return;
    }

    /* Beta-scale the UPLO triangle up front so the packed kernel can
     * always assume beta=1. */
    if (beta != one) {
#ifdef _OPENMP
        const int use_omp_beta = (N >= EGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp_beta) schedule(static, 1)
#endif
        for (int j = 0; j < N; ++j)
            egemmtr_beta_scale(j, j + 1, N, UPLO, beta, c, ldc);
    }

    int MC, KC, NC;
    egemmtr_block_sizes(&MC, &KC, &NC);
    if (NC > N) NC = N;
    if (NC < NR) NC = NR;

    const int sa_rows = egemmtr_round_up(MC, MR);
    const int sb_cols = egemmtr_round_up(NC, NR);
    const size_t ap_bytes = (size_t)sa_rows * KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * sb_cols * sizeof(T);

    T *Bp = (T *)aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (!Bp) {
        egemmtr_scalar_fallback(N, K, UPLO, ta, tb, alpha, a, lda, b, ldb, c, ldc);
        return;
    }

#ifdef _OPENMP
    const int use_omp = (N >= EGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel if(use_omp)
#endif
    {
        T *Ap = (T *)aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        if (Ap) {
            for (int jc = 0; jc < N; jc += NC) {
                const int jb = imin(NC, N - jc);
                for (int pc = 0; pc < K; pc += KC) {
                    const int pb = imin(KC, K - pc);

#ifdef _OPENMP
                    #pragma omp single
#endif
                    egemmtr_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
                    /* implicit barrier at end of `single` makes Bp safe. */

#ifdef _OPENMP
                    #pragma omp for schedule(static, 1)
#endif
                    for (int ic = 0; ic < N; ic += MC) {
                        const int ib = imin(MC, N - ic);

                        int tile_class;
                        if (UPLO == 'L') {
                            if (ic + ib <= jc)        tile_class = 0;
                            else if (ic >= jc + jb)   tile_class = 2;
                            else                      tile_class = 1;
                        } else {
                            if (ic >= jc + jb)        tile_class = 0;
                            else if (ic + ib <= jc)   tile_class = 2;
                            else                      tile_class = 1;
                        }
                        if (tile_class == 0) continue;

                        egemmtr_pack_A(a, lda, ic, pc, ib, pb, ta, Ap);

                        if (tile_class == 1)
                            egemmtr_macro_kernel_tri(ib, jb, pb, alpha, Ap, Bp,
                                                     &C_(ic, jc), ldc,
                                                     ic, jc, UPLO);
                        else
                            egemmtr_macro_kernel_rect(ib, jb, pb, alpha, Ap, Bp,
                                                      &C_(ic, jc), ldc);
                    }
                    /* implicit barrier at end of `for` keeps Bp stable. */
                }
            }
        }
        free(Ap);
    }

    free(Bp);
}

#undef C_
