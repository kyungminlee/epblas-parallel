/*
 * esymm_serial — kind10 real (long double) symmetric matrix-multiply,
 * single-thread. Owns the SYMM-aware packers and the fused serial driver.
 *
 *   C := alpha * A * B + beta * C    (SIDE='L', A symmetric M×M)
 *   C := alpha * B * A + beta * C    (SIDE='R', A symmetric N×N)
 *
 * Structure (mirrors the OpenBLAS-overlay esymm, on par's own kernels):
 * one packed GEMM. The symmetric operand is packed via esymm_pack_{a,b}_sym
 * — which mirror the UPLO triangle into egemm's packed layout — and the
 * regular operand via the stock egemm packers. The shared MR×NR macro-
 * kernel then streams diagonal and off-diagonal tiles identically. No
 * scalar diagonal kernel, no per-tile re-dispatch into egemm.
 *
 * The kernel pieces (block policy, packers, beta pre-pass, macro-kernel)
 * are the serial egemm primitives from egemm_kernel.h — calling them
 * (not egemm_) keeps esymm free of any nested OpenMP team, so it is safe
 * to run inside another routine's parallel region (the libgomp barrier
 * wedge guard, memory project-etrsm-omp4-wedge).
 */

#include "esymm_kernel.h"
#include "egemm_kernel.h"
#include <stdlib.h>
#include <ctype.h>

typedef esymm_T T;

#define MR EGEMM_MR
#define NR EGEMM_NR

/* Logical symmetric element (row, col): only the UPLO triangle is stored,
 * so a position in the other triangle is read from its transpose. */
static inline T sym_at(const T *a, int lda, int row, int col, char uplo) {
    if (uplo == 'L')
        return (row >= col) ? a[(size_t)col * lda + row]
                            : a[(size_t)row * lda + col];
    else
        return (row <= col) ? a[(size_t)col * lda + row]
                            : a[(size_t)row * lda + col];
}

void esymm_pack_a_sym(const T *a, int lda,
                      int ic, int pc, int ib, int pb,
                      char uplo, T *Ap)
{
    const int npanel = (ib + MR - 1) / MR;
    for (int q = 0; q < npanel; ++q) {
        const int i0   = ic + q * MR;
        const int rows = (q == npanel - 1) ? (ib - q * MR) : MR;
        T *Apanel = &Ap[(size_t)q * pb * MR];
        for (int p = 0; p < pb; ++p) {
            const int col = pc + p;
            T *dst = &Apanel[(size_t)p * MR];
            int ii;
            for (ii = 0; ii < rows; ++ii) dst[ii] = sym_at(a, lda, i0 + ii, col, uplo);
            for (; ii < MR; ++ii)         dst[ii] = 0.0L;
        }
    }
}

void esymm_pack_b_sym(const T *a, int lda,
                      int pc, int jc, int pb, int jb,
                      char uplo, T *Bp)
{
    const int npanel = (jb + NR - 1) / NR;
    for (int q = 0; q < npanel; ++q) {
        const int j0   = jc + q * NR;
        const int cols = (q == npanel - 1) ? (jb - q * NR) : NR;
        T *Bpanel = &Bp[(size_t)q * pb * NR];
        for (int p = 0; p < pb; ++p) {
            const int row = pc + p;
            T *dst = &Bpanel[(size_t)p * NR];
            int jj;
            for (jj = 0; jj < cols; ++jj) dst[jj] = sym_at(a, lda, row, j0 + jj, uplo);
            for (; jj < NR; ++jj)         dst[jj] = 0.0L;
        }
    }
}

void esymm_serial(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t side_len, size_t uplo_len)
{
    (void)side_len; (void)uplo_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char SIDE = (char)toupper((unsigned char)*side);
    const char UPLO = (char)toupper((unsigned char)*uplo);

    if (M <= 0 || N <= 0) return;

    egemm_beta_prepass(M, N, beta, c, ldc);   /* C := beta*C (handles beta 0/1) */
    if (alpha == 0.0L) return;

    /* K is the contraction dim = the symmetric matrix's side. */
    const int K = (SIDE == 'L') ? M : N;

    int MC, KC, NC;
    egemm_choose_blocks(K, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)egemm_round_up(MC, MR) * KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * egemm_round_up(NC, NR) * sizeof(T);
    T *Ap = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
    T *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (Ap && Bp) {
        for (int jc = 0; jc < N; jc += NC) {
            const int jb = (N - jc < NC) ? (N - jc) : NC;
            for (int pc = 0; pc < K; pc += KC) {
                const int pb = (K - pc < KC) ? (K - pc) : KC;
                /* Pack the K×N (jc-band) right operand. SIDE='L': regular B.
                 * SIDE='R': the symmetric A goes in the B slot. */
                if (SIDE == 'L')
                    egemm_pack_B(b, ldb, pc, jc, pb, jb, 'N', Bp);
                else
                    esymm_pack_b_sym(a, lda, pc, jc, pb, jb, UPLO, Bp);

                for (int ic = 0; ic < M; ic += MC) {
                    const int ib = (M - ic < MC) ? (M - ic) : MC;
                    /* Pack the M×K (ic-block) left operand. SIDE='L': the
                     * symmetric A. SIDE='R': regular B in the A slot. */
                    if (SIDE == 'L')
                        esymm_pack_a_sym(a, lda, ic, pc, ib, pb, UPLO, Ap);
                    else
                        egemm_pack_A(b, ldb, ic, pc, ib, pb, 'N', Ap);

                    egemm_macro_kernel(ib, jb, pb, alpha, Ap, Bp,
                                       &c[(size_t)jc * ldc + ic], ldc);
                }
            }
        }
    }
    free(Ap);
    free(Bp);
}
