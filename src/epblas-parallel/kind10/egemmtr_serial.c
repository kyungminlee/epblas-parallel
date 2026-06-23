/*
 * egemmtr_serial — kind10 real (long double) triangular GEMM-update,
 * single-thread. This TU owns ALL of the egemmtr math (packers, MR×NR
 * micro-kernels, rectangular + triangle-aware macro kernels, triangle
 * beta-scale, block policy, scalar fallback) and the pure-serial entry.
 * egemmtr_parallel.c reuses these pieces inside its OpenMP team.
 *
 *   C := alpha · op(A) · op(B) + beta · C   (only the UPLO triangle of C)
 *
 * Inline GotoBLAS: walk the (jc, pc, ic) loop nest, classify each (ic, jc)
 * tile against the UPLO triangle (skip / rectangular / triangle-aware).
 */

#include "egemmtr_kernel.h"
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef egemmtr_TR TR;

#define MR EGEMMTR_MR
#define NR EGEMMTR_NR
#define EGEMMTR_MC_DEFAULT  64
#define EGEMMTR_KC_DEFAULT 256
#define EGEMMTR_NC_DEFAULT 512

void egemmtr_block_sizes(ptrdiff_t *MC, ptrdiff_t *KC, ptrdiff_t *NC) {
    *MC = EGEMMTR_MC_DEFAULT; *KC = EGEMMTR_KC_DEFAULT; *NC = EGEMMTR_NC_DEFAULT;
}

ptrdiff_t egemmtr_trans_code(const char *p) {
    char c = blas_up(*p);
    return (c == 'C') ? 'T' : c;
}

ptrdiff_t egemmtr_round_up(ptrdiff_t v, ptrdiff_t m) { return ((v + m - 1) / m) * m; }
static inline ptrdiff_t imin(ptrdiff_t a, ptrdiff_t b) { return a < b ? a : b; }

#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

/* ─── Packers ──────────────────────────────────────────────────── */

void egemmtr_pack_A(const TR *restrict A, ptrdiff_t lda,
                    ptrdiff_t ic, ptrdiff_t pc, ptrdiff_t ib, ptrdiff_t pb,
                    char ta, TR *restrict Ap)
{
    const ptrdiff_t npanel = (ib + MR - 1) / MR;
    for (ptrdiff_t q = 0; q < npanel; ++q) {
        const ptrdiff_t i0 = ic + q * MR;
        const ptrdiff_t rows = (q == npanel - 1) ? (ib - q * MR) : MR;
        TR *panel = &Ap[(size_t)q * pb * MR];
        if (ta == 'N') {
            for (ptrdiff_t p = 0; p < pb; ++p) {
                const TR *src = &A[(size_t)(pc + p) * lda + i0];
                TR *dst = &panel[(size_t)p * MR];
                ptrdiff_t ii;
                for (ii = 0; ii < rows; ++ii) dst[ii] = src[ii];
                for (; ii < MR; ++ii) dst[ii] = 0.0L;
            }
        } else {
            for (ptrdiff_t p = 0; p < pb; ++p) {
                TR *dst = &panel[(size_t)p * MR];
                ptrdiff_t ii;
                for (ii = 0; ii < rows; ++ii)
                    dst[ii] = A[(size_t)(i0 + ii) * lda + (pc + p)];
                for (; ii < MR; ++ii) dst[ii] = 0.0L;
            }
        }
    }
}

void egemmtr_pack_B(const TR *restrict B, ptrdiff_t ldb,
                    ptrdiff_t pc, ptrdiff_t jc, ptrdiff_t pb, ptrdiff_t jb,
                    char tb, TR *restrict Bp)
{
    const ptrdiff_t npanel = (jb + NR - 1) / NR;
    for (ptrdiff_t q = 0; q < npanel; ++q) {
        const ptrdiff_t j0 = jc + q * NR;
        const ptrdiff_t cols = (q == npanel - 1) ? (jb - q * NR) : NR;
        TR *panel = &Bp[(size_t)q * pb * NR];
        if (tb == 'N') {
            for (ptrdiff_t p = 0; p < pb; ++p) {
                TR *dst = &panel[(size_t)p * NR];
                ptrdiff_t jj;
                for (jj = 0; jj < cols; ++jj)
                    dst[jj] = B[(size_t)(j0 + jj) * ldb + (pc + p)];
                for (; jj < NR; ++jj) dst[jj] = 0.0L;
            }
        } else {
            for (ptrdiff_t p = 0; p < pb; ++p) {
                const TR *src = &B[(size_t)(pc + p) * ldb + j0];
                TR *dst = &panel[(size_t)p * NR];
                ptrdiff_t jj;
                for (jj = 0; jj < cols; ++jj) dst[jj] = src[jj];
                for (; jj < NR; ++jj) dst[jj] = 0.0L;
            }
        }
    }
}

/* ─── MR×NR kernels ────────────────────────────────────────────── */

static inline void kernel_2x2(ptrdiff_t pb, TR alpha,
                              const TR *restrict Ap,
                              const TR *restrict Bp,
                              TR *restrict C, ptrdiff_t ldc)
{
    TR c00 = 0.0L, c01 = 0.0L, c10 = 0.0L, c11 = 0.0L;
    for (ptrdiff_t p = 0; p < pb; ++p) {
        const TR a0 = Ap[(size_t)p * MR + 0];
        const TR a1 = Ap[(size_t)p * MR + 1];
        const TR b0 = Bp[(size_t)p * NR + 0];
        const TR b1 = Bp[(size_t)p * NR + 1];
        c00 += a0 * b0;
        c10 += a1 * b0;
        c01 += a0 * b1;
        c11 += a1 * b1;
    }
    C[0]       += alpha * c00;
    C[1]       += alpha * c10;
    C[ldc]     += alpha * c01;
    C[ldc + 1] += alpha * c11;
}

static void kernel_edge(ptrdiff_t mr, ptrdiff_t nr, ptrdiff_t pb, TR alpha,
                        const TR *restrict Ap,
                        const TR *restrict Bp,
                        TR *restrict C, ptrdiff_t ldc)
{
    for (ptrdiff_t jj = 0; jj < nr; ++jj) {
        TR *cj = &C[(size_t)jj * ldc];
        for (ptrdiff_t ii = 0; ii < mr; ++ii) {
            TR s = 0.0L;
            for (ptrdiff_t p = 0; p < pb; ++p)
                s += Ap[(size_t)p * MR + ii] * Bp[(size_t)p * NR + jj];
            cj[ii] += alpha * s;
        }
    }
}

void egemmtr_macro_kernel_rect(ptrdiff_t ib, ptrdiff_t jb, ptrdiff_t pb, TR alpha,
                               const TR *restrict Ap, const TR *restrict Bp,
                               TR *restrict C, ptrdiff_t ldc)
{
    const ptrdiff_t npA = (ib + MR - 1) / MR;
    const ptrdiff_t npB = (jb + NR - 1) / NR;
    for (ptrdiff_t q = 0; q < npB; ++q) {
        const ptrdiff_t jj0  = q * NR;
        const ptrdiff_t nr_q = (q == npB - 1) ? (jb - jj0) : NR;
        const TR *Bpanel = &Bp[(size_t)q * pb * NR];
        for (ptrdiff_t r = 0; r < npA; ++r) {
            const ptrdiff_t ii0  = r * MR;
            const ptrdiff_t mr_r = (r == npA - 1) ? (ib - ii0) : MR;
            const TR *Apanel = &Ap[(size_t)r * pb * MR];
            TR *Ctile = &C[(size_t)jj0 * ldc + ii0];
            if (mr_r == MR && nr_q == NR)
                kernel_2x2(pb, alpha, Apanel, Bpanel, Ctile, ldc);
            else
                kernel_edge(mr_r, nr_q, pb, alpha, Apanel, Bpanel, Ctile, ldc);
        }
    }
}

void egemmtr_macro_kernel_tri(ptrdiff_t ib, ptrdiff_t jb, ptrdiff_t pb, TR alpha,
                              const TR *restrict Ap, const TR *restrict Bp,
                              TR *restrict C, ptrdiff_t ldc,
                              ptrdiff_t row_base, ptrdiff_t col_base, char UPLO)
{
    const ptrdiff_t npA = (ib + MR - 1) / MR;
    const ptrdiff_t npB = (jb + NR - 1) / NR;
    for (ptrdiff_t q = 0; q < npB; ++q) {
        const ptrdiff_t jj0  = q * NR;
        const ptrdiff_t nr_q = (q == npB - 1) ? (jb - jj0) : NR;
        const TR *Bpanel = &Bp[(size_t)q * pb * NR];
        const ptrdiff_t j_g0 = col_base + jj0;
        const ptrdiff_t j_g1 = j_g0 + nr_q - 1;
        for (ptrdiff_t r = 0; r < npA; ++r) {
            const ptrdiff_t ii0  = r * MR;
            const ptrdiff_t mr_r = (r == npA - 1) ? (ib - ii0) : MR;
            const TR *Apanel = &Ap[(size_t)r * pb * MR];
            const ptrdiff_t i_g0 = row_base + ii0;
            const ptrdiff_t i_g1 = i_g0 + mr_r - 1;
            TR *Ctile = &C[(size_t)jj0 * ldc + ii0];

            ptrdiff_t all_in, all_out;
            if (UPLO == 'L') {
                all_in  = (i_g0 >= j_g1);
                all_out = (i_g1 <  j_g0);
            } else {
                all_in  = (i_g1 <= j_g0);
                all_out = (i_g0 >  j_g1);
            }
            if (all_out) continue;

            if (all_in) {
                if (mr_r == MR && nr_q == NR)
                    kernel_2x2(pb, alpha, Apanel, Bpanel, Ctile, ldc);
                else
                    kernel_edge(mr_r, nr_q, pb, alpha, Apanel, Bpanel, Ctile, ldc);
            } else {
                for (ptrdiff_t jj = 0; jj < nr_q; ++jj) {
                    const ptrdiff_t j_g = col_base + jj0 + jj;
                    TR *cj = &Ctile[(size_t)jj * ldc];
                    for (ptrdiff_t ii = 0; ii < mr_r; ++ii) {
                        const ptrdiff_t i_g = row_base + ii0 + ii;
                        const ptrdiff_t keep = (UPLO == 'L') ? (i_g >= j_g) : (i_g <= j_g);
                        if (!keep) continue;
                        TR s = 0.0L;
                        for (ptrdiff_t p = 0; p < pb; ++p)
                            s += Apanel[(size_t)p * MR + ii] *
                                 Bpanel[(size_t)p * NR + jj];
                        cj[ii] += alpha * s;
                    }
                }
            }
        }
    }
}

void egemmtr_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t n, char UPLO,
                        TR beta, TR *c, ptrdiff_t ldc)
{
    const TR zero = 0.0L;
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        const ptrdiff_t is = (UPLO == 'L') ? j : 0;
        const ptrdiff_t ie = (UPLO == 'L') ? n : j + 1;
        TR *cj = &C_(0, j);
        if (beta == zero) for (ptrdiff_t i = is; i < ie; ++i) cj[i]  = zero;
        else              for (ptrdiff_t i = is; i < ie; ++i) cj[i] *= beta;
    }
}

void egemmtr_scalar_fallback(ptrdiff_t n, ptrdiff_t k, char UPLO, char ta, char tb,
                             TR alpha,
                             const TR *a, ptrdiff_t lda,
                             const TR *b, ptrdiff_t ldb,
                             TR *c, ptrdiff_t ldc)
{
    const TR zero = 0.0L;
    for (ptrdiff_t j = 0; j < n; ++j) {
        const ptrdiff_t is = (UPLO == 'L') ? j : 0;
        const ptrdiff_t ie = (UPLO == 'L') ? n : j + 1;
        TR *cj = &C_(0, j);
        for (ptrdiff_t i = is; i < ie; ++i) {
            TR s = zero;
            if (ta == 'N') {
                if (tb == 'N')
                    for (ptrdiff_t l = 0; l < k; ++l)
                        s += a[(size_t)l * lda + i] * b[(size_t)j * ldb + l];
                else
                    for (ptrdiff_t l = 0; l < k; ++l)
                        s += a[(size_t)l * lda + i] * b[(size_t)l * ldb + j];
            } else {
                if (tb == 'N')
                    for (ptrdiff_t l = 0; l < k; ++l)
                        s += a[(size_t)i * lda + l] * b[(size_t)j * ldb + l];
                else
                    for (ptrdiff_t l = 0; l < k; ++l)
                        s += a[(size_t)i * lda + l] * b[(size_t)l * ldb + j];
            }
            cj[i] += alpha * s;
        }
    }
}

/* ─── Serial entry ─────────────────────────────────────────────── */

void egemmtr_serial(char uplo, char transa, char transb,
                    ptrdiff_t n, ptrdiff_t k,
                    const TR *alpha_,
                    const TR *restrict a, ptrdiff_t lda,
                    const TR *restrict b, ptrdiff_t ldb,
                    const TR *beta_,
                    TR *restrict c, ptrdiff_t ldc)
{
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO = blas_up(uplo);
    const char ta = egemmtr_trans_code(&transa);
    const char tb = egemmtr_trans_code(&transb);

    if (n <= 0) return;
    const TR zero = 0.0L, one = 1.0L;

    if (alpha == zero || k == 0) {
        if (beta == one) return;
        egemmtr_beta_scale(0, n, n, UPLO, beta, c, ldc);
        return;
    }

    if (beta != one)
        egemmtr_beta_scale(0, n, n, UPLO, beta, c, ldc);

    ptrdiff_t MC, KC, NC;
    egemmtr_block_sizes(&MC, &KC, &NC);
    if (NC > n) NC = n;
    if (NC < NR) NC = NR;

    const ptrdiff_t sa_rows = egemmtr_round_up(MC, MR);
    const ptrdiff_t sb_cols = egemmtr_round_up(NC, NR);
    const size_t ap_bytes = (size_t)sa_rows * KC * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * sb_cols * sizeof(TR);

    TR *Bp = (TR *)aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    TR *Ap = Bp ? (TR *)aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63) : NULL;
    if (!Bp || !Ap) {
        free(Ap); free(Bp);
        egemmtr_scalar_fallback(n, k, UPLO, ta, tb, alpha, a, lda, b, ldb, c, ldc);
        return;
    }

    for (ptrdiff_t jc = 0; jc < n; jc += NC) {
        const ptrdiff_t jb = imin(NC, n - jc);
        for (ptrdiff_t pc = 0; pc < k; pc += KC) {
            const ptrdiff_t pb = imin(KC, k - pc);
            egemmtr_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
            for (ptrdiff_t ic = 0; ic < n; ic += MC) {
                const ptrdiff_t ib = imin(MC, n - ic);

                ptrdiff_t tile_class;
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
                                             &C_(ic, jc), ldc, ic, jc, UPLO);
                else
                    egemmtr_macro_kernel_rect(ib, jb, pb, alpha, Ap, Bp,
                                              &C_(ic, jc), ldc);
            }
        }
    }

    free(Ap);
    free(Bp);
}

#undef C_
