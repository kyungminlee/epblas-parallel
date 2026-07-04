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
#include "../common/blas_math.h"
#include <stdlib.h>
#include <ctype.h>
#include <stdbool.h>
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

/* ─── Unpacked tiny-N TN fast path ─────────────────────────────────
 *
 * For op(A)=A^T, op(B)=B (the TN orientation) BOTH operands are stride-1 in the
 * contraction index l: column i of A is &a[i*lda] and column j of B is &b[j*ldb],
 * each contiguous in l. So at tiny N the GotoBLAS pack of A and B buys nothing —
 * the whole problem is L2-resident and the packed kernel just pays the copy. At
 * N=64 (where egemmtr computes only the triangle, ~half the flops, so the pack is
 * ~2x the relative overhead) that copy is the entire ~5% gap to OpenBLAS's native
 * A^T·B kernel.
 *
 * This reads A/B in place (no Ap/Bp) through a 2x2 register tile — same 4
 * independent accumulator chains as the packed kernel_2x2, so it keeps the ILP
 * that made egemm route TN to blocking (a single-accumulator dot is fadd-latency
 * bound and LOSES). Summation order is l-ascending, identical to the packed path
 * and the netlib oracle → BIT-IDENTICAL output. Triangle is honoured per 2x2 tile
 * (all_in → kernel; crossing/edge → masked scalar). Gated to the non-threaded
 * tiny-N TN case only; the blocked path still owns every other shape (and the
 * omp4 N=64 cell, which already wins). */
static inline void egemmtr_tn_kernel_2x2(ptrdiff_t k, TR alpha,
        const TR *restrict ai0, const TR *restrict ai1,
        const TR *restrict bj0, const TR *restrict bj1,
        TR *restrict C, ptrdiff_t ldc)
{
    TR c00 = 0.0L, c10 = 0.0L, c01 = 0.0L, c11 = 0.0L;
    for (ptrdiff_t l = 0; l < k; ++l) {
        const TR a0 = ai0[l], a1 = ai1[l], b0 = bj0[l], b1 = bj1[l];
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

void egemmtr_unpacked_tn(char UPLO, ptrdiff_t n, ptrdiff_t k, TR alpha,
                         const TR *restrict a, ptrdiff_t lda,
                         const TR *restrict b, ptrdiff_t ldb,
                         TR *restrict c, ptrdiff_t ldc)
{
    const bool upper = (UPLO != 'L');
    for (ptrdiff_t j0 = 0; j0 < n; j0 += NR) {
        const ptrdiff_t njr = blas_imin(NR, n - j0);
        /* Only rows that can intersect the triangle for these columns. */
        const ptrdiff_t i_lo = upper ? 0  : j0;
        const ptrdiff_t i_hi = upper ? (j0 + njr) : n;
        for (ptrdiff_t i0 = i_lo; i0 < i_hi; i0 += MR) {
            const ptrdiff_t mir = blas_imin(MR, n - i0);

            bool all_in, all_out;
            if (upper) {                       /* keep i <= j */
                all_in  = (i0 + mir - 1) <= j0;
                all_out = (i0 > j0 + njr - 1);
            } else {                           /* keep i >= j */
                all_in  = (i0 >= j0 + njr - 1);
                all_out = (i0 + mir - 1) < j0;
            }
            if (all_out) continue;

            TR *Ctile = &c[(size_t)j0 * ldc + i0];
            if (all_in && mir == MR && njr == NR) {
                egemmtr_tn_kernel_2x2(k, alpha,
                    &a[(size_t)i0 * lda], &a[(size_t)(i0 + 1) * lda],
                    &b[(size_t)j0 * ldb], &b[(size_t)(j0 + 1) * ldb],
                    Ctile, ldc);
            } else {
                for (ptrdiff_t jj = 0; jj < njr; ++jj) {
                    const ptrdiff_t j = j0 + jj;
                    const TR *restrict bj = &b[(size_t)j * ldb];
                    for (ptrdiff_t ii = 0; ii < mir; ++ii) {
                        const ptrdiff_t i = i0 + ii;
                        const bool keep = upper ? (i <= j) : (i >= j);
                        if (!keep) continue;
                        const TR *restrict ai = &a[(size_t)i * lda];
                        TR s = 0.0L;
                        for (ptrdiff_t l = 0; l < k; ++l) s += ai[l] * bj[l];
                        Ctile[(size_t)jj * ldc + ii] += alpha * s;
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

/* Buffer-free triangular GEMM-update over the column band [j_lo, j_hi). Used as
 * the allocation-failure fallback; the full-range egemmtr_scalar_fallback is the
 * [0, n) case. The threaded entry calls the banded form per OOM thread so each
 * touches only its disjoint columns (no double-compute). */
void egemmtr_scalar_fallback_cols(ptrdiff_t j_lo, ptrdiff_t j_hi,
                                  ptrdiff_t n, ptrdiff_t k, char UPLO, char ta, char tb,
                                  TR alpha,
                                  const TR *a, ptrdiff_t lda,
                                  const TR *b, ptrdiff_t ldb,
                                  TR *c, ptrdiff_t ldc)
{
    const TR zero = 0.0L;
    for (ptrdiff_t j = j_lo; j < j_hi; ++j) {
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
    const char ta = blas_trans_real(transa);
    const char tb = blas_trans_real(transb);

    if (n <= 0) return;
    const TR zero = 0.0L, one = 1.0L;

    if (alpha == zero || k == 0) {
        if (beta == one) return;
        egemmtr_beta_scale(0, n, n, UPLO, beta, c, ldc);
        return;
    }

    if (beta != one)
        egemmtr_beta_scale(0, n, n, UPLO, beta, c, ldc);

    /* Tiny-N TN: skip the GotoBLAS pack (pure overhead when L2-resident) and run
     * the unpacked stride-1 2x2 kernel. Bit-identical; blocked path owns all else. */
    if (ta == 'T' && tb == 'N' && n <= EGEMMTR_UNPACKED_TN_MAX) {
        egemmtr_unpacked_tn(UPLO, n, k, alpha, a, lda, b, ldb, c, ldc);
        return;
    }

    ptrdiff_t MC, KC, NC;
    egemmtr_block_sizes(&MC, &KC, &NC);
    if (NC > n) NC = n;
    if (NC < NR) NC = NR;

    const ptrdiff_t sa_rows = blas_round_up(MC, MR);
    const ptrdiff_t sb_cols = blas_round_up(NC, NR);
    const size_t ap_bytes = (size_t)sa_rows * KC * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * sb_cols * sizeof(TR);

    /* Persistent grow-only thread-local pack arena (Bp|Ap in one block): a
     * per-call aligned_alloc+free of these mmap-threshold-sized buffers trips
     * glibc's trim heuristic and re-faults every touched page each call — a
     * pure page-fault tax at small N (see etrsm_serial.c). */
    static __thread TR *g_pack = NULL;
    static __thread size_t g_pack_cap = 0;
    const size_t bp_al = (bp_bytes + 63) & ~(size_t)63;
    const size_t need  = bp_al + ((ap_bytes + 63) & ~(size_t)63);
    if (need > g_pack_cap) {
        free(g_pack);
        size_t cap = need + (need >> 1);            /* 1.5× headroom to amortize regrow */
        cap = (cap + 63) & ~(size_t)63;
        g_pack = aligned_alloc(64, cap);
        g_pack_cap = g_pack ? cap : 0;
    }
    if (!g_pack) {
        egemmtr_scalar_fallback(n, k, UPLO, ta, tb, alpha, a, lda, b, ldb, c, ldc);
        return;
    }
    TR *Bp = g_pack;
    TR *Ap = (TR *)(void *)((char *)g_pack + bp_al);

    for (ptrdiff_t jc = 0; jc < n; jc += NC) {
        const ptrdiff_t jb = blas_imin(NC, n - jc);
        for (ptrdiff_t pc = 0; pc < k; pc += KC) {
            const ptrdiff_t pb = blas_imin(KC, k - pc);
            egemmtr_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
            for (ptrdiff_t ic = 0; ic < n; ic += MC) {
                const ptrdiff_t ib = blas_imin(MC, n - ic);

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
}

#undef C_
