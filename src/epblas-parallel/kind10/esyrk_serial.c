/*
 * esyrk_serial.c — the pure single-thread half of the kind10 (REAL(KIND=10) /
 * `long double`) symmetric rank-k update overlay. No `#pragma omp`.
 *
 * Owns ALL the leaf math shared with the cooperative parallel driver
 * (esyrk_parallel.c) through esyrk_kernel.h: the block-size policy, the
 * GotoBLAS packers, the MR×NR micro-kernel, the rectangular and
 * triangle-aware macro-kernels, the inline single-thread driver
 * (esyrk_serial_inline), and the public Fortran-ABI serial entry
 * `esyrk_serial`.
 *
 * esyrk_serial_inline is reused by esyrk_parallel.c as its OOM fallback;
 * esyrk_serial is the path the public `esyrk_` driver delegates to when
 * called from inside another routine's parallel region (the libgomp barrier
 * wedge guard, project-etrsm-omp4-wedge).
 */

#include "esyrk_kernel.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>

typedef esyrk_T T;

#define MR ESYRK_MR
#define NR ESYRK_NR

#define ESYRK_MC_DEFAULT  64
#define ESYRK_KC_DEFAULT 256
#define ESYRK_NC_DEFAULT 512
#define ESYRK_SWITCH_RATIO_DEFAULT 16

/* ─── Block-size / switch-ratio policy (lazy, env-tunable) ─────── */

static int g_mc = 0, g_kc = 0, g_nc = 0;
static int g_switch_ratio = 0;
static int g_config_done = 0;

static int env_int(const char *name, int dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    int v = atoi(s);
    return v > 0 ? v : dflt;
}

/* Lazy replacement for the old __attribute__((constructor)). All writes are
 * idempotent (identical values), so the benign first-call race is harmless;
 * the public entries always resolve config serially before opening a team. */
static void esyrk_config_init(void) {
    if (g_config_done) return;
    g_mc = env_int("ESYRK_MC", ESYRK_MC_DEFAULT);
    g_kc = env_int("ESYRK_KC", ESYRK_KC_DEFAULT);
    g_nc = env_int("ESYRK_NC", ESYRK_NC_DEFAULT);
    g_switch_ratio = env_int("ESYRK_SWITCH_RATIO", ESYRK_SWITCH_RATIO_DEFAULT);
    g_config_done = 1;
}

void esyrk_block_sizes(int *MC, int *KC, int *NC) {
    esyrk_config_init();
    if (MC) *MC = g_mc;
    if (KC) *KC = g_kc;
    if (NC) *NC = g_nc;
}

int esyrk_switch_ratio(void) {
    esyrk_config_init();
    return g_switch_ratio;
}

/* ─── Small helpers ─────────────────────────────────────────────── */

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

int esyrk_round_up(int v, int m) { return ((v + m - 1) / m) * m; }
static inline int imin(int a, int b) { return a < b ? a : b; }

/* ─── GotoBLAS packers + MR×NR micro-kernel (mirrors egemm) ───── */

/* Pack op(A) rows [i0, i0+min_i) × cols [pc, pc+min_l) into MR-row panels.
 *   layout: panel q at offset (q * min_l * MR);
 *           entry [(p * MR) + ii] = op(A)[i0 + q*MR + ii, pc + p].
 *   - TR='N': op(A) = A, so row major access at A[col=pc+p, row=i0+q*MR+ii].
 *   - TR='T': op(A) = Aᵀ, so A[row=pc+p, col=i0+q*MR+ii].
 */
void esyrk_pack_A_panel(const T *restrict A, int lda,
                        int i0, int pc, int min_i, int min_l,
                        int TR, T *restrict Apack)
{
    const int npanel = (min_i + MR - 1) / MR;
    for (int q = 0; q < npanel; ++q) {
        const int row0 = i0 + q * MR;
        const int rows = (q == npanel - 1) ? (min_i - q * MR) : MR;
        T *panel = &Apack[(size_t)q * min_l * MR];
        if (TR == 'N') {
            for (int p = 0; p < min_l; ++p) {
                const T *src = &A[(size_t)(pc + p) * lda + row0];
                T *dst = &panel[(size_t)p * MR];
                int ii;
                for (ii = 0; ii < rows; ++ii) dst[ii] = src[ii];
                for (; ii < MR; ++ii) dst[ii] = 0.0L;
            }
        } else {
            for (int p = 0; p < min_l; ++p) {
                T *dst = &panel[(size_t)p * MR];
                int ii;
                for (ii = 0; ii < rows; ++ii)
                    dst[ii] = A[(size_t)(row0 + ii) * lda + (pc + p)];
                for (; ii < MR; ++ii) dst[ii] = 0.0L;
            }
        }
    }
}

/* Pack Aᵀ (or A) cols [j0, j0+min_j) × depth [pc, pc+min_l) into NR-col panels.
 *
 * For SYRK with C = α·A·Aᵀ (TR='N'), op(B) in the underlying GEMM is Aᵀ:
 *   B[p, jj] = Aᵀ[pc+p, j0+jj] = A[j0+jj, pc+p].
 * For TR='T' (C = α·Aᵀ·A), op(B) is A:
 *   B[p, jj] = A[pc+p, j0+jj].
 */
void esyrk_pack_B_panel(const T *restrict A, int lda,
                        int j0, int pc, int min_j, int min_l,
                        int TR, T *restrict Bpack)
{
    const int npanel = (min_j + NR - 1) / NR;
    for (int q = 0; q < npanel; ++q) {
        const int col0 = j0 + q * NR;
        const int cols = (q == npanel - 1) ? (min_j - q * NR) : NR;
        T *panel = &Bpack[(size_t)q * min_l * NR];
        if (TR == 'N') {
            /* B[p, jj] = A[col0+jj, pc+p] → column (pc+p), row (col0+jj) of A */
            for (int p = 0; p < min_l; ++p) {
                const T *src = &A[(size_t)(pc + p) * lda + col0];
                T *dst = &panel[(size_t)p * NR];
                int jj;
                for (jj = 0; jj < cols; ++jj) dst[jj] = src[jj];
                for (; jj < NR; ++jj) dst[jj] = 0.0L;
            }
        } else {
            /* B[p, jj] = A[pc+p, col0+jj] → column (col0+jj), row (pc+p) of A */
            for (int p = 0; p < min_l; ++p) {
                T *dst = &panel[(size_t)p * NR];
                int jj;
                for (jj = 0; jj < cols; ++jj)
                    dst[jj] = A[(size_t)(col0 + jj) * lda + (pc + p)];
                for (; jj < NR; ++jj) dst[jj] = 0.0L;
            }
        }
    }
}

static inline void kernel_2x2(int pb, T alpha,
                              const T *restrict Ap,
                              const T *restrict Bp,
                              T *restrict C, int ldc)
{
    T c00 = 0.0L, c01 = 0.0L, c10 = 0.0L, c11 = 0.0L;
    /* K-loop unrolled by 4 — mirrors the openblas eblas_egemm_kernel dense
     * microkernel. Cuts x87 loop overhead on the fp80 inner loop (the ob1
     * serial edge over a plain rolled loop). MR == NR == 2, so the packed
     * panels are contiguous in stride-2 per p; walk Ap/Bp by 8 per step.
     * Each C accumulator stays in strict p-order → bit-identical result. */
    const T *pa = Ap, *pbb = Bp;
    int p = 0;
    for (; p + 4 <= pb; p += 4) {
        T a0 = pa[0], a1 = pa[1], b0 = pbb[0], b1 = pbb[1];
        c00 += a0 * b0; c10 += a1 * b0; c01 += a0 * b1; c11 += a1 * b1;
        a0 = pa[2]; a1 = pa[3]; b0 = pbb[2]; b1 = pbb[3];
        c00 += a0 * b0; c10 += a1 * b0; c01 += a0 * b1; c11 += a1 * b1;
        a0 = pa[4]; a1 = pa[5]; b0 = pbb[4]; b1 = pbb[5];
        c00 += a0 * b0; c10 += a1 * b0; c01 += a0 * b1; c11 += a1 * b1;
        a0 = pa[6]; a1 = pa[7]; b0 = pbb[6]; b1 = pbb[7];
        c00 += a0 * b0; c10 += a1 * b0; c01 += a0 * b1; c11 += a1 * b1;
        pa += 8; pbb += 8;
    }
    for (; p < pb; ++p) {
        const T a0 = pa[0], a1 = pa[1], b0 = pbb[0], b1 = pbb[1];
        c00 += a0 * b0; c10 += a1 * b0; c01 += a0 * b1; c11 += a1 * b1;
        pa += 2; pbb += 2;
    }
    C[0]       += alpha * c00;
    C[1]       += alpha * c10;
    C[ldc]     += alpha * c01;
    C[ldc + 1] += alpha * c11;
}

static void kernel_edge(int mr, int nr, int pb, T alpha,
                        const T *restrict Ap,
                        const T *restrict Bp,
                        T *restrict C, int ldc)
{
    for (int jj = 0; jj < nr; ++jj) {
        T *cj = &C[(size_t)jj * ldc];
        for (int ii = 0; ii < mr; ++ii) {
            T s = 0.0L;
            for (int p = 0; p < pb; ++p)
                s += Ap[(size_t)p * MR + ii] * Bp[(size_t)p * NR + jj];
            cj[ii] += alpha * s;
        }
    }
}

/* Rectangular macro-kernel: ib × jb tile, no triangle constraint. */
void esyrk_macro_kernel_rect(int ib, int jb, int pb, T alpha,
                             const T *restrict Ap, const T *restrict Bp,
                             T *restrict C, int ldc)
{
    const int npA = (ib + MR - 1) / MR;
    const int npB = (jb + NR - 1) / NR;
    for (int q = 0; q < npB; ++q) {
        const int jj0  = q * NR;
        const int nr_q = (q == npB - 1) ? (jb - jj0) : NR;
        const T *Bpanel = &Bp[(size_t)q * pb * NR];
        for (int r = 0; r < npA; ++r) {
            const int ii0  = r * MR;
            const int mr_r = (r == npA - 1) ? (ib - ii0) : MR;
            const T *Apanel = &Ap[(size_t)r * pb * MR];
            T *Ctile = &C[(size_t)jj0 * ldc + ii0];
            if (mr_r == MR && nr_q == NR) {
                kernel_2x2(pb, alpha, Apanel, Bpanel, Ctile, ldc);
            } else {
                kernel_edge(mr_r, nr_q, pb, alpha, Apanel, Bpanel, Ctile, ldc);
            }
        }
    }
}

/* Triangle-aware macro-kernel: skips sub-tiles fully outside the UPLO
 * triangle (rooted at global (row_base, col_base)) and, for sub-tiles
 * that cross the diagonal, falls back to entry-by-entry check. */
void esyrk_macro_kernel_tri(int ib, int jb, int pb, T alpha,
                            const T *restrict Ap, const T *restrict Bp,
                            T *restrict C, int ldc,
                            int row_base, int col_base, char UPLO)
{
    const int npA = (ib + MR - 1) / MR;
    const int npB = (jb + NR - 1) / NR;
    for (int q = 0; q < npB; ++q) {
        const int jj0  = q * NR;
        const int nr_q = (q == npB - 1) ? (jb - jj0) : NR;
        const T *Bpanel = &Bp[(size_t)q * pb * NR];
        const int j_g0 = col_base + jj0;
        const int j_g1 = j_g0 + nr_q - 1;
        for (int r = 0; r < npA; ++r) {
            const int ii0  = r * MR;
            const int mr_r = (r == npA - 1) ? (ib - ii0) : MR;
            const T *Apanel = &Ap[(size_t)r * pb * MR];
            const int i_g0 = row_base + ii0;
            const int i_g1 = i_g0 + mr_r - 1;
            T *Ctile = &C[(size_t)jj0 * ldc + ii0];

            int all_in, all_out;
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
                for (int jj = 0; jj < nr_q; ++jj) {
                    const int j_g = col_base + jj0 + jj;
                    T *cj = &Ctile[(size_t)jj * ldc];
                    for (int ii = 0; ii < mr_r; ++ii) {
                        const int i_g = row_base + ii0 + ii;
                        const int keep = (UPLO == 'L') ? (i_g >= j_g) : (i_g <= j_g);
                        if (!keep) continue;
                        T s = 0.0L;
                        for (int p = 0; p < pb; ++p)
                            s += Apanel[(size_t)p * MR + ii] *
                                 Bpanel[(size_t)p * NR + jj];
                        cj[ii] += alpha * s;
                    }
                }
            }
        }
    }
}

/* ─── Inline single-thread GotoBLAS path (OMP=1 / N below cooperative
 *      threshold, and the parallel driver's OOM fallback). Same MR×NR
 *      kernel as the cooperative path, but with no flag plumbing: one
 *      thread walks the (jc, pc, ic) nest and classifies each (ic, jc)
 *      tile against the UPLO triangle.
 *
 *      Three classes:
 *        skip   — tile entirely outside the stored triangle
 *        rect   — tile entirely inside (off-diagonal); rectangular kernel
 *        tri    — tile crosses the diagonal; triangle-aware kernel
 *      Tiles in 'rect' use the dense 2×2 outer-product kernel; 'tri'
 *      falls back to per-entry UPLO checks for the sub-tiles that
 *      actually straddle the diagonal.
 *
 *      Buffers (Ap, Bp) are allocated once at function entry. The old
 *      per-jc-block egemm_ call mmap'd and freed ~2 MB Bp + ~256 KB Ap
 *      on every block; inlining absorbs that. */
void esyrk_serial_inline(char UPLO, char TR, int N, int K,
                         T alpha, const T *restrict a, int lda,
                         T beta, T *restrict c, int ldc)
{
    const T zero = 0.0L, one = 1.0L;

    /* Beta-scale the UPLO triangle of C first. */
    if (beta != one) {
        for (int j = 0; j < N; ++j) {
            const int i_lo = (UPLO == 'L') ? j : 0;
            const int i_hi = (UPLO == 'L') ? N : j + 1;
            T *cj = c + (size_t)j * ldc;
            if (beta == zero) for (int i = i_lo; i < i_hi; ++i) cj[i] = zero;
            else              for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;
        }
    }

    if (alpha == zero || K == 0) return;

    int MC, KC, NC;
    esyrk_block_sizes(&MC, &KC, &NC);
    if (NC > N) NC = N;
    if (NC < NR) NC = NR;

    const int sa_rows = esyrk_round_up(MC, MR);
    const int sb_cols = esyrk_round_up(NC, NR);
    const size_t ap_bytes = (size_t)sa_rows * KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * sb_cols * sizeof(T);

    T *Ap = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
    T *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (!Ap || !Bp) {
        /* OOM — last-ditch O(N²·K) scalar fallback. */
        free(Ap); free(Bp);
        for (int j = 0; j < N; ++j) {
            const int i_lo = (UPLO == 'L') ? j : 0;
            const int i_hi = (UPLO == 'L') ? N : j + 1;
            T *cj = c + (size_t)j * ldc;
            for (int i = i_lo; i < i_hi; ++i) {
                T s = zero;
                if (TR == 'N') {
                    for (int l = 0; l < K; ++l)
                        s += a[(size_t)l * lda + i] * a[(size_t)l * lda + j];
                } else {
                    for (int l = 0; l < K; ++l)
                        s += a[(size_t)i * lda + l] * a[(size_t)j * lda + l];
                }
                cj[i] += alpha * s;
            }
        }
        return;
    }

    /* Standard GotoBLAS loop nest: jc (output cols) → pc (depth) → ic
     * (output rows). Bp packed once per (jc, pc); Ap repacked per
     * (ic, pc). */
    for (int jc = 0; jc < N; jc += NC) {
        const int jb = imin(NC, N - jc);
        for (int pc = 0; pc < K; pc += KC) {
            const int pb = imin(KC, K - pc);

            esyrk_pack_B_panel(a, lda, jc, pc, jb, pb, TR, Bp);

            for (int ic = 0; ic < N; ic += MC) {
                const int ib = imin(MC, N - ic);

                /* Tile classification against UPLO triangle. */
                int tile_class;
                if (UPLO == 'L') {
                    if (ic + ib <= jc)        tile_class = 0;  /* all i < j: skip */
                    else if (ic >= jc + jb)   tile_class = 2;  /* all i > j: rect */
                    else                      tile_class = 1;  /* crosses diag */
                } else {
                    if (ic >= jc + jb)        tile_class = 0;  /* all i > j: skip */
                    else if (ic + ib <= jc)   tile_class = 2;  /* all i < j: rect */
                    else                      tile_class = 1;
                }
                if (tile_class == 0) continue;

                esyrk_pack_A_panel(a, lda, ic, pc, ib, pb, TR, Ap);

                if (tile_class == 1) {
                    esyrk_macro_kernel_tri(ib, jb, pb, alpha, Ap, Bp,
                                           &c[(size_t)jc * ldc + ic], ldc,
                                           ic, jc, UPLO);
                } else {
                    esyrk_macro_kernel_rect(ib, jb, pb, alpha, Ap, Bp,
                                            &c[(size_t)jc * ldc + ic], ldc);
                }
            }
        }
    }

    free(Ap);
    free(Bp);
}

/* ─── Pure-serial Fortran-ABI entry (no OpenMP) ─────────────────── */

void esyrk_serial(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';

    if (N == 0) return;

    const T zero = 0.0L, one = 1.0L;

    /* α==0 or K==0: only beta-scale the UPLO triangle. */
    if (alpha == zero || K == 0) {
        if (beta == one) return;
        for (int j = 0; j < N; ++j) {
            const int i_lo = (UPLO == 'L') ? j : 0;
            const int i_hi = (UPLO == 'L') ? N : j + 1;
            T *cj = c + (size_t)j * ldc;
            if (beta == zero) for (int i = i_lo; i < i_hi; ++i) cj[i] = zero;
            else              for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;
        }
        return;
    }

    esyrk_serial_inline(UPLO, TR, N, K, alpha, a, lda, beta, c, ldc);
}
