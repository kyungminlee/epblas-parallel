/*
 * egemmtr — kind10 real (long double) triangular GEMM update.
 *   C := alpha · op(A) · op(B) + beta · C   (only UPLO triangle of C)
 *
 * Inline GotoBLAS: walk the (jc, pc, ic) loop nest, classify each
 * (ic, jc) tile against the UPLO triangle, and dispatch one of:
 *   - skip if entirely outside the stored triangle
 *   - rectangular packed kernel if entirely inside
 *   - triangle-aware kernel if crossing the diagonal (per-entry UPLO
 *     check for sub-tiles that actually straddle)
 *
 * Threading: outer `omp parallel`, Bp shared and packed via
 * `omp single` once per (jc, pc), Ap private per thread, `omp for`
 * over the ic loop with `schedule(static, 1)` (interleaves ic chunks
 * so the triangular load — early-ic threads get fewer skipped tiles
 * for LOWER, later-ic threads for UPPER — balances). Same structure
 * as egemm's parallel pattern.
 *
 * Replaces the prior structure of (jc-parallel-for) over (beta-scale
 * + scalar diag_add + recursive `egemm_` call). The recursive call
 * mmap'd ~2 MB Bp + ~256 KB Ap per jc-block, and the scalar diag
 * kernel left the x87 register-tile perf on the floor for the
 * diagonal block — that combination was responsible for the worst
 * sub-parity cells at N=64 (0.60–0.73× vs migrated).
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef long double T;

#define MR 2
#define NR 2
#define EGEMMTR_MC_DEFAULT  64
#define EGEMMTR_KC_DEFAULT 256
#define EGEMMTR_NC_DEFAULT 512

#define EGEMMTR_OMP_MIN 32

static int g_mc = 0, g_kc = 0, g_nc = 0;

static int env_int(const char *name, int dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    int v = atoi(s);
    return v > 0 ? v : dflt;
}

__attribute__((constructor))
static void egemmtr_init(void) {
    g_mc = env_int("EGEMMTR_MC", EGEMMTR_MC_DEFAULT);
    g_kc = env_int("EGEMMTR_KC", EGEMMTR_KC_DEFAULT);
    g_nc = env_int("EGEMMTR_NC", EGEMMTR_NC_DEFAULT);
}

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}
static int trans_code(const char *p) {
    char c = up(p);
    return (c == 'C') ? 'T' : c;
}

static inline int round_up(int v, int m) { return ((v + m - 1) / m) * m; }
static inline int imin(int a, int b) { return a < b ? a : b; }

#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

/* ─── Packers (egemm-style, take separate A and B) ─────────────── */

static void pack_A(const T *restrict A, int lda,
                   int ic, int pc, int ib, int pb,
                   int ta, T *restrict Ap)
{
    const int npanel = (ib + MR - 1) / MR;
    for (int q = 0; q < npanel; ++q) {
        const int i0 = ic + q * MR;
        const int rows = (q == npanel - 1) ? (ib - q * MR) : MR;
        T *panel = &Ap[(size_t)q * pb * MR];
        if (ta == 'N') {
            for (int p = 0; p < pb; ++p) {
                const T *src = &A[(size_t)(pc + p) * lda + i0];
                T *dst = &panel[(size_t)p * MR];
                int ii;
                for (ii = 0; ii < rows; ++ii) dst[ii] = src[ii];
                for (; ii < MR; ++ii) dst[ii] = 0.0L;
            }
        } else {
            for (int p = 0; p < pb; ++p) {
                T *dst = &panel[(size_t)p * MR];
                int ii;
                for (ii = 0; ii < rows; ++ii)
                    dst[ii] = A[(size_t)(i0 + ii) * lda + (pc + p)];
                for (; ii < MR; ++ii) dst[ii] = 0.0L;
            }
        }
    }
}

static void pack_B(const T *restrict B, int ldb,
                   int pc, int jc, int pb, int jb,
                   int tb, T *restrict Bp)
{
    const int npanel = (jb + NR - 1) / NR;
    for (int q = 0; q < npanel; ++q) {
        const int j0 = jc + q * NR;
        const int cols = (q == npanel - 1) ? (jb - q * NR) : NR;
        T *panel = &Bp[(size_t)q * pb * NR];
        if (tb == 'N') {
            /* op(B) = B: B[p, jj] = B_global[pc+p, j0+jj]
             *           = b[(j0+jj)*ldb + (pc+p)]. */
            for (int p = 0; p < pb; ++p) {
                T *dst = &panel[(size_t)p * NR];
                int jj;
                for (jj = 0; jj < cols; ++jj)
                    dst[jj] = B[(size_t)(j0 + jj) * ldb + (pc + p)];
                for (; jj < NR; ++jj) dst[jj] = 0.0L;
            }
        } else {
            /* op(B) = Bᵀ: B[p, jj] = B_global[j0+jj, pc+p]
             *            = b[(pc+p)*ldb + (j0+jj)]. */
            for (int p = 0; p < pb; ++p) {
                const T *src = &B[(size_t)(pc + p) * ldb + j0];
                T *dst = &panel[(size_t)p * NR];
                int jj;
                for (jj = 0; jj < cols; ++jj) dst[jj] = src[jj];
                for (; jj < NR; ++jj) dst[jj] = 0.0L;
            }
        }
    }
}

/* ─── MR×NR kernels ────────────────────────────────────────────── */

static inline void kernel_2x2(int pb, T alpha,
                              const T *restrict Ap,
                              const T *restrict Bp,
                              T *restrict C, int ldc)
{
    T c00 = 0.0L, c01 = 0.0L, c10 = 0.0L, c11 = 0.0L;
    for (int p = 0; p < pb; ++p) {
        const T a0 = Ap[(size_t)p * MR + 0];
        const T a1 = Ap[(size_t)p * MR + 1];
        const T b0 = Bp[(size_t)p * NR + 0];
        const T b1 = Bp[(size_t)p * NR + 1];
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

static void macro_kernel_rect(int ib, int jb, int pb, T alpha,
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
            if (mr_r == MR && nr_q == NR)
                kernel_2x2(pb, alpha, Apanel, Bpanel, Ctile, ldc);
            else
                kernel_edge(mr_r, nr_q, pb, alpha, Apanel, Bpanel, Ctile, ldc);
        }
    }
}

static void macro_kernel_tri(int ib, int jb, int pb, T alpha,
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

/* ─── Entry point ──────────────────────────────────────────────── */

void egemmtr_(const char *uplo, const char *transa, const char *transb,
              const int *n_, const int *k_,
              const T *alpha_,
              const T *restrict a, const int *lda_,
              const T *restrict b, const int *ldb_,
              const T *beta_,
              T *restrict c, const int *ldc_,
              size_t uplo_len, size_t ta_len, size_t tb_len)
{
    (void)uplo_len; (void)ta_len; (void)tb_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);
    const int ta = trans_code(transa);
    const int tb = trans_code(transb);

    if (N <= 0) return;
    const T zero = 0.0L, one = 1.0L;

    /* alpha==0 or K==0: only beta-scale the UPLO triangle. */
    if (alpha == zero || K == 0) {
        if (beta == one) return;
#ifdef _OPENMP
        const int use_omp0 = (N >= EGEMMTR_OMP_MIN
                              && blas_omp_max_threads() > 1
                              && !omp_in_parallel());
        #pragma omp parallel for if(use_omp0) schedule(static, 1)
#endif
        for (int j = 0; j < N; ++j) {
            const int is = (UPLO == 'L') ? j : 0;
            const int ie = (UPLO == 'L') ? N : j + 1;
            T *cj = &C_(0, j);
            if (beta == zero) for (int i = is; i < ie; ++i) cj[i] = zero;
            else              for (int i = is; i < ie; ++i) cj[i] *= beta;
        }
        return;
    }

    /* Beta-scale UPLO triangle of C up front (separate phase so the
     * packed kernel below can always assume beta=1). */
    if (beta != one) {
#ifdef _OPENMP
        const int use_omp_beta = (N >= EGEMMTR_OMP_MIN
                                  && blas_omp_max_threads() > 1
                                  && !omp_in_parallel());
        #pragma omp parallel for if(use_omp_beta) schedule(static, 1)
#endif
        for (int j = 0; j < N; ++j) {
            const int is = (UPLO == 'L') ? j : 0;
            const int ie = (UPLO == 'L') ? N : j + 1;
            T *cj = &C_(0, j);
            if (beta == zero) for (int i = is; i < ie; ++i) cj[i] = zero;
            else              for (int i = is; i < ie; ++i) cj[i] *= beta;
        }
    }

    const int MC = g_mc, KC = g_kc;
    int NC = g_nc;
    if (NC > N) NC = N;
    if (NC < NR) NC = NR;

    const int sa_rows = round_up(MC, MR);
    const int sb_cols = round_up(NC, NR);
    const size_t ap_bytes = (size_t)sa_rows * KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * sb_cols * sizeof(T);

#ifdef _OPENMP
    const int use_omp = (N >= EGEMMTR_OMP_MIN
                         && blas_omp_max_threads() > 1
                         && !omp_in_parallel());
#else
    const int use_omp = 0;
#endif

    T *Bp = (T *)aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (!Bp) {
        /* Last-ditch O(N²·K) scalar fallback. */
        for (int j = 0; j < N; ++j) {
            const int is = (UPLO == 'L') ? j : 0;
            const int ie = (UPLO == 'L') ? N : j + 1;
            T *cj = &C_(0, j);
            for (int i = is; i < ie; ++i) {
                T s = zero;
                if (ta == 'N') {
                    if (tb == 'N')
                        for (int l = 0; l < K; ++l)
                            s += a[(size_t)l * lda + i] * b[(size_t)j * ldb + l];
                    else
                        for (int l = 0; l < K; ++l)
                            s += a[(size_t)l * lda + i] * b[(size_t)l * ldb + j];
                } else {
                    if (tb == 'N')
                        for (int l = 0; l < K; ++l)
                            s += a[(size_t)i * lda + l] * b[(size_t)j * ldb + l];
                    else
                        for (int l = 0; l < K; ++l)
                            s += a[(size_t)i * lda + l] * b[(size_t)l * ldb + j];
                }
                cj[i] += alpha * s;
            }
        }
        return;
    }

#ifdef _OPENMP
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
                    pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
                    /* implicit barrier at end of `single` makes Bp safe to
                     * read in the for below. */

#ifdef _OPENMP
                    #pragma omp for schedule(static, 1)
#endif
                    for (int ic = 0; ic < N; ic += MC) {
                        const int ib = imin(MC, N - ic);

                        /* Classify tile against UPLO triangle. */
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

                        pack_A(a, lda, ic, pc, ib, pb, ta, Ap);

                        if (tile_class == 1) {
                            macro_kernel_tri(ib, jb, pb, alpha, Ap, Bp,
                                             &C_(ic, jc), ldc,
                                             ic, jc, UPLO);
                        } else {
                            macro_kernel_rect(ib, jb, pb, alpha, Ap, Bp,
                                              &C_(ic, jc), ldc);
                        }
                    }
                    /* implicit barrier at end of `for` keeps Bp stable
                     * for the next (jc, pc) iteration. */
                }
            }
        }
        free(Ap);
    }

    free(Bp);
}

#undef C_
