/*
 * egemm — kind10 (REAL(KIND=10), x86-64 80-bit long double) GEMM overlay.
 *
 * Structure: GotoBLAS / OpenBLAS — three-level cache blocking
 * (NC × KC × MC), copy-and-conquer packing (op(A), op(B) absorbed into
 * Ap/Bp), register-blocked MR×NR outer-product micro-kernel, sub-NC
 * chunking by NR to keep the C-tile hot across the K sweep, adaptive
 * MC when K is small.
 *
 * Differences from OpenBLAS dgemm:
 *   - No assembly. Pure C inner kernel; gcc keeps the 4 MR×NR fp80
 *     accumulators on the x87 stack across the K-loop (MR=2, NR=2 is
 *     deliberately small to fit the 8-deep x87 register stack).
 *   - No SIMD on `long double` (x86-64 has no AVX path for 80-bit).
 *   - Edge tiles for the M-tail and N-tail go through a scalar dot path.
 *
 * Packing layouts (panel-packed, OpenBLAS-style):
 *   Ap: tiled by MR rows. For each MR-panel within (ic..ic+ib),
 *       Ap_panel[p*MR + ii] = op(A)[ic + panel_off + ii, pc + p].
 *   Bp: tiled by NR cols. For each NR-panel within (jc..jc+jb),
 *       Bp_panel[p*NR + jj] = op(B)[pc + p, jc + panel_off + jj].
 *   Inner kernel reads MR consecutive A elements and NR consecutive B
 *   elements per p — both stride-1 in the packed layout.
 *
 * Block sizes (env-overridable):
 *   EBLAS_MC=64   panel rows
 *   EBLAS_KC=256  panel depth
 *   EBLAS_NC=512  column band per thread
 * Register-tile dims MR=2, NR=2 are compile-time constants.
 *
 * Fortran ABI:
 *   - subroutine name lowercased + trailing underscore: `egemm_`
 *   - scalars passed by pointer
 *   - character args followed by hidden trailing `size_t` lengths
 *   - REAL(KIND=10) ↔ `long double` (x86-64 80-bit extended)
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef long double T;

#define MR 2
#define NR 2

/* ── Block sizes ──────────────────────────────────────────────── */

static int env_int(const char *name, int dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    int v = atoi(s);
    return v > 0 ? v : dflt;
}

static int g_mc = 0, g_kc = 0, g_nc = 0;
static void init_blocks(void) {
    if (g_mc) return;
    g_mc = env_int("EBLAS_MC",  64);
    g_kc = env_int("EBLAS_KC", 256);
    g_nc = env_int("EBLAS_NC", 512);
}

static int trans_code(const char *p, size_t len) {
    (void)len;
    char c = (char)toupper((unsigned char)*p);
    return (c == 'C') ? 'T' : c;  /* real: 'C' ≡ 'T' */
}

static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }

/* ── Packers (panel-packed, OpenBLAS-style) ───────────────────── */

/*
 * Pack op(A)(ic..ic+ib, pc..pc+pb) into Ap as a stack of MR-row panels.
 * Panel layout:  Ap[(ii_panel * pb + p) * MR + ii] = op(A)[ic + ii_panel*MR + ii, pc + p].
 *
 * The last panel is zero-padded to MR rows when ib % MR != 0.
 */
static void pack_A(const T *restrict A, int lda,
                   int ic, int pc, int ib, int pb,
                   int ta, T *restrict Ap)
{
    const int npanel = (ib + MR - 1) / MR;
    for (int q = 0; q < npanel; ++q) {
        const int i0 = ic + q * MR;
        const int rows = (q == npanel - 1) ? (ib - q * MR) : MR;
        T *Apanel = &Ap[(size_t)q * pb * MR];
        if (ta == 'N') {
            for (int p = 0; p < pb; ++p) {
                const T *src = &A[(size_t)(pc + p) * lda + i0];
                T *dst = &Apanel[(size_t)p * MR];
                int ii;
                for (ii = 0; ii < rows; ++ii) dst[ii] = src[ii];
                for (; ii < MR; ++ii) dst[ii] = 0.0L;
            }
        } else {
            for (int p = 0; p < pb; ++p) {
                T *dst = &Apanel[(size_t)p * MR];
                int ii;
                for (ii = 0; ii < rows; ++ii)
                    dst[ii] = A[(size_t)(i0 + ii) * lda + (pc + p)];
                for (; ii < MR; ++ii) dst[ii] = 0.0L;
            }
        }
    }
}

/*
 * Pack op(B)(pc..pc+pb, jc..jc+jb) into Bp as a stack of NR-col panels.
 * Panel layout:  Bp[(jj_panel * pb + p) * NR + jj] = op(B)[pc + p, jc + jj_panel*NR + jj].
 */
static void pack_B(const T *restrict B, int ldb,
                   int pc, int jc, int pb, int jb,
                   int tb, T *restrict Bp)
{
    const int npanel = (jb + NR - 1) / NR;
    for (int q = 0; q < npanel; ++q) {
        const int j0 = jc + q * NR;
        const int cols = (q == npanel - 1) ? (jb - q * NR) : NR;
        T *Bpanel = &Bp[(size_t)q * pb * NR];
        if (tb == 'N') {
            for (int p = 0; p < pb; ++p) {
                T *dst = &Bpanel[(size_t)p * NR];
                int jj;
                for (jj = 0; jj < cols; ++jj)
                    dst[jj] = B[(size_t)(j0 + jj) * ldb + (pc + p)];
                for (; jj < NR; ++jj) dst[jj] = 0.0L;
            }
        } else {
            for (int p = 0; p < pb; ++p) {
                const T *src = &B[(size_t)(pc + p) * ldb + j0];
                T *dst = &Bpanel[(size_t)p * NR];
                int jj;
                for (jj = 0; jj < cols; ++jj) dst[jj] = src[jj];
                for (; jj < NR; ++jj) dst[jj] = 0.0L;
            }
        }
    }
}

/* ── Inner kernel: MR × NR outer-product over K ──────────────── */

/*
 * Full MR × NR tile. C strip is column-major (stride ldc between cols).
 * Four scalar accumulators live on the x87 register stack across the
 * K loop; each iteration loads MR A-elements and NR B-elements from
 * packed buffers and issues MR*NR independent multiply-adds.
 */
static inline void kernel_2x2(int pb, T alpha,
                              const T *restrict Apanel,
                              const T *restrict Bpanel,
                              T *restrict C, int ldc)
{
    T c00 = 0.0L, c01 = 0.0L, c10 = 0.0L, c11 = 0.0L;
    for (int p = 0; p < pb; ++p) {
        const T a0 = Apanel[(size_t)p * MR + 0];
        const T a1 = Apanel[(size_t)p * MR + 1];
        const T b0 = Bpanel[(size_t)p * NR + 0];
        const T b1 = Bpanel[(size_t)p * NR + 1];
        c00 += a0 * b0;
        c10 += a1 * b0;
        c01 += a0 * b1;
        c11 += a1 * b1;
    }
    C[0]         += alpha * c00;
    C[1]         += alpha * c10;
    C[ldc + 0]   += alpha * c01;
    C[ldc + 1]   += alpha * c11;
}

/* Edge tile: arbitrary mr ∈ [1, MR], nr ∈ [1, NR] — scalar fallback. */
static void kernel_edge(int mr, int nr, int pb, T alpha,
                        const T *restrict Apanel,
                        const T *restrict Bpanel,
                        T *restrict C, int ldc)
{
    for (int jj = 0; jj < nr; ++jj) {
        T *cj = &C[(size_t)jj * ldc];
        for (int ii = 0; ii < mr; ++ii) {
            T sum = 0.0L;
            for (int p = 0; p < pb; ++p)
                sum += Apanel[(size_t)p * MR + ii] *
                       Bpanel[(size_t)p * NR + jj];
            cj[ii] += alpha * sum;
        }
    }
}

/* Drive one (ib, jb, pb) macro-tile via MR×NR sub-tiles. */
static void macro_kernel(int ib, int jb, int pb, T alpha,
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

/* ── Entry point ──────────────────────────────────────────────── */

void egemm_(
    const char *transa, const char *transb,
    const int *m_, const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t transa_len, size_t transb_len)
{
    const int M = *m_, N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const int ta = trans_code(transa, transa_len);
    const int tb = trans_code(transb, transb_len);

    if (M <= 0 || N <= 0) return;

    /* BETA pre-pass: handles K==0 / alpha==0 in one place. */
    for (int j = 0; j < N; ++j) {
        T *cj = &c[(size_t)j * ldc];
        if (beta == 0.0L)      for (int i = 0; i < M; ++i) cj[i]  = 0.0L;
        else if (beta != 1.0L) for (int i = 0; i < M; ++i) cj[i] *= beta;
    }
    if (alpha == 0.0L || K == 0) return;

    /* Fast path: TA = 'T' (≡ 'C' for real), TB = 'N'. With column-major
     * storage the inner k-loop reads A column i and B column j both
     * stride-1 — near peak x87 throughput. Packing adds overhead the
     * blocked path can never recover here; the explicit reference body
     * matches migrated at ~1.0× across all sizes. */
    if (ta == 'T' && tb == 'N') {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int j2 = 0; j2 < N; ++j2) {
            T *cj = &c[(size_t)j2 * ldc];
            const T *bj = &b[(size_t)j2 * ldb];
            for (int i2 = 0; i2 < M; ++i2) {
                const T *ai = &a[(size_t)i2 * lda];
                T acc = 0.0L;
                for (int l = 0; l < K; ++l) acc += ai[l] * bj[l];
                cj[i2] += alpha * acc;
            }
        }
        return;
    }

    init_blocks();
    int MC = g_mc, KC = g_kc, NC = g_nc;

    /* OpenBLAS-style adaptive MC: when K fits in one panel, grow MC
     * so MC*KC stays roughly L2-sized (rounded to MR). Helps small-K
     * shapes where the default MC under-uses cache. */
    if (K <= KC) {
        const long L2_TARGET_BYTES = 768 * 1024;   /* ~3/4 of i3-1315U P-core L2 */
        long target_mc = L2_TARGET_BYTES / ((long)K * (long)sizeof(T));
        if (target_mc > MC) {
            if (target_mc > 4L * g_mc) target_mc = 4L * g_mc;
            MC = round_up((int)target_mc, MR);
            if (MC < g_mc) MC = g_mc;
        }
    }

    /*
     * Threading: single outer `omp parallel`, shared Bp packed once per
     * (jc, pc) via `omp single` (implicit barrier), then `omp for` over
     * the ic loop. Each thread keeps a private Ap.
     *
     * The previous 1D `omp for` over jc gave zero scaling whenever
     * N ≤ NC (default NC=512): only one jc iteration to share across
     * threads. Splitting along ic (the M axis) restores parallelism;
     * keeping Bp shared avoids per-tile re-packing that a naive
     * collapse(2) would force. Effective parallelism is bounded by
     * (M / MC) per jc-band, which is ample for our typical square
     * problems (HPC-target overlay, many cores) (Addendum 27 / Rule 35
     * spirit applied with single-Bp-pack refinement).
     */
    const size_t ap_bytes = (size_t)round_up(MC, MR) * KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * round_up(NC, NR) * sizeof(T);
    T *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (!Bp) return;
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        T *Ap = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        if (Ap) {
            for (int jc = 0; jc < N; jc += NC) {
                const int jb = (N - jc < NC) ? (N - jc) : NC;
                for (int pc = 0; pc < K; pc += KC) {
                    const int pb = (K - pc < KC) ? (K - pc) : KC;
#ifdef _OPENMP
                    #pragma omp single
#endif
                    pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
                    /* implicit barrier at end of `single` makes Bp safe to
                     * read in the for below. */
#ifdef _OPENMP
                    #pragma omp for schedule(static)
#endif
                    for (int ic = 0; ic < M; ic += MC) {
                        const int ib = (M - ic < MC) ? (M - ic) : MC;
                        pack_A(a, lda, ic, pc, ib, pb, ta, Ap);
                        macro_kernel(ib, jb, pb, alpha, Ap, Bp,
                                     &c[(size_t)jc * ldc + ic], ldc);
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
