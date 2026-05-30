/*
 * esyrk — kind10 (REAL(KIND=10) / `long double`) symmetric rank-k update.
 *
 *   C := alpha · A · Aᵀ + beta · C          (TRANS='N')
 *   C := alpha · Aᵀ · A + beta · C          (TRANS='T'/'C')
 *
 * C is N×N symmetric; only the UPLO triangle is touched.
 *
 * Two parallel strategies:
 *
 *   - Cooperative GotoBLAS port (OpenBLAS DSYRK pattern):
 *       Used when N ≥ nthreads · ESYRK_SWITCH_RATIO. Quadratic N-partition
 *       balances triangular work across threads; each thread packs an A
 *       row panel (sa) and an A column panel (buffer), then computes its
 *       own diagonal block plus all off-diagonal output rows it owns.
 *       Cross-thread buffer sharing via per-(producer,consumer,bufferside)
 *       atomic flags eliminates duplicate packing of A.
 *
 *   - Single-thread inline GotoBLAS:
 *       Used at OMP=1 and at small N below the cooperative threshold.
 *       Same packers and MR×NR kernels as the cooperative path; one
 *       thread walks (jc, pc, ic) and classifies each (ic, jc) tile
 *       against the UPLO triangle (skip / rect / triangle-aware).
 *       Buffers are allocated once at entry, not per block.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef long double T;

#define MR 2
#define NR 2
#define ESYRK_MC_DEFAULT  64
#define ESYRK_KC_DEFAULT 256
#define ESYRK_NC_DEFAULT 512

#define ESYRK_OMP_MIN        32
#define ESYRK_SWITCH_RATIO   16

#define DIVIDE_RATE   2
#define CACHE_LINE_T  8   /* 8 × sizeof(uintptr_t) = 64 bytes — one cache line */

static int g_mc = 0, g_kc = 0, g_nc = 0;
static int g_switch_ratio = 0;

static int env_int(const char *name, int dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    int v = atoi(s);
    return v > 0 ? v : dflt;
}

__attribute__((constructor))
static void esyrk_init(void) {
    g_mc = env_int("ESYRK_MC", ESYRK_MC_DEFAULT);
    g_kc = env_int("ESYRK_KC", ESYRK_KC_DEFAULT);
    g_nc = env_int("ESYRK_NC", ESYRK_NC_DEFAULT);
    g_switch_ratio = env_int("ESYRK_SWITCH_RATIO", ESYRK_SWITCH_RATIO);
}

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static inline int round_up(int v, int m) { return ((v + m - 1) / m) * m; }
static inline int imin(int a, int b) { return a < b ? a : b; }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

/* ─── GotoBLAS packers + MR×NR micro-kernel (mirrors egemm) ───── */

/* Pack op(A) rows [i0, i0+min_i) × cols [pc, pc+min_l) into MR-row panels.
 *   layout: panel q at offset (q * min_l * MR);
 *           entry [(p * MR) + ii] = op(A)[i0 + q*MR + ii, pc + p].
 *   - TR='N': op(A) = A, so row major access at A[col=pc+p, row=i0+q*MR+ii].
 *   - TR='T': op(A) = Aᵀ, so A[row=pc+p, col=i0+q*MR+ii].
 */
static void pack_A_panel(const T *restrict A, int lda,
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
static void pack_B_panel(const T *restrict A, int lda,
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

/* ─── Quadratic N partition for equal triangular work per thread ─ */

/* Quadratic partition for COOPERATIVE work balance (matches OpenBLAS
 * driver/level3/level3_syrk_threaded.c). Width sequence is the same for
 * both UPLOs:
 *     w_t = sqrt(i_t^2 + N^2/nthreads) - i_t,    i_t = sum_{s<t} w_s.
 *
 * For LOWER, fill forward: thread 0 owns the widest band at the LOWEST
 * column indices (its diagonal triangle is the dominant work, no off-
 * diagonal contribution since no lower-index threads exist).
 *
 * For UPPER, fill backward: thread 0 owns the NARROWEST band at the
 * LOWEST column indices (its diagonal is tiny but it contributes
 * off-diagonal slabs to every higher-index thread's column band).
 *
 * Either way, each thread's total work — diagonal triangle plus all
 * rectangles it produces using OWN sa × OTHER buffer — equals N²/(2·nt).
 */
static void syrk_quadratic_partition(int N, int nthreads, int mask,
                                     char UPLO, int *range)
{
    const int seg = mask + 1;
    const double dnum = (double)N * (double)N / (double)nthreads;

    if (UPLO == 'L') {
        int i = 0, num_cpu = 0;
        range[0] = 0;
        while (i < N && num_cpu < nthreads) {
            int width;
            if (nthreads - num_cpu > 1) {
                const double di = (double)i;
                const double dinum = di * di + dnum;
                width = ((int)((sqrt(dinum) - di + mask) / seg)) * seg;
                if (width <= 0 || width > N - i) width = N - i;
            } else {
                width = N - i;
            }
            range[num_cpu + 1] = range[num_cpu] + width;
            num_cpu++;
            i += width;
        }
        while (num_cpu < nthreads) {
            range[num_cpu + 1] = N;
            num_cpu++;
        }
    } else {
        /* UPPER — backward fill */
        range[nthreads] = N;
        int i = 0, num_cpu = 0;
        while (i < N && num_cpu < nthreads) {
            int width;
            if (nthreads - num_cpu > 1) {
                const double di = (double)i;
                const double dinum = di * di + dnum;
                width = ((int)((sqrt(dinum) - di + mask) / seg)) * seg;
                if (width <= 0 || width > N - i) width = N - i;
            } else {
                width = N - i;
            }
            range[nthreads - num_cpu - 1] = range[nthreads - num_cpu] - width;
            num_cpu++;
            i += width;
        }
        while (num_cpu < nthreads) {
            range[nthreads - num_cpu - 1] = 0;
            num_cpu++;
        }
    }
}

/* ─── Cooperative flag plumbing ────────────────────────────────── */

/* flags[(producer * nt + consumer) * DIVIDE_RATE + bs] occupies one cache
 * line. Value = 0 ⇒ buffer is empty (or consumed). Non-zero ⇒ pointer
 * to producer's buffer[bs], safe for consumer to read. */
static inline volatile uintptr_t *flag_at(volatile uintptr_t *flags,
                                          int producer, int consumer, int bs,
                                          int nt)
{
    return &flags[(((size_t)producer * nt + consumer) * DIVIDE_RATE + bs)
                  * CACHE_LINE_T];
}

#ifdef _OPENMP
static inline void cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__("pause" ::: "memory");
#else
    __asm__ __volatile__("" ::: "memory");
#endif
}
#define WMB() __atomic_thread_fence(__ATOMIC_RELEASE)
#define RMB() __atomic_thread_fence(__ATOMIC_ACQUIRE)
#else
static inline void cpu_relax(void) { }
#define WMB() do { } while (0)
#define RMB() do { } while (0)
#endif

/* ─── Cooperative inner kernel ─────────────────────────────────── */

/* One thread's contribution to the SYRK. mypos is the thread's index in
 * [0, nt). Produces a row panel (sa) and a column panel (buffer[bs]) per
 * K chunk; signals buffer-ready via flags; consumes other threads'
 * buffers for off-diagonal work. */
static void inner_syrk(int N, int K, T alpha, char UPLO, char TR,
                       const T *restrict a, int lda,
                       T *restrict c, int ldc,
                       const int *range, int nt, int mypos,
                       volatile uintptr_t *flags,
                       T *restrict sa, T *restrict buffer_base,
                       int sa_rows_padded, int buf_div_n,
                       int MC, int KC)
{
    const int m_from = range[mypos];
    const int m_to   = range[mypos + 1];
    const int own_w  = m_to - m_from;
    if (own_w <= 0) return;
    const int lower = (UPLO == 'L');

    /* DIVIDE_RATE sub-buffers, each sized KC × buf_div_n (in T). */
    T *buffer[DIVIDE_RATE];
    for (int b = 0; b < DIVIDE_RATE; ++b)
        buffer[b] = buffer_base + (size_t)b * KC * round_up(buf_div_n, NR);

    /* Per-thread own-band sub-division for buffer sharing.
     * Each bufferside holds up to div_n cols of own band, packed. */
    const int div_n = round_up((own_w + DIVIDE_RATE - 1) / DIVIDE_RATE, NR);
    if (div_n != buf_div_n) {
        /* defensive — caller computed buf_div_n as the MAX across threads;
         * each thread's own div_n is ≤ buf_div_n, so layout still fits. */
    }

    for (int ls = 0; ls < K; ls += KC) {
        const int min_l = imin(KC, K - ls);

        /* Pick first row chunk size: split own_w into ~halves rounded
         * to MR; clamp to MC. */
        int min_i;
        if (own_w >= 2 * MC) {
            min_i = MC;
        } else if (own_w > MC) {
            min_i = round_up(own_w / 2, MR);
            if (min_i > MC) min_i = MC;
        } else {
            min_i = own_w;
        }
        const int start_i = lower ? min_i : 0;
        const int first_row = lower ? (m_to - min_i) : m_from;

        /* PHASE 1: pack own A row panel (sa) for first chunk, and own
         * column panel (buffer[bs]) sub-pieces; compute diagonal block. */
        pack_A_panel(a, lda, first_row, ls, min_i, min_l, TR, sa);

        int bs = 0;
        for (int xxx = m_from; xxx < m_to; xxx += div_n, ++bs) {
            /* wait for own working[i][bs] == 0 for cross-thread consumers
             * (self-flag is intentionally never set, so skipping it is safe). */
            const int i_lo = lower ? mypos     : 0;
            const int i_hi = lower ? nt        : mypos + 1;
            for (int i = i_lo; i < i_hi; ++i) {
                if (i == mypos) continue;
                volatile uintptr_t *f = flag_at(flags, mypos, i, bs, nt);
                while (*f != 0) cpu_relax();
            }
            RMB();

            const int sub_w = imin(div_n, m_to - xxx);
            pack_B_panel(a, lda, xxx, ls, sub_w, min_l, TR, buffer[bs]);

            /* Diagonal sub-block kernel (triangle-aware). */
            macro_kernel_tri(min_i, sub_w, min_l, alpha, sa, buffer[bs],
                             &C_(first_row, xxx), ldc,
                             first_row, xxx, UPLO);

            WMB();
            for (int i = i_lo; i < i_hi; ++i) {
                if (i == mypos) continue;
                volatile uintptr_t *f = flag_at(flags, mypos, i, bs, nt);
                *f = (uintptr_t)buffer[bs];
            }
        }

        /* PHASE 2: work-steal own sa × OTHER threads' buffers for the
         * off-diagonal slab spanned by this row chunk. */
        if (lower) {
            for (int current = mypos - 1; current >= 0; --current) {
                const int cw = range[current + 1] - range[current];
                const int cdiv = round_up((cw + DIVIDE_RATE - 1) / DIVIDE_RATE, NR);
                int cbs = 0;
                for (int xxx = range[current]; xxx < range[current + 1];
                     xxx += cdiv, ++cbs) {
                    volatile uintptr_t *f = flag_at(flags, current, mypos, cbs, nt);
                    while (*f == 0) cpu_relax();
                    RMB();
                    T *their = (T *)*f;
                    const int sub_w = imin(cdiv, range[current + 1] - xxx);
                    macro_kernel_rect(min_i, sub_w, min_l, alpha, sa, their,
                                      &C_(first_row, xxx), ldc);
                    if (own_w == min_i) {
                        WMB();
                        *f = 0;
                    }
                }
            }
        } else {
            for (int current = mypos + 1; current < nt; ++current) {
                const int cw = range[current + 1] - range[current];
                const int cdiv = round_up((cw + DIVIDE_RATE - 1) / DIVIDE_RATE, NR);
                int cbs = 0;
                for (int xxx = range[current]; xxx < range[current + 1];
                     xxx += cdiv, ++cbs) {
                    volatile uintptr_t *f = flag_at(flags, current, mypos, cbs, nt);
                    while (*f == 0) cpu_relax();
                    RMB();
                    T *their = (T *)*f;
                    const int sub_w = imin(cdiv, range[current + 1] - xxx);
                    macro_kernel_rect(min_i, sub_w, min_l, alpha, sa, their,
                                      &C_(first_row, xxx), ldc);
                    if (own_w == min_i) {
                        WMB();
                        *f = 0;
                    }
                }
            }
        }

        /* PHASE 3: remaining own row chunks. For LOWER, walk upward from
         * m_from to m_to-start_i (the first chunk was at the bottom).
         * For UPPER, walk downward from m_from+min_i to m_to. */
        const int is_lo = lower ? m_from           : (m_from + min_i);
        const int is_hi = lower ? (m_to - start_i) : m_to;

        for (int is = is_lo; is < is_hi; is += MC) {
            int chunk_i = imin(MC, is_hi - is);
            pack_A_panel(a, lda, is, ls, chunk_i, min_l, TR, sa);

            const int last_chunk = (is + chunk_i >= is_hi);

            if (lower) {
                for (int current = mypos; current >= 0; --current) {
                    const int cw = range[current + 1] - range[current];
                    const int cdiv = round_up((cw + DIVIDE_RATE - 1) / DIVIDE_RATE, NR);
                    int cbs = 0;
                    for (int xxx = range[current]; xxx < range[current + 1];
                         xxx += cdiv, ++cbs) {
                        T *their;
                        if (current == mypos) {
                            their = buffer[cbs];
                        } else {
                            volatile uintptr_t *f = flag_at(flags, current, mypos, cbs, nt);
                            RMB();
                            their = (T *)*f;
                        }
                        const int sub_w = imin(cdiv, range[current + 1] - xxx);
                        if (current == mypos) {
                            macro_kernel_tri(chunk_i, sub_w, min_l, alpha, sa, their,
                                             &C_(is, xxx), ldc, is, xxx, UPLO);
                        } else {
                            macro_kernel_rect(chunk_i, sub_w, min_l, alpha, sa, their,
                                              &C_(is, xxx), ldc);
                            if (last_chunk) {
                                volatile uintptr_t *f = flag_at(flags, current, mypos, cbs, nt);
                                WMB();
                                *f = 0;
                            }
                        }
                    }
                }
            } else {
                for (int current = mypos; current < nt; ++current) {
                    const int cw = range[current + 1] - range[current];
                    const int cdiv = round_up((cw + DIVIDE_RATE - 1) / DIVIDE_RATE, NR);
                    int cbs = 0;
                    for (int xxx = range[current]; xxx < range[current + 1];
                         xxx += cdiv, ++cbs) {
                        T *their;
                        if (current == mypos) {
                            their = buffer[cbs];
                        } else {
                            volatile uintptr_t *f = flag_at(flags, current, mypos, cbs, nt);
                            RMB();
                            their = (T *)*f;
                        }
                        const int sub_w = imin(cdiv, range[current + 1] - xxx);
                        if (current == mypos) {
                            macro_kernel_tri(chunk_i, sub_w, min_l, alpha, sa, their,
                                             &C_(is, xxx), ldc, is, xxx, UPLO);
                        } else {
                            macro_kernel_rect(chunk_i, sub_w, min_l, alpha, sa, their,
                                              &C_(is, xxx), ldc);
                            if (last_chunk) {
                                volatile uintptr_t *f = flag_at(flags, current, mypos, cbs, nt);
                                WMB();
                                *f = 0;
                            }
                        }
                    }
                }
            }
        }

        /* If own_w == min_i then the second pass loop was empty, and
         * own-buffer flags toward LOWER consumers (other threads) were
         * already cleared in PHASE 2. */
    }

    /* Drain: wait until every consumer (other than self) has cleared the
     * flags producer mypos wrote during PHASE 1. */
    for (int bs2 = 0; bs2 < DIVIDE_RATE; ++bs2) {
        const int i_lo = lower ? mypos     : 0;
        const int i_hi = lower ? nt        : mypos + 1;
        for (int i = i_lo; i < i_hi; ++i) {
            if (i == mypos) continue;
            volatile uintptr_t *f = flag_at(flags, mypos, i, bs2, nt);
            while (*f != 0) cpu_relax();
        }
    }
}

/* ─── Inline single-thread GotoBLAS path (OMP=1 / N below cooperative
 *      threshold). Same MR×NR kernel as the cooperative path, but with
 *      no flag plumbing: one thread walks the (jc, pc, ic) nest and
 *      classifies each (ic, jc) tile against the UPLO triangle.
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
static void esyrk_serial_inline(char UPLO, char TR, int N, int K,
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

    const int MC = g_mc, KC = g_kc;
    int NC = g_nc;
    if (NC > N) NC = N;
    if (NC < NR) NC = NR;

    const int sa_rows = round_up(MC, MR);
    const int sb_cols = round_up(NC, NR);
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

            pack_B_panel(a, lda, jc, pc, jb, pb, TR, Bp);

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

                pack_A_panel(a, lda, ic, pc, ib, pb, TR, Ap);

                if (tile_class == 1) {
                    macro_kernel_tri(ib, jb, pb, alpha, Ap, Bp,
                                     &c[(size_t)jc * ldc + ic], ldc,
                                     ic, jc, UPLO);
                } else {
                    macro_kernel_rect(ib, jb, pb, alpha, Ap, Bp,
                                      &c[(size_t)jc * ldc + ic], ldc);
                }
            }
        }
    }

    free(Ap);
    free(Bp);
}

/* ─── Entry point ──────────────────────────────────────────────── */

void esyrk_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *beta_,
    T *restrict c, const int *ldc_,
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
#ifdef _OPENMP
        const int use_omp_beta = (N >= ESYRK_OMP_MIN
                                  && blas_omp_max_threads() > 1
                                  && !omp_in_parallel());
        #pragma omp parallel for if(use_omp_beta) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const int i_lo = (UPLO == 'L') ? j : 0;
            const int i_hi = (UPLO == 'L') ? N : j + 1;
            T *cj = c + (size_t)j * ldc;
            if (beta == zero) for (int i = i_lo; i < i_hi; ++i) cj[i] = zero;
            else              for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;
        }
        return;
    }

#ifdef _OPENMP
    const int max_threads = blas_omp_max_threads();
    const int can_par = (max_threads > 1 && !omp_in_parallel()
                         && N >= ESYRK_OMP_MIN);
    const int nt = can_par ? max_threads : 1;
    const int use_cooperative = can_par && (N >= nt * g_switch_ratio);
#else
    const int can_par = 0;
    const int nt = 1;
    const int use_cooperative = 0;
#endif

    if (!use_cooperative) {
        esyrk_serial_inline(UPLO, TR, N, K, alpha, a, lda, beta, c, ldc);
        return;
    }

    /* ─── Cooperative parallel path ─────────────────────────────── */

    const int MC = g_mc, KC = g_kc;

    /* Partition own col bands. */
    int *range = (int *)malloc((size_t)(nt + 1) * sizeof(int));
    if (!range) {
        esyrk_serial_inline(UPLO, TR, N, K, alpha, a, lda, beta, c, ldc);
        return;
    }
    syrk_quadratic_partition(N, nt, MR - 1, UPLO, range);

    /* Compute max own-band width → buffer sizing.
     * Fall back to serial if any thread got zero width (would deadlock
     * the flag-based protocol). */
    int max_w = 0, min_w = N + 1;
    for (int t = 0; t < nt; ++t) {
        const int w = range[t + 1] - range[t];
        if (w > max_w) max_w = w;
        if (w < min_w) min_w = w;
    }
    if (min_w <= 0) {
        free(range);
        esyrk_serial_inline(UPLO, TR, N, K, alpha, a, lda, beta, c, ldc);
        return;
    }
    const int buf_div_n = round_up((max_w + DIVIDE_RATE - 1) / DIVIDE_RATE, NR);

    /* Allocate flag array (zeroed). */
    const size_t flag_count = (size_t)nt * nt * DIVIDE_RATE * CACHE_LINE_T;
    volatile uintptr_t *flags =
        (volatile uintptr_t *)aligned_alloc(64,
            ((flag_count * sizeof(uintptr_t)) + 63) & ~(size_t)63);
    if (!flags) {
        free(range);
        esyrk_serial_inline(UPLO, TR, N, K, alpha, a, lda, beta, c, ldc);
        return;
    }
    memset((void *)flags, 0, flag_count * sizeof(uintptr_t));

    const int sa_rows_padded = round_up(MC, MR);
    const size_t sa_bytes  = (size_t)sa_rows_padded * KC * sizeof(T);
    const size_t buf_bytes = (size_t)DIVIDE_RATE * KC * buf_div_n * sizeof(T);

    int alloc_failed = 0;

#ifdef _OPENMP
    #pragma omp parallel num_threads(nt)
#endif
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
        const int nt_inside = omp_get_num_threads();
#else
        const int tid = 0;
        const int nt_inside = 1;
#endif
        (void)nt_inside;

        /* Per-thread beta scale of own UPLO column band. */
        if (beta != one) {
            const int m_from = range[tid];
            const int m_to   = range[tid + 1];
            for (int j = m_from; j < m_to; ++j) {
                const int i_lo = (UPLO == 'L') ? j : 0;
                const int i_hi = (UPLO == 'L') ? N : j + 1;
                T *cj = c + (size_t)j * ldc;
                if (beta == zero) for (int i = i_lo; i < i_hi; ++i) cj[i] = zero;
                else              for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;
            }
        }

        T *sa  = (T *)aligned_alloc(64, (sa_bytes  + 63) & ~(size_t)63);
        T *buf = (T *)aligned_alloc(64, (buf_bytes + 63) & ~(size_t)63);
        if (!sa || !buf) {
            __atomic_store_n(&alloc_failed, 1, __ATOMIC_RELAXED);
        }

#ifdef _OPENMP
        #pragma omp barrier
#endif

        if (!__atomic_load_n(&alloc_failed, __ATOMIC_RELAXED) && sa && buf) {
            inner_syrk(N, K, alpha, UPLO, TR, a, lda, c, ldc,
                       range, nt, tid, flags, sa, buf,
                       sa_rows_padded, buf_div_n, MC, KC);
        }

        free(sa);
        free(buf);
    }

    free((void *)flags);
    free(range);

    if (alloc_failed) {
        /* Lost the parallel run to OOM — re-run via the serial fallback so
         * the caller still gets a correct C. The parallel section already
         * beta-scaled each thread's own column band before any thread hit
         * OOM, so pass beta=1 here to avoid double-scaling. */
        esyrk_serial_inline(UPLO, TR, N, K, alpha, a, lda, one, c, ldc);
    }
}

#undef A_
#undef C_
