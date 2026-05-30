/*
 * esyrk_parallel.c — the public Fortran entry `esyrk_` and threading-only half
 * of the kind10 (REAL(KIND=10) / `long double`) symmetric rank-k overlay (see
 * esyrk_kernel.h; all the leaf math lives in esyrk_serial.c).
 *
 * Owns the cooperative GotoBLAS port (OpenBLAS DSYRK pattern): a quadratic
 * N-partition that balances triangular work across threads, cross-thread
 * buffer-sharing via per-(producer,consumer,bufferside) atomic flags, and the
 * cooperative inner kernel (inner_syrk). The leaf packers / micro-kernels /
 * macro-kernels are reused from the header.
 *
 * Nesting guard: when esyrk_ is itself called from inside another routine's
 * parallel region it delegates to esyrk_serial and opens no region of its own
 * (the libgomp barrier wedge guard, project-etrsm-omp4-wedge). esyrk_serial_inline
 * is the OOM fallback for every allocation that can fail along the way.
 */

#include "esyrk_kernel.h"
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

typedef esyrk_T T;

#define MR ESYRK_MR
#define NR ESYRK_NR

#define ESYRK_OMP_MIN  32

#define DIVIDE_RATE   2
#define CACHE_LINE_T  8   /* 8 × sizeof(uintptr_t) = 64 bytes — one cache line */

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}
static inline int imin(int a, int b) { return a < b ? a : b; }

#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

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
    (void)sa_rows_padded;
    const int m_from = range[mypos];
    const int m_to   = range[mypos + 1];
    const int own_w  = m_to - m_from;
    if (own_w <= 0) return;
    const int lower = (UPLO == 'L');

    /* DIVIDE_RATE sub-buffers, each sized KC × buf_div_n (in T). */
    T *buffer[DIVIDE_RATE];
    for (int b = 0; b < DIVIDE_RATE; ++b)
        buffer[b] = buffer_base + (size_t)b * KC * esyrk_round_up(buf_div_n, NR);

    /* Per-thread own-band sub-division for buffer sharing.
     * Each bufferside holds up to div_n cols of own band, packed. */
    const int div_n = esyrk_round_up((own_w + DIVIDE_RATE - 1) / DIVIDE_RATE, NR);
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
            min_i = esyrk_round_up(own_w / 2, MR);
            if (min_i > MC) min_i = MC;
        } else {
            min_i = own_w;
        }
        const int start_i = lower ? min_i : 0;
        const int first_row = lower ? (m_to - min_i) : m_from;

        /* PHASE 1: pack own A row panel (sa) for first chunk, and own
         * column panel (buffer[bs]) sub-pieces; compute diagonal block. */
        esyrk_pack_A_panel(a, lda, first_row, ls, min_i, min_l, TR, sa);

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
            esyrk_pack_B_panel(a, lda, xxx, ls, sub_w, min_l, TR, buffer[bs]);

            /* Diagonal sub-block kernel (triangle-aware). */
            esyrk_macro_kernel_tri(min_i, sub_w, min_l, alpha, sa, buffer[bs],
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
                const int cdiv = esyrk_round_up((cw + DIVIDE_RATE - 1) / DIVIDE_RATE, NR);
                int cbs = 0;
                for (int xxx = range[current]; xxx < range[current + 1];
                     xxx += cdiv, ++cbs) {
                    volatile uintptr_t *f = flag_at(flags, current, mypos, cbs, nt);
                    while (*f == 0) cpu_relax();
                    RMB();
                    T *their = (T *)*f;
                    const int sub_w = imin(cdiv, range[current + 1] - xxx);
                    esyrk_macro_kernel_rect(min_i, sub_w, min_l, alpha, sa, their,
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
                const int cdiv = esyrk_round_up((cw + DIVIDE_RATE - 1) / DIVIDE_RATE, NR);
                int cbs = 0;
                for (int xxx = range[current]; xxx < range[current + 1];
                     xxx += cdiv, ++cbs) {
                    volatile uintptr_t *f = flag_at(flags, current, mypos, cbs, nt);
                    while (*f == 0) cpu_relax();
                    RMB();
                    T *their = (T *)*f;
                    const int sub_w = imin(cdiv, range[current + 1] - xxx);
                    esyrk_macro_kernel_rect(min_i, sub_w, min_l, alpha, sa, their,
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
            esyrk_pack_A_panel(a, lda, is, ls, chunk_i, min_l, TR, sa);

            const int last_chunk = (is + chunk_i >= is_hi);

            if (lower) {
                for (int current = mypos; current >= 0; --current) {
                    const int cw = range[current + 1] - range[current];
                    const int cdiv = esyrk_round_up((cw + DIVIDE_RATE - 1) / DIVIDE_RATE, NR);
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
                            esyrk_macro_kernel_tri(chunk_i, sub_w, min_l, alpha, sa, their,
                                                   &C_(is, xxx), ldc, is, xxx, UPLO);
                        } else {
                            esyrk_macro_kernel_rect(chunk_i, sub_w, min_l, alpha, sa, their,
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
                    const int cdiv = esyrk_round_up((cw + DIVIDE_RATE - 1) / DIVIDE_RATE, NR);
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
                            esyrk_macro_kernel_tri(chunk_i, sub_w, min_l, alpha, sa, their,
                                                   &C_(is, xxx), ldc, is, xxx, UPLO);
                        } else {
                            esyrk_macro_kernel_rect(chunk_i, sub_w, min_l, alpha, sa, their,
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
#ifdef _OPENMP
    /* Already inside a team → run serially, no nested region (wedge guard). */
    if (omp_in_parallel()) {
        esyrk_serial(uplo, trans, n_, k_, alpha_, a, lda_, beta_, c, ldc_,
                     uplo_len, trans_len);
        return;
    }
#endif
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
                                  && blas_omp_max_threads() > 1);
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
    const int can_par = (max_threads > 1 && N >= ESYRK_OMP_MIN);
    const int nt = can_par ? max_threads : 1;
    const int use_cooperative = can_par && (N >= nt * esyrk_switch_ratio());
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

    int MC, KC, NC_unused;
    esyrk_block_sizes(&MC, &KC, &NC_unused);

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
    const int buf_div_n = esyrk_round_up((max_w + DIVIDE_RATE - 1) / DIVIDE_RATE, NR);

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

    const int sa_rows_padded = esyrk_round_up(MC, MR);
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
