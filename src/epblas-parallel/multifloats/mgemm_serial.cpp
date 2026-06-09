/*
 * mgemm_serial.cpp — multifloats real GEMM (float64x2, double-double),
 * single-thread core. Owns ALL the numerics shared by the serial and
 * parallel entries: the trans decode, the block-size policy, the scalar
 * and AVX2-SIMD packers and micro-kernels (declared in mgemm_kernel.h),
 * plus the public `mgemm_serial` Fortran entry. No OpenMP anywhere on this
 * call path — safe to invoke from inside another routine's parallel region.
 *
 * The math is bitwise-identical to the previous single mgemm.cpp: the
 * parallel entry (mgemm_parallel.cpp) drives these exact leaves over a
 * jc-partitioned team, so the two paths agree to the last bit.
 *
 * Per-element cost: ~8 fp ops for an add, ~12 for a mul (Dekker / Knuth
 * EFTs). Arithmetic-bound; OMP scales near-linearly.
 */

#include "mgemm_kernel.h"
#include <cstdlib>
#include <cctype>

#ifdef MBLAS_SIMD_DD
#include "mgemm_simd_kernel.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {

int g_mc = 0, g_kc = 0, g_nc = 0;
void init_blocks() {
    if (g_mc) return;
    g_mc =  64;
    g_kc = 128;
    g_nc = 256;
}

const T zero_dd{0.0, 0.0};
const T one_dd {1.0, 0.0};

}  // namespace

int mgemm_trans_code(const char *p, std::size_t /*len*/) {
    char c = static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
    return (c == 'C') ? 'T' : c;  /* real type: C == T */
}

void mgemm_choose_blocks(int *MC, int *KC, int *NC) {
    init_blocks();
    *MC = g_mc; *KC = g_kc; *NC = g_nc;
}

void mgemm_pack_A(const T * __restrict__ A, int lda,
                  int ic, int pc, int ib, int pb,
                  int ta, T * __restrict__ Ap)
{
    if (ta == 'N') {
        for (int p = 0; p < pb; ++p) {
            const T *src = &A[static_cast<std::size_t>(pc + p) * lda + ic];
            T *dst = &Ap[static_cast<std::size_t>(p) * ib];
            for (int i = 0; i < ib; ++i) dst[i] = src[i];
        }
    } else {
        for (int i = 0; i < ib; ++i) {
            const T *src = &A[static_cast<std::size_t>(ic + i) * lda + pc];
            for (int p = 0; p < pb; ++p)
                Ap[static_cast<std::size_t>(p) * ib + i] = src[p];
        }
    }
}

void mgemm_pack_B(const T * __restrict__ B, int ldb,
                  int pc, int jc, int pb, int jb,
                  int tb, T * __restrict__ Bp)
{
    if (tb == 'N') {
        for (int j = 0; j < jb; ++j) {
            const T *src = &B[static_cast<std::size_t>(jc + j) * ldb + pc];
            T *dst = &Bp[static_cast<std::size_t>(j) * pb];
            for (int p = 0; p < pb; ++p) dst[p] = src[p];
        }
    } else {
        for (int p = 0; p < pb; ++p) {
            const T *src = &B[static_cast<std::size_t>(pc + p) * ldb + jc];
            for (int j = 0; j < jb; ++j)
                Bp[static_cast<std::size_t>(j) * pb + p] = src[j];
        }
    }
}

void mgemm_inner_kernel(int ib, int jb, int pb, T alpha,
                        const T * __restrict__ Ap, const T * __restrict__ Bp,
                        T * __restrict__ C, int ldc)
{
    for (int j = 0; j < jb; ++j) {
        T *cj = &C[static_cast<std::size_t>(j) * ldc];
        const T *bj = &Bp[static_cast<std::size_t>(j) * pb];
        for (int p = 0; p < pb; ++p) {
            const T t = alpha * bj[p];
            const T *ap = &Ap[static_cast<std::size_t>(p) * ib];
            for (int i = 0; i < ib; ++i) cj[i] += t * ap[i];
        }
    }
}

#ifdef MBLAS_SIMD_DD

/*
 * SoA pack_B for SIMD path.
 * Layout: NR-column panels along the j-axis. Within each panel, the
 * (p, j_local) plane is stored as NR contiguous doubles per p in two
 * parallel arrays (hi, lo) — so loading 4 doubles into a ymm register
 * for one p iteration is a straight `vmovupd`.
 *
 *   Bp_hi[ panel*(pb*NR) + p*NR + j_local ] = op(B)[pc+p, jc+panel*NR+j_local].limbs[0]
 *   Bp_lo[ same offset ]                    = ... .limbs[1]
 *
 * Trailing panel (jb mod NR != 0) is zero-padded so the SIMD kernel
 * can always run the full NR-wide tile; the writeback masks padded
 * lanes.
 */
/* Panel width W = simd_dd::NR * MGEMM_SIMD_NR_PAN (4 / 8 / …) */
constexpr int simd_pack_W() { return simd_dd::NR * MGEMM_SIMD_NR_PAN; }

void mgemm_pack_B_soa(const T * __restrict__ B, int ldb,
                      int pc, int jc, int pb, int jb, int tb,
                      double * __restrict__ Bp_hi, double * __restrict__ Bp_lo)
{
    const int W = simd_pack_W();
    const int npanels = (jb + W - 1) / W;
    for (int panel = 0; panel < npanels; ++panel) {
        const int j0 = panel * W;
        const int w_eff = (jb - j0 < W) ? (jb - j0) : W;
        double *dst_hi = &Bp_hi[static_cast<std::size_t>(panel) * pb * W];
        double *dst_lo = &Bp_lo[static_cast<std::size_t>(panel) * pb * W];
        if (tb == 'N') {
            for (int c = 0; c < w_eff; ++c) {
                const T *col = &B[static_cast<std::size_t>(jc + j0 + c) * ldb + pc];
                for (int p = 0; p < pb; ++p) {
                    dst_hi[p * W + c] = col[p].limbs[0];
                    dst_lo[p * W + c] = col[p].limbs[1];
                }
            }
            for (int c = w_eff; c < W; ++c)
                for (int p = 0; p < pb; ++p) {
                    dst_hi[p * W + c] = 0.0;
                    dst_lo[p * W + c] = 0.0;
                }
        } else {
            for (int p = 0; p < pb; ++p) {
                const T *row = &B[static_cast<std::size_t>(pc + p) * ldb + (jc + j0)];
                for (int c = 0; c < w_eff; ++c) {
                    dst_hi[p * W + c] = row[c].limbs[0];
                    dst_lo[p * W + c] = row[c].limbs[1];
                }
                for (int c = w_eff; c < W; ++c) {
                    dst_hi[p * W + c] = 0.0;
                    dst_lo[p * W + c] = 0.0;
                }
            }
        }
    }
}

int mgemm_simd_pack_W(void) { return simd_pack_W(); }

/*
 * SIMD writeback helper: alpha-scale a DD accumulator (acc_h, acc_l)
 * and merge into C[i, j0..j0+nr_eff-1].
 */
static inline __attribute__((always_inline)) void
simd_writeback(__m256d alpha_h, __m256d alpha_l,
               __m256d acc_h, __m256d acc_l,
               T *C_row_i, int ldc, int j0, int nr_eff)
{
    constexpr int NR = simd_dd::NR;
    __m256d ph, pl;
    simd_dd::dd_mul(alpha_h, alpha_l, acc_h, acc_l, ph, pl);
    alignas(32) double ph_a[NR], pl_a[NR];
    _mm256_store_pd(ph_a, ph);
    _mm256_store_pd(pl_a, pl);
    for (int j = 0; j < nr_eff; ++j) {
        T r;
        r.limbs[0] = ph_a[j];
        r.limbs[1] = pl_a[j];
        T &dst = C_row_i[static_cast<std::size_t>(j0 + j) * ldc];
        dst = dst + r;
    }
}

/*
 * Templated SIMD inner micro-kernel:
 *   MR rows of A × W=4*NR_PAN cols of B per call.
 *
 *   NR_PAN = number of 4-lane ymm panels stacked along the j-axis.
 *     1 → kernel processes NR=4 cols   (1 ymm-wide)
 *     2 → kernel processes NR=8 cols   (2 ymm-wide)
 *
 * Both loops are constexpr-bounded so gcc unrolls them. Each
 * (row k, panel n) pair gets its own pair of ymm accumulators
 * (ach[k][n], acl[k][n]) = 2 regs; total acc = 2 * MR * NR_PAN.
 *
 * Register budget on AVX2 (16 ymm regs):
 *   total = 2*MR*NR_PAN (acc) + 2 (broadcasts) + 2*NR_PAN (B loads)
 *         + 2 (scratch)
 * For MR=3 NR_PAN=1 → 12, MR=2 NR_PAN=2 → 16, MR=3 NR_PAN=2 → 20 (spills).
 *
 * MGEMM_SIMD_MR / MGEMM_SIMD_NR_PAN (cmake cache vars) pick MR and
 * NR_PAN at compile time.
 */
#ifndef MGEMM_SIMD_MR
#define MGEMM_SIMD_MR 3
#endif
#ifndef MGEMM_SIMD_NR_PAN
#define MGEMM_SIMD_NR_PAN 1
#endif

template <int MR, int NR_PAN>
static __attribute__((noinline)) void
inner_kernel_simd_t(int ib, int jb, int pb, T alpha,
                    const T * __restrict__ Ap,
                    const double * __restrict__ Bp_hi,
                    const double * __restrict__ Bp_lo,
                    T * __restrict__ C, int ldc)
{
    constexpr int NR_LANE = simd_dd::NR;   /* = 4 (one ymm) */
    constexpr int W = NR_LANE * NR_PAN;    /* total cols per call */
    const int j_panels = (jb + W - 1) / W;
    const __m256d alpha_h = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d alpha_l = _mm256_set1_pd(alpha.limbs[1]);
    for (int jp = 0; jp < j_panels; ++jp) {
        const int j0 = jp * W;
        const int w_eff = (jb - j0 < W) ? (jb - j0) : W;
        const double *Bp_h_panel = &Bp_hi[static_cast<std::size_t>(jp) * pb * W];
        const double *Bp_l_panel = &Bp_lo[static_cast<std::size_t>(jp) * pb * W];
        int i = 0;
        /* Main MR×NR_PAN tile loop */
        for (; i + MR <= ib; i += MR) {
            __m256d ach[MR][NR_PAN], acl[MR][NR_PAN];
            #pragma GCC unroll 8
            for (int k = 0; k < MR; ++k)
                #pragma GCC unroll 8
                for (int n = 0; n < NR_PAN; ++n) {
                    ach[k][n] = _mm256_setzero_pd();
                    acl[k][n] = _mm256_setzero_pd();
                }
            for (int p = 0; p < pb; ++p) {
                __m256d bh[NR_PAN], bl[NR_PAN];
                #pragma GCC unroll 8
                for (int n = 0; n < NR_PAN; ++n) {
                    bh[n] = _mm256_loadu_pd(&Bp_h_panel[p * W + n * NR_LANE]);
                    bl[n] = _mm256_loadu_pd(&Bp_l_panel[p * W + n * NR_LANE]);
                }
                #pragma GCC unroll 8
                for (int k = 0; k < MR; ++k) {
                    const T &aval = Ap[static_cast<std::size_t>(p) * ib + i + k];
                    __m256d ah = _mm256_set1_pd(aval.limbs[0]);
                    __m256d al = _mm256_set1_pd(aval.limbs[1]);
                    #pragma GCC unroll 8
                    for (int n = 0; n < NR_PAN; ++n) {
                        __m256d rh, rl;
                        simd_dd::dd_mul(ah, al, bh[n], bl[n], rh, rl);
                        simd_dd::dd_add(ach[k][n], acl[k][n], rh, rl,
                                        ach[k][n], acl[k][n]);
                    }
                }
            }
            /* Writeback: NR_PAN ymm-panels per row. */
            #pragma GCC unroll 8
            for (int k = 0; k < MR; ++k) {
                #pragma GCC unroll 8
                for (int n = 0; n < NR_PAN; ++n) {
                    const int panel_j0 = j0 + n * NR_LANE;
                    const int panel_eff = (w_eff - n * NR_LANE < NR_LANE)
                        ? (w_eff - n * NR_LANE) : NR_LANE;
                    if (panel_eff > 0)
                        simd_writeback(alpha_h, alpha_l, ach[k][n], acl[k][n],
                                       &C[i + k], ldc, panel_j0, panel_eff);
                }
            }
        }
        /* MR=1, NR_PAN=1 tail (just iterate remaining rows over panel 0). */
        for (; i < ib; ++i) {
            #pragma GCC unroll 8
            for (int n = 0; n < NR_PAN; ++n) {
                const int panel_j0 = j0 + n * NR_LANE;
                const int panel_eff = (w_eff - n * NR_LANE < NR_LANE)
                    ? (w_eff - n * NR_LANE) : NR_LANE;
                if (panel_eff <= 0) continue;
                __m256d acc_h = _mm256_setzero_pd();
                __m256d acc_l = _mm256_setzero_pd();
                for (int p = 0; p < pb; ++p) {
                    const T &aval = Ap[static_cast<std::size_t>(p) * ib + i];
                    __m256d ah = _mm256_set1_pd(aval.limbs[0]);
                    __m256d al = _mm256_set1_pd(aval.limbs[1]);
                    __m256d bh = _mm256_loadu_pd(&Bp_h_panel[p * W + n * NR_LANE]);
                    __m256d bl = _mm256_loadu_pd(&Bp_l_panel[p * W + n * NR_LANE]);
                    __m256d rh, rl;
                    simd_dd::dd_mul(ah, al, bh, bl, rh, rl);
                    simd_dd::dd_add(acc_h, acc_l, rh, rl, acc_h, acc_l);
                }
                simd_writeback(alpha_h, alpha_l, acc_h, acc_l,
                               &C[i], ldc, panel_j0, panel_eff);
            }
        }
    }
}

void mgemm_inner_kernel_simd(int ib, int jb, int pb, T alpha,
                             const T * __restrict__ Ap,
                             const double * __restrict__ Bp_hi,
                             const double * __restrict__ Bp_lo,
                             T * __restrict__ C, int ldc)
{
    inner_kernel_simd_t<MGEMM_SIMD_MR, MGEMM_SIMD_NR_PAN>(
        ib, jb, pb, alpha, Ap, Bp_hi, Bp_lo, C, ldc);
}

#endif /* MBLAS_SIMD_DD */

extern "C" void mgemm_serial(
    const char *transa, const char *transb,
    const int *m_, const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    std::size_t transa_len, std::size_t transb_len)
{
    const int M = *m_, N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const int ta = mgemm_trans_code(transa, transa_len);
    const int tb = mgemm_trans_code(transb, transb_len);

    if (M <= 0 || N <= 0) return;

    for (int j = 0; j < N; ++j) {
        T *cj = &c[static_cast<std::size_t>(j) * ldc];
        if (beta == zero_dd) {
            for (int i = 0; i < M; ++i) cj[i] = zero_dd;
        } else if (beta != one_dd) {
            for (int i = 0; i < M; ++i) cj[i] = cj[i] * beta;
        }
    }
    if (alpha == zero_dd || K == 0) return;

    int MC, KC, NC;
    mgemm_choose_blocks(&MC, &KC, &NC);

    T *Ap = static_cast<T *>(std::aligned_alloc(
        64, static_cast<std::size_t>(MC) * KC * sizeof(T)));
#ifdef MBLAS_SIMD_DD
    const int W_simd = mgemm_simd_pack_W();
    const int NC_pad = ((NC + W_simd - 1) / W_simd) * W_simd;
    double *Bp_hi = static_cast<double *>(std::aligned_alloc(
        64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
    double *Bp_lo = static_cast<double *>(std::aligned_alloc(
        64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
    if (Ap && Bp_hi && Bp_lo) {
        for (int jc = 0; jc < N; jc += NC) {
            const int jb = (N - jc < NC) ? (N - jc) : NC;
            for (int pc = 0; pc < K; pc += KC) {
                const int pb = (K - pc < KC) ? (K - pc) : KC;
                mgemm_pack_B_soa(b, ldb, pc, jc, pb, jb, tb, Bp_hi, Bp_lo);
                for (int ic = 0; ic < M; ic += MC) {
                    const int ib = (M - ic < MC) ? (M - ic) : MC;
                    mgemm_pack_A(a, lda, ic, pc, ib, pb, ta, Ap);
                    mgemm_inner_kernel_simd(ib, jb, pb, alpha, Ap, Bp_hi, Bp_lo,
                                            &c[static_cast<std::size_t>(jc) * ldc + ic],
                                            ldc);
                }
            }
        }
    }
    std::free(Ap);
    std::free(Bp_hi);
    std::free(Bp_lo);
#else
    T *Bp = static_cast<T *>(std::aligned_alloc(
        64, static_cast<std::size_t>(KC) * NC * sizeof(T)));
    if (Ap && Bp) {
        for (int jc = 0; jc < N; jc += NC) {
            const int jb = (N - jc < NC) ? (N - jc) : NC;
            for (int pc = 0; pc < K; pc += KC) {
                const int pb = (K - pc < KC) ? (K - pc) : KC;
                mgemm_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
                for (int ic = 0; ic < M; ic += MC) {
                    const int ib = (M - ic < MC) ? (M - ic) : MC;
                    mgemm_pack_A(a, lda, ic, pc, ib, pb, ta, Ap);
                    mgemm_inner_kernel(ib, jb, pb, alpha, Ap, Bp,
                                       &c[static_cast<std::size_t>(jc) * ldc + ic],
                                       ldc);
                }
            }
        }
    }
    std::free(Ap);
    std::free(Bp);
#endif /* MBLAS_SIMD_DD */
}
