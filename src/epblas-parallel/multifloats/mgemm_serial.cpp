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
#include "mf_pred.h"
#include "../common/blas_char.h"
#include <cstdlib>
#include <cctype>
#include "mf_dispatch.h"   /* MF_SIMD_TARGET + mf_have_avx2_fma() runtime gate */

#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using TR = mf::float64x2;

namespace {

/* Cache-block sizes — compile-time constants (nothing writes them). */
constexpr std::ptrdiff_t g_mc =  64;
constexpr std::ptrdiff_t g_kc = 128;
constexpr std::ptrdiff_t g_nc = 256;

using mf_pred::zero_dd;   /* shared DD constants — mf_pred.h */
using mf_pred::one_dd;

}  // namespace

void mgemm_choose_blocks(std::ptrdiff_t *MC, std::ptrdiff_t *KC, std::ptrdiff_t *NC) {
    *MC = g_mc; *KC = g_kc; *NC = g_nc;
}

void mgemm_pack_A(const TR * __restrict__ A, std::ptrdiff_t lda,
                  std::ptrdiff_t ic, std::ptrdiff_t pc, std::ptrdiff_t ib, std::ptrdiff_t pb,
                  std::ptrdiff_t ta, TR * __restrict__ Ap)
{
    if (ta == 'N') {
        for (std::ptrdiff_t p = 0; p < pb; ++p) {
            const TR *src = &A[static_cast<std::size_t>(pc + p) * lda + ic];
            TR *dst = &Ap[static_cast<std::size_t>(p) * ib];
            for (std::ptrdiff_t i = 0; i < ib; ++i) dst[i] = src[i];
        }
    } else {
        for (std::ptrdiff_t i = 0; i < ib; ++i) {
            const TR *src = &A[static_cast<std::size_t>(ic + i) * lda + pc];
            for (std::ptrdiff_t p = 0; p < pb; ++p)
                Ap[static_cast<std::size_t>(p) * ib + i] = src[p];
        }
    }
}

void mgemm_pack_B(const TR * __restrict__ B, std::ptrdiff_t ldb,
                  std::ptrdiff_t pc, std::ptrdiff_t jc, std::ptrdiff_t pb, std::ptrdiff_t jb,
                  std::ptrdiff_t tb, TR * __restrict__ Bp)
{
    if (tb == 'N') {
        for (std::ptrdiff_t j = 0; j < jb; ++j) {
            const TR *src = &B[static_cast<std::size_t>(jc + j) * ldb + pc];
            TR *dst = &Bp[static_cast<std::size_t>(j) * pb];
            for (std::ptrdiff_t p = 0; p < pb; ++p) dst[p] = src[p];
        }
    } else {
        for (std::ptrdiff_t p = 0; p < pb; ++p) {
            const TR *src = &B[static_cast<std::size_t>(pc + p) * ldb + jc];
            for (std::ptrdiff_t j = 0; j < jb; ++j)
                Bp[static_cast<std::size_t>(j) * pb + p] = src[j];
        }
    }
}

void mgemm_inner_kernel(std::ptrdiff_t ib, std::ptrdiff_t jb, std::ptrdiff_t pb, TR alpha,
                        const TR * __restrict__ Ap, const TR * __restrict__ Bp,
                        TR * __restrict__ C, std::ptrdiff_t ldc)
{
    for (std::ptrdiff_t j = 0; j < jb; ++j) {
        TR *cj = &C[static_cast<std::size_t>(j) * ldc];
        const TR *bj = &Bp[static_cast<std::size_t>(j) * pb];
        for (std::ptrdiff_t p = 0; p < pb; ++p) {
            const TR t = alpha * bj[p];
            const TR *ap = &Ap[static_cast<std::size_t>(p) * ib];
            for (std::ptrdiff_t i = 0; i < ib; ++i) cj[i] += t * ap[i];
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
/* Panel width W = simd_fast::NR * MGEMM_SIMD_NR_PAN (4 / 8 / …) */
constexpr std::ptrdiff_t simd_pack_W() { return simd_fast::NR * MGEMM_SIMD_NR_PAN; }

void mgemm_pack_B_soa(const TR * __restrict__ B, std::ptrdiff_t ldb,
                      std::ptrdiff_t pc, std::ptrdiff_t jc, std::ptrdiff_t pb, std::ptrdiff_t jb, std::ptrdiff_t tb,
                      double * __restrict__ Bp_hi, double * __restrict__ Bp_lo)
{
    const std::ptrdiff_t W = simd_pack_W();
    const std::ptrdiff_t npanels = (jb + W - 1) / W;
    for (std::ptrdiff_t panel = 0; panel < npanels; ++panel) {
        const std::ptrdiff_t j0 = panel * W;
        const std::ptrdiff_t w_eff = (jb - j0 < W) ? (jb - j0) : W;
        double *dst_hi = &Bp_hi[static_cast<std::size_t>(panel) * pb * W];
        double *dst_lo = &Bp_lo[static_cast<std::size_t>(panel) * pb * W];
        if (tb == 'N') {
            for (std::ptrdiff_t c = 0; c < w_eff; ++c) {
                const TR *col = &B[static_cast<std::size_t>(jc + j0 + c) * ldb + pc];
                for (std::ptrdiff_t p = 0; p < pb; ++p) {
                    dst_hi[p * W + c] = col[p].limbs[0];
                    dst_lo[p * W + c] = col[p].limbs[1];
                }
            }
            for (std::ptrdiff_t c = w_eff; c < W; ++c)
                for (std::ptrdiff_t p = 0; p < pb; ++p) {
                    dst_hi[p * W + c] = 0.0;
                    dst_lo[p * W + c] = 0.0;
                }
        } else {
            for (std::ptrdiff_t p = 0; p < pb; ++p) {
                const TR *row = &B[static_cast<std::size_t>(pc + p) * ldb + (jc + j0)];
                for (std::ptrdiff_t c = 0; c < w_eff; ++c) {
                    dst_hi[p * W + c] = row[c].limbs[0];
                    dst_lo[p * W + c] = row[c].limbs[1];
                }
                for (std::ptrdiff_t c = w_eff; c < W; ++c) {
                    dst_hi[p * W + c] = 0.0;
                    dst_lo[p * W + c] = 0.0;
                }
            }
        }
    }
}

std::ptrdiff_t mgemm_simd_pack_W(void) { return simd_pack_W(); }

/* AVX2+FMA under a possibly pre-Haswell baseline -march: the SIMD micro-kernels
 * below are compiled with the feature enabled and reached only behind
 * mf_have_avx2_fma() at the call site. The scalar packers above stay OUTSIDE this
 * region so they compile at the baseline ISA and are SNB-safe. See mf_dispatch.h. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")

/*
 * SIMD writeback helper: alpha-scale a DD accumulator (acc_h, acc_l)
 * and merge into C[i, j0..j0+nr_eff-1].
 */
static inline __attribute__((always_inline)) void
simd_writeback(__m256d alpha_h, __m256d alpha_l,
               __m256d acc_h, __m256d acc_l,
               TR *C_row_i, std::ptrdiff_t ldc, std::ptrdiff_t j0, std::ptrdiff_t nr_eff)
{
    constexpr std::ptrdiff_t NR = simd_fast::NR;
    __m256d ph, pl;
    simd_fast::mul(alpha_h, alpha_l, acc_h, acc_l, ph, pl);
    alignas(32) double ph_a[NR], pl_a[NR];
    _mm256_store_pd(ph_a, ph);
    _mm256_store_pd(pl_a, pl);
    for (std::ptrdiff_t j = 0; j < nr_eff; ++j) {
        TR r;
        r.limbs[0] = ph_a[j];
        r.limbs[1] = pl_a[j];
        TR &dst = C_row_i[static_cast<std::size_t>(j0 + j) * ldc];
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
#define MGEMM_SIMD_MR 4
#endif
#ifndef MGEMM_SIMD_NR_PAN
#define MGEMM_SIMD_NR_PAN 1
#endif

template <std::ptrdiff_t MR, std::ptrdiff_t NR_PAN>
static __attribute__((noinline)) void
inner_kernel_simd_t(std::ptrdiff_t ib, std::ptrdiff_t jb, std::ptrdiff_t pb, TR alpha,
                    const TR * __restrict__ Ap,
                    const double * __restrict__ Bp_hi,
                    const double * __restrict__ Bp_lo,
                    TR * __restrict__ C, std::ptrdiff_t ldc)
{
    constexpr std::ptrdiff_t NR_LANE = simd_fast::NR;   /* = 4 (one ymm) */
    constexpr std::ptrdiff_t W = NR_LANE * NR_PAN;    /* total cols per call */
    const std::ptrdiff_t j_panels = (jb + W - 1) / W;
    const __m256d alpha_h = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d alpha_l = _mm256_set1_pd(alpha.limbs[1]);
    for (std::ptrdiff_t jp = 0; jp < j_panels; ++jp) {
        const std::ptrdiff_t j0 = jp * W;
        const std::ptrdiff_t w_eff = (jb - j0 < W) ? (jb - j0) : W;
        const double *Bp_h_panel = &Bp_hi[static_cast<std::size_t>(jp) * pb * W];
        const double *Bp_l_panel = &Bp_lo[static_cast<std::size_t>(jp) * pb * W];
        std::ptrdiff_t i = 0;
        /* Main MR×NR_PAN tile loop */
        for (; i + MR <= ib; i += MR) {
            __m256d ach[MR][NR_PAN], acl[MR][NR_PAN];
            #pragma GCC unroll 8
            for (std::ptrdiff_t k = 0; k < MR; ++k)
                #pragma GCC unroll 8
                for (std::ptrdiff_t n = 0; n < NR_PAN; ++n) {
                    ach[k][n] = _mm256_setzero_pd();
                    acl[k][n] = _mm256_setzero_pd();
                }
            for (std::ptrdiff_t p = 0; p < pb; ++p) {
                __m256d bh[NR_PAN], bl[NR_PAN];
                #pragma GCC unroll 8
                for (std::ptrdiff_t n = 0; n < NR_PAN; ++n) {
                    bh[n] = _mm256_loadu_pd(&Bp_h_panel[p * W + n * NR_LANE]);
                    bl[n] = _mm256_loadu_pd(&Bp_l_panel[p * W + n * NR_LANE]);
                }
                #pragma GCC unroll 8
                for (std::ptrdiff_t k = 0; k < MR; ++k) {
                    const TR &aval = Ap[static_cast<std::size_t>(p) * ib + i + k];
                    __m256d ah = _mm256_set1_pd(aval.limbs[0]);
                    __m256d al = _mm256_set1_pd(aval.limbs[1]);
                    #pragma GCC unroll 8
                    for (std::ptrdiff_t n = 0; n < NR_PAN; ++n) {
                        __m256d rh, rl;
                        simd_fast::mul(ah, al, bh[n], bl[n], rh, rl);
                        simd_fast::add(ach[k][n], acl[k][n], rh, rl,
                                        ach[k][n], acl[k][n]);
                    }
                }
            }
            /* Writeback: NR_PAN ymm-panels per row. */
            #pragma GCC unroll 8
            for (std::ptrdiff_t k = 0; k < MR; ++k) {
                #pragma GCC unroll 8
                for (std::ptrdiff_t n = 0; n < NR_PAN; ++n) {
                    const std::ptrdiff_t panel_j0 = j0 + n * NR_LANE;
                    const std::ptrdiff_t panel_eff = (w_eff - n * NR_LANE < NR_LANE)
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
            for (std::ptrdiff_t n = 0; n < NR_PAN; ++n) {
                const std::ptrdiff_t panel_j0 = j0 + n * NR_LANE;
                const std::ptrdiff_t panel_eff = (w_eff - n * NR_LANE < NR_LANE)
                    ? (w_eff - n * NR_LANE) : NR_LANE;
                if (panel_eff <= 0) continue;
                __m256d acc_h = _mm256_setzero_pd();
                __m256d acc_l = _mm256_setzero_pd();
                for (std::ptrdiff_t p = 0; p < pb; ++p) {
                    const TR &aval = Ap[static_cast<std::size_t>(p) * ib + i];
                    __m256d ah = _mm256_set1_pd(aval.limbs[0]);
                    __m256d al = _mm256_set1_pd(aval.limbs[1]);
                    __m256d bh = _mm256_loadu_pd(&Bp_h_panel[p * W + n * NR_LANE]);
                    __m256d bl = _mm256_loadu_pd(&Bp_l_panel[p * W + n * NR_LANE]);
                    __m256d rh, rl;
                    simd_fast::mul(ah, al, bh, bl, rh, rl);
                    simd_fast::add(acc_h, acc_l, rh, rl, acc_h, acc_l);
                }
                simd_writeback(alpha_h, alpha_l, acc_h, acc_l,
                               &C[i], ldc, panel_j0, panel_eff);
            }
        }
    }
}

void mgemm_inner_kernel_simd(std::ptrdiff_t ib, std::ptrdiff_t jb, std::ptrdiff_t pb, TR alpha,
                             const TR * __restrict__ Ap,
                             const double * __restrict__ Bp_hi,
                             const double * __restrict__ Bp_lo,
                             TR * __restrict__ C, std::ptrdiff_t ldc)
{
    inner_kernel_simd_t<MGEMM_SIMD_MR, MGEMM_SIMD_NR_PAN>(
        ib, jb, pb, alpha, Ap, Bp_hi, Bp_lo, C, ldc);
}

#pragma GCC pop_options

#endif /* MBLAS_SIMD_DD */

extern "C" void mgemm_serial(
    char transa, char transb,
    std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
    const TR *alpha_,
    const TR *a, std::ptrdiff_t lda,
    const TR *b, std::ptrdiff_t ldb,
    const TR *beta_,
    TR *c, std::ptrdiff_t ldc)
{
    const TR alpha = *alpha_, beta = *beta_;
    const char ta = blas_trans_real(transa);
    const char tb = blas_trans_real(transb);

    if (m <= 0 || n <= 0) return;

    for (std::ptrdiff_t j = 0; j < n; ++j) {
        TR *cj = &c[static_cast<std::size_t>(j) * ldc];
        if (beta == zero_dd) {
            for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = zero_dd;
        } else if (beta != one_dd) {
            for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = cj[i] * beta;
        }
    }
    if (alpha == zero_dd || k == 0) return;

    std::ptrdiff_t MC, KC, NC;
    mgemm_choose_blocks(&MC, &KC, &NC);

    TR *Ap = static_cast<TR *>(std::aligned_alloc(
        64, static_cast<std::size_t>(MC) * KC * sizeof(TR)));
#ifdef MBLAS_SIMD_DD
    /* AVX2/FMA SIMD path at runtime on Haswell+; scalar fallback below is always
     * compiled and taken on pre-Haswell CPUs. See mf_dispatch.h. */
    if (mf_have_avx2_fma()) {
        const std::ptrdiff_t W_simd = mgemm_simd_pack_W();
        const std::ptrdiff_t NC_pad = ((NC + W_simd - 1) / W_simd) * W_simd;
        double *Bp_hi = static_cast<double *>(std::aligned_alloc(
            64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
        double *Bp_lo = static_cast<double *>(std::aligned_alloc(
            64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
        if (Ap && Bp_hi && Bp_lo) {
            for (std::ptrdiff_t jc = 0; jc < n; jc += NC) {
                const std::ptrdiff_t jb = (n - jc < NC) ? (n - jc) : NC;
                for (std::ptrdiff_t pc = 0; pc < k; pc += KC) {
                    const std::ptrdiff_t pb = (k - pc < KC) ? (k - pc) : KC;
                    mgemm_pack_B_soa(b, ldb, pc, jc, pb, jb, tb, Bp_hi, Bp_lo);
                    for (std::ptrdiff_t ic = 0; ic < m; ic += MC) {
                        const std::ptrdiff_t ib = (m - ic < MC) ? (m - ic) : MC;
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
        return;
    }
#endif /* MBLAS_SIMD_DD */
    TR *Bp = static_cast<TR *>(std::aligned_alloc(
        64, static_cast<std::size_t>(KC) * NC * sizeof(TR)));
    if (Ap && Bp) {
        for (std::ptrdiff_t jc = 0; jc < n; jc += NC) {
            const std::ptrdiff_t jb = (n - jc < NC) ? (n - jc) : NC;
            for (std::ptrdiff_t pc = 0; pc < k; pc += KC) {
                const std::ptrdiff_t pb = (k - pc < KC) ? (k - pc) : KC;
                mgemm_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
                for (std::ptrdiff_t ic = 0; ic < m; ic += MC) {
                    const std::ptrdiff_t ib = (m - ic < MC) ? (m - ic) : MC;
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
}
