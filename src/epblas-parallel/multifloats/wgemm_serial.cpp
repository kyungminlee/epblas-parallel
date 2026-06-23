/*
 * wgemm_serial.cpp — multifloats complex GEMM (complex64x2, complex double-
 * double), single-thread core. Owns ALL the numerics shared by the serial
 * and parallel entries: the trans decode, the block-size policy, the scalar
 * and AVX2-SIMD complex packers and micro-kernels (declared in
 * wgemm_kernel.h), plus the public `wgemm_serial` Fortran entry. No OpenMP
 * anywhere on this call path — safe to invoke from inside another routine's
 * parallel region.
 *
 * Math is bitwise-identical to the previous single wgemm.cpp: the parallel
 * entry (wgemm_parallel.cpp) drives these exact leaves over a jc-partitioned
 * team, so the two paths agree to the last bit.
 */

#include "wgemm_kernel.h"
#include "../common/blas_char.h"
#include "mf_kernels.h"
#include "mf_pred.h"
#include <cstdlib>
#include <cctype>

#ifdef WBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using TC = mf::complex64x2;

namespace {

std::ptrdiff_t g_mc = 0, g_kc = 0, g_nc = 0;
void init_blocks() {
    if (g_mc) return;
    g_mc =  64;
    g_kc = 128;
    g_nc = 256;
}

using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::cconj;
using mf_pred::ceq0;
using mf_pred::ceq1;

const TC zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };

}  // namespace

void wgemm_choose_blocks(std::ptrdiff_t *MC, std::ptrdiff_t *KC, std::ptrdiff_t *NC) {
    init_blocks();
    *MC = g_mc; *KC = g_kc; *NC = g_nc;
}

void wgemm_pack_A(const TC * __restrict__ A, std::ptrdiff_t lda,
                  std::ptrdiff_t ic, std::ptrdiff_t pc, std::ptrdiff_t ib, std::ptrdiff_t pb,
                  std::ptrdiff_t ta, TC * __restrict__ Ap)
{
    if (ta == 'N') {
        for (std::ptrdiff_t p = 0; p < pb; ++p) {
            const TC *src = &A[static_cast<std::size_t>(pc + p) * lda + ic];
            TC *dst = &Ap[static_cast<std::size_t>(p) * ib];
            for (std::ptrdiff_t i = 0; i < ib; ++i) dst[i] = src[i];
        }
    } else if (ta == 'T') {
        for (std::ptrdiff_t i = 0; i < ib; ++i) {
            const TC *src = &A[static_cast<std::size_t>(ic + i) * lda + pc];
            for (std::ptrdiff_t p = 0; p < pb; ++p)
                Ap[static_cast<std::size_t>(p) * ib + i] = src[p];
        }
    } else {  /* 'C' */
        for (std::ptrdiff_t i = 0; i < ib; ++i) {
            const TC *src = &A[static_cast<std::size_t>(ic + i) * lda + pc];
            for (std::ptrdiff_t p = 0; p < pb; ++p)
                Ap[static_cast<std::size_t>(p) * ib + i] = cconj(src[p]);
        }
    }
}

void wgemm_pack_B(const TC * __restrict__ B, std::ptrdiff_t ldb,
                  std::ptrdiff_t pc, std::ptrdiff_t jc, std::ptrdiff_t pb, std::ptrdiff_t jb,
                  std::ptrdiff_t tb, TC * __restrict__ Bp)
{
    if (tb == 'N') {
        for (std::ptrdiff_t j = 0; j < jb; ++j) {
            const TC *src = &B[static_cast<std::size_t>(jc + j) * ldb + pc];
            TC *dst = &Bp[static_cast<std::size_t>(j) * pb];
            for (std::ptrdiff_t p = 0; p < pb; ++p) dst[p] = src[p];
        }
    } else if (tb == 'T') {
        for (std::ptrdiff_t p = 0; p < pb; ++p) {
            const TC *src = &B[static_cast<std::size_t>(pc + p) * ldb + jc];
            for (std::ptrdiff_t j = 0; j < jb; ++j)
                Bp[static_cast<std::size_t>(j) * pb + p] = src[j];
        }
    } else {  /* 'C' */
        for (std::ptrdiff_t p = 0; p < pb; ++p) {
            const TC *src = &B[static_cast<std::size_t>(pc + p) * ldb + jc];
            for (std::ptrdiff_t j = 0; j < jb; ++j)
                Bp[static_cast<std::size_t>(j) * pb + p] = cconj(src[j]);
        }
    }
}

void wgemm_inner_kernel(std::ptrdiff_t ib, std::ptrdiff_t jb, std::ptrdiff_t pb, TC alpha,
                        const TC * __restrict__ Ap, const TC * __restrict__ Bp,
                        TC * __restrict__ C, std::ptrdiff_t ldc)
{
    for (std::ptrdiff_t j = 0; j < jb; ++j) {
        TC *cj = &C[static_cast<std::size_t>(j) * ldc];
        const TC *bj = &Bp[static_cast<std::size_t>(j) * pb];
        for (std::ptrdiff_t p = 0; p < pb; ++p) {
            const TC t = cmul(alpha, bj[p]);
            const TC *ap = &Ap[static_cast<std::size_t>(p) * ib];
            for (std::ptrdiff_t i = 0; i < ib; ++i) cj[i] = cadd(cj[i], cmul(t, ap[i]));
        }
    }
}

#ifdef WBLAS_SIMD_DD

#ifndef WGEMM_SIMD_MR
#define WGEMM_SIMD_MR 1
#endif
#ifndef WGEMM_SIMD_NR_PAN
#define WGEMM_SIMD_NR_PAN 1
#endif

/*
 * Complex SoA pack_B: produces 4 separate SoA arrays per panel,
 * one per real component (re_hi / re_lo / im_hi / im_lo). Conjugate
 * transpose 'C' negates the im_h / im_l limbs during pack.
 *
 * Panel width W = simd_fast::NR * WGEMM_SIMD_NR_PAN. Each array has
 * W elements per p iteration.
 */
constexpr std::ptrdiff_t wsimd_pack_W() { return simd_fast::NR * WGEMM_SIMD_NR_PAN; }

void wgemm_pack_B_soa_complex(const TC * __restrict__ B, std::ptrdiff_t ldb,
                              std::ptrdiff_t pc, std::ptrdiff_t jc, std::ptrdiff_t pb, std::ptrdiff_t jb, std::ptrdiff_t tb,
                              double * __restrict__ Bp_rh,
                              double * __restrict__ Bp_rl,
                              double * __restrict__ Bp_ih,
                              double * __restrict__ Bp_il)
{
    const std::ptrdiff_t W = wsimd_pack_W();
    const std::ptrdiff_t npanels = (jb + W - 1) / W;
    const bool conj = (tb == 'C');
    auto store_elem = [&](double *dst_rh, double *dst_rl,
                          double *dst_ih, double *dst_il,
                          std::size_t idx, const TC &v) {
        dst_rh[idx] = v.re.limbs[0];
        dst_rl[idx] = v.re.limbs[1];
        if (conj) {
            dst_ih[idx] = -v.im.limbs[0];
            dst_il[idx] = -v.im.limbs[1];
        } else {
            dst_ih[idx] = v.im.limbs[0];
            dst_il[idx] = v.im.limbs[1];
        }
    };
    auto store_zero = [&](double *dst_rh, double *dst_rl,
                          double *dst_ih, double *dst_il,
                          std::size_t idx) {
        dst_rh[idx] = 0.0;
        dst_rl[idx] = 0.0;
        dst_ih[idx] = 0.0;
        dst_il[idx] = 0.0;
    };
    for (std::ptrdiff_t panel = 0; panel < npanels; ++panel) {
        const std::ptrdiff_t j0 = panel * W;
        const std::ptrdiff_t w_eff = (jb - j0 < W) ? (jb - j0) : W;
        double *dst_rh = &Bp_rh[static_cast<std::size_t>(panel) * pb * W];
        double *dst_rl = &Bp_rl[static_cast<std::size_t>(panel) * pb * W];
        double *dst_ih = &Bp_ih[static_cast<std::size_t>(panel) * pb * W];
        double *dst_il = &Bp_il[static_cast<std::size_t>(panel) * pb * W];
        if (tb == 'N') {
            for (std::ptrdiff_t c = 0; c < w_eff; ++c) {
                const TC *col = &B[static_cast<std::size_t>(jc + j0 + c) * ldb + pc];
                for (std::ptrdiff_t p = 0; p < pb; ++p)
                    store_elem(dst_rh, dst_rl, dst_ih, dst_il, p * W + c, col[p]);
            }
            for (std::ptrdiff_t c = w_eff; c < W; ++c)
                for (std::ptrdiff_t p = 0; p < pb; ++p)
                    store_zero(dst_rh, dst_rl, dst_ih, dst_il, p * W + c);
        } else {
            /* 'T' or 'C' — for both, op(B)[p, j] = B[j, p].
             * 'C' additionally negates the imaginary part (handled in
             * store_elem via the `conj` flag set above). */
            for (std::ptrdiff_t p = 0; p < pb; ++p) {
                const TC *row = &B[static_cast<std::size_t>(pc + p) * ldb + (jc + j0)];
                for (std::ptrdiff_t c = 0; c < w_eff; ++c)
                    store_elem(dst_rh, dst_rl, dst_ih, dst_il, p * W + c, row[c]);
                for (std::ptrdiff_t c = w_eff; c < W; ++c)
                    store_zero(dst_rh, dst_rl, dst_ih, dst_il, p * W + c);
            }
        }
    }
}

std::ptrdiff_t wgemm_simd_pack_W(void) { return wsimd_pack_W(); }

/* Writeback one ymm-panel of complex DD accumulators into C. */
static inline __attribute__((always_inline)) void
simd_writeback_complex(__m256d alpha_rh, __m256d alpha_rl,
                       __m256d alpha_ih, __m256d alpha_il,
                       __m256d acc_rh, __m256d acc_rl,
                       __m256d acc_ih, __m256d acc_il,
                       TC *C_row_i, std::ptrdiff_t ldc, std::ptrdiff_t j0, std::ptrdiff_t nr_eff)
{
    constexpr std::ptrdiff_t NR = simd_fast::NR;
    __m256d ph_rh, ph_rl, ph_ih, ph_il;
    simd_fast::cmul(alpha_rh, alpha_rl, alpha_ih, alpha_il,
                     acc_rh, acc_rl, acc_ih, acc_il,
                     ph_rh, ph_rl, ph_ih, ph_il);
    alignas(32) double rh_a[NR], rl_a[NR], ih_a[NR], il_a[NR];
    _mm256_store_pd(rh_a, ph_rh);
    _mm256_store_pd(rl_a, ph_rl);
    _mm256_store_pd(ih_a, ph_ih);
    _mm256_store_pd(il_a, ph_il);
    for (std::ptrdiff_t j = 0; j < nr_eff; ++j) {
        TC r;
        r.re.limbs[0] = rh_a[j];
        r.re.limbs[1] = rl_a[j];
        r.im.limbs[0] = ih_a[j];
        r.im.limbs[1] = il_a[j];
        TC &dst = C_row_i[static_cast<std::size_t>(j0 + j) * ldc];
        dst = cadd(dst, r);
    }
}

template <std::ptrdiff_t MR, std::ptrdiff_t NR_PAN>
static __attribute__((noinline)) void
inner_kernel_simd_complex_t(std::ptrdiff_t ib, std::ptrdiff_t jb, std::ptrdiff_t pb, TC alpha,
                            const TC * __restrict__ Ap,
                            const double * __restrict__ Bp_rh,
                            const double * __restrict__ Bp_rl,
                            const double * __restrict__ Bp_ih,
                            const double * __restrict__ Bp_il,
                            TC * __restrict__ C, std::ptrdiff_t ldc)
{
    constexpr std::ptrdiff_t NR_LANE = simd_fast::NR;
    constexpr std::ptrdiff_t W = NR_LANE * NR_PAN;
    const std::ptrdiff_t j_panels = (jb + W - 1) / W;
    const __m256d alpha_rh = _mm256_set1_pd(alpha.re.limbs[0]);
    const __m256d alpha_rl = _mm256_set1_pd(alpha.re.limbs[1]);
    const __m256d alpha_ih = _mm256_set1_pd(alpha.im.limbs[0]);
    const __m256d alpha_il = _mm256_set1_pd(alpha.im.limbs[1]);
    for (std::ptrdiff_t jp = 0; jp < j_panels; ++jp) {
        const std::ptrdiff_t j0 = jp * W;
        const std::ptrdiff_t w_eff = (jb - j0 < W) ? (jb - j0) : W;
        const double *p_rh = &Bp_rh[static_cast<std::size_t>(jp) * pb * W];
        const double *p_rl = &Bp_rl[static_cast<std::size_t>(jp) * pb * W];
        const double *p_ih = &Bp_ih[static_cast<std::size_t>(jp) * pb * W];
        const double *p_il = &Bp_il[static_cast<std::size_t>(jp) * pb * W];
        std::ptrdiff_t i = 0;
        for (; i + MR <= ib; i += MR) {
            /* 4 ymm regs per acc cell: re_h, re_l, im_h, im_l */
            __m256d ac_rh[MR][NR_PAN], ac_rl[MR][NR_PAN];
            __m256d ac_ih[MR][NR_PAN], ac_il[MR][NR_PAN];
            #pragma GCC unroll 8
            for (std::ptrdiff_t k = 0; k < MR; ++k)
                #pragma GCC unroll 8
                for (std::ptrdiff_t n = 0; n < NR_PAN; ++n) {
                    ac_rh[k][n] = _mm256_setzero_pd();
                    ac_rl[k][n] = _mm256_setzero_pd();
                    ac_ih[k][n] = _mm256_setzero_pd();
                    ac_il[k][n] = _mm256_setzero_pd();
                }
            for (std::ptrdiff_t p = 0; p < pb; ++p) {
                __m256d brh[NR_PAN], brl[NR_PAN], bih[NR_PAN], bil[NR_PAN];
                #pragma GCC unroll 8
                for (std::ptrdiff_t n = 0; n < NR_PAN; ++n) {
                    brh[n] = _mm256_loadu_pd(&p_rh[p * W + n * NR_LANE]);
                    brl[n] = _mm256_loadu_pd(&p_rl[p * W + n * NR_LANE]);
                    bih[n] = _mm256_loadu_pd(&p_ih[p * W + n * NR_LANE]);
                    bil[n] = _mm256_loadu_pd(&p_il[p * W + n * NR_LANE]);
                }
                #pragma GCC unroll 8
                for (std::ptrdiff_t k = 0; k < MR; ++k) {
                    const TC &aval = Ap[static_cast<std::size_t>(p) * ib + i + k];
                    __m256d arh = _mm256_set1_pd(aval.re.limbs[0]);
                    __m256d arl = _mm256_set1_pd(aval.re.limbs[1]);
                    __m256d aih = _mm256_set1_pd(aval.im.limbs[0]);
                    __m256d ail = _mm256_set1_pd(aval.im.limbs[1]);
                    #pragma GCC unroll 8
                    for (std::ptrdiff_t n = 0; n < NR_PAN; ++n) {
                        __m256d r_rh, r_rl, r_ih, r_il;
                        simd_fast::cmul(arh, arl, aih, ail,
                                         brh[n], brl[n], bih[n], bil[n],
                                         r_rh, r_rl, r_ih, r_il);
                        simd_fast::cadd(ac_rh[k][n], ac_rl[k][n],
                                         ac_ih[k][n], ac_il[k][n],
                                         r_rh, r_rl, r_ih, r_il,
                                         ac_rh[k][n], ac_rl[k][n],
                                         ac_ih[k][n], ac_il[k][n]);
                    }
                }
            }
            /* Writeback NR_PAN panels per row */
            #pragma GCC unroll 8
            for (std::ptrdiff_t k = 0; k < MR; ++k)
                #pragma GCC unroll 8
                for (std::ptrdiff_t n = 0; n < NR_PAN; ++n) {
                    const std::ptrdiff_t panel_j0 = j0 + n * NR_LANE;
                    const std::ptrdiff_t panel_eff = (w_eff - n * NR_LANE < NR_LANE)
                        ? (w_eff - n * NR_LANE) : NR_LANE;
                    if (panel_eff > 0)
                        simd_writeback_complex(
                            alpha_rh, alpha_rl, alpha_ih, alpha_il,
                            ac_rh[k][n], ac_rl[k][n],
                            ac_ih[k][n], ac_il[k][n],
                            &C[i + k], ldc, panel_j0, panel_eff);
                }
        }
        /* MR=1 tail */
        for (; i < ib; ++i) {
            #pragma GCC unroll 8
            for (std::ptrdiff_t n = 0; n < NR_PAN; ++n) {
                const std::ptrdiff_t panel_j0 = j0 + n * NR_LANE;
                const std::ptrdiff_t panel_eff = (w_eff - n * NR_LANE < NR_LANE)
                    ? (w_eff - n * NR_LANE) : NR_LANE;
                if (panel_eff <= 0) continue;
                __m256d ac_rh = _mm256_setzero_pd();
                __m256d ac_rl = _mm256_setzero_pd();
                __m256d ac_ih = _mm256_setzero_pd();
                __m256d ac_il = _mm256_setzero_pd();
                for (std::ptrdiff_t p = 0; p < pb; ++p) {
                    const TC &aval = Ap[static_cast<std::size_t>(p) * ib + i];
                    __m256d arh = _mm256_set1_pd(aval.re.limbs[0]);
                    __m256d arl = _mm256_set1_pd(aval.re.limbs[1]);
                    __m256d aih = _mm256_set1_pd(aval.im.limbs[0]);
                    __m256d ail = _mm256_set1_pd(aval.im.limbs[1]);
                    __m256d brh = _mm256_loadu_pd(&p_rh[p * W + n * NR_LANE]);
                    __m256d brl = _mm256_loadu_pd(&p_rl[p * W + n * NR_LANE]);
                    __m256d bih = _mm256_loadu_pd(&p_ih[p * W + n * NR_LANE]);
                    __m256d bil = _mm256_loadu_pd(&p_il[p * W + n * NR_LANE]);
                    __m256d r_rh, r_rl, r_ih, r_il;
                    simd_fast::cmul(arh, arl, aih, ail, brh, brl, bih, bil,
                                     r_rh, r_rl, r_ih, r_il);
                    simd_fast::cadd(ac_rh, ac_rl, ac_ih, ac_il,
                                     r_rh, r_rl, r_ih, r_il,
                                     ac_rh, ac_rl, ac_ih, ac_il);
                }
                simd_writeback_complex(
                    alpha_rh, alpha_rl, alpha_ih, alpha_il,
                    ac_rh, ac_rl, ac_ih, ac_il,
                    &C[i], ldc, panel_j0, panel_eff);
            }
        }
    }
}

void wgemm_inner_kernel_simd_complex(std::ptrdiff_t ib, std::ptrdiff_t jb, std::ptrdiff_t pb, TC alpha,
                                     const TC * __restrict__ Ap,
                                     const double * __restrict__ Bp_rh,
                                     const double * __restrict__ Bp_rl,
                                     const double * __restrict__ Bp_ih,
                                     const double * __restrict__ Bp_il,
                                     TC * __restrict__ C, std::ptrdiff_t ldc)
{
    inner_kernel_simd_complex_t<WGEMM_SIMD_MR, WGEMM_SIMD_NR_PAN>(
        ib, jb, pb, alpha, Ap, Bp_rh, Bp_rl, Bp_ih, Bp_il, C, ldc);
}

#endif /* WBLAS_SIMD_DD */

extern "C" void wgemm_serial(
    char transa, char transb,
    std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
    const TC *alpha_,
    const TC *a, std::ptrdiff_t lda,
    const TC *b, std::ptrdiff_t ldb,
    const TC *beta_,
    TC *c, std::ptrdiff_t ldc)
{
    const TC alpha = *alpha_, beta = *beta_;
    const char ta = blas_trans_complex(transa);
    const char tb = blas_trans_complex(transb);

    if (m <= 0 || n <= 0) return;

    for (std::ptrdiff_t j = 0; j < n; ++j) {
        TC *cj = &c[static_cast<std::size_t>(j) * ldc];
        if (ceq0(beta)) {
            for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = zero_cdd;
        } else if (!ceq1(beta)) {
            for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = cmul(cj[i], beta);
        }
    }
    if (ceq0(alpha) || k == 0) return;

    std::ptrdiff_t MC, KC, NC;
    wgemm_choose_blocks(&MC, &KC, &NC);

    TC *Ap = static_cast<TC *>(std::aligned_alloc(
        64, static_cast<std::size_t>(MC) * KC * sizeof(TC)));
#ifdef WBLAS_SIMD_DD
    const std::ptrdiff_t W_simd = wgemm_simd_pack_W();
    const std::ptrdiff_t NC_pad = ((NC + W_simd - 1) / W_simd) * W_simd;
    double *Bp_rh = static_cast<double *>(std::aligned_alloc(
        64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
    double *Bp_rl = static_cast<double *>(std::aligned_alloc(
        64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
    double *Bp_ih = static_cast<double *>(std::aligned_alloc(
        64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
    double *Bp_il = static_cast<double *>(std::aligned_alloc(
        64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
    if (Ap && Bp_rh && Bp_rl && Bp_ih && Bp_il) {
        for (std::ptrdiff_t jc = 0; jc < n; jc += NC) {
            const std::ptrdiff_t jb = (n - jc < NC) ? (n - jc) : NC;
            for (std::ptrdiff_t pc = 0; pc < k; pc += KC) {
                const std::ptrdiff_t pb = (k - pc < KC) ? (k - pc) : KC;
                wgemm_pack_B_soa_complex(b, ldb, pc, jc, pb, jb, tb,
                                         Bp_rh, Bp_rl, Bp_ih, Bp_il);
                for (std::ptrdiff_t ic = 0; ic < m; ic += MC) {
                    const std::ptrdiff_t ib = (m - ic < MC) ? (m - ic) : MC;
                    wgemm_pack_A(a, lda, ic, pc, ib, pb, ta, Ap);
                    wgemm_inner_kernel_simd_complex(ib, jb, pb, alpha, Ap,
                                                    Bp_rh, Bp_rl, Bp_ih, Bp_il,
                                                    &c[static_cast<std::size_t>(jc) * ldc + ic],
                                                    ldc);
                }
            }
        }
    }
    std::free(Ap);
    std::free(Bp_rh);
    std::free(Bp_rl);
    std::free(Bp_ih);
    std::free(Bp_il);
#else
    TC *Bp = static_cast<TC *>(std::aligned_alloc(
        64, static_cast<std::size_t>(KC) * NC * sizeof(TC)));
    if (Ap && Bp) {
        for (std::ptrdiff_t jc = 0; jc < n; jc += NC) {
            const std::ptrdiff_t jb = (n - jc < NC) ? (n - jc) : NC;
            for (std::ptrdiff_t pc = 0; pc < k; pc += KC) {
                const std::ptrdiff_t pb = (k - pc < KC) ? (k - pc) : KC;
                wgemm_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
                for (std::ptrdiff_t ic = 0; ic < m; ic += MC) {
                    const std::ptrdiff_t ib = (m - ic < MC) ? (m - ic) : MC;
                    wgemm_pack_A(a, lda, ic, pc, ib, pb, ta, Ap);
                    wgemm_inner_kernel(ib, jb, pb, alpha, Ap, Bp,
                                       &c[static_cast<std::size_t>(jc) * ldc + ic],
                                       ldc);
                }
            }
        }
    }
    std::free(Ap);
    std::free(Bp);
#endif /* WBLAS_SIMD_DD */
}
