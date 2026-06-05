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
#include <cstdlib>
#include <cctype>

#ifdef WBLAS_SIMD_DD
#include "mgemm_simd_kernel.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {

int env_int(const char *name, int dflt) {
    const char *s = std::getenv(name);
    if (!s || !*s) return dflt;
    int v = std::atoi(s);
    return v > 0 ? v : dflt;
}

int g_mc = 0, g_kc = 0, g_nc = 0;
void init_blocks() {
    if (g_mc) return;
    g_mc = env_int("MBLAS_MC",  64);
    g_kc = env_int("MBLAS_KC", 128);
    g_nc = env_int("MBLAS_NC", 256);
}

inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im,
              a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) {
    return T{ a.re + b.re, a.im + b.im };
}
inline T cconj(T const &a) { return T{ a.re, -a.im }; }

inline bool ciszero(T const &a) {
    return a.re.limbs[0] == 0.0 && a.re.limbs[1] == 0.0
        && a.im.limbs[0] == 0.0 && a.im.limbs[1] == 0.0;
}
inline bool cisone(T const &a) {
    return a.re.limbs[0] == 1.0 && a.re.limbs[1] == 0.0
        && a.im.limbs[0] == 0.0 && a.im.limbs[1] == 0.0;
}

const T zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };

}  // namespace

int wgemm_trans_code(const char *p, std::size_t /*len*/) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}

void wgemm_choose_blocks(int *MC, int *KC, int *NC) {
    init_blocks();
    *MC = g_mc; *KC = g_kc; *NC = g_nc;
}

void wgemm_pack_A(const T * __restrict__ A, int lda,
                  int ic, int pc, int ib, int pb,
                  int ta, T * __restrict__ Ap)
{
    if (ta == 'N') {
        for (int p = 0; p < pb; ++p) {
            const T *src = &A[static_cast<std::size_t>(pc + p) * lda + ic];
            T *dst = &Ap[static_cast<std::size_t>(p) * ib];
            for (int i = 0; i < ib; ++i) dst[i] = src[i];
        }
    } else if (ta == 'T') {
        for (int i = 0; i < ib; ++i) {
            const T *src = &A[static_cast<std::size_t>(ic + i) * lda + pc];
            for (int p = 0; p < pb; ++p)
                Ap[static_cast<std::size_t>(p) * ib + i] = src[p];
        }
    } else {  /* 'C' */
        for (int i = 0; i < ib; ++i) {
            const T *src = &A[static_cast<std::size_t>(ic + i) * lda + pc];
            for (int p = 0; p < pb; ++p)
                Ap[static_cast<std::size_t>(p) * ib + i] = cconj(src[p]);
        }
    }
}

void wgemm_pack_B(const T * __restrict__ B, int ldb,
                  int pc, int jc, int pb, int jb,
                  int tb, T * __restrict__ Bp)
{
    if (tb == 'N') {
        for (int j = 0; j < jb; ++j) {
            const T *src = &B[static_cast<std::size_t>(jc + j) * ldb + pc];
            T *dst = &Bp[static_cast<std::size_t>(j) * pb];
            for (int p = 0; p < pb; ++p) dst[p] = src[p];
        }
    } else if (tb == 'T') {
        for (int p = 0; p < pb; ++p) {
            const T *src = &B[static_cast<std::size_t>(pc + p) * ldb + jc];
            for (int j = 0; j < jb; ++j)
                Bp[static_cast<std::size_t>(j) * pb + p] = src[j];
        }
    } else {  /* 'C' */
        for (int p = 0; p < pb; ++p) {
            const T *src = &B[static_cast<std::size_t>(pc + p) * ldb + jc];
            for (int j = 0; j < jb; ++j)
                Bp[static_cast<std::size_t>(j) * pb + p] = cconj(src[j]);
        }
    }
}

void wgemm_inner_kernel(int ib, int jb, int pb, T alpha,
                        const T * __restrict__ Ap, const T * __restrict__ Bp,
                        T * __restrict__ C, int ldc)
{
    for (int j = 0; j < jb; ++j) {
        T *cj = &C[static_cast<std::size_t>(j) * ldc];
        const T *bj = &Bp[static_cast<std::size_t>(j) * pb];
        for (int p = 0; p < pb; ++p) {
            const T t = cmul(alpha, bj[p]);
            const T *ap = &Ap[static_cast<std::size_t>(p) * ib];
            for (int i = 0; i < ib; ++i) cj[i] = cadd(cj[i], cmul(t, ap[i]));
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
 * Panel width W = simd_dd::NR * WGEMM_SIMD_NR_PAN. Each array has
 * W elements per p iteration.
 */
constexpr int wsimd_pack_W() { return simd_dd::NR * WGEMM_SIMD_NR_PAN; }

void wgemm_pack_B_soa_complex(const T * __restrict__ B, int ldb,
                              int pc, int jc, int pb, int jb, int tb,
                              double * __restrict__ Bp_rh,
                              double * __restrict__ Bp_rl,
                              double * __restrict__ Bp_ih,
                              double * __restrict__ Bp_il)
{
    const int W = wsimd_pack_W();
    const int npanels = (jb + W - 1) / W;
    const bool conj = (tb == 'C');
    auto store_elem = [&](double *dst_rh, double *dst_rl,
                          double *dst_ih, double *dst_il,
                          std::size_t idx, const T &v) {
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
    for (int panel = 0; panel < npanels; ++panel) {
        const int j0 = panel * W;
        const int w_eff = (jb - j0 < W) ? (jb - j0) : W;
        double *dst_rh = &Bp_rh[static_cast<std::size_t>(panel) * pb * W];
        double *dst_rl = &Bp_rl[static_cast<std::size_t>(panel) * pb * W];
        double *dst_ih = &Bp_ih[static_cast<std::size_t>(panel) * pb * W];
        double *dst_il = &Bp_il[static_cast<std::size_t>(panel) * pb * W];
        if (tb == 'N') {
            for (int c = 0; c < w_eff; ++c) {
                const T *col = &B[static_cast<std::size_t>(jc + j0 + c) * ldb + pc];
                for (int p = 0; p < pb; ++p)
                    store_elem(dst_rh, dst_rl, dst_ih, dst_il, p * W + c, col[p]);
            }
            for (int c = w_eff; c < W; ++c)
                for (int p = 0; p < pb; ++p)
                    store_zero(dst_rh, dst_rl, dst_ih, dst_il, p * W + c);
        } else {
            /* 'T' or 'C' — for both, op(B)[p, j] = B[j, p].
             * 'C' additionally negates the imaginary part (handled in
             * store_elem via the `conj` flag set above). */
            for (int p = 0; p < pb; ++p) {
                const T *row = &B[static_cast<std::size_t>(pc + p) * ldb + (jc + j0)];
                for (int c = 0; c < w_eff; ++c)
                    store_elem(dst_rh, dst_rl, dst_ih, dst_il, p * W + c, row[c]);
                for (int c = w_eff; c < W; ++c)
                    store_zero(dst_rh, dst_rl, dst_ih, dst_il, p * W + c);
            }
        }
    }
}

int wgemm_simd_pack_W(void) { return wsimd_pack_W(); }

/* Writeback one ymm-panel of complex DD accumulators into C. */
static inline __attribute__((always_inline)) void
simd_writeback_complex(__m256d alpha_rh, __m256d alpha_rl,
                       __m256d alpha_ih, __m256d alpha_il,
                       __m256d acc_rh, __m256d acc_rl,
                       __m256d acc_ih, __m256d acc_il,
                       T *C_row_i, int ldc, int j0, int nr_eff)
{
    constexpr int NR = simd_dd::NR;
    __m256d ph_rh, ph_rl, ph_ih, ph_il;
    simd_dd::cdd_mul(alpha_rh, alpha_rl, alpha_ih, alpha_il,
                     acc_rh, acc_rl, acc_ih, acc_il,
                     ph_rh, ph_rl, ph_ih, ph_il);
    alignas(32) double rh_a[NR], rl_a[NR], ih_a[NR], il_a[NR];
    _mm256_store_pd(rh_a, ph_rh);
    _mm256_store_pd(rl_a, ph_rl);
    _mm256_store_pd(ih_a, ph_ih);
    _mm256_store_pd(il_a, ph_il);
    for (int j = 0; j < nr_eff; ++j) {
        T r;
        r.re.limbs[0] = rh_a[j];
        r.re.limbs[1] = rl_a[j];
        r.im.limbs[0] = ih_a[j];
        r.im.limbs[1] = il_a[j];
        T &dst = C_row_i[static_cast<std::size_t>(j0 + j) * ldc];
        dst = cadd(dst, r);
    }
}

template <int MR, int NR_PAN>
static __attribute__((noinline)) void
inner_kernel_simd_complex_t(int ib, int jb, int pb, T alpha,
                            const T * __restrict__ Ap,
                            const double * __restrict__ Bp_rh,
                            const double * __restrict__ Bp_rl,
                            const double * __restrict__ Bp_ih,
                            const double * __restrict__ Bp_il,
                            T * __restrict__ C, int ldc)
{
    constexpr int NR_LANE = simd_dd::NR;
    constexpr int W = NR_LANE * NR_PAN;
    const int j_panels = (jb + W - 1) / W;
    const __m256d alpha_rh = _mm256_set1_pd(alpha.re.limbs[0]);
    const __m256d alpha_rl = _mm256_set1_pd(alpha.re.limbs[1]);
    const __m256d alpha_ih = _mm256_set1_pd(alpha.im.limbs[0]);
    const __m256d alpha_il = _mm256_set1_pd(alpha.im.limbs[1]);
    for (int jp = 0; jp < j_panels; ++jp) {
        const int j0 = jp * W;
        const int w_eff = (jb - j0 < W) ? (jb - j0) : W;
        const double *p_rh = &Bp_rh[static_cast<std::size_t>(jp) * pb * W];
        const double *p_rl = &Bp_rl[static_cast<std::size_t>(jp) * pb * W];
        const double *p_ih = &Bp_ih[static_cast<std::size_t>(jp) * pb * W];
        const double *p_il = &Bp_il[static_cast<std::size_t>(jp) * pb * W];
        int i = 0;
        for (; i + MR <= ib; i += MR) {
            /* 4 ymm regs per acc cell: re_h, re_l, im_h, im_l */
            __m256d ac_rh[MR][NR_PAN], ac_rl[MR][NR_PAN];
            __m256d ac_ih[MR][NR_PAN], ac_il[MR][NR_PAN];
            #pragma GCC unroll 8
            for (int k = 0; k < MR; ++k)
                #pragma GCC unroll 8
                for (int n = 0; n < NR_PAN; ++n) {
                    ac_rh[k][n] = _mm256_setzero_pd();
                    ac_rl[k][n] = _mm256_setzero_pd();
                    ac_ih[k][n] = _mm256_setzero_pd();
                    ac_il[k][n] = _mm256_setzero_pd();
                }
            for (int p = 0; p < pb; ++p) {
                __m256d brh[NR_PAN], brl[NR_PAN], bih[NR_PAN], bil[NR_PAN];
                #pragma GCC unroll 8
                for (int n = 0; n < NR_PAN; ++n) {
                    brh[n] = _mm256_loadu_pd(&p_rh[p * W + n * NR_LANE]);
                    brl[n] = _mm256_loadu_pd(&p_rl[p * W + n * NR_LANE]);
                    bih[n] = _mm256_loadu_pd(&p_ih[p * W + n * NR_LANE]);
                    bil[n] = _mm256_loadu_pd(&p_il[p * W + n * NR_LANE]);
                }
                #pragma GCC unroll 8
                for (int k = 0; k < MR; ++k) {
                    const T &aval = Ap[static_cast<std::size_t>(p) * ib + i + k];
                    __m256d arh = _mm256_set1_pd(aval.re.limbs[0]);
                    __m256d arl = _mm256_set1_pd(aval.re.limbs[1]);
                    __m256d aih = _mm256_set1_pd(aval.im.limbs[0]);
                    __m256d ail = _mm256_set1_pd(aval.im.limbs[1]);
                    #pragma GCC unroll 8
                    for (int n = 0; n < NR_PAN; ++n) {
                        __m256d r_rh, r_rl, r_ih, r_il;
                        simd_dd::cdd_mul(arh, arl, aih, ail,
                                         brh[n], brl[n], bih[n], bil[n],
                                         r_rh, r_rl, r_ih, r_il);
                        simd_dd::cdd_add(ac_rh[k][n], ac_rl[k][n],
                                         ac_ih[k][n], ac_il[k][n],
                                         r_rh, r_rl, r_ih, r_il,
                                         ac_rh[k][n], ac_rl[k][n],
                                         ac_ih[k][n], ac_il[k][n]);
                    }
                }
            }
            /* Writeback NR_PAN panels per row */
            #pragma GCC unroll 8
            for (int k = 0; k < MR; ++k)
                #pragma GCC unroll 8
                for (int n = 0; n < NR_PAN; ++n) {
                    const int panel_j0 = j0 + n * NR_LANE;
                    const int panel_eff = (w_eff - n * NR_LANE < NR_LANE)
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
            for (int n = 0; n < NR_PAN; ++n) {
                const int panel_j0 = j0 + n * NR_LANE;
                const int panel_eff = (w_eff - n * NR_LANE < NR_LANE)
                    ? (w_eff - n * NR_LANE) : NR_LANE;
                if (panel_eff <= 0) continue;
                __m256d ac_rh = _mm256_setzero_pd();
                __m256d ac_rl = _mm256_setzero_pd();
                __m256d ac_ih = _mm256_setzero_pd();
                __m256d ac_il = _mm256_setzero_pd();
                for (int p = 0; p < pb; ++p) {
                    const T &aval = Ap[static_cast<std::size_t>(p) * ib + i];
                    __m256d arh = _mm256_set1_pd(aval.re.limbs[0]);
                    __m256d arl = _mm256_set1_pd(aval.re.limbs[1]);
                    __m256d aih = _mm256_set1_pd(aval.im.limbs[0]);
                    __m256d ail = _mm256_set1_pd(aval.im.limbs[1]);
                    __m256d brh = _mm256_loadu_pd(&p_rh[p * W + n * NR_LANE]);
                    __m256d brl = _mm256_loadu_pd(&p_rl[p * W + n * NR_LANE]);
                    __m256d bih = _mm256_loadu_pd(&p_ih[p * W + n * NR_LANE]);
                    __m256d bil = _mm256_loadu_pd(&p_il[p * W + n * NR_LANE]);
                    __m256d r_rh, r_rl, r_ih, r_il;
                    simd_dd::cdd_mul(arh, arl, aih, ail, brh, brl, bih, bil,
                                     r_rh, r_rl, r_ih, r_il);
                    simd_dd::cdd_add(ac_rh, ac_rl, ac_ih, ac_il,
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

void wgemm_inner_kernel_simd_complex(int ib, int jb, int pb, T alpha,
                                     const T * __restrict__ Ap,
                                     const double * __restrict__ Bp_rh,
                                     const double * __restrict__ Bp_rl,
                                     const double * __restrict__ Bp_ih,
                                     const double * __restrict__ Bp_il,
                                     T * __restrict__ C, int ldc)
{
    inner_kernel_simd_complex_t<WGEMM_SIMD_MR, WGEMM_SIMD_NR_PAN>(
        ib, jb, pb, alpha, Ap, Bp_rh, Bp_rl, Bp_ih, Bp_il, C, ldc);
}

#endif /* WBLAS_SIMD_DD */

extern "C" void wgemm_serial(
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
    const int ta = wgemm_trans_code(transa, transa_len);
    const int tb = wgemm_trans_code(transb, transb_len);

    if (M <= 0 || N <= 0) return;

    for (int j = 0; j < N; ++j) {
        T *cj = &c[static_cast<std::size_t>(j) * ldc];
        if (ciszero(beta)) {
            for (int i = 0; i < M; ++i) cj[i] = zero_cdd;
        } else if (!cisone(beta)) {
            for (int i = 0; i < M; ++i) cj[i] = cmul(cj[i], beta);
        }
    }
    if (ciszero(alpha) || K == 0) return;

    int MC, KC, NC;
    wgemm_choose_blocks(&MC, &KC, &NC);

    T *Ap = static_cast<T *>(std::aligned_alloc(
        64, static_cast<std::size_t>(MC) * KC * sizeof(T)));
#ifdef WBLAS_SIMD_DD
    const int W_simd = wgemm_simd_pack_W();
    const int NC_pad = ((NC + W_simd - 1) / W_simd) * W_simd;
    double *Bp_rh = static_cast<double *>(std::aligned_alloc(
        64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
    double *Bp_rl = static_cast<double *>(std::aligned_alloc(
        64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
    double *Bp_ih = static_cast<double *>(std::aligned_alloc(
        64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
    double *Bp_il = static_cast<double *>(std::aligned_alloc(
        64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
    if (Ap && Bp_rh && Bp_rl && Bp_ih && Bp_il) {
        for (int jc = 0; jc < N; jc += NC) {
            const int jb = (N - jc < NC) ? (N - jc) : NC;
            for (int pc = 0; pc < K; pc += KC) {
                const int pb = (K - pc < KC) ? (K - pc) : KC;
                wgemm_pack_B_soa_complex(b, ldb, pc, jc, pb, jb, tb,
                                         Bp_rh, Bp_rl, Bp_ih, Bp_il);
                for (int ic = 0; ic < M; ic += MC) {
                    const int ib = (M - ic < MC) ? (M - ic) : MC;
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
    T *Bp = static_cast<T *>(std::aligned_alloc(
        64, static_cast<std::size_t>(KC) * NC * sizeof(T)));
    if (Ap && Bp) {
        for (int jc = 0; jc < N; jc += NC) {
            const int jb = (N - jc < NC) ? (N - jc) : NC;
            for (int pc = 0; pc < K; pc += KC) {
                const int pb = (K - pc < KC) ? (K - pc) : KC;
                wgemm_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
                for (int ic = 0; ic < M; ic += MC) {
                    const int ib = (M - ic < MC) ? (M - ic) : MC;
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
