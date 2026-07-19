/* wtrsv — multifloats complex DD triangular solve.
 * SIMD: pre-pack x to SoA scratch; per i, scalar divide then SIMD
 * inner loop using cmul/cadd. TRANS='C' applies neg to A.im
 * before cmul to implement conj(A). */

#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <vector>
#include <multifloats.h>
#include "mf_kernels.h"
#include "mf_util.h"
#include "mf_pred.h"
#include "mf_dispatch.h"   /* MF_SIMD_TARGET + mf_have_avx2_fma() runtime gate */
#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define WTRSV_OMP_MIN 256   /* below this, run the serial SIMD path */
#define WTRSV_BLK     128   /* diagonal-block size for the blocked solve */
#define WTRSV_MAX_CPUS 256
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using R = mf::float64x2;
using TC = mf::complex64x2;


/* zero/one predicates — see mf_pred.h */
using mf_pred::ceq0;

using mf_util::up;  /* char flag uppercase — mf_util.h */
namespace {
using mf_pred::zero_cdd;   /* shared DD constants — mf_pred.h */
using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::csub;
using mf_kernels::cconj;
inline TC cdiv(TC const &a, TC const &b) {
    /* a / b = a·conj(b) / |b|², direct DD divide (canonical form shared with
     * wtbsv/wtpsv/wtrsm_serial). */
    const R denom = b.re * b.re + b.im * b.im;
    return TC{ (a.re * b.re + a.im * b.im) / denom,
              (a.im * b.re - a.re * b.im) / denom };
}

#ifdef MBLAS_SIMD_DD
/* AVX2+FMA under a possibly pre-Haswell baseline -march: these SIMD kernels are
 * compiled with the feature enabled and reached only behind mf_have_avx2_fma()
 * at the call sites below. See mf_dispatch.h. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")
using simd_exact::cload4;
using simd_exact::cstore4;
using simd_fast::chreduce;
/* Off-diagonal SIMD kernels for the blocked threaded solve.
 * msub: x[k] = csub(x[k], cmul(xi, ai[k]))  for k in [lo,hi)  (NoTrans).
 * dot : returns sum_{k in [lo,hi)} (conj?conj(ai[k]):ai[k]) * x[k]  (Trans). */
static inline void
wtrsv_col_msub_simd(TC *x, const TC *ai, TC xi, std::ptrdiff_t lo, std::ptrdiff_t hi)
{
    double *xp = reinterpret_cast<double *>(x);
    const double *aip = reinterpret_cast<const double *>(ai);
    const __m256d xrh = _mm256_set1_pd(xi.re.limbs[0]);
    const __m256d xrl = _mm256_set1_pd(xi.re.limbs[1]);
    const __m256d xih = _mm256_set1_pd(xi.im.limbs[0]);
    const __m256d xil = _mm256_set1_pd(xi.im.limbs[1]);
    std::ptrdiff_t k = lo;
    for (; k + 4 <= hi; k += 4) {
        __m256d arh, arl, aih, ail;
        cload4(aip + 4 * k, arh, arl, aih, ail);
        __m256d prh, prl, pih, pil;
        simd_fast::cmul(xrh, xrl, xih, xil, arh, arl, aih, ail,
                         prh, prl, pih, pil);
        simd_fast::neg(prh, prl);
        simd_fast::neg(pih, pil);
        __m256d crh, crl, cih, cil;
        cload4(xp + 4 * k, crh, crl, cih, cil);
        __m256d rrh, rrl, rih, ril;
        simd_fast::cadd(crh, crl, cih, cil, prh, prl, pih, pil,
                         rrh, rrl, rih, ril);
        cstore4(xp + 4 * k, rrh, rrl, rih, ril);
    }
    for (; k < hi; ++k) x[k] = csub(x[k], cmul(xi, ai[k]));
}
static inline TC
wtrsv_dot_range_simd(const TC *ai, const TC *x, std::ptrdiff_t lo, std::ptrdiff_t hi, bool conj_a)
{
    const double *aip = reinterpret_cast<const double *>(ai);
    const double *xp  = reinterpret_cast<const double *>(x);
    __m256d srh = _mm256_setzero_pd(), srl = _mm256_setzero_pd();
    __m256d sih = _mm256_setzero_pd(), sil = _mm256_setzero_pd();
    std::ptrdiff_t k = lo;
    for (; k + 4 <= hi; k += 4) {
        __m256d arh, arl, aih, ail;
        cload4(aip + 4 * k, arh, arl, aih, ail);
        if (conj_a) simd_fast::neg(aih, ail);
        __m256d xrh, xrl, xih, xil;
        cload4(xp + 4 * k, xrh, xrl, xih, xil);
        __m256d prh, prl, pih, pil;
        simd_fast::cmul(arh, arl, aih, ail, xrh, xrl, xih, xil,
                         prh, prl, pih, pil);
        simd_fast::cadd(srh, srl, sih, sil, prh, prl, pih, pil,
                         srh, srl, sih, sil);
    }
    TC s = chreduce(srh, srl, sih, sil);
    for (; k < hi; ++k) {
        const TC e = conj_a ? cconj(ai[k]) : ai[k];
        s = cadd(s, cmul(e, x[k]));
    }
    return s;
}
#pragma GCC pop_options
#endif
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

#ifdef MBLAS_SIMD_DD
/* Packed-SoA AVX2/FMA incx==1 solve, extracted from wtrsv_serial so its
 * intrinsics compile under target("avx2,fma") on a pre-Haswell baseline -march.
 * Reached only behind mf_have_avx2_fma() from wtrsv_serial. See mf_dispatch.h. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")
static void wtrsv_serial_simd_unit(char UPLO, char TRANS, bool nounit,
                         std::ptrdiff_t n, const TC *a, std::ptrdiff_t lda, TC *x)
{
        const std::size_t N_pad = (static_cast<std::size_t>(n) + 3) & ~static_cast<std::size_t>(3);
        double *x_rh = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        double *x_rl = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        double *x_ih = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        double *x_il = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            x_rh[i] = x[i].re.limbs[0]; x_rl[i] = x[i].re.limbs[1];
            x_ih[i] = x[i].im.limbs[0]; x_il[i] = x[i].im.limbs[1];
        }
        for (std::size_t i = static_cast<std::size_t>(n); i < N_pad; ++i) {
            x_rh[i] = 0.0; x_rl[i] = 0.0; x_ih[i] = 0.0; x_il[i] = 0.0;
        }
        const __m256d zerov = _mm256_setzero_pd();

        auto load_x = [&](std::ptrdiff_t k) -> TC {
            return TC{ R{x_rh[k], x_rl[k]}, R{x_ih[k], x_il[k]} };
        };
        auto store_x = [&](std::ptrdiff_t k, const TC &v) {
            x_rh[k] = v.re.limbs[0]; x_rl[k] = v.re.limbs[1];
            x_ih[k] = v.im.limbs[0]; x_il[k] = v.im.limbs[1];
        };

        if (TRANS == 'N') {
            auto do_axpy_range = [&](std::ptrdiff_t i, std::ptrdiff_t k_lo, std::ptrdiff_t k_hi) {
                TC xi = load_x(i);
                if (ceq0(xi)) return;
                if (nounit) { xi = cdiv(xi, A_(i, i)); store_x(i, xi); }
                const __m256d xrh = _mm256_set1_pd(xi.re.limbs[0]);
                const __m256d xrl = _mm256_set1_pd(xi.re.limbs[1]);
                const __m256d xih = _mm256_set1_pd(xi.im.limbs[0]);
                const __m256d xil = _mm256_set1_pd(xi.im.limbs[1]);
                const double *aip = reinterpret_cast<const double *>(&A_(0, i));
                std::ptrdiff_t k = k_lo;
                for (; k < k_hi && (k & 3) != 0; ++k) {
                    TC aki{ R{aip[4*k], aip[4*k+1]}, R{aip[4*k+2], aip[4*k+3]} };
                    store_x(k, csub(load_x(k), cmul(xi, aki)));
                }
                for (; k + 3 < k_hi; k += 4) {
                    __m256d a_rh, a_rl, a_ih, a_il;
                    cload4(aip + 4 * k, a_rh, a_rl, a_ih, a_il);
                    __m256d xkrh = _mm256_loadu_pd(x_rh + k);
                    __m256d xkrl = _mm256_loadu_pd(x_rl + k);
                    __m256d xkih = _mm256_loadu_pd(x_ih + k);
                    __m256d xkil = _mm256_loadu_pd(x_il + k);
                    __m256d p_rh, p_rl, p_ih, p_il;
                    simd_fast::cmul(xrh, xrl, xih, xil, a_rh, a_rl, a_ih, a_il,
                                     p_rh, p_rl, p_ih, p_il);
                    simd_fast::neg(p_rh, p_rl);
                    simd_fast::neg(p_ih, p_il);
                    __m256d nrh, nrl, nih, nil;
                    simd_fast::cadd(xkrh, xkrl, xkih, xkil, p_rh, p_rl, p_ih, p_il,
                                     nrh, nrl, nih, nil);
                    _mm256_storeu_pd(x_rh + k, nrh);
                    _mm256_storeu_pd(x_rl + k, nrl);
                    _mm256_storeu_pd(x_ih + k, nih);
                    _mm256_storeu_pd(x_il + k, nil);
                }
                for (; k < k_hi; ++k) {
                    TC aki{ R{aip[4*k], aip[4*k+1]}, R{aip[4*k+2], aip[4*k+3]} };
                    store_x(k, csub(load_x(k), cmul(xi, aki)));
                }
            };
            if (UPLO == 'L') {
                for (std::ptrdiff_t i = 0; i < n; ++i) do_axpy_range(i, i + 1, n);
            } else {
                for (std::ptrdiff_t i = n - 1; i >= 0; --i) do_axpy_range(i, 0, i);
            }
        } else {
            const bool conj_a = (TRANS == 'C');
            auto do_dot_range = [&](std::ptrdiff_t i, std::ptrdiff_t k_lo, std::ptrdiff_t k_hi) {
                const double *aip = reinterpret_cast<const double *>(&A_(0, i));
                __m256d s_rh = zerov, s_rl = zerov, s_ih = zerov, s_il = zerov;
                TC t = load_x(i);
                std::ptrdiff_t k = k_lo;
                for (; k < k_hi && (k & 3) != 0; ++k) {
                    TC aki{ R{aip[4*k], aip[4*k+1]}, R{aip[4*k+2], aip[4*k+3]} };
                    if (conj_a) aki = cconj(aki);
                    t = csub(t, cmul(aki, load_x(k)));
                }
                for (; k + 3 < k_hi; k += 4) {
                    __m256d a_rh, a_rl, a_ih, a_il;
                    cload4(aip + 4 * k, a_rh, a_rl, a_ih, a_il);
                    if (conj_a) simd_fast::neg(a_ih, a_il);
                    __m256d xkrh = _mm256_loadu_pd(x_rh + k);
                    __m256d xkrl = _mm256_loadu_pd(x_rl + k);
                    __m256d xkih = _mm256_loadu_pd(x_ih + k);
                    __m256d xkil = _mm256_loadu_pd(x_il + k);
                    __m256d p_rh, p_rl, p_ih, p_il;
                    simd_fast::cmul(a_rh, a_rl, a_ih, a_il, xkrh, xkrl, xkih, xkil,
                                     p_rh, p_rl, p_ih, p_il);
                    __m256d nsrh, nsrl, nsih, nsil;
                    simd_fast::cadd(s_rh, s_rl, s_ih, s_il, p_rh, p_rl, p_ih, p_il,
                                     nsrh, nsrl, nsih, nsil);
                    s_rh = nsrh; s_rl = nsrl; s_ih = nsih; s_il = nsil;
                }
                TC s_red = chreduce(s_rh, s_rl, s_ih, s_il);
                t = csub(t, s_red);
                for (; k < k_hi; ++k) {
                    TC aki{ R{aip[4*k], aip[4*k+1]}, R{aip[4*k+2], aip[4*k+3]} };
                    if (conj_a) aki = cconj(aki);
                    t = csub(t, cmul(aki, load_x(k)));
                }
                if (nounit) {
                    TC diag_v{ R{aip[4*i], aip[4*i+1]}, R{aip[4*i+2], aip[4*i+3]} };
                    if (conj_a) diag_v = cconj(diag_v);
                    t = cdiv(t, diag_v);
                }
                store_x(i, t);
            };
            if (UPLO == 'L') {
                for (std::ptrdiff_t i = n - 1; i >= 0; --i) do_dot_range(i, i + 1, n);
            } else {
                for (std::ptrdiff_t i = 0; i < n; ++i) do_dot_range(i, 0, i);
            }
        }
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            x[i].re.limbs[0] = x_rh[i]; x[i].re.limbs[1] = x_rl[i];
            x[i].im.limbs[0] = x_ih[i]; x[i].im.limbs[1] = x_il[i];
        }
        std::free(x_rh); std::free(x_rl); std::free(x_ih); std::free(x_il);
}
#pragma GCC pop_options
#endif  /* MBLAS_SIMD_DD */

/* Bit-exact serial path (the SIMD-packed reference). Also reused as the
 * diagonal-block solver by the threaded path below. */
static void wtrsv_serial(char UPLO, char TRANS, bool nounit,
                         std::ptrdiff_t n, const TC *a, std::ptrdiff_t lda, TC *x, std::ptrdiff_t incx)
{
    if (n == 0) return;

    if (incx == 1) {
#ifdef MBLAS_SIMD_DD
        if (mf_have_avx2_fma()) {
            wtrsv_serial_simd_unit(UPLO, TRANS, nounit, n, a, lda, x);
        } else
#endif
        {
        if (TRANS == 'N') {
            if (UPLO == 'L') {
                for (std::ptrdiff_t i = 0; i < n; ++i) {
                    if (!ceq0(x[i])) {
                        if (nounit) x[i] = cdiv(x[i], A_(i, i));
                        const TC xi = x[i];
                        const TC *ai = &A_(0, i);
                        for (std::ptrdiff_t k = i + 1; k < n; ++k) x[k] = csub(x[k], cmul(xi, ai[k]));
                    }
                }
            } else {
                for (std::ptrdiff_t i = n - 1; i >= 0; --i) {
                    if (!ceq0(x[i])) {
                        if (nounit) x[i] = cdiv(x[i], A_(i, i));
                        const TC xi = x[i];
                        const TC *ai = &A_(0, i);
                        for (std::ptrdiff_t k = 0; k < i; ++k) x[k] = csub(x[k], cmul(xi, ai[k]));
                    }
                }
            }
        } else {
            const bool conj_a = (TRANS == 'C');
            if (UPLO == 'L') {
                for (std::ptrdiff_t i = n - 1; i >= 0; --i) {
                    TC t = x[i];
                    const TC *ai = &A_(0, i);
                    if (conj_a) {
                        for (std::ptrdiff_t k = i + 1; k < n; ++k) t = csub(t, cmul(cconj(ai[k]), x[k]));
                        if (nounit) t = cdiv(t, cconj(ai[i]));
                    } else {
                        for (std::ptrdiff_t k = i + 1; k < n; ++k) t = csub(t, cmul(ai[k], x[k]));
                        if (nounit) t = cdiv(t, ai[i]);
                    }
                    x[i] = t;
                }
            } else {
                for (std::ptrdiff_t i = 0; i < n; ++i) {
                    TC t = x[i];
                    const TC *ai = &A_(0, i);
                    if (conj_a) {
                        for (std::ptrdiff_t k = 0; k < i; ++k) t = csub(t, cmul(cconj(ai[k]), x[k]));
                        if (nounit) t = cdiv(t, cconj(ai[i]));
                    } else {
                        for (std::ptrdiff_t k = 0; k < i; ++k) t = csub(t, cmul(ai[k], x[k]));
                        if (nounit) t = cdiv(t, ai[i]);
                    }
                    x[i] = t;
                }
            }
        }
        }
    } else {
        /* Strided: gather x to a contiguous scratch, run the SIMD incx==1 core,
         * scatter back. O(N) gather vs the O(N^2) strided scalar sweep. */
        TC *xbase = (incx < 0) ? x - (std::ptrdiff_t)(n - 1) * incx : x;
        std::vector<TC> xs(static_cast<std::size_t>(n));
        for (std::ptrdiff_t i = 0; i < n; ++i) xs[i] = xbase[(std::ptrdiff_t)i * incx];
        wtrsv_serial(UPLO, TRANS, nounit, n, a, lda, xs.data(), 1);
        for (std::ptrdiff_t i = 0; i < n; ++i) xbase[(std::ptrdiff_t)i * incx] = xs[i];
    }
}

#ifdef _OPENMP
/* Blocked threaded complex triangular solve, incx==1 only. Diagonal blocks
 * (MTRSV_BLK) are solved serially via the bit-exact wtrsv_serial; the bulk
 * off-diagonal coupling is a rectangular GEMV threaded over disjoint output
 * rows. Serial fallback stays bit-exact; threaded path matches within DD fuzz
 * tol. Returns true if it handled the call. */
__attribute__((noinline)) static bool wtrsv_omp(
    char UPLO, char TRANS, bool nounit, std::ptrdiff_t n, const TC *a, std::ptrdiff_t lda, TC *x)
{
    if (n < WTRSV_OMP_MIN || !blas_omp_should_thread())
        return false;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > WTRSV_MAX_CPUS) nthreads = WTRSV_MAX_CPUS;
    const bool lower = (UPLO == 'L');
    const bool trans = (TRANS != 'N');
    const bool conj_a = (TRANS == 'C');

    if (!trans) {
        /* NoTrans: axpy form. Solve a diagonal block, then propagate its solved
         * columns into the not-yet-solved rows (trailing for L, leading for U). */
        if (lower) {
            for (std::ptrdiff_t j0 = 0; j0 < n; j0 += WTRSV_BLK) {
                std::ptrdiff_t j1 = j0 + WTRSV_BLK; if (j1 > n) j1 = n;
                wtrsv_serial(UPLO, TRANS, nounit, j1 - j0, &A_(j0, j0), lda, x + j0, 1);
                if (j1 >= n) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    std::ptrdiff_t tid = omp_get_thread_num();
                    std::ptrdiff_t rlo = j1 + blas_part_bound(n - j1, tid, omp_get_num_threads());
                    std::ptrdiff_t rhi = j1 + blas_part_bound(n - j1, tid + 1, omp_get_num_threads());
                    for (std::ptrdiff_t i = j0; i < j1; ++i) {
                        const TC xi = x[i];
                        if (ceq0(xi)) continue;
                        const TC *ai = &A_(0, i);
#ifdef MBLAS_SIMD_DD
                        if (mf_have_avx2_fma())
                            wtrsv_col_msub_simd(x, ai, xi, rlo, rhi);
                        else
#endif
                            for (std::ptrdiff_t k = rlo; k < rhi; ++k) x[k] = csub(x[k], cmul(xi, ai[k]));
                    }
                }
            }
        } else {
            for (std::ptrdiff_t j1 = n; j1 > 0; j1 -= WTRSV_BLK) {
                std::ptrdiff_t j0 = j1 - WTRSV_BLK; if (j0 < 0) j0 = 0;
                wtrsv_serial(UPLO, TRANS, nounit, j1 - j0, &A_(j0, j0), lda, x + j0, 1);
                if (j0 <= 0) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    std::ptrdiff_t tid = omp_get_thread_num();
                    std::ptrdiff_t rlo = blas_part_bound(j0, tid, omp_get_num_threads());
                    std::ptrdiff_t rhi = blas_part_bound(j0, tid + 1, omp_get_num_threads());
                    for (std::ptrdiff_t i = j0; i < j1; ++i) {
                        const TC xi = x[i];
                        if (ceq0(xi)) continue;
                        const TC *ai = &A_(0, i);
#ifdef MBLAS_SIMD_DD
                        if (mf_have_avx2_fma())
                            wtrsv_col_msub_simd(x, ai, xi, rlo, rhi);
                        else
#endif
                            for (std::ptrdiff_t k = rlo; k < rhi; ++k) x[k] = csub(x[k], cmul(xi, ai[k]));
                    }
                }
            }
        }
    } else {
        /* Trans/ConjTrans: dot form. Fold the already-solved out-of-block
         * tail/head into the block rows (threaded, disjoint rows), then solve
         * the diagonal block serially (within-block coupling + divide). */
        if (lower) {                                  /* backward, k > i */
            for (std::ptrdiff_t j1 = n; j1 > 0; j1 -= WTRSV_BLK) {
                std::ptrdiff_t j0 = j1 - WTRSV_BLK; if (j0 < 0) j0 = 0;
                if (j1 < n) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        std::ptrdiff_t tid = omp_get_thread_num();
                        std::ptrdiff_t ilo = j0 + blas_part_bound(j1 - j0, tid, omp_get_num_threads());
                        std::ptrdiff_t ihi = j0 + blas_part_bound(j1 - j0, tid + 1, omp_get_num_threads());
                        for (std::ptrdiff_t i = ilo; i < ihi; ++i) {
                            const TC *ai = &A_(0, i);
                            TC s;
#ifdef MBLAS_SIMD_DD
                            if (mf_have_avx2_fma())
                                s = wtrsv_dot_range_simd(ai, x, j1, n, conj_a);
                            else
#endif
                            {
                                s = zero_cdd;
                                for (std::ptrdiff_t k = j1; k < n; ++k) {
                                    const TC e = conj_a ? cconj(ai[k]) : ai[k];
                                    s = cadd(s, cmul(e, x[k]));
                                }
                            }
                            x[i] = csub(x[i], s);
                        }
                    }
                }
                wtrsv_serial(UPLO, TRANS, nounit, j1 - j0, &A_(j0, j0), lda, x + j0, 1);
            }
        } else {                                      /* forward, k < i */
            for (std::ptrdiff_t j0 = 0; j0 < n; j0 += WTRSV_BLK) {
                std::ptrdiff_t j1 = j0 + WTRSV_BLK; if (j1 > n) j1 = n;
                if (j0 > 0) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        std::ptrdiff_t tid = omp_get_thread_num();
                        std::ptrdiff_t ilo = j0 + blas_part_bound(j1 - j0, tid, omp_get_num_threads());
                        std::ptrdiff_t ihi = j0 + blas_part_bound(j1 - j0, tid + 1, omp_get_num_threads());
                        for (std::ptrdiff_t i = ilo; i < ihi; ++i) {
                            const TC *ai = &A_(0, i);
                            TC s;
#ifdef MBLAS_SIMD_DD
                            if (mf_have_avx2_fma())
                                s = wtrsv_dot_range_simd(ai, x, 0, j0, conj_a);
                            else
#endif
                            {
                                s = zero_cdd;
                                for (std::ptrdiff_t k = 0; k < j0; ++k) {
                                    const TC e = conj_a ? cconj(ai[k]) : ai[k];
                                    s = cadd(s, cmul(e, x[k]));
                                }
                            }
                            x[i] = csub(x[i], s);
                        }
                    }
                }
                wtrsv_serial(UPLO, TRANS, nounit, j1 - j0, &A_(j0, j0), lda, x + j0, 1);
            }
        }
    }
    return true;
}
#endif

static void wtrsv_core(
    char uplo, char trans, char diag,
    std::ptrdiff_t n,
    const TC *a, std::ptrdiff_t lda,
    TC *x, std::ptrdiff_t incx)
{
    const char UPLO = up(&uplo);
    const char TRANS   = up(&trans);
    const char DIAG = up(&diag);
    const bool nounit = (DIAG != 'U');

    if (n == 0) return;

#ifdef _OPENMP
    if (incx == 1 && n >= WTRSV_OMP_MIN && blas_omp_available()
        && wtrsv_omp(UPLO, TRANS, nounit, n, a, lda, x))
        return;
#endif

    wtrsv_serial(UPLO, TRANS, nounit, n, a, lda, x, incx);
}

extern "C" {
EPBLAS_FACADE_TRMV(wtrsv, TC)
}

#undef A_
