/* imamax — multifloats real DD: 1-based argmax(|X|). */
#include <stddef.h>
#include <multifloats.h>
#include "mf_pred.h"
#include <multifloats/float64x2.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#endif
#ifdef MBLAS_SIMD_DD
#include <immintrin.h>
#endif
#include "mf_dispatch.h"   /* MF_SIMD_TARGET + mf_have_avx2_fma() runtime gate */
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using TR = mf::float64x2;

namespace {
/* Inline magnitude/compare from mf_pred — mf_pred::mag stays header-inline (the
 * public fabsdd() is out-of-line; per-element use would emit a PLT call in the
 * hot loop, ~2x slower than ob's inlined abs). */
using mf_pred::mag;
using mf_pred::gt;

#ifdef MBLAS_SIMD_DD
/* AVX2+FMA SoA argmax — target("avx2,fma") so it builds under a pre-Haswell
 * baseline -march; reached only behind the mf_have_avx2_fma() probe below. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")
/* SoA argmax over a contiguous DD range. ob (and the scalar fallback below) walk
 * one element/iter; this processes 4 lanes/iter on the deinterleaved hi/lo
 * limbs, so par genuinely beats the scalar reference instead of tying it.
 *
 * mag(x) is exactly (|hi|, sign(hi)*lo): |hi| = andnot(signbit,hi); the lo
 * part flips iff hi<0, i.e. lo ^ (hi & signbit). The lexicographic compare on
 * (|hi|, adj_lo) reproduces the scalar DD '>' bit-for-bit. Per-lane strict-'>'
 * keeps the lowest index within a lane; the 4-lane horizontal merge and scalar
 * tail keep the lowest index on exact ties — identical result to the left-to-
 * right scan. (Normalised DDs never have hi==0 with lo!=0, the one input where
 * the branchless sign-of-value would diverge from the scalar a<0 test.) */
inline void deint4_simd(const TR *p, __m256d &hi, __m256d &lo) {
    __m256d v0 = _mm256_loadu_pd(reinterpret_cast<const double *>(p));
    __m256d v1 = _mm256_loadu_pd(reinterpret_cast<const double *>(p + 2));
    __m256d a = _mm256_unpacklo_pd(v0, v1);
    __m256d b = _mm256_unpackhi_pd(v0, v1);
    hi = _mm256_permute4x64_pd(a, 0xD8);
    lo = _mm256_permute4x64_pd(b, 0xD8);
}

/* One 4-lane running-max update: fold |x[p..p+3]| (indices `idx`) into the
 * lexicographic max state (vmh,vml,vidx). Pulled out so the scan can carry TWO
 * independent max accumulators — the loop-carried chain here is
 * cmp→and→or→blendv (~8 cy / 4 elts) and a single accumulator leaves the scan
 * latency-bound at compute-bound sizes; two interleaved chains overlap and let
 * the loop hit its throughput ceiling instead. */
inline void amax_step_simd(const TR *p, __m256i idx, __m256d sgn,
                      __m256d &vmh, __m256d &vml, __m256i &vidx) {
    __m256d h, l;
    deint4_simd(p, h, l);
    __m256d ah = _mm256_andnot_pd(sgn, h);
    __m256d al = _mm256_xor_pd(l, _mm256_and_pd(h, sgn));
    __m256d gt  = _mm256_cmp_pd(ah, vmh, _CMP_GT_OQ);
    __m256d eq  = _mm256_cmp_pd(ah, vmh, _CMP_EQ_OQ);
    __m256d glo = _mm256_cmp_pd(al, vml, _CMP_GT_OQ);
    __m256d upd = _mm256_or_pd(gt, _mm256_and_pd(eq, glo));
    vmh  = _mm256_blendv_pd(vmh, ah, upd);
    vml  = _mm256_blendv_pd(vml, al, upd);
    vidx = _mm256_castpd_si256(
               _mm256_blendv_pd(_mm256_castsi256_pd(vidx),
                                _mm256_castsi256_pd(idx), upd));
}

__attribute__((noinline)) ptrdiff_t imamax_scan_simd(ptrdiff_t n, const TR *x, TR *bv_out)
{
    if (n < 8) {            /* SIMD setup not worth it for tiny ranges */
        ptrdiff_t best = 0;
        TR bv = mag(x[0]);
        for (ptrdiff_t i = 1; i < n; ++i) {
            TR v = mag(x[i]);
            if (v > bv) { bv = v; best = i; }
        }
        *bv_out = bv;
        return best;
    }

    const __m256d sgn = _mm256_set1_pd(-0.0);   /* 0x8000…0 sign bit */
    const __m256d ninf = _mm256_set1_pd(-1.0 / 0.0);
    /* Two independent 4-lane accumulators (group A = elts i..i+3, group B =
     * i+4..i+7). -inf seed makes the first fold of each lane unconditional. */
    __m256d vmhA = ninf, vmlA = ninf, vmhB = ninf, vmlB = ninf;
    __m256i vidxA = _mm256_setzero_si256(), vidxB = _mm256_setzero_si256();
    __m256i curA = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i curB = _mm256_set_epi64x(7, 6, 5, 4);
    const __m256i eight = _mm256_set1_epi64x(8);

    ptrdiff_t i = 0;
    for (; i + 8 <= n; i += 8) {
        amax_step_simd(x + i,     curA, sgn, vmhA, vmlA, vidxA);
        amax_step_simd(x + i + 4, curB, sgn, vmhB, vmlB, vidxB);
        curA = _mm256_add_epi64(curA, eight);
        curB = _mm256_add_epi64(curB, eight);
    }
    if (i + 4 <= n) {      /* a leftover 4-block folds into group A (idx == i..) */
        amax_step_simd(x + i, curA, sgn, vmhA, vmlA, vidxA);
        i += 4;
    }

    /* Horizontal merge of all 8 lanes — lexicographic max, lowest index on tie. */
    alignas(32) double mh[8], ml[8];
    alignas(32) long long li[8];
    _mm256_store_pd(mh,     vmhA);  _mm256_store_pd(mh + 4, vmhB);
    _mm256_store_pd(ml,     vmlA);  _mm256_store_pd(ml + 4, vmlB);
    _mm256_store_si256(reinterpret_cast<__m256i *>(li),     vidxA);
    _mm256_store_si256(reinterpret_cast<__m256i *>(li + 4), vidxB);
    ptrdiff_t best = (ptrdiff_t)li[0];
    TR bv{mh[0], ml[0]};
    for (std::ptrdiff_t k = 1; k < 8; ++k) {
        TR cv{mh[k], ml[k]};
        if (gt(cv, bv) || (cv.limbs[0] == bv.limbs[0] &&
                              cv.limbs[1] == bv.limbs[1] && li[k] < best)) {
            bv = cv; best = (ptrdiff_t)li[k];
        }
    }
    /* Scalar tail (< 4 leftover); strict '>' preserves the lower held index. */
    for (; i < n; ++i) {
        TR v = mag(x[i]);
        if (v > bv) { bv = v; best = i; }
    }
    *bv_out = bv;
    return best;
}
#pragma GCC pop_options
#endif

/* Scan a contiguous unit-stride range [0,n); return the 0-based index of the
 * first element with maximal |x| and store that magnitude in *bv_out.
 * Strictly-greater update keeps the lowest index on ties. Runtime dispatch:
 * SoA-SIMD scan on Haswell+, scalar (always compiled) on Sandybridge/Ivy. */
__attribute__((noinline)) ptrdiff_t imamax_scan(ptrdiff_t n, const TR *x, TR *bv_out)
{
#ifdef MBLAS_SIMD_DD
    if (mf_have_avx2_fma()) return imamax_scan_simd(n, x, bv_out);
#endif
    ptrdiff_t best = 0;
    TR bv = mag(x[0]);
    for (ptrdiff_t i = 1; i < n; ++i) {
        TR v = mag(x[i]);
        if (v > bv) { bv = v; best = i; }   /* lexicographic >, matches ob clone */
    }
    *bv_out = bv;
    return best;
}
}

#ifdef _OPENMP
/* Threaded argmax for large unit-stride X — shared partial[]-reduce wrapper
 * (mf_omp::partial_argmax, pre-initialized slots). Each thread scans its
 * contiguous slice keeping the first (lowest-index) maximum, then partials
 * merge in ascending-tid order with a strict-greater test — so the lowest
 * global index wins any tie, bit-identical to the serial left-to-right scan. */
#define IMAMAX_OMP_MIN 8192
__attribute__((noinline)) static std::ptrdiff_t imamax_omp(std::ptrdiff_t n, const TR *x, std::ptrdiff_t *out)
{
    if (n <= IMAMAX_OMP_MIN || !blas_omp_should_thread())
        return 0;
    *out = mf_omp::partial_argmax(n, TR{0.0, 0.0},
        [x](std::ptrdiff_t lo, std::ptrdiff_t hi, TR &bv) {
            ptrdiff_t li = imamax_scan(hi - lo, x + lo, &bv);
            return lo + (std::ptrdiff_t)li + 1;   /* 1-based global index */
        },
        [](const TR &a, const TR &b) { return gt(a, b); });
    return 1;
}
#endif

static std::ptrdiff_t imamax_core(std::ptrdiff_t n, const TR *x, std::ptrdiff_t incx)
{
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
#ifdef _OPENMP
    if (incx == 1) {
        std::ptrdiff_t r;
        if (imamax_omp(n, x, &r)) return r;
    }
#endif
    if (incx == 1) {
        TR bv;
        return (std::ptrdiff_t)imamax_scan(n, x, &bv) + 1;
    }
    std::ptrdiff_t best = 1;
    TR bestv = mag(x[0]);
    std::ptrdiff_t ix = incx;
    for (std::ptrdiff_t i = 2; i <= n; ++i) {
        TR av = mag(x[ix]);
        if (gt(av, bestv)) { bestv = av; best = i; }
        ix += incx;
    }
    return best;
}

extern "C" { EPBLAS_FACADE_IAMAX(imamax, TR) }
