/* iwamax — multifloats complex DD: 1-based argmax(|re|+|im|). */
#include <stddef.h>
#include <multifloats.h>
#include "mf_pred.h"
#include <multifloats/float64x2.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#endif
#ifdef WBLAS_SIMD_DD
#include <immintrin.h>
#include "mf_simd_exact.h"
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
/* Inline magnitude/compare from mf_pred — mf_pred::mag stays header-inline (the
 * public fabsdd() is out-of-line; cabs1 would emit two PLT calls per element). */
using mf_pred::mag;
using mf_pred::gt;
inline R cabs1(T const &z) { return mag(z.re) + mag(z.im); }

#ifdef WBLAS_SIMD_DD
/* SoA argmax over a contiguous complex-DD range, 4 lanes/iter, so par beats the
 * scalar reference instead of tying it (see imamax.cpp for the real twin).
 *
 * The magnitude is cabs1 = abs(re) + abs(im), a DD add. Bit-exactness with
 * the scalar argmax therefore demands the *same* error-free transform the scalar
 * float64x2::operator+ uses (two_sum on BOTH limbs, then fast_two_sum) — i.e.
 * simd_exact::add (NOT the cheaper simd_fast::add, whose lo limb differs by ~1
 * ulp and would flip near-ties). The non-finite / both-hi-zero short-circuits of
 * operator+ are skipped: inputs here are abs values, finite for finite x, and a
 * zero element yields a zero magnitude on the main path too (sign of zero is
 * irrelevant to the comparison). */
/* Deinterleave 4 complex-DD elements (re.hi,re.lo,im.hi,im.lo each) via a 4x4
 * transpose into SoA limb vectors. */
inline void deint4c(const T *p, __m256d &reh, __m256d &rel,
                    __m256d &imh, __m256d &iml) {
    const double *d = reinterpret_cast<const double *>(p);
    __m256d r0 = _mm256_loadu_pd(d);
    __m256d r1 = _mm256_loadu_pd(d + 4);
    __m256d r2 = _mm256_loadu_pd(d + 8);
    __m256d r3 = _mm256_loadu_pd(d + 12);
    __m256d t0 = _mm256_unpacklo_pd(r0, r1);
    __m256d t1 = _mm256_unpackhi_pd(r0, r1);
    __m256d t2 = _mm256_unpacklo_pd(r2, r3);
    __m256d t3 = _mm256_unpackhi_pd(r2, r3);
    reh = _mm256_permute2f128_pd(t0, t2, 0x20);
    rel = _mm256_permute2f128_pd(t1, t3, 0x20);
    imh = _mm256_permute2f128_pd(t0, t2, 0x31);
    iml = _mm256_permute2f128_pd(t1, t3, 0x31);
}
/* |re|+|im| for 4 lanes: abs(x) = (|hi|, sign(hi)*lo) branchlessly, summed
 * with the exact EFT add. */
inline void cabs1_4(const T *p, __m256d &mh, __m256d &ml) {
    const __m256d sgn = _mm256_set1_pd(-0.0);
    __m256d reh, rel, imh, iml;
    deint4c(p, reh, rel, imh, iml);
    __m256d arh = _mm256_andnot_pd(sgn, reh);
    __m256d arl = _mm256_xor_pd(rel, _mm256_and_pd(reh, sgn));
    __m256d aih = _mm256_andnot_pd(sgn, imh);
    __m256d ail = _mm256_xor_pd(iml, _mm256_and_pd(imh, sgn));
    simd_exact::add(arh, arl, aih, ail, mh, ml);
}

inline ptrdiff_t iwamax_scan(ptrdiff_t n, const T *x, R *bv_out)
{
    if (n < 8) {
        ptrdiff_t best = 0;
        R bv = cabs1(x[0]);
        for (ptrdiff_t i = 1; i < n; ++i) {
            R v = cabs1(x[i]);
            if (v > bv) { bv = v; best = i; }
        }
        *bv_out = bv;
        return best;
    }
    __m256d vmh, vml;
    cabs1_4(x, vmh, vml);
    __m256i vidx = _mm256_set_epi64x(3, 2, 1, 0);
    __m256i cur  = vidx;
    const __m256i four = _mm256_set1_epi64x(4);

    ptrdiff_t i = 4;
    for (; i + 4 <= n; i += 4) {
        cur = _mm256_add_epi64(cur, four);
        __m256d mh, ml;
        cabs1_4(x + i, mh, ml);
        __m256d gt  = _mm256_cmp_pd(mh, vmh, _CMP_GT_OQ);
        __m256d eq  = _mm256_cmp_pd(mh, vmh, _CMP_EQ_OQ);
        __m256d glo = _mm256_cmp_pd(ml, vml, _CMP_GT_OQ);
        __m256d upd = _mm256_or_pd(gt, _mm256_and_pd(eq, glo));
        vmh  = _mm256_blendv_pd(vmh, mh, upd);
        vml  = _mm256_blendv_pd(vml, ml, upd);
        vidx = _mm256_castpd_si256(
                   _mm256_blendv_pd(_mm256_castsi256_pd(vidx),
                                    _mm256_castsi256_pd(cur), upd));
    }

    alignas(32) double mh[4], ml[4];
    alignas(32) long long li[4];
    _mm256_store_pd(mh, vmh);
    _mm256_store_pd(ml, vml);
    _mm256_store_si256(reinterpret_cast<__m256i *>(li), vidx);
    ptrdiff_t best = (ptrdiff_t)li[0];
    R bv{mh[0], ml[0]};
    for (std::ptrdiff_t k = 1; k < 4; ++k) {
        R cv{mh[k], ml[k]};
        if (gt(cv, bv) || (cv.limbs[0] == bv.limbs[0] &&
                              cv.limbs[1] == bv.limbs[1] && li[k] < best)) {
            bv = cv; best = (ptrdiff_t)li[k];
        }
    }
    for (; i < n; ++i) {
        R v = cabs1(x[i]);
        if (v > bv) { bv = v; best = i; }
    }
    *bv_out = bv;
    return best;
}
#else
/* Contiguous unit-stride scan: 0-based index of first max-|re|+|im| element. */
inline ptrdiff_t iwamax_scan(ptrdiff_t n, const T *x, R *bv_out)
{
    ptrdiff_t best = 0;
    R bv = cabs1(x[0]);
    for (ptrdiff_t i = 1; i < n; ++i) {
        R v = cabs1(x[i]);
        if (v > bv) { bv = v; best = i; }   /* lexicographic >, matches ob clone */
    }
    *bv_out = bv;
    return best;
}
#endif
}

#ifdef _OPENMP
/* Threaded argmax for large unit-stride X. Each thread scans its contiguous
 * slice keeping the first (lowest-index) maximum, then partials merge in
 * ascending-tid order with a strict-greater test — so the lowest global index
 * wins any tie, bit-identical to the serial left-to-right scan. */
#define IWAMAX_OMP_MIN 8192
#define IWAMAX_MAX_CPUS 64
__attribute__((noinline)) static std::ptrdiff_t iwamax_omp(std::ptrdiff_t n, const T *x, std::ptrdiff_t *out)
{
    if (n <= IWAMAX_OMP_MIN || !blas_omp_available() || omp_in_parallel())
        return 0;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > IWAMAX_MAX_CPUS) nthreads = IWAMAX_MAX_CPUS;
    std::ptrdiff_t   idx[IWAMAX_MAX_CPUS];
    R     val[IWAMAX_MAX_CPUS];
    #pragma omp parallel num_threads(nthreads)
    {
        std::ptrdiff_t tid = omp_get_thread_num();
        std::ptrdiff_t nth = omp_get_num_threads();
        std::ptrdiff_t lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        std::ptrdiff_t b = 0;
        R bv{0.0, 0.0};
        if (lo < hi) {
            R sv;
            ptrdiff_t li = iwamax_scan(hi - lo, x + lo, &sv);
            b = lo + (std::ptrdiff_t)li + 1;   /* 1-based global index */
            bv = sv;
        }
        idx[tid] = b; val[tid] = bv;
    }
    std::ptrdiff_t best = 0;
    R bestv{0.0, 0.0};
    for (std::ptrdiff_t t = 0; t < nthreads; ++t) {
        if (idx[t] == 0) continue;
        if (best == 0 || gt(val[t], bestv)) { best = idx[t]; bestv = val[t]; }
    }
    *out = best;
    return 1;
}
#endif

static std::ptrdiff_t iwamax_core(std::ptrdiff_t n, const T *x, std::ptrdiff_t incx)
{
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
#ifdef _OPENMP
    if (incx == 1) {
        std::ptrdiff_t r;
        if (iwamax_omp(n, x, &r)) return r;
    }
#endif
    if (incx == 1) {
        R bv;
        return (std::ptrdiff_t)iwamax_scan(n, x, &bv) + 1;
    }
    std::ptrdiff_t best = 1;
    R bestv = cabs1(x[0]);
    std::ptrdiff_t ix = incx;
    for (std::ptrdiff_t i = 2; i <= n; ++i) {
        R av = cabs1(x[ix]);
        if (gt(av, bestv)) { bestv = av; best = i; }
        ix += incx;
    }
    return best;
}

extern "C" {
EPBLAS_FACADE_IAMAX(iwamax, T)
}
