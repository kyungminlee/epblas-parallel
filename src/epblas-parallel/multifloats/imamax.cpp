/* imamax — multifloats real DD: 1-based argmax(|X|). */
#include <stddef.h>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#ifdef MBLAS_SIMD_DD
#include <immintrin.h>
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
/* Inline magnitude — the public fabsdd() is an out-of-line library call, so
 * using it per element emits a PLT call in the hot loop (~2x slower than ob's
 * inlined abs). a < 0 ? -a : a uses only header-inline constexpr ops and yields
 * the identical canonical magnitude for the comparison below. */
inline T t_abs(T a) { return a < 0 ? -a : a; }

inline bool dd_gt(T a, T b) {
    return a.limbs[0] > b.limbs[0]
        || (a.limbs[0] == b.limbs[0] && a.limbs[1] > b.limbs[1]);
}

#ifdef MBLAS_SIMD_DD
/* SoA argmax over a contiguous DD range. ob (and the scalar fallback below) walk
 * one element/iter; this processes 4 lanes/iter on the deinterleaved hi/lo
 * limbs, so par genuinely beats the scalar reference instead of tying it.
 *
 * t_abs(x) is exactly (|hi|, sign(hi)*lo): |hi| = andnot(signbit,hi); the lo
 * part flips iff hi<0, i.e. lo ^ (hi & signbit). The lexicographic compare on
 * (|hi|, adj_lo) reproduces the scalar DD '>' bit-for-bit. Per-lane strict-'>'
 * keeps the lowest index within a lane; the 4-lane horizontal merge and scalar
 * tail keep the lowest index on exact ties — identical result to the left-to-
 * right scan. (Normalised DDs never have hi==0 with lo!=0, the one input where
 * the branchless sign-of-value would diverge from the scalar a<0 test.) */
inline void deint4(const T *p, __m256d &hi, __m256d &lo) {
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
inline void amax_step(const T *p, __m256i idx, __m256d sgn,
                      __m256d &vmh, __m256d &vml, __m256i &vidx) {
    __m256d h, l;
    deint4(p, h, l);
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

__attribute__((noinline)) ptrdiff_t imamax_scan(ptrdiff_t n, const T *x, T *bv_out)
{
    if (n < 8) {            /* SIMD setup not worth it for tiny ranges */
        ptrdiff_t best = 0;
        T bv = t_abs(x[0]);
        for (ptrdiff_t i = 1; i < n; ++i) {
            T v = t_abs(x[i]);
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
        amax_step(x + i,     curA, sgn, vmhA, vmlA, vidxA);
        amax_step(x + i + 4, curB, sgn, vmhB, vmlB, vidxB);
        curA = _mm256_add_epi64(curA, eight);
        curB = _mm256_add_epi64(curB, eight);
    }
    if (i + 4 <= n) {      /* a leftover 4-block folds into group A (idx == i..) */
        amax_step(x + i, curA, sgn, vmhA, vmlA, vidxA);
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
    T bv{mh[0], ml[0]};
    for (int k = 1; k < 8; ++k) {
        T cv{mh[k], ml[k]};
        if (dd_gt(cv, bv) || (cv.limbs[0] == bv.limbs[0] &&
                              cv.limbs[1] == bv.limbs[1] && li[k] < best)) {
            bv = cv; best = (ptrdiff_t)li[k];
        }
    }
    /* Scalar tail (< 4 leftover); strict '>' preserves the lower held index. */
    for (; i < n; ++i) {
        T v = t_abs(x[i]);
        if (v > bv) { bv = v; best = i; }
    }
    *bv_out = bv;
    return best;
}
#else
/* Scan a contiguous unit-stride range [0,n); return the 0-based index of the
 * first element with maximal |x| and store that magnitude in *bv_out.
 * Strictly-greater update keeps the lowest index on ties. */
__attribute__((noinline)) ptrdiff_t imamax_scan(ptrdiff_t n, const T *x, T *bv_out)
{
    ptrdiff_t best = 0;
    T bv = t_abs(x[0]);
    for (ptrdiff_t i = 1; i < n; ++i) {
        T v = t_abs(x[i]);
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
#define IMAMAX_OMP_MIN 8192
#define IMAMAX_MAX_CPUS 64
__attribute__((noinline)) static int imamax_omp(int n, const T *x, int *out)
{
    if (n <= IMAMAX_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > IMAMAX_MAX_CPUS) nthreads = IMAMAX_MAX_CPUS;
    int   idx[IMAMAX_MAX_CPUS];
    T     val[IMAMAX_MAX_CPUS];
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
        int b = 0;
        T bv{0.0, 0.0};
        if (lo < hi) {
            T sv;
            ptrdiff_t li = imamax_scan(hi - lo, x + lo, &sv);
            b = lo + (int)li + 1;   /* 1-based global index */
            bv = sv;
        }
        idx[tid] = b; val[tid] = bv;
    }
    int best = 0;
    T bestv{0.0, 0.0};
    for (int t = 0; t < nthreads; ++t) {
        if (idx[t] == 0) continue;
        if (best == 0 || dd_gt(val[t], bestv)) { best = idx[t]; bestv = val[t]; }
    }
    *out = best;
    return 1;
}
#endif

extern "C" int imamax_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
#ifdef _OPENMP
    if (incx == 1) {
        int r;
        if (imamax_omp(n, x, &r)) return r;
    }
#endif
    if (incx == 1) {
        T bv;
        return (int)imamax_scan(n, x, &bv) + 1;
    }
    int best = 1;
    T bestv = t_abs(x[0]);
    int ix = incx;
    for (int i = 2; i <= n; ++i) {
        T av = t_abs(x[ix]);
        if (dd_gt(av, bestv)) { bestv = av; best = i; }
        ix += incx;
    }
    return best;
}
