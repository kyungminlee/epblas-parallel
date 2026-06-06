/* mnrm2 — multifloats real DD: ||X||₂ via two-pass scaled.
 *
 * Pass 1 (SIMD vmaxpd): find scale = max(|x[i].hi|).
 * Pass 2 (SIMD wide-acc): Σ (x[i]/scale)² then scale·√sum.
 */
#include <cstddef>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define MNRM2_OMP_MIN 8192
#define MNRM2_MAX_CPUS 64
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline bool dd_iszero(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_gt(T a, T b) {
    return a.limbs[0] > b.limbs[0]
        || (a.limbs[0] == b.limbs[0] && a.limbs[1] > b.limbs[1]);
}
}

#ifdef MBLAS_SIMD_DD
#include <immintrin.h>

namespace {
inline void load_4cell_soa(const T *p, __m256d &h, __m256d &l) {
    __m256d v0 = _mm256_loadu_pd(reinterpret_cast<const double*>(p));
    __m256d v1 = _mm256_loadu_pd(reinterpret_cast<const double*>(p + 2));
    __m256d lo = _mm256_unpacklo_pd(v0, v1);
    __m256d hi = _mm256_unpackhi_pd(v0, v1);
    h = _mm256_permute4x64_pd(lo, 0xD8);
    l = _mm256_permute4x64_pd(hi, 0xD8);
}
inline void simd_twoprod(__m256d a, __m256d b, __m256d &p, __m256d &e) {
    p = _mm256_mul_pd(a, b);
    e = _mm256_fmsub_pd(a, b, p);
}
inline void simd_twosum(__m256d a, __m256d b, __m256d &s, __m256d &e) {
    s = _mm256_add_pd(a, b);
    __m256d bb = _mm256_sub_pd(s, a);
    __m256d aa = _mm256_sub_pd(s, bb);
    e = _mm256_add_pd(_mm256_sub_pd(a, aa), _mm256_sub_pd(b, bb));
}
inline void simd_fast_twosum(__m256d a, __m256d b, __m256d &s, __m256d &e) {
    s = _mm256_add_pd(a, b);
    e = _mm256_sub_pd(b, _mm256_sub_pd(s, a));
}
inline T horizontal_dd(__m256d h, __m256d l) {
    alignas(32) double ha[4], la[4];
    _mm256_store_pd(ha, h); _mm256_store_pd(la, l);
    T s{ha[0], la[0]};
    for (int k = 1; k < 4; ++k) s = s + T{ha[k], la[k]};
    return s;
}
}
#endif

/* Pass-1 unit kernel: max|x.hi| over a contiguous (incx==1) slice. Max is
 * exact (no rounding), so the combined global scale is order-independent and
 * BIT-EXACT regardless of how the slice is split. */
static double mnrm2_maxabs_unit(int n, const T *x)
{
    double scale_hi = 0.0;
#ifdef MBLAS_SIMD_DD
    __m256d mx = _mm256_setzero_pd();
    const __m256d absmask = _mm256_castsi256_pd(
        _mm256_set1_epi64x(static_cast<long long>(0x7FFFFFFFFFFFFFFFULL)));
    const int n4 = n & ~3;
    for (int i = 0; i < n4; i += 4) {
        __m256d xh, xl; load_4cell_soa(&x[i], xh, xl); (void)xl;
        __m256d ax = _mm256_and_pd(xh, absmask);
        mx = _mm256_max_pd(mx, ax);
    }
    alignas(32) double mxa[4];
    _mm256_store_pd(mxa, mx);
    for (int k = 0; k < 4; ++k) if (mxa[k] > scale_hi) scale_hi = mxa[k];
    for (int i = n4; i < n; ++i) {
        T ax = fabsdd(x[i]);
        if (ax.limbs[0] > scale_hi) scale_hi = ax.limbs[0];
    }
#else
    for (int i = 0; i < n; ++i) {
        T ax = fabsdd(x[i]);
        if (ax.limbs[0] > scale_hi) scale_hi = ax.limbs[0];
    }
#endif
    return scale_hi;
}

/* Pass-2 unit kernel: Σ (x/scale)² over a contiguous (incx==1) slice. */
static T mnrm2_ssq_unit(int n, const T *x, T scale)
{
    T s{0.0, 0.0};
#ifdef MBLAS_SIMD_DD
    /* Precompute inverse so the inner loop avoids dd_div. */
    T inv = T{1.0, 0.0} / scale;
    __m256d invh = _mm256_set1_pd(inv.limbs[0]);
    __m256d invl = _mm256_set1_pd(inv.limbs[1]);
    __m256d a0 = _mm256_setzero_pd();
    __m256d a1 = _mm256_setzero_pd();
    __m256d a2 = _mm256_setzero_pd();
    constexpr int K = 64;
    int counter = K;
    const int n4 = n & ~3;
    for (int i = 0; i < n4; i += 4) {
        __m256d xh, xl; load_4cell_soa(&x[i], xh, xl);
        /* t = x · inv  (DD mul, drop xl*invl) */
        __m256d th, tl;
        simd_twoprod(xh, invh, th, tl);
        tl = _mm256_add_pd(tl,
                _mm256_add_pd(_mm256_mul_pd(xh, invl), _mm256_mul_pd(xl, invh)));
        /* p = t · t */
        __m256d ph, pl;
        simd_twoprod(th, th, ph, pl);
        pl = _mm256_add_pd(pl,
                _mm256_add_pd(_mm256_mul_pd(th, tl), _mm256_mul_pd(tl, th)));
        __m256d e0, e1, e2;
        simd_twosum(a0, ph, a0, e0);
        simd_twosum(a1, pl, a1, e1);
        simd_twosum(a1, e0, a1, e2);
        a2 = _mm256_add_pd(a2, _mm256_add_pd(e1, e2));
        if (--counter == 0) {
            __m256d t, e;
            simd_fast_twosum(a1, a2, t, e);
            a1 = t; a2 = e;
            simd_fast_twosum(a0, a1, a0, a1);
            a1 = _mm256_add_pd(a1, a2);
            simd_fast_twosum(a0, a1, a0, a1);
            a2 = _mm256_setzero_pd();
            counter = K;
        }
    }
    __m256d t = _mm256_add_pd(a1, a2);
    s = horizontal_dd(a0, t);
    for (int i = n4; i < n; ++i) { T u = x[i] / scale; s = s + u * u; }
#else
    for (int i = 0; i < n; ++i) { T u = x[i] / scale; s = s + u * u; }
#endif
    return s;
}

#ifdef _OPENMP
/* Threaded two-pass nrm2 (incx==1). Pass 1 reduces max|x.hi| (exact → global
 * scale BIT-EXACT vs serial); pass 2 reduces Σ(x/scale)² in tid order (the
 * cross-slice summation reorders, so the squared-sum matches serial within
 * fuzz tol). Returns false (caller falls through to serial) when below the
 * threshold or already inside a parallel region. */
__attribute__((noinline)) static bool mnrm2_omp(int n, const T *x, T *out)
{
    if (n <= MNRM2_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MNRM2_MAX_CPUS) nthreads = MNRM2_MAX_CPUS;

    double scale_hi = 0.0;
    #pragma omp parallel num_threads(nthreads) reduction(max:scale_hi)
    {
        int tid = omp_get_thread_num(), nth = omp_get_num_threads();
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
        if (lo < hi) {
            double local = mnrm2_maxabs_unit(hi - lo, x + lo);
            if (local > scale_hi) scale_hi = local;
        }
    }
    if (scale_hi == 0.0) { *out = T{0.0, 0.0}; return true; }
    T scale{scale_hi, 0.0};

    T partial[MNRM2_MAX_CPUS];
    for (int t = 0; t < nthreads; ++t) partial[t] = T{0.0, 0.0};
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num(), nth = omp_get_num_threads();
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
        if (lo < hi) partial[tid] = mnrm2_ssq_unit(hi - lo, x + lo, scale);
    }
    T s{0.0, 0.0};
    for (int t = 0; t < nthreads; ++t) s = s + partial[t];
    *out = scale * sqrtdd(s);
    return true;
}
#endif

extern "C" T mnrm2_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    T zero{0.0, 0.0};
    if (n < 1 || incx < 1) return zero;
    if (n == 1) return fabsdd(x[0]);

    if (incx == 1) {
#ifdef _OPENMP
        T out;
        if (mnrm2_omp(n, x, &out)) return out;
#endif
        double scale_hi = mnrm2_maxabs_unit(n, x);
        if (scale_hi == 0.0) return zero;
        T scale{scale_hi, 0.0};
        return scale * sqrtdd(mnrm2_ssq_unit(n, x, scale));
    }

    /* Strided fallback (scalar two-pass). */
    double scale_hi = 0.0;
    int ix = 0;
    for (int i = 0; i < n; ++i) {
        T ax = fabsdd(x[ix]);
        if (ax.limbs[0] > scale_hi) scale_hi = ax.limbs[0];
        ix += incx;
    }
    if (scale_hi == 0.0) return zero;
    T scale{scale_hi, 0.0};
    T s = zero;
    ix = 0;
    for (int i = 0; i < n; ++i) {
        T t = x[ix] / scale;
        s = s + t * t;
        ix += incx;
    }
    return scale * sqrtdd(s);
}
