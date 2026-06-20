/* mrotm — multifloats real DD: apply modified Givens rotation.
 * H · (X, Y) determined by flag in dparam[0] ∈ {-2, -1, 0, +1}.
 *
 * flag is loop-invariant, so it is unswitched OUT of the element loop into three
 * flag-specific kernels (source-level unswitching). Findings rule 7 — do NOT
 * fold back to one inner loop with a per-element branch; gcc then loses the
 * unswitch and emits ~3x the stores (the prior lambda-per-element form cost
 * ~7-10% serially vs ob).
 *
 * Each kernel runs an AVX2 SoA-DD body (4 cells/iter, simd_fast::mul/add)
 * with a scalar tail; the serial, threaded, and gathered-strided paths all drive
 * the same kernels, so every increment combination is vectorized + threadable.
 */
#include <cstddef>
#include <vector>
#include <multifloats.h>
#include "mf_pred.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define MROTM_OMP_MIN 1024
#endif
#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
using mf_pred::lt0;
using mf_pred::eq0;

#ifdef MBLAS_SIMD_DD
inline void load4(const T *p, __m256d &h, __m256d &l) {
    __m256d v0 = _mm256_loadu_pd(reinterpret_cast<const double *>(p));
    __m256d v1 = _mm256_loadu_pd(reinterpret_cast<const double *>(p + 2));
    __m256d lo = _mm256_unpacklo_pd(v0, v1);
    __m256d hi = _mm256_unpackhi_pd(v0, v1);
    h = _mm256_permute4x64_pd(lo, 0xD8);
    l = _mm256_permute4x64_pd(hi, 0xD8);
}
inline void store4(T *p, __m256d h, __m256d l) {
    __m256d lo = _mm256_unpacklo_pd(h, l);
    __m256d hi = _mm256_unpackhi_pd(h, l);
    __m256d v0 = _mm256_permute2f128_pd(lo, hi, 0x20);
    __m256d v1 = _mm256_permute2f128_pd(lo, hi, 0x31);
    _mm256_storeu_pd(reinterpret_cast<double *>(p),     v0);
    _mm256_storeu_pd(reinterpret_cast<double *>(p + 2), v1);
}
struct Bcast { __m256d h, l; };
inline Bcast bcast(T v) { return Bcast{_mm256_set1_pd(v.limbs[0]), _mm256_set1_pd(v.limbs[1])}; }
#endif

void rotm_neg(std::ptrdiff_t lo, std::ptrdiff_t hi, T *x, T *y,
              T h11, T h12, T h21, T h22) {
    std::ptrdiff_t i = lo;
#ifdef MBLAS_SIMD_DD
    const Bcast b11 = bcast(h11), b12 = bcast(h12), b21 = bcast(h21), b22 = bcast(h22);
    for (; i + 4 <= hi; i += 4) {
        __m256d wh, wl, zh, zl;
        load4(&x[i], wh, wl); load4(&y[i], zh, zl);
        __m256d a_h, a_l, b_h, b_l, nh, nl;
        simd_fast::mul(wh, wl, b11.h, b11.l, a_h, a_l);
        simd_fast::mul(zh, zl, b12.h, b12.l, b_h, b_l);
        simd_fast::add(a_h, a_l, b_h, b_l, nh, nl);
        store4(&x[i], nh, nl);
        simd_fast::mul(wh, wl, b21.h, b21.l, a_h, a_l);
        simd_fast::mul(zh, zl, b22.h, b22.l, b_h, b_l);
        simd_fast::add(a_h, a_l, b_h, b_l, nh, nl);
        store4(&y[i], nh, nl);
    }
#endif
    for (; i < hi; ++i) {
        T w = x[i], z = y[i];
        x[i] = w * h11 + z * h12;
        y[i] = w * h21 + z * h22;
    }
}
void rotm_zero(std::ptrdiff_t lo, std::ptrdiff_t hi, T *x, T *y, T h12, T h21) {
    std::ptrdiff_t i = lo;
#ifdef MBLAS_SIMD_DD
    const Bcast b12 = bcast(h12), b21 = bcast(h21);
    for (; i + 4 <= hi; i += 4) {
        __m256d wh, wl, zh, zl;
        load4(&x[i], wh, wl); load4(&y[i], zh, zl);
        __m256d p_h, p_l, nh, nl;
        simd_fast::mul(zh, zl, b12.h, b12.l, p_h, p_l);
        simd_fast::add(wh, wl, p_h, p_l, nh, nl);
        store4(&x[i], nh, nl);
        simd_fast::mul(wh, wl, b21.h, b21.l, p_h, p_l);
        simd_fast::add(p_h, p_l, zh, zl, nh, nl);
        store4(&y[i], nh, nl);
    }
#endif
    for (; i < hi; ++i) {
        T w = x[i], z = y[i];
        x[i] = w + z * h12;
        y[i] = w * h21 + z;
    }
}
void rotm_pos(std::ptrdiff_t lo, std::ptrdiff_t hi, T *x, T *y, T h11, T h22) {
    std::ptrdiff_t i = lo;
#ifdef MBLAS_SIMD_DD
    const Bcast b11 = bcast(h11), b22 = bcast(h22);
    const __m256d zerov = _mm256_setzero_pd();
    for (; i + 4 <= hi; i += 4) {
        __m256d wh, wl, zh, zl;
        load4(&x[i], wh, wl); load4(&y[i], zh, zl);
        __m256d p_h, p_l, nh, nl;
        simd_fast::mul(wh, wl, b11.h, b11.l, p_h, p_l);
        simd_fast::add(p_h, p_l, zh, zl, nh, nl);
        store4(&x[i], nh, nl);
        simd_fast::mul(b22.h, b22.l, zh, zl, p_h, p_l);
        simd_fast::add(_mm256_sub_pd(zerov, wh), _mm256_sub_pd(zerov, wl), p_h, p_l, nh, nl);
        store4(&y[i], nh, nl);
    }
#endif
    for (; i < hi; ++i) {
        T w = x[i], z = y[i];
        x[i] = w * h11 + z;
        y[i] = T{-w.limbs[0], -w.limbs[1]} + h22 * z;
    }
}

/* Dispatch the flag-selected kernel over a contiguous [0,n) range, threaded over
 * disjoint slices when large enough (each slice runs the same SIMD kernel). */
void rotm_contig(bool neg, bool zero, std::ptrdiff_t n, T *x, T *y,
                 T h11, T h12, T h21, T h22) {
#ifdef _OPENMP
    if (n > MROTM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel()) {
        int nthreads = blas_omp_max_threads();
        #pragma omp parallel num_threads(nthreads)
        {
            int tid = omp_get_thread_num();
            int nth = omp_get_num_threads();
            std::ptrdiff_t chunk = (n + nth - 1) / nth;
            std::ptrdiff_t s = (std::ptrdiff_t)tid * chunk;
            std::ptrdiff_t e = s + chunk;
            if (e > n) e = n;
            if (s < e) {
                if (neg)       rotm_neg(s, e, x, y, h11, h12, h21, h22);
                else if (zero) rotm_zero(s, e, x, y, h12, h21);
                else           rotm_pos(s, e, x, y, h11, h22);
            }
        }
        return;
    }
#endif
    if (neg)       rotm_neg(0, n, x, y, h11, h12, h21, h22);
    else if (zero) rotm_zero(0, n, x, y, h12, h21);
    else           rotm_pos(0, n, x, y, h11, h22);
}
}

extern "C" void mrotm_(const int *n_,
                       T *x, const int *incx_,
                       T *y, const int *incy_,
                       const T *dparam)
{
    const std::ptrdiff_t n = *n_, incx = *incx_, incy = *incy_;
    const T flag = dparam[0];
    /* flag == -2: identity, do nothing */
    if (n <= 0 || eq0(flag + T{2.0, 0.0})) return;

    const bool neg = lt0(flag), zero = eq0(flag);
    const T h11 = dparam[1], h21 = dparam[2], h12 = dparam[3], h22 = dparam[4];

    if (incx == 1 && incy == 1) {
        rotm_contig(neg, zero, n, x, y, h11, h12, h21, h22);
        return;
    }

    /* Strided: gather x,y to unit stride, run the SIMD kernel, scatter back
     * (O(n) gather; bit-exact element-wise — no cross-element dependency). */
    T *xbase = (incx < 0) ? x - (n - 1) * incx : x;
    T *ybase = (incy < 0) ? y - (n - 1) * incy : y;
    std::vector<T> xs(static_cast<std::size_t>(n)), ys(static_cast<std::size_t>(n));
    for (std::ptrdiff_t i = 0; i < n; ++i) { xs[i] = xbase[i * incx]; ys[i] = ybase[i * incy]; }
    rotm_contig(neg, zero, n, xs.data(), ys.data(), h11, h12, h21, h22);
    for (std::ptrdiff_t i = 0; i < n; ++i) { xbase[i * incx] = xs[i]; ybase[i * incy] = ys[i]; }
}
