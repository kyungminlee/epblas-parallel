/*
 * mf_tri_simd.h — the SoA DD loop kernels shared by the triangular / band /
 * packed matvec & solve family (mtrmv, mtbmv, mtbsv, mtpmv, msbmv, mspmv, ...).
 *
 * Every one of those routines reduces, per column, to one of two primitives
 * over a contiguous run of double-double values:
 *
 *   axpy_add(len, xp, cp, t):  xp[i] += t * cp[i]   (NoTrans matvec band update)
 *   axpy_sub(len, xp, cp, t):  xp[i] -= t * cp[i]   (NoTrans solve band update)
 *   dot(len, cp, xp):          sum_i cp[i] * xp[i]  (Trans dot reduction)
 *
 * The axpys are independent writes over i (the scaling temp is a finalized
 * element), so 4-wide-over-i is order-free and BIT-IDENTICAL to the scalar
 * float64x2 operators. The dot is a reduction: it accumulates into a 4-lane
 * vector and folds once with mf_simd::hreduce, which reorders vs the scalar
 * left-fold — so a dot-based leaf matches its reference only within the
 * consistency tolerance (the cross-element recurrence, when there is one, stays
 * in the scalar caller). Both fall back to a plain scalar loop when SIMD is off.
 *
 * Built on the faithful primitives in mf_simd_dd.h; a no-op include otherwise.
 */
#pragma once

#include <cstddef>
#include <multifloats.h>
#include "mf_simd_dd.h"

namespace mf_tri {

using T = multifloats::float64x2;

#ifdef MBLAS_SIMD_DD

/* xp[i] += t*cp[i] — 4-wide SoA, scalar tail. Order-free -> bit-exact. */
static inline void axpy_add(std::ptrdiff_t len, T *xp, const T *cp, const T &t) {
    const __m256d th = _mm256_set1_pd(t.limbs[0]);
    const __m256d tl = _mm256_set1_pd(t.limbs[1]);
    std::ptrdiff_t i = 0;
    for (; i + 4 <= len; i += 4) {
        __m256d mh, ml; mf_simd::load_dd4(cp + i, mh, ml);
        __m256d ph, pl; mf_simd::dd_mul(mh, ml, th, tl, ph, pl);
        __m256d xh, xl; mf_simd::load_dd4(xp + i, xh, xl);
        __m256d rh, rl; mf_simd::dd_add(xh, xl, ph, pl, rh, rl);
        mf_simd::store_dd4(xp + i, rh, rl);
    }
    for (; i < len; ++i) xp[i] = xp[i] + t * cp[i];
}

/* xp[i] -= t*cp[i] — 4-wide SoA, scalar tail. Order-free -> bit-exact. */
static inline void axpy_sub(std::ptrdiff_t len, T *xp, const T *cp, const T &t) {
    const __m256d th = _mm256_set1_pd(t.limbs[0]);
    const __m256d tl = _mm256_set1_pd(t.limbs[1]);
    std::ptrdiff_t i = 0;
    for (; i + 4 <= len; i += 4) {
        __m256d mh, ml; mf_simd::load_dd4(cp + i, mh, ml);
        __m256d ph, pl; mf_simd::dd_mul(mh, ml, th, tl, ph, pl);
        __m256d xh, xl; mf_simd::load_dd4(xp + i, xh, xl);
        __m256d rh, rl; mf_simd::dd_sub(xh, xl, ph, pl, rh, rl);
        mf_simd::store_dd4(xp + i, rh, rl);
    }
    for (; i < len; ++i) xp[i] = xp[i] - t * cp[i];
}

/* sum_i cp[i]*xp[i] — vector accumulator + one hreduce, scalar tail. The fold
 * reorders the reduction -> within tolerance, not bit-exact. */
static inline T dot(std::ptrdiff_t len, const T *cp, const T *xp) {
    __m256d sh = _mm256_setzero_pd(), sl = _mm256_setzero_pd();
    std::ptrdiff_t i = 0;
    for (; i + 4 <= len; i += 4) {
        __m256d mh, ml; mf_simd::load_dd4(cp + i, mh, ml);
        __m256d xh, xl; mf_simd::load_dd4(xp + i, xh, xl);
        __m256d ph, pl; mf_simd::dd_mul(mh, ml, xh, xl, ph, pl);
        mf_simd::dd_add(sh, sl, ph, pl, sh, sl);
    }
    T s = (i >= 4) ? mf_simd::hreduce(sh, sl) : T{0.0, 0.0};
    for (; i < len; ++i) s = s + cp[i] * xp[i];
    return s;
}

#else  /* scalar fallbacks */

static inline void axpy_add(std::ptrdiff_t len, T *xp, const T *cp, const T &t) {
    for (std::ptrdiff_t i = 0; i < len; ++i) xp[i] = xp[i] + t * cp[i];
}
static inline void axpy_sub(std::ptrdiff_t len, T *xp, const T *cp, const T &t) {
    for (std::ptrdiff_t i = 0; i < len; ++i) xp[i] = xp[i] - t * cp[i];
}
static inline T dot(std::ptrdiff_t len, const T *cp, const T *xp) {
    T s{0.0, 0.0};
    for (std::ptrdiff_t i = 0; i < len; ++i) s = s + cp[i] * xp[i];
    return s;
}

#endif  /* MBLAS_SIMD_DD */

}  // namespace mf_tri
