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
using CT = multifloats::complex64x2;

namespace cdetail {
/* scalar complex ops matching the wtrmv/wtpmv references (faithful tails). */
inline CT cmul(const CT &a, const CT &b) {
    return CT{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline CT cadd(const CT &a, const CT &b) { return CT{ a.re + b.re, a.im + b.im }; }
inline CT csub(const CT &a, const CT &b) { return CT{ a.re - b.re, a.im - b.im }; }
inline CT cconj(const CT &a) {
    return CT{ a.re, T{ -a.im.limbs[0], -a.im.limbs[1] } };
}
}

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

/* ---- complex double-double kernels (w family) --------------------------- */

static inline mf_simd::cx4 cbcast(const CT &t) {
    return mf_simd::cx4{ _mm256_set1_pd(t.re.limbs[0]), _mm256_set1_pd(t.re.limbs[1]),
                         _mm256_set1_pd(t.im.limbs[0]), _mm256_set1_pd(t.im.limbs[1]) };
}

/* xp[i] += t*cp[i] — 4-wide complex SoA, scalar tail. Order-free -> bit-exact. */
static inline void caxpy_add(std::ptrdiff_t len, CT *xp, const CT *cp, const CT &t) {
    const mf_simd::cx4 tt = cbcast(t);
    std::ptrdiff_t i = 0;
    for (; i + 4 <= len; i += 4) {
        mf_simd::cx4 c = mf_simd::cload4(cp + i);
        mf_simd::cx4 p = mf_simd::cmul_soa(tt, c);
        mf_simd::cx4 y = mf_simd::cload4(xp + i);
        mf_simd::cstore4(xp + i, mf_simd::cadd_soa(y, p));
    }
    for (; i < len; ++i) xp[i] = cdetail::cadd(xp[i], cdetail::cmul(t, cp[i]));
}

/* xp[i] -= t*cp[i] — 4-wide complex SoA, scalar tail. Order-free -> bit-exact. */
static inline void caxpy_sub(std::ptrdiff_t len, CT *xp, const CT *cp, const CT &t) {
    const mf_simd::cx4 tt = cbcast(t);
    std::ptrdiff_t i = 0;
    for (; i + 4 <= len; i += 4) {
        mf_simd::cx4 c = mf_simd::cload4(cp + i);
        mf_simd::cx4 p = mf_simd::cmul_soa(tt, c);
        mf_simd::cx4 y = mf_simd::cload4(xp + i);
        mf_simd::cstore4(xp + i, mf_simd::csub_soa(y, p));
    }
    for (; i < len; ++i) xp[i] = cdetail::csub(xp[i], cdetail::cmul(t, cp[i]));
}

/* sum_i [conj]cp[i]*xp[i] — vector accumulator + one chreduce, scalar tail.
 * Reorders the reduction -> within tolerance, not bit-exact. */
static inline CT cdot(std::ptrdiff_t len, const CT *cp, const CT *xp, bool conj) {
    mf_simd::cx4 s{ _mm256_setzero_pd(), _mm256_setzero_pd(),
                    _mm256_setzero_pd(), _mm256_setzero_pd() };
    std::ptrdiff_t i = 0;
    for (; i + 4 <= len; i += 4) {
        mf_simd::cx4 a = mf_simd::cload4(cp + i);
        if (conj) a = mf_simd::cconj_soa(a);
        mf_simd::cx4 x = mf_simd::cload4(xp + i);
        s = mf_simd::cadd_soa(s, mf_simd::cmul_soa(a, x));
    }
    CT acc = (i >= 4) ? mf_simd::chreduce(s) : CT{ T{0.0, 0.0}, T{0.0, 0.0} };
    for (; i < len; ++i) {
        const CT e = conj ? cdetail::cconj(cp[i]) : cp[i];
        acc = cdetail::cadd(acc, cdetail::cmul(e, xp[i]));
    }
    return acc;
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

static inline void caxpy_add(std::ptrdiff_t len, CT *xp, const CT *cp, const CT &t) {
    for (std::ptrdiff_t i = 0; i < len; ++i) xp[i] = cdetail::cadd(xp[i], cdetail::cmul(t, cp[i]));
}
static inline void caxpy_sub(std::ptrdiff_t len, CT *xp, const CT *cp, const CT &t) {
    for (std::ptrdiff_t i = 0; i < len; ++i) xp[i] = cdetail::csub(xp[i], cdetail::cmul(t, cp[i]));
}
static inline CT cdot(std::ptrdiff_t len, const CT *cp, const CT *xp, bool conj) {
    CT acc{ T{0.0, 0.0}, T{0.0, 0.0} };
    for (std::ptrdiff_t i = 0; i < len; ++i) {
        const CT e = conj ? cdetail::cconj(cp[i]) : cp[i];
        acc = cdetail::cadd(acc, cdetail::cmul(e, xp[i]));
    }
    return acc;
}

#endif  /* MBLAS_SIMD_DD */

}  // namespace mf_tri
