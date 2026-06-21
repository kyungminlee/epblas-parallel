/*
 * mf_kernels.h — the shaped double-double loop kernels, built on the faithful
 * SoA primitives in mf_simd_exact.h. One namespace (`mf_kernels`) for what were
 * three BLAS-lineage-split files:
 *
 *   - tri/band/packed matvec & solve column primitives (axpy_add/axpy_sub/dot
 *     and complex twins caxpy_add/caxpy_sub/cdot/caxpy2_add) — mtrmv, mtbmv,
 *     mtbsv, mtpmv, msbmv, mspmv, and the w-complex family.
 *   - rank-update "axpy over a column" kernels (dd_axpy/dd_axpy2) — mspr/mspr2/
 *     msyr/msyr2 and complex twins.
 *   - the contiguous complex-dot kernels (wdotu_unit/wdotc_unit), declared here
 *     and defined out-of-line in wdotu.cpp/wdotc.cpp so the packed/banded
 *     triangular matvecs can share the wide Bailey accumulator.
 *
 * All three were "shaped loop kernels on the faithful set with a scalar #else,"
 * artificially split by BLAS family. The numeric contract is in the namespace at
 * every call site (mf_kernels:: shaped loops on simd_exact:: primitives).
 *
 * The axpys are independent writes over i (the scaling temp is a finalized
 * element), so 4-wide-over-i is order-free and BIT-IDENTICAL to the scalar
 * float64x2 operators. The dots are reductions: they accumulate into a 4-lane
 * vector and fold once with simd_exact::hreduce, which reorders vs the scalar
 * left-fold — so a dot-based leaf matches its reference only within the
 * consistency tolerance (the cross-element recurrence, when there is one, stays
 * in the scalar caller). Everything falls back to a plain scalar loop when SIMD
 * is off, so callers need no #ifdef of their own.
 */
#pragma once

#include <cstddef>
#include <multifloats.h>
#include "mf_simd_exact.h"
#include "mf_pred.h"

namespace mf_kernels {

using T = multifloats::float64x2;
using CT = multifloats::complex64x2;

/* scalar complex ops matching the wtrmv/wtpmv references (faithful tails); the
 * shared scalar-complex API for the w family (whbmv/whpmv/wgbmv/wher/.../wtbsv). */
inline CT cmul(const CT &a, const CT &b) {
    return CT{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline CT cadd(const CT &a, const CT &b) { return CT{ a.re + b.re, a.im + b.im }; }
inline CT csub(const CT &a, const CT &b) { return CT{ a.re - b.re, a.im - b.im }; }
inline CT cconj(const CT &a) {
    return CT{ a.re, T{ -a.im.limbs[0], -a.im.limbs[1] } };
}
/* real scalar * complex (the old scale_r / cmul_r / rcmul trio; scalar-first.
 * DD multiply is commutative bit-for-bit, so the operand order is cosmetic). */
inline CT rcmul(const T &r, const CT &z) { return CT{ r * z.re, r * z.im }; }

/* y := beta*y, or y := 0 when beta == 0, over the length-n vector y with stride
 * inc — the beta prologue every dense/banded/packed L2 matvec runs on its output
 * before the alpha*A*x accumulation. No-op when beta == 1. A negative stride
 * starts at the far end so the walk visits elements in the same order as the
 * matvec update. Cold O(n) setup (one pass, never a SIMD hot loop); the eq tests
 * come from mf_pred. DD (and complex-DD) multiply is commutative bit-for-bit, so
 * the operand order is cosmetic — call sites that wrote beta*y and y*beta both
 * map here unchanged. */
static inline void scale_y(int n, const T &beta, T *y, int inc) {
    if (mf_pred::eq1(beta)) return;
    int iy = (inc < 0) ? -(n - 1) * inc : 0;
    if (mf_pred::eq0(beta)) for (int i = 0; i < n; ++i) { y[iy] = T{}; iy += inc; }
    else                    for (int i = 0; i < n; ++i) { y[iy] = beta * y[iy]; iy += inc; }
}
static inline void cscale_y(int n, const CT &beta, CT *y, int inc) {
    if (mf_pred::ceq1(beta)) return;
    int iy = (inc < 0) ? -(n - 1) * inc : 0;
    if (mf_pred::ceq0(beta)) for (int i = 0; i < n; ++i) { y[iy] = CT{}; iy += inc; }
    else                     for (int i = 0; i < n; ++i) { y[iy] = cmul(beta, y[iy]); iy += inc; }
}

/* Gather the length-n strided vector x (stride inc) into the contiguous scratch
 * dst, or scatter the contiguous src back into x — the O(n) data movement every
 * strided L2 matvec runs to reach its unit-stride SIMD core (gather x/y, run the
 * core, scatter y). A negative stride starts at the far end so logical element i
 * lands at dst[i] either way, matching the matvec's element order. Pass the RAW
 * caller pointer (the far-end adjustment is computed here, NOT pre-applied). Cold
 * straight copies, zero arithmetic -> no rounding-order change, bit-identical to
 * the hand-written gather/scatter loops they replace. Same class as scale_y. */
template <class V>
static inline void gather_strided(int n, const V *x, int inc, V *dst) {
    const V *base = (inc < 0) ? x - (std::ptrdiff_t)(n - 1) * inc : x;
    for (int i = 0; i < n; ++i) dst[i] = base[(std::ptrdiff_t)i * inc];
}
template <class V>
static inline void scatter_strided(int n, V *x, int inc, const V *src) {
    V *base = (inc < 0) ? x - (std::ptrdiff_t)(n - 1) * inc : x;
    for (int i = 0; i < n; ++i) base[(std::ptrdiff_t)i * inc] = src[i];
}

#ifdef MBLAS_SIMD_DD

/* xp[i] += t*cp[i] — 4-wide SoA, scalar tail. Order-free -> bit-exact. */
static inline void axpy_add(std::ptrdiff_t len, T *xp, const T *cp, const T &t) {
    const __m256d th = _mm256_set1_pd(t.limbs[0]);
    const __m256d tl = _mm256_set1_pd(t.limbs[1]);
    std::ptrdiff_t i = 0;
    for (; i + 4 <= len; i += 4) {
        __m256d mh, ml; simd_exact::load_dd4(cp + i, mh, ml);
        __m256d ph, pl; simd_exact::mul(mh, ml, th, tl, ph, pl);
        __m256d xh, xl; simd_exact::load_dd4(xp + i, xh, xl);
        __m256d rh, rl; simd_exact::add(xh, xl, ph, pl, rh, rl);
        simd_exact::store_dd4(xp + i, rh, rl);
    }
    for (; i < len; ++i) xp[i] = xp[i] + t * cp[i];
}

/* xp[i] -= t*cp[i] — 4-wide SoA, scalar tail. Order-free -> bit-exact. */
static inline void axpy_sub(std::ptrdiff_t len, T *xp, const T *cp, const T &t) {
    const __m256d th = _mm256_set1_pd(t.limbs[0]);
    const __m256d tl = _mm256_set1_pd(t.limbs[1]);
    std::ptrdiff_t i = 0;
    for (; i + 4 <= len; i += 4) {
        __m256d mh, ml; simd_exact::load_dd4(cp + i, mh, ml);
        __m256d ph, pl; simd_exact::mul(mh, ml, th, tl, ph, pl);
        __m256d xh, xl; simd_exact::load_dd4(xp + i, xh, xl);
        __m256d rh, rl; simd_exact::sub(xh, xl, ph, pl, rh, rl);
        simd_exact::store_dd4(xp + i, rh, rl);
    }
    for (; i < len; ++i) xp[i] = xp[i] - t * cp[i];
}

/* sum_i cp[i]*xp[i] — vector accumulator + one hreduce, scalar tail. The fold
 * reorders the reduction -> within tolerance, not bit-exact. */
static inline T dot(std::ptrdiff_t len, const T *cp, const T *xp) {
    __m256d sh = _mm256_setzero_pd(), sl = _mm256_setzero_pd();
    std::ptrdiff_t i = 0;
    for (; i + 4 <= len; i += 4) {
        __m256d mh, ml; simd_exact::load_dd4(cp + i, mh, ml);
        __m256d xh, xl; simd_exact::load_dd4(xp + i, xh, xl);
        __m256d ph, pl; simd_exact::mul(mh, ml, xh, xl, ph, pl);
        simd_exact::add(sh, sl, ph, pl, sh, sl);
    }
    T s = (i >= 4) ? simd_exact::hreduce(sh, sl) : T{0.0, 0.0};
    for (; i < len; ++i) s = s + cp[i] * xp[i];
    return s;
}

/* ---- complex double-double kernels (w family) --------------------------- */

/* xp[i] += t*cp[i] — 4-wide complex SoA, scalar tail. Order-free -> bit-exact. */
static inline void caxpy_add(std::ptrdiff_t len, CT *xp, const CT *cp, const CT &t) {
    const simd_exact::cx4 tt = simd_exact::vbcast(t);
    std::ptrdiff_t i = 0;
    /* 8-wide head: two INDEPENDENT 4-lane complex chains. A complex MAC
     * (cmul_soa = 4 mul + sub + add) is a long latency chain with no native
     * scalar-DD SIMD; one chain leaves it exposed while the column streams from
     * cache -> ~12-15% cache-resident & serial. Unlike the FUSED rank-2 form
     * (caxpy2_add note: held both scalars + both source vecs + two products ->
     * spilled), this holds one broadcast scalar + two independent groups, so
     * peak ymm pressure stays under 16 (verified: no spill). Bit-identical. */
    for (; i + 8 <= len; i += 8) {
        simd_exact::cx4 c0 = simd_exact::cload4(cp + i);
        simd_exact::cx4 c1 = simd_exact::cload4(cp + i + 4);
        simd_exact::cx4 p0 = simd_exact::cmul_soa(tt, c0);
        simd_exact::cx4 p1 = simd_exact::cmul_soa(tt, c1);
        simd_exact::cx4 y0 = simd_exact::cload4(xp + i);
        simd_exact::cx4 y1 = simd_exact::cload4(xp + i + 4);
        simd_exact::cstore4(xp + i,     simd_exact::cadd_soa(y0, p0));
        simd_exact::cstore4(xp + i + 4, simd_exact::cadd_soa(y1, p1));
    }
    for (; i + 4 <= len; i += 4) {
        simd_exact::cx4 c = simd_exact::cload4(cp + i);
        simd_exact::cx4 p = simd_exact::cmul_soa(tt, c);
        simd_exact::cx4 y = simd_exact::cload4(xp + i);
        simd_exact::cstore4(xp + i, simd_exact::cadd_soa(y, p));
    }
    for (; i < len; ++i) xp[i] = cadd(xp[i], cmul(t, cp[i]));
}

/* xp[i] -= t*cp[i] — 4-wide complex SoA, scalar tail. Order-free -> bit-exact. */
static inline void caxpy_sub(std::ptrdiff_t len, CT *xp, const CT *cp, const CT &t) {
    const simd_exact::cx4 tt = simd_exact::vbcast(t);
    std::ptrdiff_t i = 0;
    for (; i + 4 <= len; i += 4) {
        simd_exact::cx4 c = simd_exact::cload4(cp + i);
        simd_exact::cx4 p = simd_exact::cmul_soa(tt, c);
        simd_exact::cx4 y = simd_exact::cload4(xp + i);
        simd_exact::cstore4(xp + i, simd_exact::csub_soa(y, p));
    }
    for (; i < len; ++i) xp[i] = csub(xp[i], cmul(t, cp[i]));
}

/* sum_i [conj]cp[i]*xp[i] — vector accumulator + one chreduce, scalar tail.
 * Reorders the reduction -> within tolerance, not bit-exact. */
static inline CT cdot(std::ptrdiff_t len, const CT *cp, const CT *xp, bool conj) {
    simd_exact::cx4 s{ _mm256_setzero_pd(), _mm256_setzero_pd(),
                    _mm256_setzero_pd(), _mm256_setzero_pd() };
    std::ptrdiff_t i = 0;
    for (; i + 4 <= len; i += 4) {
        simd_exact::cx4 a = simd_exact::cload4(cp + i);
        if (conj) a = simd_exact::cconj_soa(a);
        simd_exact::cx4 x = simd_exact::cload4(xp + i);
        s = simd_exact::cadd_soa(s, simd_exact::cmul_soa(a, x));
    }
    CT acc = (i >= 4) ? simd_exact::chreduce(s) : CT{ T{0.0, 0.0}, T{0.0, 0.0} };
    for (; i < len; ++i) {
        const CT e = conj ? cconj(cp[i]) : cp[i];
        acc = cadd(acc, cmul(e, xp[i]));
    }
    return acc;
}

/* ap[i] := ap[i] + {xh[i], xl[i]} * {th, tl} — the rank-update column axpy
 * (mspr/mspr2/msyr/msyr2 + complex twins); x pre-split into SoA limb arrays by
 * the caller (doubles as the strided-x gather, so the kernel is unit-stride). */
static inline void
dd_axpy(int n, const double *xh, const double *xl,
        double th, double tl, multifloats::float64x2 *ap)
{
    const __m256d thb = _mm256_set1_pd(th);
    const __m256d tlb = _mm256_set1_pd(tl);
    int i = 0;
    /* 8-wide head: two INDEPENDENT 4-lane DD chains. The mul->add EFT chain
     * is long-latency with no native SIMD; a single chain leaves that latency
     * exposed even when the column streams from cache (matrix ~16 MB at N=1024 is
     * only partly L3-resident -> not yet pure-bandwidth-bound). A second chain
     * fills the pipeline -> ~7% at N=1024, ~11% cache-resident. Bit-identical:
     * each 4-lane group is computed exactly as the 4-wide loop would. */
    for (; i + 8 <= n; i += 8) {
        __m256d p0h, p0l, p1h, p1l;
        simd_exact::mul(_mm256_loadu_pd(xh + i),     _mm256_loadu_pd(xl + i),     thb, tlb, p0h, p0l);
        simd_exact::mul(_mm256_loadu_pd(xh + i + 4), _mm256_loadu_pd(xl + i + 4), thb, tlb, p1h, p1l);
        __m256d a0h, a0l, a1h, a1l;
        simd_exact::load_dd4(ap + i,     a0h, a0l);
        simd_exact::load_dd4(ap + i + 4, a1h, a1l);
        __m256d r0h, r0l, r1h, r1l;
        simd_exact::add(a0h, a0l, p0h, p0l, r0h, r0l);
        simd_exact::add(a1h, a1l, p1h, p1l, r1h, r1l);
        simd_exact::store_dd4(ap + i,     r0h, r0l);
        simd_exact::store_dd4(ap + i + 4, r1h, r1l);
    }
    for (; i + 4 <= n; i += 4) {
        __m256d xhv = _mm256_loadu_pd(xh + i);
        __m256d xlv = _mm256_loadu_pd(xl + i);
        __m256d ph, pl;
        simd_exact::mul(xhv, xlv, thb, tlb, ph, pl);
        __m256d ahv, alv;
        simd_exact::load_dd4(ap + i, ahv, alv);
        __m256d rh, rl;
        simd_exact::add(ahv, alv, ph, pl, rh, rl);
        simd_exact::store_dd4(ap + i, rh, rl);
    }
    for (; i < n; ++i) {  /* scalar tail — identical EFTs via the operators */
        multifloats::float64x2 xv{xh[i], xl[i]};
        multifloats::float64x2 tv{th, tl};
        ap[i] = ap[i] + xv * tv;
    }
}

/* Fused rank-2: ap[i] := (ap[i] + {xh,xl}[i]*{th,tl}) + {yh,yl}[i]*{sh,sl},
 * the left-associative order of the reference `ap + x*t1 + y*t2`. One ap
 * read/write (vs two dd_axpy passes). */
static inline void
dd_axpy2(int n, const double *xh, const double *xl, double th, double tl,
         const double *yh, const double *yl, double sh, double sl,
         multifloats::float64x2 *ap)
{
    const __m256d thb = _mm256_set1_pd(th), tlb = _mm256_set1_pd(tl);
    const __m256d shb = _mm256_set1_pd(sh), slb = _mm256_set1_pd(sl);
    int i = 0;
    /* 8-wide head: two INDEPENDENT 4-lane chains. The fused rank-2 element chain
     * (load -> +x*t1 -> +y*t2 -> store) is twice as long as the rank-1 one and
     * stays compute/latency-bound even at N=1024 omp4 (the bandwidth wall that
     * caps the rank-1 worst cell does not bind here), so a second chain wins ~9-10%
     * across all sizes incl. N=1024. Bit-identical — each group matches the 4-wide
     * loop op-for-op; no register spill (verified, AVX2 16 ymm). */
    for (; i + 8 <= n; i += 8) {
        __m256d a0h, a0l, a1h, a1l;
        simd_exact::load_dd4(ap + i,     a0h, a0l);
        simd_exact::load_dd4(ap + i + 4, a1h, a1l);
        __m256d p0h, p0l, p1h, p1l;
        simd_exact::mul(_mm256_loadu_pd(xh + i),     _mm256_loadu_pd(xl + i),     thb, tlb, p0h, p0l);
        simd_exact::mul(_mm256_loadu_pd(xh + i + 4), _mm256_loadu_pd(xl + i + 4), thb, tlb, p1h, p1l);
        simd_exact::add(a0h, a0l, p0h, p0l, a0h, a0l);        /* ap + x*t1 */
        simd_exact::add(a1h, a1l, p1h, p1l, a1h, a1l);
        simd_exact::mul(_mm256_loadu_pd(yh + i),     _mm256_loadu_pd(yl + i),     shb, slb, p0h, p0l);
        simd_exact::mul(_mm256_loadu_pd(yh + i + 4), _mm256_loadu_pd(yl + i + 4), shb, slb, p1h, p1l);
        simd_exact::add(a0h, a0l, p0h, p0l, a0h, a0l);        /* + y*t2    */
        simd_exact::add(a1h, a1l, p1h, p1l, a1h, a1l);
        simd_exact::store_dd4(ap + i,     a0h, a0l);
        simd_exact::store_dd4(ap + i + 4, a1h, a1l);
    }
    for (; i + 4 <= n; i += 4) {
        __m256d ahv, alv;
        simd_exact::load_dd4(ap + i, ahv, alv);
        __m256d ph, pl;
        simd_exact::mul(_mm256_loadu_pd(xh + i), _mm256_loadu_pd(xl + i), thb, tlb, ph, pl);
        simd_exact::add(ahv, alv, ph, pl, ahv, alv);          /* ap + x*t1 */
        simd_exact::mul(_mm256_loadu_pd(yh + i), _mm256_loadu_pd(yl + i), shb, slb, ph, pl);
        simd_exact::add(ahv, alv, ph, pl, ahv, alv);          /* + y*t2    */
        simd_exact::store_dd4(ap + i, ahv, alv);
    }
    for (; i < n; ++i) {
        const multifloats::float64x2 xv{xh[i], xl[i]}, tv{th, tl};
        const multifloats::float64x2 yv{yh[i], yl[i]}, sv{sh, sl};
        ap[i] = ap[i] + xv * tv + yv * sv;
    }
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
    for (std::ptrdiff_t i = 0; i < len; ++i) xp[i] = cadd(xp[i], cmul(t, cp[i]));
}
static inline void caxpy_sub(std::ptrdiff_t len, CT *xp, const CT *cp, const CT &t) {
    for (std::ptrdiff_t i = 0; i < len; ++i) xp[i] = csub(xp[i], cmul(t, cp[i]));
}
static inline CT cdot(std::ptrdiff_t len, const CT *cp, const CT *xp, bool conj) {
    CT acc{ T{0.0, 0.0}, T{0.0, 0.0} };
    for (std::ptrdiff_t i = 0; i < len; ++i) {
        const CT e = conj ? cconj(cp[i]) : cp[i];
        acc = cadd(acc, cmul(e, xp[i]));
    }
    return acc;
}

static inline void
dd_axpy(int n, const double *xh, const double *xl,
        double th, double tl, multifloats::float64x2 *ap)
{
    const multifloats::float64x2 tv{th, tl};
    for (int i = 0; i < n; ++i) {
        const multifloats::float64x2 xv{xh[i], xl[i]};
        ap[i] = ap[i] + xv * tv;
    }
}

static inline void
dd_axpy2(int n, const double *xh, const double *xl, double th, double tl,
         const double *yh, const double *yl, double sh, double sl,
         multifloats::float64x2 *ap)
{
    const multifloats::float64x2 tv{th, tl}, sv{sh, sl};
    for (int i = 0; i < n; ++i) {
        const multifloats::float64x2 xv{xh[i], xl[i]}, yv{yh[i], yl[i]};
        ap[i] = ap[i] + xv * tv + yv * sv;
    }
}

#endif  /* MBLAS_SIMD_DD */

/* xp[i] += t1*c1[i] + t2*c2[i] — complex rank-2 column AXPY run as two
 * independent rank-1 caxpy_add passes. The fused single-pass form spilled (it
 * must hold both broadcast scalars + both source vectors + the two products,
 * exceeding the 16 ymm registers, ~3.4x slower per flop); two passes of the
 * low-pressure kernel are ~3x faster. Reassociating the per-element sum
 * ((xp+t1*c1)+t2*c2 vs xp+(t1*c1+t2*c2)) matches the scalar reference within DD
 * fuzz tol, not bit-for-bit. */
static inline void caxpy2_add(std::ptrdiff_t len, CT *xp,
                              const CT *c1, const CT &t1,
                              const CT *c2, const CT &t2) {
    caxpy_add(len, xp, c1, t1);
    caxpy_add(len, xp, c2, t2);
}

/* Σ X·Y / Σ conj(X)·Y over contiguous unit-stride ranges — the wide Bailey
 * 3-limb accumulator kernels, defined out-of-line in wdotu.cpp/wdotc.cpp and
 * shared by the OpenMP partial-reduction and the packed/banded complex Trans
 * matvecs (wtpmv). */
CT wdotu_unit(int n, const CT *x, const CT *y);
CT wdotc_unit(int n, const CT *x, const CT *y);

}  // namespace mf_kernels
