/*
 * mf_simd_fast.h — the FAST (1-ulp, FMA-fused) 4-wide SoA double-double
 * primitive vocabulary, shared by the GEMM/GEMV/GER/SYMV/L3 family and the
 * tolerance-bound L1 RMW routines. Counterpart to the bit-exact set in
 * mf_simd_exact.h.
 *
 * 4-wide SoA: each DD value (hi, lo) becomes one lane of a pair of
 * __m256d vectors. The inner micro-kernel processes one row of A
 * against NR=4 columns of B in parallel, producing 4 DD-accumulator
 * cells across the contraction.
 *
 * AVX2 + FMA assumed (Haswell+; gcc requires -mavx2 -mfma). This
 * header is only included when MBLAS_SIMD_DD is on, and the cmake
 * gate also checks the target compiler flags.
 *
 * Algorithms:
 *   twoprod: p = a*b, e = fma(a, b, -p)              (exact, 1 mul + 1 fma)
 *   twosum : s = a+b, e = (a - (s-(s-a))) + (b - (s-a))  (exact, 6 ops)
 *   mul :  ~17 fp ops per lane, 1-ulp accurate
 *   add :  ~10 fp ops per lane, 1-ulp accurate
 *
 * Three-way numeric split across the multifloats SIMD headers (documented
 * here so it is not re-flagged): (1) the FAST 1-ulp 2-limb mul/add set, below;
 * (2) the FAITHFUL 2-limb set + loads/gathers/hreduce in mf_simd_exact.h;
 * (3) the BAILEY 3-limb wide accumulator (absorb/renorm3/dd_prod) used by the
 * long L1 reductions — it lives here, alongside the fast set, with
 * horizontal_dd as its 2-limb finalizer. All three share ONE EFT copy
 * (twoprod/twosum/fast2sum, below); there is deliberately no second EFT home.
 */
#pragma once

#include <immintrin.h>
#include <multifloats.h>

namespace simd_fast {

constexpr int NR = 4;

/* Error-free transforms ---------------------------------------------- */

static inline __attribute__((always_inline)) void
twoprod(__m256d a, __m256d b, __m256d &p, __m256d &e)
{
    p = _mm256_mul_pd(a, b);
    /* e = a*b - p, computed exactly via FMA */
    e = _mm256_fmsub_pd(a, b, p);
}

static inline __attribute__((always_inline)) void
twosum(__m256d a, __m256d b, __m256d &s, __m256d &e)
{
    s = _mm256_add_pd(a, b);
    __m256d bb = _mm256_sub_pd(s, a);
    __m256d aa = _mm256_sub_pd(s, bb);
    __m256d da = _mm256_sub_pd(a, aa);
    __m256d db = _mm256_sub_pd(b, bb);
    e = _mm256_add_pd(da, db);
}

/* fast2sum: assumes |a| >= |b|; cheaper than twosum (3 ops vs 6). */
static inline __attribute__((always_inline)) void
fast2sum(__m256d a, __m256d b, __m256d &s, __m256d &e)
{
    s = _mm256_add_pd(a, b);
    __m256d t = _mm256_sub_pd(s, a);
    e = _mm256_sub_pd(b, t);
}

/* DD * DD = DD ------------------------------------------------------- */

static inline __attribute__((always_inline)) void
mul(__m256d ah, __m256d al, __m256d bh, __m256d bl,
       __m256d &rh, __m256d &rl)
{
    __m256d ph, pe;
    twoprod(ah, bh, ph, pe);
    /* pe accumulates the cross terms: ah*bl + al*bh */
    pe = _mm256_fmadd_pd(ah, bl, pe);
    pe = _mm256_fmadd_pd(al, bh, pe);
    fast2sum(ph, pe, rh, rl);   /* |ph| dominates by 2^53 */
}

/* DD + DD = DD ------------------------------------------------------- */

static inline __attribute__((always_inline)) void
add(__m256d ah, __m256d al, __m256d bh, __m256d bl,
       __m256d &rh, __m256d &rl)
{
    __m256d sh, se;
    twosum(ah, bh, sh, se);
    __m256d t = _mm256_add_pd(al, bl);
    se = _mm256_add_pd(se, t);
    fast2sum(sh, se, rh, rl);
}

/* Complex-DD primitives (SoA: each ymm carries 4 lanes of one real
 * component — re_hi / re_lo / im_hi / im_lo treated independently). */

/* Negate a DD pair (xor-flip the sign bit of both limbs). */
static inline __attribute__((always_inline)) void
neg(__m256d &h, __m256d &l)
{
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    h = _mm256_xor_pd(h, sign_mask);
    l = _mm256_xor_pd(l, sign_mask);
}

/* (a + b·i) · (c + d·i) = (ac - bd) + (ad + bc)·i — all inputs and
 * outputs are SoA DD pairs (one ymm per limb). 4 mul + 1 add
 * (for r_im) + 1 negate-and-add (for r_re) per call. */
static inline __attribute__((always_inline)) void
cmul(__m256d a_re_h, __m256d a_re_l, __m256d a_im_h, __m256d a_im_l,
        __m256d b_re_h, __m256d b_re_l, __m256d b_im_h, __m256d b_im_l,
        __m256d &r_re_h, __m256d &r_re_l, __m256d &r_im_h, __m256d &r_im_l)
{
    __m256d p_rh, p_rl, p_ih, p_il;
    /* r.re = a.re·b.re - a.im·b.im */
    mul(a_re_h, a_re_l, b_re_h, b_re_l, p_rh, p_rl);
    mul(a_im_h, a_im_l, b_im_h, b_im_l, p_ih, p_il);
    neg(p_ih, p_il);
    add(p_rh, p_rl, p_ih, p_il, r_re_h, r_re_l);
    /* r.im = a.re·b.im + a.im·b.re */
    mul(a_re_h, a_re_l, b_im_h, b_im_l, p_rh, p_rl);
    mul(a_im_h, a_im_l, b_re_h, b_re_l, p_ih, p_il);
    add(p_rh, p_rl, p_ih, p_il, r_im_h, r_im_l);
}

/* (a + b·i) + (c + d·i) = (a + c) + (b + d)·i — 2 dd_adds. */
static inline __attribute__((always_inline)) void
cadd(__m256d a_re_h, __m256d a_re_l, __m256d a_im_h, __m256d a_im_l,
        __m256d b_re_h, __m256d b_re_l, __m256d b_im_h, __m256d b_im_l,
        __m256d &r_re_h, __m256d &r_re_l, __m256d &r_im_h, __m256d &r_im_l)
{
    add(a_re_h, a_re_l, b_re_h, b_re_l, r_re_h, r_re_l);
    add(a_im_h, a_im_l, b_im_h, b_im_l, r_im_h, r_im_l);
}

/* Bailey 3-limb wide accumulator (long L1 reductions) ----------------- *
 * A THIRD numeric scheme, distinct from the 1-ulp 2-limb mul/add above and
 * from the faithful 2-limb set in mf_simd_exact.h: a 3-double-per-lane
 * accumulator (a0,a1,a2) for long dot/asum/nrm2 reductions, renormalized
 * periodically. Shared by masum/mdot/mnrm2/mwasum/mwnrm2/wdotu/wdotc.
 * (Names are NOT subject to the prefix drop — they are the Bailey layer.) */

/* 2-double finalizer: fold the 4 SoA lanes (h,l) into one DD scalar. NOT the
 * accumulator — the tail fold after the wide loop. */
static inline __attribute__((always_inline)) multifloats::float64x2
horizontal_dd(__m256d h, __m256d l)
{
    alignas(32) double ha[4], la[4];
    _mm256_store_pd(ha, h); _mm256_store_pd(la, l);
    multifloats::float64x2 s{ha[0], la[0]};
    for (int k = 1; k < 4; ++k) s = s + multifloats::float64x2{ha[k], la[k]};
    return s;
}

/* Absorb a DD product (ph, pl) into the wide acc (a0, a1, a2). */
static inline __attribute__((always_inline)) void
absorb(__m256d ph, __m256d pl, __m256d &a0, __m256d &a1, __m256d &a2)
{
    __m256d e0, e1, e2;
    twosum(a0, ph, a0, e0);
    twosum(a1, pl, a1, e1);
    twosum(a1, e0, a1, e2);
    a2 = _mm256_add_pd(a2, _mm256_add_pd(e1, e2));
}

/* Renormalize the wide acc back toward (a0, a1, 0). */
static inline __attribute__((always_inline)) void
renorm3(__m256d &a0, __m256d &a1, __m256d &a2)
{
    __m256d t, e;
    fast2sum(a1, a2, t, e);
    a1 = t; a2 = e;
    fast2sum(a0, a1, a0, a1);
    a1 = _mm256_add_pd(a1, a2);
    fast2sum(a0, a1, a0, a1);
    a2 = _mm256_setzero_pd();
}

/* DD product dropping the xl*yl term -> (ph, pl). */
static inline __attribute__((always_inline)) void
dd_prod(__m256d xh, __m256d xl, __m256d yh, __m256d yl, __m256d &ph, __m256d &pl)
{
    twoprod(xh, yh, ph, pl);
    pl = _mm256_add_pd(pl,
            _mm256_add_pd(_mm256_mul_pd(xh, yl), _mm256_mul_pd(xl, yh)));
}

}  // namespace simd_fast
