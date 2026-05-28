/*
 * mgemm_simd_kernel.h — AVX2 SIMD kernel for double-double GEMM.
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
 *   dd_mul :  ~17 fp ops per lane, 1-ulp accurate
 *   dd_add :  ~10 fp ops per lane, 1-ulp accurate
 */
#pragma once

#include <immintrin.h>
#include <multifloats.h>

namespace simd_dd {

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
dd_mul(__m256d ah, __m256d al, __m256d bh, __m256d bl,
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
dd_add(__m256d ah, __m256d al, __m256d bh, __m256d bl,
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
dd_neg(__m256d &h, __m256d &l)
{
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    h = _mm256_xor_pd(h, sign_mask);
    l = _mm256_xor_pd(l, sign_mask);
}

/* (a + b·i) · (c + d·i) = (ac - bd) + (ad + bc)·i — all inputs and
 * outputs are SoA DD pairs (one ymm per limb). 4 dd_mul + 1 dd_add
 * (for r_im) + 1 negate-and-dd_add (for r_re) per call. */
static inline __attribute__((always_inline)) void
cdd_mul(__m256d a_re_h, __m256d a_re_l, __m256d a_im_h, __m256d a_im_l,
        __m256d b_re_h, __m256d b_re_l, __m256d b_im_h, __m256d b_im_l,
        __m256d &r_re_h, __m256d &r_re_l, __m256d &r_im_h, __m256d &r_im_l)
{
    __m256d p_rh, p_rl, p_ih, p_il;
    /* r.re = a.re·b.re - a.im·b.im */
    dd_mul(a_re_h, a_re_l, b_re_h, b_re_l, p_rh, p_rl);
    dd_mul(a_im_h, a_im_l, b_im_h, b_im_l, p_ih, p_il);
    dd_neg(p_ih, p_il);
    dd_add(p_rh, p_rl, p_ih, p_il, r_re_h, r_re_l);
    /* r.im = a.re·b.im + a.im·b.re */
    dd_mul(a_re_h, a_re_l, b_im_h, b_im_l, p_rh, p_rl);
    dd_mul(a_im_h, a_im_l, b_re_h, b_re_l, p_ih, p_il);
    dd_add(p_rh, p_rl, p_ih, p_il, r_im_h, r_im_l);
}

/* (a + b·i) + (c + d·i) = (a + c) + (b + d)·i — 2 dd_adds. */
static inline __attribute__((always_inline)) void
cdd_add(__m256d a_re_h, __m256d a_re_l, __m256d a_im_h, __m256d a_im_l,
        __m256d b_re_h, __m256d b_re_l, __m256d b_im_h, __m256d b_im_l,
        __m256d &r_re_h, __m256d &r_re_l, __m256d &r_im_h, __m256d &r_im_l)
{
    dd_add(a_re_h, a_re_l, b_re_h, b_re_l, r_re_h, r_re_l);
    dd_add(a_im_h, a_im_l, b_im_h, b_im_l, r_im_h, r_im_l);
}

}  // namespace simd_dd
