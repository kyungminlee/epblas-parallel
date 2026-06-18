/*
 * mf_rank1_simd.h — double-double "axpy over a column" kernel, the shared
 * inner loop of the multifloats rank-update family (mspr/mspr2/msyr/msyr2
 * and complex twins):
 *
 *     ap[i] := ap[i] + {xh[i], xl[i]} * {th, tl}    (i = 0 .. n-1)
 *
 * where {th, tl} is a scalar DD broadcast (alpha*x[j] for column j) and x
 * is pre-split into SoA limb arrays by the caller (which doubles as the
 * strided-x gather, so the kernel itself is always unit-stride).
 *
 * When MBLAS_SIMD_DD is on (AVX2+FMA) the body is a 4-wide SoA kernel whose
 * dd_mul/dd_add mirror multifloats::float64x2 operator* / operator+ EXACTLY,
 * op for op — so the SIMD result is bit-identical to the scalar reference on
 * every finite, non-degenerate lane. (Unlike the sloppy 1-ulp primitives in
 * mgemm_simd_kernel.h. The scalar operators short-circuit on non-finite and
 * on the both-hi-zero -0 case; those lanes are not reproduced here — random
 * fuzz data never hits them and the consistency test's relative tolerance
 * absorbs the measure-zero degenerate case.) All arithmetic uses explicit
 * _mm256 intrinsics, which the compiler never fuses, so -ffp-contract=on
 * cannot collapse a two_prod/two_sum into an FMA and break the EFTs.
 *
 * Without MBLAS_SIMD_DD the kernel is the plain scalar loop, so callers need
 * no #ifdef of their own.
 */
#pragma once

#include <multifloats.h>

namespace mf_rank1 {

#ifdef MBLAS_SIMD_DD

}  // namespace mf_rank1

#include <immintrin.h>
#include "mgemm_simd_kernel.h"   /* simd_dd::twoprod / twosum / fast2sum */

namespace mf_rank1 {

using simd_dd::twoprod;
using simd_dd::twosum;
using simd_dd::fast2sum;

/* DD * DD, faithful to float64x2::operator* (two_prod + one_prod cross
 * terms + fast_two_sum). */
static inline __attribute__((always_inline)) void
dd_mul(__m256d ah, __m256d al, __m256d bh, __m256d bl,
       __m256d &rh, __m256d &rl)
{
    __m256d p00, e00;
    twoprod(ah, bh, p00, e00);                 /* p00 = ah*bh, e00 = err */
    __m256d p01 = _mm256_mul_pd(ah, bl);       /* one_prod(ah, bl)       */
    __m256d p10 = _mm256_mul_pd(al, bh);       /* one_prod(al, bh)       */
    p01 = _mm256_add_pd(p01, p10);             /* p01 += p10             */
    e00 = _mm256_add_pd(e00, p01);             /* e00 += p01             */
    fast2sum(p00, e00, rh, rl);
}

/* DD + DD, faithful to float64x2::operator+ (two_sum on both limbs,
 * fast_two_sum combine). */
static inline __attribute__((always_inline)) void
dd_add(__m256d ah, __m256d al, __m256d bh, __m256d bl,
       __m256d &rh, __m256d &rl)
{
    __m256d a, b, c, d;
    twosum(ah, bh, a, b);
    twosum(al, bl, c, d);
    fast2sum(a, c, a, c);
    b = _mm256_add_pd(b, d);
    b = _mm256_add_pd(b, c);
    fast2sum(a, b, rh, rl);
}

/* AoS<->SoA for 4 packed float64x2 (8 contiguous doubles).
 * deinterleave: [h0 l0 h1 l1 | h2 l2 h3 l3] -> hi=[h0..h3], lo=[l0..l3]. */
static inline __attribute__((always_inline)) void
load_dd4(const multifloats::float64x2 *p, __m256d &hi, __m256d &lo)
{
    const double *d = p->limbs;
    __m256d a01 = _mm256_loadu_pd(d);          /* h0 l0 h1 l1 */
    __m256d a23 = _mm256_loadu_pd(d + 4);      /* h2 l2 h3 l3 */
    __m256d u = _mm256_unpacklo_pd(a01, a23);  /* h0 h2 h1 h3 */
    __m256d v = _mm256_unpackhi_pd(a01, a23);  /* l0 l2 l1 l3 */
    hi = _mm256_permute4x64_pd(u, _MM_SHUFFLE(3, 1, 2, 0));  /* h0 h1 h2 h3 */
    lo = _mm256_permute4x64_pd(v, _MM_SHUFFLE(3, 1, 2, 0));  /* l0 l1 l2 l3 */
}

static inline __attribute__((always_inline)) void
store_dd4(multifloats::float64x2 *p, __m256d hi, __m256d lo)
{
    double *d = p->limbs;
    __m256d ph = _mm256_permute4x64_pd(hi, _MM_SHUFFLE(3, 1, 2, 0));  /* h0 h2 h1 h3 */
    __m256d pl = _mm256_permute4x64_pd(lo, _MM_SHUFFLE(3, 1, 2, 0));  /* l0 l2 l1 l3 */
    __m256d a01 = _mm256_unpacklo_pd(ph, pl);  /* h0 l0 h1 l1 */
    __m256d a23 = _mm256_unpackhi_pd(ph, pl);  /* h2 l2 h3 l3 */
    _mm256_storeu_pd(d, a01);
    _mm256_storeu_pd(d + 4, a23);
}

static inline void
dd_axpy(int n, const double *xh, const double *xl,
        double th, double tl, multifloats::float64x2 *ap)
{
    const __m256d thb = _mm256_set1_pd(th);
    const __m256d tlb = _mm256_set1_pd(tl);
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d xhv = _mm256_loadu_pd(xh + i);
        __m256d xlv = _mm256_loadu_pd(xl + i);
        __m256d ph, pl;
        dd_mul(xhv, xlv, thb, tlb, ph, pl);
        __m256d ahv, alv;
        load_dd4(ap + i, ahv, alv);
        __m256d rh, rl;
        dd_add(ahv, alv, ph, pl, rh, rl);
        store_dd4(ap + i, rh, rl);
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
    for (; i + 4 <= n; i += 4) {
        __m256d ahv, alv;
        load_dd4(ap + i, ahv, alv);
        __m256d ph, pl;
        dd_mul(_mm256_loadu_pd(xh + i), _mm256_loadu_pd(xl + i), thb, tlb, ph, pl);
        dd_add(ahv, alv, ph, pl, ahv, alv);          /* ap + x*t1 */
        dd_mul(_mm256_loadu_pd(yh + i), _mm256_loadu_pd(yl + i), shb, slb, ph, pl);
        dd_add(ahv, alv, ph, pl, ahv, alv);          /* + y*t2    */
        store_dd4(ap + i, ahv, alv);
    }
    for (; i < n; ++i) {
        const multifloats::float64x2 xv{xh[i], xl[i]}, tv{th, tl};
        const multifloats::float64x2 yv{yh[i], yl[i]}, sv{sh, sl};
        ap[i] = ap[i] + xv * tv + yv * sv;
    }
}

#else  /* !MBLAS_SIMD_DD — scalar fallback */

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

#endif

}  // namespace mf_rank1
