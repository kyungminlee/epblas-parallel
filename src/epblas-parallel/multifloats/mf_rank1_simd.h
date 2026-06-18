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
 * The faithful 4-wide SoA primitives this is built on (dd_mul/dd_add/load_dd4/
 * store_dd4, bit-identical to the float64x2 operators) live in mf_simd_dd.h —
 * the shared vocabulary; this header adds only the rank-shaped axpy kernels.
 * Without MBLAS_SIMD_DD they fall back to the plain scalar loop, so callers
 * need no #ifdef of their own.
 */
#pragma once

#include <multifloats.h>
#include "mf_simd_dd.h"   /* faithful SoA dd_mul/dd_add/load_dd4/store_dd4 */

namespace mf_rank1 {

#ifdef MBLAS_SIMD_DD

using mf_simd::dd_mul;
using mf_simd::dd_add;
using mf_simd::load_dd4;
using mf_simd::store_dd4;

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
