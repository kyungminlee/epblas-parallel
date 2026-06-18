/*
 * mf_simd_dd.h — the faithful 4-wide SoA double-double primitive vocabulary,
 * shared by every multifloats routine that drops below the scalar-DD floor
 * (band matvecs mtbmv/wtbmv, the rank kernels in mf_rank1_simd.h, ...).
 *
 * DD is arithmetic-bound — there is no native SIMD for a single scalar DD — so
 * the lever is packing 4 INDEPENDENT DD values across the ymm lanes (hi limbs
 * in one __m256d, lo limbs in another; for complex, real/imag × hi/lo each in
 * its own register) so one vector op does four scalar DD ops at once.
 *
 * FAITHFUL means every primitive here mirrors the multifloats::float64x2 /
 * complex64x2 scalar operators OP FOR OP, so a kernel built from them is
 * bit-identical to its scalar reference on every finite, non-degenerate lane.
 * This is the deliberate opposite of the sloppy 1-ulp simd_dd::dd_mul/dd_add in
 * mgemm_simd_kernel.h (which fuse cross terms into FMAs): reach for those only
 * where bit-exactness is not required, and for those that are, ONLY use the
 * mf_simd:: ops below. (The scalar operators short-circuit on non-finite and on
 * the both-hi-zero -0 case; those lanes are not reproduced — random fuzz never
 * hits them and the consistency tolerance absorbs the measure-zero case.) All
 * arithmetic uses explicit _mm256 intrinsics the compiler never fuses, so
 * -ffp-contract cannot collapse a two_prod/two_sum into an FMA and break it.
 *
 * The whole header is a no-op unless MBLAS_SIMD_DD (AVX2+FMA) is defined, so
 * callers may include it unconditionally.
 */
#pragma once

#ifdef MBLAS_SIMD_DD

#include <cstddef>
#include <immintrin.h>
#include <multifloats.h>
#include "mgemm_simd_kernel.h"   /* simd_dd::twoprod / twosum / fast2sum (EFTs) */

namespace mf_simd {

using simd_dd::twoprod;
using simd_dd::twosum;
using simd_dd::fast2sum;

/* ---- real double-double ------------------------------------------------- */

/* DD * DD, faithful to float64x2::operator* (two_prod + one_prod cross terms +
 * fast_two_sum — cross terms added, NOT FMA-fused, to match the operator). */
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

/* DD - DD == DD + (-DD): flip the sign bits with xor so -0.0 is preserved (a
 * plain 0-b subtract would drop it), matching float64x2::operator- limb-for-
 * limb. */
static inline __attribute__((always_inline)) void
dd_sub(__m256d ah, __m256d al, __m256d bh, __m256d bl,
       __m256d &rh, __m256d &rl)
{
    const __m256d sgn = _mm256_set1_pd(-0.0);
    dd_add(ah, al, _mm256_xor_pd(bh, sgn), _mm256_xor_pd(bl, sgn), rh, rl);
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

/* Horizontal reduce of a 4-lane DD accumulator to a single float64x2, via a
 * faithful dd_add tree (lane0+lane2, lane1+lane3, then the two survivors). The
 * reduction reorders vs a scalar left-fold, so a dot built on a vector
 * accumulator + this reduce matches its scalar reference only within the
 * consistency tolerance (not bit-for-bit) — which is exactly what the band/tri
 * dot reductions need. Mirrors the simd_dd hreduce in the trsv kernels but
 * stays in the faithful (non-FMA-fused) vocabulary. */
static inline __attribute__((always_inline)) multifloats::float64x2
hreduce(__m256d sh, __m256d sl)
{
    __m256d sh_sw = _mm256_permute2f128_pd(sh, sh, 0x01);
    __m256d sl_sw = _mm256_permute2f128_pd(sl, sl, 0x01);
    __m256d ph, pl;
    dd_add(sh, sl, sh_sw, sl_sw, ph, pl);
    __m256d ph_sw = _mm256_shuffle_pd(ph, ph, 0x5);
    __m256d pl_sw = _mm256_shuffle_pd(pl, pl, 0x5);
    __m256d rh, rl;
    dd_add(ph, pl, ph_sw, pl_sw, rh, rl);
    double h[4], l[4];
    _mm256_storeu_pd(h, rh); _mm256_storeu_pd(l, rl);
    return multifloats::float64x2{h[0], l[0]};
}

/* Gather the hi/lo limbs of 4 DD values at p[0], p[s], p[2s], p[3s] into SoA
 * lanes (lane t <- p[t*s]). For a band matvec the 4 adjacent COLUMNS a Trans
 * row-group reads sit lda apart -> a strided gather; the source block (a few
 * thin columns) is L1-resident, so this is latency- not bandwidth-bound. */
static inline __attribute__((always_inline)) void
gather_dd4(const multifloats::float64x2 *p, std::ptrdiff_t s,
           __m256d &hi, __m256d &lo)
{
    hi = _mm256_set_pd(p[3 * s].limbs[0], p[2 * s].limbs[0], p[s].limbs[0], p[0].limbs[0]);
    lo = _mm256_set_pd(p[3 * s].limbs[1], p[2 * s].limbs[1], p[s].limbs[1], p[0].limbs[1]);
}

/* ---- complex double-double ---------------------------------------------- */

/* 4 complex64x2 values held in SoA lanes (real/imag, hi/lo limbs). */
struct cx4 { __m256d reh, rel, imh, iml; };

/* (a.re + i a.im)(b.re + i b.im) = (re·re − im·im) + i(re·im + im·re), each DD
 * product/sum via the faithful primitives — bit-identical to the scalar cmul.
 * cmul is commutative bit-exact in DD (two_prod/two_sum commute), so callers may
 * pass (x,matrix) or (matrix,x) interchangeably to match the reference order. */
static inline cx4 cmul_soa(const cx4 &a, const cx4 &b)
{
    __m256d p1h, p1l, p2h, p2l, p3h, p3l, p4h, p4l;
    dd_mul(a.reh, a.rel, b.reh, b.rel, p1h, p1l);   /* re·re */
    dd_mul(a.imh, a.iml, b.imh, b.iml, p2h, p2l);   /* im·im */
    dd_mul(a.reh, a.rel, b.imh, b.iml, p3h, p3l);   /* re·im */
    dd_mul(a.imh, a.iml, b.reh, b.rel, p4h, p4l);   /* im·re */
    cx4 r;
    dd_sub(p1h, p1l, p2h, p2l, r.reh, r.rel);
    dd_add(p3h, p3l, p4h, p4l, r.imh, r.iml);
    return r;
}

static inline cx4 cadd_soa(const cx4 &a, const cx4 &b)
{
    cx4 r;
    dd_add(a.reh, a.rel, b.reh, b.rel, r.reh, r.rel);
    dd_add(a.imh, a.iml, b.imh, b.iml, r.imh, r.iml);
    return r;
}

static inline cx4 cconj_soa(const cx4 &a)
{
    const __m256d sgn = _mm256_set1_pd(-0.0);
    return cx4{ a.reh, a.rel, _mm256_xor_pd(a.imh, sgn), _mm256_xor_pd(a.iml, sgn) };
}

/* A complex vector held as four parallel SoA limb arrays. */
struct cvec { double *reh, *rel, *imh, *iml; };

static inline cx4 vload(const cvec &v, int i)
{
    return cx4{ _mm256_loadu_pd(v.reh + i), _mm256_loadu_pd(v.rel + i),
                _mm256_loadu_pd(v.imh + i), _mm256_loadu_pd(v.iml + i) };
}
static inline void vstore(const cvec &v, int i, const cx4 &c)
{
    _mm256_storeu_pd(v.reh + i, c.reh); _mm256_storeu_pd(v.rel + i, c.rel);
    _mm256_storeu_pd(v.imh + i, c.imh); _mm256_storeu_pd(v.iml + i, c.iml);
}
static inline cx4 vbcast(const cvec &v, int j)
{
    return cx4{ _mm256_set1_pd(v.reh[j]), _mm256_set1_pd(v.rel[j]),
                _mm256_set1_pd(v.imh[j]), _mm256_set1_pd(v.iml[j]) };
}
static inline multifloats::complex64x2 vload1(const cvec &v, int i)
{
    return multifloats::complex64x2{ multifloats::float64x2{v.reh[i], v.rel[i]},
                                     multifloats::float64x2{v.imh[i], v.iml[i]} };
}
static inline void vstore1(const cvec &v, int i, const multifloats::complex64x2 &c)
{
    v.reh[i] = c.re.limbs[0]; v.rel[i] = c.re.limbs[1];
    v.imh[i] = c.im.limbs[0]; v.iml[i] = c.im.limbs[1];
}

/* Deinterleave 4 contiguous complex64x2 (16 contiguous doubles) into SoA lanes
 * via a 4×4 double transpose — the matrix band segment a NoTrans column reads. */
static inline cx4 cload4(const multifloats::complex64x2 *p)
{
    __m256d c0 = _mm256_loadu_pd(reinterpret_cast<const double *>(&p[0]));
    __m256d c1 = _mm256_loadu_pd(reinterpret_cast<const double *>(&p[1]));
    __m256d c2 = _mm256_loadu_pd(reinterpret_cast<const double *>(&p[2]));
    __m256d c3 = _mm256_loadu_pd(reinterpret_cast<const double *>(&p[3]));
    __m256d t0 = _mm256_unpacklo_pd(c0, c1);   /* reh0 reh1 imh0 imh1 */
    __m256d t1 = _mm256_unpackhi_pd(c0, c1);   /* rel0 rel1 iml0 iml1 */
    __m256d t2 = _mm256_unpacklo_pd(c2, c3);   /* reh2 reh3 imh2 imh3 */
    __m256d t3 = _mm256_unpackhi_pd(c2, c3);   /* rel2 rel3 iml2 iml3 */
    cx4 r;
    r.reh = _mm256_permute2f128_pd(t0, t2, 0x20);
    r.imh = _mm256_permute2f128_pd(t0, t2, 0x31);
    r.rel = _mm256_permute2f128_pd(t1, t3, 0x20);
    r.iml = _mm256_permute2f128_pd(t1, t3, 0x31);
    return r;
}

/* Gather 4 complex64x2 at p[0], p[s], p[2s], p[3s] into SoA lanes (lane t<-p[ts]).
 * The Trans row-group reads 4 adjacent COLUMNS, lda apart -> a strided gather;
 * the source block (a few thin columns) is L1-resident, latency- not bw-bound. */
static inline cx4 cgather4(const multifloats::complex64x2 *p, std::ptrdiff_t s)
{
    cx4 r;
    r.reh = _mm256_set_pd(p[3*s].re.limbs[0], p[2*s].re.limbs[0], p[s].re.limbs[0], p[0].re.limbs[0]);
    r.rel = _mm256_set_pd(p[3*s].re.limbs[1], p[2*s].re.limbs[1], p[s].re.limbs[1], p[0].re.limbs[1]);
    r.imh = _mm256_set_pd(p[3*s].im.limbs[0], p[2*s].im.limbs[0], p[s].im.limbs[0], p[0].im.limbs[0]);
    r.iml = _mm256_set_pd(p[3*s].im.limbs[1], p[2*s].im.limbs[1], p[s].im.limbs[1], p[0].im.limbs[1]);
    return r;
}

}  // namespace mf_simd

#endif  // MBLAS_SIMD_DD
