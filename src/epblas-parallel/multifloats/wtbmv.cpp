/* wtbmv — multifloats complex DD triangular band matrix-vector. */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef MBLAS_SIMD_DD
#include <cstdlib>
#include <immintrin.h>
#include "mf_rank1_simd.h"   /* faithful SoA dd_mul/dd_add + transpose loads */
#endif
#ifdef _OPENMP
#include <cstdlib>
#include <omp.h>
#include "../common/blas_omp.h"
#define WTBMV_OMP_MIN 256
#define WTBMV_MAX_CPUS 256
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
inline bool dd_iszero(const R &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool cdd_iszero(const T &x) { return dd_iszero(x.re) && dd_iszero(x.im); }
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }

/* Band inner-loop kernels, carved out as noinline so each compiles once in
 * isolation: the DD complex MAC then stays register-resident with the
 * error-free-transform float64x2 ops inlined inside a single tight body,
 * instead of GCC emitting a per-element out-of-line cmul call (with its
 * 32-byte sret spill and an AVX->SSE vzeroupper) amid the big wtbmv_ scaffold.
 *
 * caxpy_run updates len independent outputs x[t] += tmp*a[t]; since each x[t]
 * is touched exactly once the traversal order is irrelevant, so both UPLOs use
 * the same ascending kernel and stay bit-exact. The Trans dot accumulates into
 * a single acc, so its rounding depends on order: cdot_fwd walks ascending
 * (lower), cdot_rev walks descending from the base (upper), each matching the
 * original loop direction exactly. */
__attribute__((noinline)) static void caxpy_run(T *x, const T *a, T tmp, int len) {
    for (int t = 0; t < len; ++t) x[t] = cadd(x[t], cmul(tmp, a[t]));
}
__attribute__((noinline)) static T cdot_fwd(const T *a, const T *x, int len, bool conj, T acc) {
    if (!conj) for (int t = 0; t < len; ++t) acc = cadd(acc, cmul(a[t], x[t]));
    else       for (int t = 0; t < len; ++t) acc = cadd(acc, cmul(cconj(a[t]), x[t]));
    return acc;
}
__attribute__((noinline)) static T cdot_rev(const T *a, const T *x, int len, bool conj, T acc) {
    if (!conj) for (int t = 0; t < len; ++t) acc = cadd(acc, cmul(a[-t], x[-t]));
    else       for (int t = 0; t < len; ++t) acc = cadd(acc, cmul(cconj(a[-t]), x[-t]));
    return acc;
}
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

#ifdef MBLAS_SIMD_DD
/* 4-wide SoA AVX2 for complex double-double.  DD is arithmetic-bound (no native
 * SIMD for a scalar DD), so packing 4 INDEPENDENT complex64x2 values across the
 * ymm lanes — real/imag hi/lo each in its own register — quarters the op count.
 * The mf_rank1 dd_mul/dd_add primitives are bit-identical to the float64x2
 * multiply/add operators on every non-degenerate lane; dd_sub below is the
 * faithful twin of float64x2::operator- (operator+ with the rhs limbs negated). */

/* a - b == a + (-b): flip the sign bits with xor so -0.0 is preserved (a plain
 * 0-b subtract would drop it), matching float64x2::operator- limb-for-limb. */
static inline void dd_sub(__m256d ah, __m256d al, __m256d bh, __m256d bl,
                          __m256d &rh, __m256d &rl) {
    const __m256d sgn = _mm256_set1_pd(-0.0);
    mf_rank1::dd_add(ah, al, _mm256_xor_pd(bh, sgn), _mm256_xor_pd(bl, sgn), rh, rl);
}

/* 4 complex64x2 values held in SoA lanes (real/imag, hi/lo limbs). */
struct cx4 { __m256d reh, rel, imh, iml; };

/* (a.re + i a.im)(b.re + i b.im) = (re·re − im·im) + i(re·im + im·re), each DD
 * product/sum via the faithful primitives — bit-identical to the scalar cmul.
 * cmul is commutative bit-exact in DD (two_prod/two_sum commute), so callers may
 * pass (x,matrix) or (matrix,x) interchangeably to match the reference order. */
static inline cx4 cmul_soa(const cx4 &a, const cx4 &b) {
    __m256d p1h, p1l, p2h, p2l, p3h, p3l, p4h, p4l;
    mf_rank1::dd_mul(a.reh, a.rel, b.reh, b.rel, p1h, p1l);   /* re·re */
    mf_rank1::dd_mul(a.imh, a.iml, b.imh, b.iml, p2h, p2l);   /* im·im */
    mf_rank1::dd_mul(a.reh, a.rel, b.imh, b.iml, p3h, p3l);   /* re·im */
    mf_rank1::dd_mul(a.imh, a.iml, b.reh, b.rel, p4h, p4l);   /* im·re */
    cx4 r;
    dd_sub(p1h, p1l, p2h, p2l, r.reh, r.rel);
    mf_rank1::dd_add(p3h, p3l, p4h, p4l, r.imh, r.iml);
    return r;
}
static inline cx4 cadd_soa(const cx4 &a, const cx4 &b) {
    cx4 r;
    mf_rank1::dd_add(a.reh, a.rel, b.reh, b.rel, r.reh, r.rel);
    mf_rank1::dd_add(a.imh, a.iml, b.imh, b.iml, r.imh, r.iml);
    return r;
}
static inline cx4 cconj_soa(const cx4 &a) {
    const __m256d sgn = _mm256_set1_pd(-0.0);
    return cx4{ a.reh, a.rel, _mm256_xor_pd(a.imh, sgn), _mm256_xor_pd(a.iml, sgn) };
}

/* A vector held as four parallel SoA limb arrays. */
struct cvec { double *reh, *rel, *imh, *iml; };
static inline cx4 vload(const cvec &v, int i) {
    return cx4{ _mm256_loadu_pd(v.reh + i), _mm256_loadu_pd(v.rel + i),
                _mm256_loadu_pd(v.imh + i), _mm256_loadu_pd(v.iml + i) };
}
static inline void vstore(const cvec &v, int i, const cx4 &c) {
    _mm256_storeu_pd(v.reh + i, c.reh); _mm256_storeu_pd(v.rel + i, c.rel);
    _mm256_storeu_pd(v.imh + i, c.imh); _mm256_storeu_pd(v.iml + i, c.iml);
}
static inline cx4 vbcast(const cvec &v, int j) {
    return cx4{ _mm256_set1_pd(v.reh[j]), _mm256_set1_pd(v.rel[j]),
                _mm256_set1_pd(v.imh[j]), _mm256_set1_pd(v.iml[j]) };
}
static inline T vload1(const cvec &v, int i) {
    return T{ R{v.reh[i], v.rel[i]}, R{v.imh[i], v.iml[i]} };
}
static inline void vstore1(const cvec &v, int i, const T &c) {
    v.reh[i] = c.re.limbs[0]; v.rel[i] = c.re.limbs[1];
    v.imh[i] = c.im.limbs[0]; v.iml[i] = c.im.limbs[1];
}

/* Deinterleave 4 contiguous complex64x2 (16 contiguous doubles) into SoA lanes
 * via a 4×4 double transpose — the matrix band segment a NoTrans column reads. */
static inline cx4 cload4(const T *p) {
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
static inline cx4 cgather4(const T *p, std::ptrdiff_t s) {
    cx4 r;
    r.reh = _mm256_set_pd(p[3*s].re.limbs[0], p[2*s].re.limbs[0], p[s].re.limbs[0], p[0].re.limbs[0]);
    r.rel = _mm256_set_pd(p[3*s].re.limbs[1], p[2*s].re.limbs[1], p[s].re.limbs[1], p[0].re.limbs[1]);
    r.imh = _mm256_set_pd(p[3*s].im.limbs[0], p[2*s].im.limbs[0], p[s].im.limbs[0], p[0].im.limbs[0]);
    r.iml = _mm256_set_pd(p[3*s].im.limbs[1], p[2*s].im.limbs[1], p[s].im.limbs[1], p[0].im.limbs[1]);
    return r;
}

/* 4-wide SoA Trans/ConjTrans row-gather (x := A^T*x / A^H*x).  Output rows are
 * independent dots; four ADJACENT rows run in the lanes, each accumulating its
 * own dot over the shared reduction index d=1..k in exact scalar order -> bit-
 * identical (no reductive reassociation).  The 4 rows read 4 adjacent band
 * columns lda apart (strided cgather4); x is contiguous in the reduction index.
 * conj negates each gathered matrix element's imag before the multiply, matching
 * the reference cmul(cconj(a),x).  Interior full-band groups vectorize; boundary
 * and straddling rows fall to the scalar per-row path. */
static void wtbmv_rowgather_t_soa(bool upper, bool conj, bool nounit, int n, int k,
                                  int lo, int hi, const T *a, std::ptrdiff_t lda,
                                  const cvec &x, const cvec &y)
{
    int r = lo;
    if (upper) {
        while (r < hi) {
            if (r >= k && r + 4 <= hi) {                 /* full band: llen == k */
                cx4 s;
                if (nounit) {
                    cx4 d = cgather4(&A_(k, r), lda); if (conj) d = cconj_soa(d);
                    s = cmul_soa(d, vload(x, r));
                } else { s = vload(x, r); }
                for (int d = 1; d <= k; ++d) {
                    cx4 m = cgather4(&A_(k - d, r), lda); if (conj) m = cconj_soa(m);
                    s = cadd_soa(s, cmul_soa(m, vload(x, r - d)));
                }
                vstore(y, r, s);
                r += 4;
            } else {                                     /* scalar boundary/tail row */
                const T *base = &A_(0, r);
                const int llen = (r < k) ? r : k;
                T diagc = base[k]; if (conj) diagc = cconj(diagc);
                T s = nounit ? cmul(diagc, vload1(x, r)) : vload1(x, r);
                for (int d = 1; d <= llen; ++d) { T e = base[k - d]; if (conj) e = cconj(e); s = cadd(s, cmul(e, vload1(x, r - d))); }
                vstore1(y, r, s);
                ++r;
            }
        }
    } else {
        while (r < hi) {
            if (r + 3 <= n - 1 - k && r + 4 <= hi) {     /* full band: rlen == k */
                cx4 s;
                if (nounit) {
                    cx4 d = cgather4(&A_(0, r), lda); if (conj) d = cconj_soa(d);
                    s = cmul_soa(d, vload(x, r));
                } else { s = vload(x, r); }
                for (int d = 1; d <= k; ++d) {
                    cx4 m = cgather4(&A_(d, r), lda); if (conj) m = cconj_soa(m);
                    s = cadd_soa(s, cmul_soa(m, vload(x, r + d)));
                }
                vstore(y, r, s);
                r += 4;
            } else {
                const T *base = &A_(0, r);
                const int rlen = (n - 1 - r < k) ? (n - 1 - r) : k;
                T diagc = base[0]; if (conj) diagc = cconj(diagc);
                T s = nounit ? cmul(diagc, vload1(x, r)) : vload1(x, r);
                for (int d = 1; d <= rlen; ++d) { T e = base[d]; if (conj) e = cconj(e); s = cadd(s, cmul(e, vload1(x, r + d))); }
                vstore1(y, r, s);
                ++r;
            }
        }
    }
}

/* Serial NoTrans, unit-stride: 4-wide SoA in-place column walk.  Per column j,
 * x := A*x is an axpy of the band segment scaled by broadcast x[j]; the matrix
 * column (read once) is deinterleaved inline with cload4, 4 rows updated at a
 * time.  Columns run in reference order and every write within a column lands on
 * a distinct row i != j -> order-free, bit-identical to the scalar reference.
 * Zero columns are skipped exactly as the reference does.  false on alloc fail. */
static bool wtbmv_notrans_soa(bool upper, bool nounit, int n, int k,
                              const T *a, std::ptrdiff_t lda, T *x)
{
    const std::size_t np = (static_cast<std::size_t>(n) + 3) & ~static_cast<std::size_t>(3);
    double *reh = static_cast<double *>(std::aligned_alloc(32, np * sizeof(double)));
    double *rel = static_cast<double *>(std::aligned_alloc(32, np * sizeof(double)));
    double *imh = static_cast<double *>(std::aligned_alloc(32, np * sizeof(double)));
    double *iml = static_cast<double *>(std::aligned_alloc(32, np * sizeof(double)));
    if (!reh || !rel || !imh || !iml) { std::free(reh); std::free(rel); std::free(imh); std::free(iml); return false; }
    const cvec v{reh, rel, imh, iml};
    for (int i = 0; i < n; ++i) vstore1(v, i, x[i]);
    for (std::size_t i = n; i < np; ++i) { reh[i] = rel[i] = imh[i] = iml[i] = 0.0; }

    if (upper) {
        for (int j = 0; j < n; ++j) {
            if (reh[j] == 0.0 && rel[j] == 0.0 && imh[j] == 0.0 && iml[j] == 0.0) continue;
            const cx4 bj = vbcast(v, j);
            const T *col = &A_(0, j);
            const std::ptrdiff_t off = k - j;
            int i = (j > k) ? j - k : 0;
            for (; i + 4 <= j; i += 4) {
                cx4 p = cmul_soa(bj, cload4(&col[off + i]));
                vstore(v, i, cadd_soa(vload(v, i), p));
            }
            const T xj = vload1(v, j);
            for (; i < j; ++i) vstore1(v, i, cadd(vload1(v, i), cmul(xj, col[off + i])));
            if (nounit) vstore1(v, j, cmul(xj, col[k]));
        }
    } else {
        for (int j = n - 1; j >= 0; --j) {
            if (reh[j] == 0.0 && rel[j] == 0.0 && imh[j] == 0.0 && iml[j] == 0.0) continue;
            const cx4 bj = vbcast(v, j);
            const T *col = &A_(0, j);
            const std::ptrdiff_t off = -j;
            const int i_hi = (j + k < n - 1) ? j + k : n - 1;   /* inclusive top row */
            int i = j + 1;
            for (; i + 4 <= i_hi + 1; i += 4) {
                cx4 p = cmul_soa(bj, cload4(&col[off + i]));
                vstore(v, i, cadd_soa(vload(v, i), p));
            }
            const T xj = vload1(v, j);
            for (; i <= i_hi; ++i) vstore1(v, i, cadd(vload1(v, i), cmul(xj, col[off + i])));
            if (nounit) vstore1(v, j, cmul(xj, col[0]));
        }
    }

    for (int i = 0; i < n; ++i) x[i] = vload1(v, i);
    std::free(reh); std::free(rel); std::free(imh); std::free(iml);
    return true;
}

/* Serial Trans/ConjTrans, unit-stride: split x to SoA, run the row-gather over
 * [0,n), merge back.  Bit-identical to the scalar Trans cores. false on alloc. */
static bool wtbmv_trans_soa(bool upper, bool conj, bool nounit, int n, int k,
                            const T *a, std::ptrdiff_t lda, T *x)
{
    const std::size_t np = (static_cast<std::size_t>(n) + 3) & ~static_cast<std::size_t>(3);
    double *xb = static_cast<double *>(std::aligned_alloc(32, 4 * np * sizeof(double)));
    double *yb = static_cast<double *>(std::aligned_alloc(32, 4 * np * sizeof(double)));
    if (!xb || !yb) { std::free(xb); std::free(yb); return false; }
    const cvec xv{xb, xb + np, xb + 2*np, xb + 3*np};
    const cvec yv{yb, yb + np, yb + 2*np, yb + 3*np};
    for (int i = 0; i < n; ++i) vstore1(xv, i, x[i]);
    for (std::size_t i = n; i < np; ++i) { xv.reh[i] = xv.rel[i] = xv.imh[i] = xv.iml[i] = 0.0; }
    wtbmv_rowgather_t_soa(upper, conj, nounit, n, k, 0, n, a, lda, xv, yv);
    for (int i = 0; i < n; ++i) x[i] = vload1(yv, i);
    std::free(xb); std::free(yb);
    return true;
}
#endif  /* MBLAS_SIMD_DD */

#ifdef _OPENMP
/* Row-gather: x := A*x / A^T*x / A^H*x, in-place triangular band. Each output row
 * r is an independent dot once original x is copied off. NoTrans reads matrix row
 * r (anti-diagonal stride lda-1, elements direct); Trans/ConjTrans reads column r
 * (contiguous), conjugating elements when conj. Diagonal scales x[r] when
 * non-unit (conjugated when conj). xin is the contiguous copy, xout the
 * destination. Mirrors the serial accumulation order → within DD fuzz tol; the
 * serial path stays bit-exact. */
static void wtbmv_rowgather(bool upper, bool trans, bool conj, bool nounit,
                            int n, int k, int lo, int hi,
                            const T *a, std::size_t lda,
                            const T *xin, T *xout, int incx)
{
    const std::ptrdiff_t s1 = static_cast<std::ptrdiff_t>(lda) - 1;
    for (int r = lo; r < hi; ++r) {
        const T *base = &A_(0, r);
        T diagc = upper ? base[k] : base[0];
        if (conj) diagc = cconj(diagc);
        T s = nounit ? cmul(diagc, xin[r]) : xin[r];
        if (!trans) {
            if (upper) {
                const int rlen = (n - 1 - r < k) ? (n - 1 - r) : k;
                for (int d = 1; d <= rlen; ++d)
                    s = cadd(s, cmul(base[k + (std::ptrdiff_t)d * s1], xin[r + d]));
            } else {
                const int llen = (r < k) ? r : k;
                for (int d = 1; d <= llen; ++d)
                    s = cadd(s, cmul(base[-(std::ptrdiff_t)d * s1], xin[r - d]));
            }
        } else {
            if (upper) {
                const int llen = (r < k) ? r : k;
                for (int d = 1; d <= llen; ++d) {
                    T e = base[k - d]; if (conj) e = cconj(e);
                    s = cadd(s, cmul(e, xin[r - d]));
                }
            } else {
                const int rlen = (n - 1 - r < k) ? (n - 1 - r) : k;
                for (int d = 1; d <= rlen; ++d) {
                    T e = base[d]; if (conj) e = cconj(e);
                    s = cadd(s, cmul(e, xin[r + d]));
                }
            }
        }
        xout[(std::ptrdiff_t)r * incx] = s;
    }
}

#ifdef MBLAS_SIMD_DD
/* 4-wide SoA NoTrans column-scatter (the OMP twin of wtbmv_notrans_soa): each
 * thread owns output rows [lo,hi), iterating the columns that touch them and
 * reading every column's band segment CONTIGUOUSLY (cload4) — the row-gather's
 * anti-diagonal read does not vectorize, so NoTrans threads via scatter instead.
 * Writes are disjoint across threads; iterating columns ascending (upper) /
 * descending (lower) seeds each owned row at its diagonal column first, then
 * accumulates off-diagonals in column order -> identical per-row association as
 * the serial scatter (bit-exact). y[lo,hi) needs no pre-zero. */
static void wtbmv_colscatter_soa(bool upper, bool nounit, int n, int k,
                                 int lo, int hi, const T *a, std::ptrdiff_t lda,
                                 const cvec &x, const cvec &y)
{
    if (upper) {
        const int jmax = (hi + k < n) ? (hi + k) : n;
        for (int j = lo; j < jmax; ++j) {
            const cx4 bj = vbcast(x, j);
            const T *col = &A_(0, j);
            const std::ptrdiff_t off = k - j;            /* A(i,j) = col[off+i] */
            const int i_lo = (j - k > lo) ? (j - k) : lo;
            const int i_hi = (j < hi) ? j : hi;          /* off-diagonal rows < j */
            int i = i_lo;
            for (; i + 4 <= i_hi; i += 4) {
                cx4 p = cmul_soa(bj, cload4(&col[off + i]));
                vstore(y, i, cadd_soa(vload(y, i), p));
            }
            const T tmp = vload1(x, j);
            for (; i < i_hi; ++i) vstore1(y, i, cadd(vload1(y, i), cmul(tmp, col[off + i])));
            if (j >= lo && j < hi)                        /* diagonal seed */
                vstore1(y, j, nounit ? cmul(tmp, col[k]) : tmp);
        }
    } else {
        const int jmin = (lo - k > 0) ? (lo - k) : 0;
        for (int j = hi - 1; j >= jmin; --j) {
            const cx4 bj = vbcast(x, j);
            const T *col = &A_(0, j);
            const std::ptrdiff_t off = -j;               /* A(i,j) = col[off+i] */
            const int i_lo = (j + 1 > lo) ? (j + 1) : lo;
            const int i_hi = (j + k + 1 < hi) ? (j + k + 1) : hi;  /* rows > j */
            int i = i_lo;
            for (; i + 4 <= i_hi; i += 4) {
                cx4 p = cmul_soa(bj, cload4(&col[off + i]));
                vstore(y, i, cadd_soa(vload(y, i), p));
            }
            const T tmp = vload1(x, j);
            for (; i < i_hi; ++i) vstore1(y, i, cadd(vload1(y, i), cmul(tmp, col[off + i])));
            if (j >= lo && j < hi)                        /* diagonal seed */
                vstore1(y, j, nounit ? cmul(tmp, col[0]) : tmp);
        }
    }
}
#endif  /* MBLAS_SIMD_DD */

/* Threaded in-place complex triangular band matvec. Partition the output rows
 * across threads, each filling its disjoint range, barrier, write back. Returns
 * true if handled. */
__attribute__((noinline)) static bool wtbmv_omp(
    bool upper, bool trans, bool conj, bool nounit, int n, int k,
    const T *a, std::size_t lda, T *x, int incx)
{
    if (n < WTBMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > WTBMV_MAX_CPUS) nthreads = WTBMV_MAX_CPUS;

    T *xbase = (incx < 0) ? x - (std::ptrdiff_t)(n - 1) * incx : x;

#ifdef MBLAS_SIMD_DD
    /* Both triangles thread the 4-wide SoA kernels: split x to SoA limb arrays
     * once, each thread fills its owned rows (NoTrans column-scatter / Trans
     * row-gather), barrier, merge back. Disjoint row ownership + scalar per-row
     * column/d order keep it bit-exact. Alloc failure -> scalar path below. */
    {
        const std::size_t np = (static_cast<std::size_t>(n) + 3) & ~static_cast<std::size_t>(3);
        double *xb = static_cast<double *>(std::aligned_alloc(32, 4 * np * sizeof(double)));
        double *yb = static_cast<double *>(std::aligned_alloc(32, 4 * np * sizeof(double)));
        if (xb && yb) {
            const cvec xv{xb, xb + np, xb + 2*np, xb + 3*np};
            const cvec yv{yb, yb + np, yb + 2*np, yb + 3*np};
            for (int i = 0; i < n; ++i) vstore1(xv, i, xbase[(std::ptrdiff_t)i * incx]);
            for (std::size_t i = n; i < np; ++i) { xv.reh[i] = xv.rel[i] = xv.imh[i] = xv.iml[i] = 0.0; }
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int lo = (int)((long long)n * tid / nthreads);
                int hi = (int)((long long)n * (tid + 1) / nthreads);
                if (!trans) wtbmv_colscatter_soa(upper, nounit, n, k, lo, hi, a, lda, xv, yv);
                else        wtbmv_rowgather_t_soa(upper, conj, nounit, n, k, lo, hi, a, lda, xv, yv);
                #pragma omp barrier          /* all reads of x done before write-back */
                for (int i = lo; i < hi; ++i)
                    xbase[(std::ptrdiff_t)i * incx] = vload1(yv, i);
            }
            std::free(xb); std::free(yb);
            return true;
        }
        std::free(xb); std::free(yb);
        /* alloc failure -> scalar path below */
    }
#endif

    T *xbuf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
    if (!xbuf) return false;
    for (int i = 0; i < n; ++i) xbuf[i] = xbase[(std::ptrdiff_t)i * incx];

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int lo = (int)((long long)n * tid / nthreads);
        int hi = (int)((long long)n * (tid + 1) / nthreads);
        wtbmv_rowgather(upper, trans, conj, nounit, n, k, lo, hi, a, lda, xbuf, xbase, incx);
    }
    std::free(xbuf);
    return true;
}
#endif

extern "C" void wtbmv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_, const int *k_,
    const T *a, const int *lda_,
    T *x, const int *incx_,
    std::size_t uplo_len, std::size_t trans_len, std::size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, incx = *incx_;
    const char UPLO = up(uplo);
    const char TR = up(trans);
    const int noconj = (TR == 'T');
    const int nounit = (up(diag) != 'U');

    if (N == 0) return;

#ifdef _OPENMP
    if (N >= WTBMV_OMP_MIN && blas_omp_max_threads() > 1
        && wtbmv_omp(UPLO == 'U', TR != 'N', TR == 'C', nounit != 0, N, K, a, lda, x, incx))
        return;
#endif

#ifdef MBLAS_SIMD_DD
    /* Unit-stride serial: 4-wide SoA — NoTrans axpy-per-column, Trans/ConjTrans
     * row-gather. Alloc failure falls through to the scalar cores below. */
    if (incx == 1 && TR == 'N'
        && wtbmv_notrans_soa(UPLO == 'U', nounit != 0, N, K, a, lda, x))
        return;
    if (incx == 1 && (TR == 'T' || TR == 'C')
        && wtbmv_trans_soa(UPLO == 'U', TR == 'C', nounit != 0, N, K, a, lda, x))
        return;
#endif

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'U') {
                for (int j = 0; j < N; ++j) {
                    if (!cdd_iszero(x[j])) {
                        const T tmp = x[j];
                        const int L = K - j;
                        const int i_lo = (j - K > 0) ? (j - K) : 0;
                        if (j > i_lo) caxpy_run(&x[i_lo], &A_(L + i_lo, j), tmp, j - i_lo);
                        if (nounit) x[j] = cmul(x[j], A_(K, j));
                    }
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    if (!cdd_iszero(x[j])) {
                        const T tmp = x[j];
                        const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                        if (i_hi - 1 > j) caxpy_run(&x[j + 1], &A_(1, j), tmp, i_hi - j - 1);
                        if (nounit) x[j] = cmul(x[j], A_(0, j));
                    }
                }
            }
        } else {
            if (UPLO == 'U') {
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    const int L = K - j;
                    if (nounit) tmp = cmul(tmp, (noconj ? A_(K, j) : cconj(A_(K, j))));
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    if (j > i_lo) tmp = cdot_rev(&A_(L + j - 1, j), &x[j - 1], j - i_lo, !noconj, tmp);
                    x[j] = tmp;
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    T tmp = x[j];
                    if (nounit) tmp = cmul(tmp, (noconj ? A_(0, j) : cconj(A_(0, j))));
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    if (i_hi > j + 1) tmp = cdot_fwd(&A_(1, j), &x[j + 1], i_hi - j - 1, !noconj, tmp);
                    x[j] = tmp;
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'U') {
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    if (!cdd_iszero(x[jx])) {
                        const T tmp = x[jx];
                        int ix = kx;
                        const int L = K - j;
                        const int i_lo = (j - K > 0) ? (j - K) : 0;
                        for (int i = i_lo; i < j; ++i) {
                            x[ix] = cadd(x[ix], cmul(tmp, A_(L + i, j)));
                            ix += incx;
                        }
                        if (nounit) x[jx] = cmul(x[jx], A_(K, j));
                    }
                    jx += incx;
                    if (j >= K) kx += incx;
                }
            } else {
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    if (!cdd_iszero(x[jx])) {
                        const T tmp = x[jx];
                        int ix = kx;
                        const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                        for (int i = i_hi - 1; i > j; --i) {
                            x[ix] = cadd(x[ix], cmul(tmp, A_(i - j, j)));
                            ix -= incx;
                        }
                        if (nounit) x[jx] = cmul(x[jx], A_(0, j));
                    }
                    jx -= incx;
                    if ((N - 1 - j) >= K) kx -= incx;
                }
            }
        } else {
            if (UPLO == 'U') {
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    kx -= incx;
                    int ix = kx;
                    const int L = K - j;
                    if (nounit) tmp = cmul(tmp, (noconj ? A_(K, j) : cconj(A_(K, j))));
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    for (int i = j - 1; i >= i_lo; --i) {
                        const T aij = noconj ? A_(L + i, j) : cconj(A_(L + i, j));
                        tmp = cadd(tmp, cmul(aij, x[ix]));
                        ix -= incx;
                    }
                    x[jx] = tmp;
                    jx -= incx;
                }
            } else {
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    T tmp = x[jx];
                    kx += incx;
                    int ix = kx;
                    if (nounit) tmp = cmul(tmp, (noconj ? A_(0, j) : cconj(A_(0, j))));
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (int i = j + 1; i < i_hi; ++i) {
                        const T aij = noconj ? A_(i - j, j) : cconj(A_(i - j, j));
                        tmp = cadd(tmp, cmul(aij, x[ix]));
                        ix += incx;
                    }
                    x[jx] = tmp;
                    jx += incx;
                }
            }
        }
    }
}

#undef A_
