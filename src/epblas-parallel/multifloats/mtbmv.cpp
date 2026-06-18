/* mtbmv — multifloats real DD triangular band matrix-vector.
 *   x := A*x or A^T*x, A triangular band with K+1 diagonals.
 *
 * Serial — data dependencies in x force the Fortran reference order.  The eight
 * UPLO×TRANS×stride cores are a faithful port of the OpenBLAS clone's serial
 * paths: each column hoists its base pointer (col = &A_(0,j), off) so the inner
 * loop carries no j*lda recompute, and a strided x is walked IN PLACE through a
 * small K-wide window (which stays in cache) rather than gathered to scratch —
 * an O(N) gather pass costs more than the band work for a thin band, which is
 * why the gather variant trailed ob ~5% on the strided NoTrans-lower leaves.
 */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef MBLAS_SIMD_DD
#include <cstdlib>
#include <immintrin.h>
#include "mf_rank1_simd.h"   /* faithful SoA dd_mul/dd_add + load_dd4 */
#endif
#ifdef _OPENMP
#include <cstdlib>
#include <omp.h>
#include "../common/blas_omp.h"
#define MTBMV_OMP_MIN 256
#define MTBMV_MAX_CPUS 256
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
inline bool dd_iszero(const T &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

/* Serial path — faithful port of the OpenBLAS clone's dtbmv reference cores.
 * Eight branches: UPLO × TRANS × (unit-stride / strided), each hoisting the
 * column base pointer so the inner loop has no j*lda recompute. Strided walks
 * x in place (caller has already shifted x to logical index 0 for incx<0). */
static void mtbmv_serial(bool upper, bool trans, bool nounit,
                         std::ptrdiff_t n, std::ptrdiff_t k,
                         const T *a, std::ptrdiff_t lda, T *x, std::ptrdiff_t incx)
{
    std::ptrdiff_t kx = 0;
    if (!trans) {
        if (upper) {
            if (incx == 1) {
                for (std::ptrdiff_t j = 0; j < n; ++j) {
                    if (!dd_iszero(x[j])) {
                        const T temp = x[j];
                        std::ptrdiff_t i_lo = (j > k) ? j - k : 0;
                        const T *col = &A_(0, j);
                        std::ptrdiff_t off = k - j;
                        for (std::ptrdiff_t i = i_lo; i < j; ++i) x[i] = x[i] + temp * col[off + i];
                        if (nounit) x[j] = x[j] * col[k];
                    }
                }
            } else {
                std::ptrdiff_t jx = kx;
                for (std::ptrdiff_t j = 0; j < n; ++j) {
                    if (!dd_iszero(x[jx])) {
                        const T temp = x[jx];
                        std::ptrdiff_t i_lo = (j > k) ? j - k : 0;
                        std::ptrdiff_t ix = kx;
                        const T *col = &A_(0, j);
                        std::ptrdiff_t off = k - j;
                        for (std::ptrdiff_t i = i_lo; i < j; ++i) { x[ix] = x[ix] + temp * col[off + i]; ix += incx; }
                        if (nounit) x[jx] = x[jx] * col[k];
                    }
                    jx += incx;
                    if (j >= k) kx += incx;
                }
            }
        } else {
            if (incx == 1) {
                for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (!dd_iszero(x[j])) {
                        const T temp = x[j];
                        std::ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                        const T *col = &A_(0, j);
                        std::ptrdiff_t off = -j;
                        for (std::ptrdiff_t i = i_hi; i > j; --i) x[i] = x[i] + temp * col[off + i];
                        if (nounit) x[j] = x[j] * col[0];
                    }
                }
            } else {
                kx += (n - 1) * incx;
                std::ptrdiff_t jx = kx;
                for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (!dd_iszero(x[jx])) {
                        const T temp = x[jx];
                        std::ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                        std::ptrdiff_t ix = kx;
                        const T *col = &A_(0, j);
                        std::ptrdiff_t off = -j;
                        for (std::ptrdiff_t i = i_hi; i > j; --i) { x[ix] = x[ix] + temp * col[off + i]; ix -= incx; }
                        if (nounit) x[jx] = x[jx] * col[0];
                    }
                    jx -= incx;
                    if (n - 1 - j >= k) kx -= incx;
                }
            }
        }
    } else {
        if (upper) {
            if (incx == 1) {
                for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                    T temp = x[j];
                    const T *col = &A_(0, j);
                    std::ptrdiff_t off = k - j;
                    if (nounit) temp = temp * col[k];
                    std::ptrdiff_t i_lo = (j > k) ? j - k : 0;
                    for (std::ptrdiff_t i = j - 1; i >= i_lo; --i) temp = temp + col[off + i] * x[i];
                    x[j] = temp;
                }
            } else {
                kx += (n - 1) * incx;
                std::ptrdiff_t jx = kx;
                for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                    T temp = x[jx];
                    kx -= incx;
                    std::ptrdiff_t ix = kx;
                    const T *col = &A_(0, j);
                    std::ptrdiff_t off = k - j;
                    if (nounit) temp = temp * col[k];
                    std::ptrdiff_t i_lo = (j > k) ? j - k : 0;
                    for (std::ptrdiff_t i = j - 1; i >= i_lo; --i) { temp = temp + col[off + i] * x[ix]; ix -= incx; }
                    x[jx] = temp;
                    jx -= incx;
                }
            }
        } else {
            if (incx == 1) {
                for (std::ptrdiff_t j = 0; j < n; ++j) {
                    T temp = x[j];
                    const T *col = &A_(0, j);
                    std::ptrdiff_t off = -j;
                    if (nounit) temp = temp * col[0];
                    std::ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                    for (std::ptrdiff_t i = j + 1; i <= i_hi; ++i) temp = temp + col[off + i] * x[i];
                    x[j] = temp;
                }
            } else {
                std::ptrdiff_t jx = kx;
                for (std::ptrdiff_t j = 0; j < n; ++j) {
                    T temp = x[jx];
                    kx += incx;
                    std::ptrdiff_t ix = kx;
                    const T *col = &A_(0, j);
                    std::ptrdiff_t off = -j;
                    if (nounit) temp = temp * col[0];
                    std::ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                    for (std::ptrdiff_t i = j + 1; i <= i_hi; ++i) { temp = temp + col[off + i] * x[ix]; ix += incx; }
                    x[jx] = temp;
                    jx += incx;
                }
            }
        }
    }
}

#ifdef MBLAS_SIMD_DD
/* NoTrans unit-stride, 4-wide SoA.  x := A*x is, per column j, an axpy of the
 * band segment of column j scaled by the broadcast x[j].  x is split to SoA
 * limb arrays ONCE (reused as both the broadcast source and the accumulation
 * target, so the split also serves every column); the matrix column — read
 * once — is deinterleaved inline with load_dd4.  Columns run in reference order
 * (upper ascending / lower descending) and within a column every write lands on
 * a distinct row i != j, so 4-wide-over-i is order-free and the result is
 * bit-identical to the scalar reference on every non-degenerate lane (mf_rank1
 * dd_mul/dd_add mirror the float64x2 multiply/add operators op-for-op).  x is
 * already at logical index 0 (caller shifted for incx<0); a strided x is gathered
 * into the SoA limb arrays up front and scattered back at the end — the O(N)
 * gather is repaid by the SoA core quartering the O(N*K) band work (gather alone,
 * feeding a scalar core, never closes the strided gap on a thin band; feeding the
 * SoA core it does).  Returns true if it ran; false (alloc) -> scalar core. */
static bool mtbmv_notrans_soa(bool upper, bool nounit, int n, int k,
                              const T *a, std::ptrdiff_t lda, T *x, int incx)
{
    const std::size_t np = (static_cast<std::size_t>(n) + 3) & ~static_cast<std::size_t>(3);
    double *xh = static_cast<double *>(std::aligned_alloc(32, np * sizeof(double)));
    double *xl = static_cast<double *>(std::aligned_alloc(32, np * sizeof(double)));
    if (!xh || !xl) { std::free(xh); std::free(xl); return false; }
    for (int i = 0; i < n; ++i) { const T v = x[(std::ptrdiff_t)i * incx]; xh[i] = v.limbs[0]; xl[i] = v.limbs[1]; }
    for (std::size_t i = n; i < np; ++i) { xh[i] = 0.0; xl[i] = 0.0; }

    if (upper) {
        for (int j = 0; j < n; ++j) {
            if (xh[j] == 0.0 && xl[j] == 0.0) continue;
            const __m256d bh = _mm256_set1_pd(xh[j]), bl = _mm256_set1_pd(xl[j]);
            const T *col = &A_(0, j);
            const std::ptrdiff_t off = k - j;
            int i = (j > k) ? j - k : 0;
            for (; i + 4 <= j; i += 4) {
                __m256d mh, ml; mf_rank1::load_dd4(&col[off + i], mh, ml);
                __m256d ph, pl; mf_rank1::dd_mul(mh, ml, bh, bl, ph, pl);
                __m256d rh, rl;
                mf_rank1::dd_add(_mm256_loadu_pd(xh + i), _mm256_loadu_pd(xl + i), ph, pl, rh, rl);
                _mm256_storeu_pd(xh + i, rh); _mm256_storeu_pd(xl + i, rl);
            }
            const T xj{xh[j], xl[j]};
            for (; i < j; ++i) { T xi{xh[i], xl[i]}; xi = xi + xj * col[off + i]; xh[i] = xi.limbs[0]; xl[i] = xi.limbs[1]; }
            if (nounit) { T d{xh[j], xl[j]}; d = d * col[k]; xh[j] = d.limbs[0]; xl[j] = d.limbs[1]; }
        }
    } else {
        for (int j = n - 1; j >= 0; --j) {
            if (xh[j] == 0.0 && xl[j] == 0.0) continue;
            const __m256d bh = _mm256_set1_pd(xh[j]), bl = _mm256_set1_pd(xl[j]);
            const T *col = &A_(0, j);
            const std::ptrdiff_t off = -j;
            const int i_hi = (j + k < n - 1) ? j + k : n - 1;   /* inclusive top row */
            int i = j + 1;
            for (; i + 4 <= i_hi + 1; i += 4) {
                __m256d mh, ml; mf_rank1::load_dd4(&col[off + i], mh, ml);
                __m256d ph, pl; mf_rank1::dd_mul(mh, ml, bh, bl, ph, pl);
                __m256d rh, rl;
                mf_rank1::dd_add(_mm256_loadu_pd(xh + i), _mm256_loadu_pd(xl + i), ph, pl, rh, rl);
                _mm256_storeu_pd(xh + i, rh); _mm256_storeu_pd(xl + i, rl);
            }
            const T xj{xh[j], xl[j]};
            for (; i <= i_hi; ++i) { T xi{xh[i], xl[i]}; xi = xi + xj * col[off + i]; xh[i] = xi.limbs[0]; xl[i] = xi.limbs[1]; }
            if (nounit) { T d{xh[j], xl[j]}; d = d * col[0]; xh[j] = d.limbs[0]; xl[j] = d.limbs[1]; }
        }
    }

    for (int i = 0; i < n; ++i) x[(std::ptrdiff_t)i * incx] = T{xh[i], xl[i]};
    std::free(xh); std::free(xl);
    return true;
}

/* Gather the hi/lo limbs of 4 DD values at p[0], p[s], p[2s], p[3s] into SoA
 * lanes (lane t <- p[t*s]). The matrix is band-stored column-major, so the 4
 * adjacent COLUMNS a Trans row-group reads sit lda apart -> a strided gather.
 * Assembled from scalar loads; the source block (4 thin columns, ~1 KB) is
 * L1-resident, so this is latency- not bandwidth-bound, and the DD arithmetic
 * it feeds — the actual bottleneck — drops to a quarter of the scalar op count. */
static inline void gather_dd4(const T *p, std::ptrdiff_t s,
                              __m256d &hi, __m256d &lo)
{
    hi = _mm256_set_pd(p[3 * s].limbs[0], p[2 * s].limbs[0], p[s].limbs[0], p[0].limbs[0]);
    lo = _mm256_set_pd(p[3 * s].limbs[1], p[2 * s].limbs[1], p[s].limbs[1], p[0].limbs[1]);
}

/* 4-wide SoA twin of the Trans row-gather (x := A^T*x). Output rows are
 * independent dots; here four ADJACENT rows run in the SIMD lanes, each lane
 * accumulating its own dot over the shared reduction index d=1..k in the exact
 * scalar order — so the result is bit-identical (no reductive reassociation,
 * unlike a within-row 4-accumulator split). x stays AoS-split into xh/xl
 * (read across [0,n) — a row reaches outside [lo,hi)); y is written SoA in
 * [lo,hi). Only interior rows with a full k-wide band group; boundary/tail
 * rows (and any group straddling [lo,hi)) fall to the scalar per-row path. */
static void mtbmv_rowgather_t_soa(bool upper, bool nounit, int n, int k,
                                  int lo, int hi, const T *a, std::ptrdiff_t lda,
                                  const double *xh, const double *xl,
                                  double *yh, double *yl)
{
    int r = lo;
    if (upper) {
        while (r < hi) {
            if (r >= k && r + 4 <= hi) {                 /* full band: llen == k */
                __m256d sh, sl;
                if (nounit) {
                    __m256d dh, dl; gather_dd4(&A_(k, r), lda, dh, dl);
                    mf_rank1::dd_mul(dh, dl, _mm256_loadu_pd(xh + r), _mm256_loadu_pd(xl + r), sh, sl);
                } else { sh = _mm256_loadu_pd(xh + r); sl = _mm256_loadu_pd(xl + r); }
                for (int d = 1; d <= k; ++d) {
                    __m256d mh, ml; gather_dd4(&A_(k - d, r), lda, mh, ml);
                    __m256d ph, pl;
                    mf_rank1::dd_mul(mh, ml, _mm256_loadu_pd(xh + (r - d)), _mm256_loadu_pd(xl + (r - d)), ph, pl);
                    mf_rank1::dd_add(sh, sl, ph, pl, sh, sl);
                }
                _mm256_storeu_pd(yh + r, sh); _mm256_storeu_pd(yl + r, sl);
                r += 4;
            } else {                                     /* scalar boundary/tail row */
                const T *base = &A_(0, r);
                const int llen = (r < k) ? r : k;
                T s = nounit ? base[k] * T{xh[r], xl[r]} : T{xh[r], xl[r]};
                for (int d = 1; d <= llen; ++d) { const T xv{xh[r - d], xl[r - d]}; s = s + base[k - d] * xv; }
                yh[r] = s.limbs[0]; yl[r] = s.limbs[1];
                ++r;
            }
        }
    } else {
        while (r < hi) {
            if (r + 3 <= n - 1 - k && r + 4 <= hi) {     /* full band: rlen == k */
                __m256d sh, sl;
                if (nounit) {
                    __m256d dh, dl; gather_dd4(&A_(0, r), lda, dh, dl);
                    mf_rank1::dd_mul(dh, dl, _mm256_loadu_pd(xh + r), _mm256_loadu_pd(xl + r), sh, sl);
                } else { sh = _mm256_loadu_pd(xh + r); sl = _mm256_loadu_pd(xl + r); }
                for (int d = 1; d <= k; ++d) {
                    __m256d mh, ml; gather_dd4(&A_(d, r), lda, mh, ml);
                    __m256d ph, pl;
                    mf_rank1::dd_mul(mh, ml, _mm256_loadu_pd(xh + (r + d)), _mm256_loadu_pd(xl + (r + d)), ph, pl);
                    mf_rank1::dd_add(sh, sl, ph, pl, sh, sl);
                }
                _mm256_storeu_pd(yh + r, sh); _mm256_storeu_pd(yl + r, sl);
                r += 4;
            } else {
                const T *base = &A_(0, r);
                const int rlen = (n - 1 - r < k) ? (n - 1 - r) : k;
                T s = nounit ? base[0] * T{xh[r], xl[r]} : T{xh[r], xl[r]};
                for (int d = 1; d <= rlen; ++d) { const T xv{xh[r + d], xl[r + d]}; s = s + base[d] * xv; }
                yh[r] = s.limbs[0]; yl[r] = s.limbs[1];
                ++r;
            }
        }
    }
}

/* Serial Trans: SoA over the whole vector. Splits x to limb arrays, runs the
 * row-gather across [0,n), merges back. A strided x is gathered into the SoA
 * limb arrays up front and scattered back at the end — like NoTrans, the O(N)
 * gather is repaid by the SoA core quartering the band work (gather feeding a
 * scalar core would not close the strided gap on a thin band; feeding the SoA
 * core it does). Bit-identical to the scalar Trans cores (same per-row
 * d-order). false on alloc failure. */
static bool mtbmv_trans_soa(bool upper, bool nounit, int n, int k,
                            const T *a, std::ptrdiff_t lda, T *x, int incx)
{
    const std::size_t np = ((std::size_t)n + 3) & ~(std::size_t)3;
    double *xh = static_cast<double *>(std::aligned_alloc(32, np * sizeof(double)));
    double *xl = static_cast<double *>(std::aligned_alloc(32, np * sizeof(double)));
    double *yh = static_cast<double *>(std::aligned_alloc(32, np * sizeof(double)));
    double *yl = static_cast<double *>(std::aligned_alloc(32, np * sizeof(double)));
    if (!xh || !xl || !yh || !yl) { std::free(xh); std::free(xl); std::free(yh); std::free(yl); return false; }
    for (int i = 0; i < n; ++i) { const T v = x[(std::ptrdiff_t)i * incx]; xh[i] = v.limbs[0]; xl[i] = v.limbs[1]; }
    for (std::size_t i = n; i < np; ++i) { xh[i] = 0.0; xl[i] = 0.0; }
    mtbmv_rowgather_t_soa(upper, nounit, n, k, 0, n, a, lda, xh, xl, yh, yl);
    for (int i = 0; i < n; ++i) x[(std::ptrdiff_t)i * incx] = T{yh[i], yl[i]};
    std::free(xh); std::free(xl); std::free(yh); std::free(yl);
    return true;
}
#endif

#ifdef _OPENMP
/* Row-gather for the Trans triangle: x := A^T*x is an in-place band matvec, so
 * each output row r is an independent dot once the original x is preserved in a
 * copy. Trans reads matrix column r CONTIGUOUSLY (base[k-d] upper / base[d]
 * lower), so this already scales well; the NoTrans triangle, whose row read is
 * anti-diagonal, uses the column-scatter below instead. Diagonal scales x[r]
 * when non-unit. Mirrors the serial accumulation order (bit-exact). */
static void mtbmv_rowgather_t(bool upper, bool nounit,
                              int n, int k, int lo, int hi,
                              const T *a, std::size_t lda,
                              const T *xin, T *y)
{
    for (int r = lo; r < hi; ++r) {
        const T *base = &A_(0, r);
        const T diagc = upper ? base[k] : base[0];
        T s = nounit ? diagc * xin[r] : xin[r];
        if (upper) {
            const int llen = (r < k) ? r : k;
            for (int d = 1; d <= llen; ++d) s = s + base[k - d] * xin[r - d];
        } else {
            const int rlen = (n - 1 - r < k) ? (n - 1 - r) : k;
            for (int d = 1; d <= rlen; ++d) s = s + base[d] * xin[r + d];
        }
        y[r] = s;
    }
}

/* Column-scatter for the NoTrans triangle: each thread owns output rows [lo,hi)
 * and walks the columns touching them, reading every column's band segment
 * CONTIGUOUSLY (stride 1 in the row index, vs the row-gather's anti-diagonal
 * lda-1 stride) and scattering only into its owned rows. Writes are disjoint
 * across threads -> no race, no O(nthreads*n) fold. Iterating columns ascending
 * (upper) / descending (lower) seeds each owned row at its diagonal column
 * first, then accumulates off-diagonals in column order -> identical per-row
 * association as the serial scatter (bit-exact). y[lo,hi) need not be pre-zeroed:
 * every owned row is assigned at its diagonal column before any += reaches it. */
static void mtbmv_colscatter(bool upper, bool nounit, int n, int k,
                             int lo, int hi, const T *a, std::size_t lda,
                             const T *xin, T *y)
{
    if (upper) {
        const int jmax = (hi + k < n) ? (hi + k) : n;
        for (int j = lo; j < jmax; ++j) {
            const T tmp = xin[j];
            const int L = k - j;                         /* A(i,j) = A_(L+i, j) */
            const int i_lo = (j - k > lo) ? (j - k) : lo;
            const int i_hi = (j < hi) ? j : hi;          /* off-diagonal rows < j */
            const T *col = &A_(L + i_lo, j);             /* contiguous in i */
            for (int i = i_lo; i < i_hi; ++i) y[i] = y[i] + tmp * (*col++);
            if (j >= lo && j < hi)                       /* diagonal seed */
                y[j] = nounit ? tmp * A_(k, j) : tmp;
        }
    } else {
        const int jmin = (lo - k > 0) ? (lo - k) : 0;
        for (int j = hi - 1; j >= jmin; --j) {
            const T tmp = xin[j];
            const int i_lo = (j + 1 > lo) ? (j + 1) : lo;
            const int i_hi = (j + k + 1 < hi) ? (j + k + 1) : hi;  /* rows > j */
            const T *col = &A_(i_lo - j, j);             /* A(i,j)=A_(i-j,j), contiguous */
            for (int i = i_lo; i < i_hi; ++i) y[i] = y[i] + tmp * (*col++);
            if (j >= lo && j < hi)                       /* diagonal seed */
                y[j] = nounit ? tmp * A_(0, j) : tmp;
        }
    }
}

#ifdef MBLAS_SIMD_DD
/* 4-wide SoA twin of mtbmv_colscatter: same disjoint-row ownership and same
 * per-row column order (hence bit-exact), with x in/y out as SoA limb arrays so
 * the inner axpy runs packed. The matrix column is deinterleaved inline; y rows
 * are plain (loadu) since y is SoA. y[lo,hi) is seeded at each row's diagonal
 * column before any += reaches it (no pre-zero needed). */
static void mtbmv_colscatter_soa(bool upper, bool nounit, int n, int k,
                                 int lo, int hi, const T *a, std::ptrdiff_t lda,
                                 const double *xh, const double *xl,
                                 double *yh, double *yl)
{
    if (upper) {
        const int jmax = (hi + k < n) ? (hi + k) : n;
        for (int j = lo; j < jmax; ++j) {
            const __m256d bh = _mm256_set1_pd(xh[j]), bl = _mm256_set1_pd(xl[j]);
            const T *col = &A_(0, j);
            const std::ptrdiff_t off = k - j;            /* A(i,j) = col[off+i] */
            const int i_lo = (j - k > lo) ? (j - k) : lo;
            const int i_hi = (j < hi) ? j : hi;          /* off-diagonal rows < j */
            int i = i_lo;
            for (; i + 4 <= i_hi; i += 4) {
                __m256d mh, ml; mf_rank1::load_dd4(&col[off + i], mh, ml);
                __m256d ph, pl; mf_rank1::dd_mul(mh, ml, bh, bl, ph, pl);
                __m256d rh, rl;
                mf_rank1::dd_add(_mm256_loadu_pd(yh + i), _mm256_loadu_pd(yl + i), ph, pl, rh, rl);
                _mm256_storeu_pd(yh + i, rh); _mm256_storeu_pd(yl + i, rl);
            }
            const T tmp{xh[j], xl[j]};
            for (; i < i_hi; ++i) { T yi{yh[i], yl[i]}; yi = yi + tmp * col[off + i]; yh[i] = yi.limbs[0]; yl[i] = yi.limbs[1]; }
            if (j >= lo && j < hi) {                      /* diagonal seed */
                if (nounit) { const T d = tmp * col[k]; yh[j] = d.limbs[0]; yl[j] = d.limbs[1]; }
                else { yh[j] = xh[j]; yl[j] = xl[j]; }
            }
        }
    } else {
        const int jmin = (lo - k > 0) ? (lo - k) : 0;
        for (int j = hi - 1; j >= jmin; --j) {
            const __m256d bh = _mm256_set1_pd(xh[j]), bl = _mm256_set1_pd(xl[j]);
            const T *col = &A_(0, j);
            const std::ptrdiff_t off = -j;               /* A(i,j) = col[off+i] */
            const int i_lo = (j + 1 > lo) ? (j + 1) : lo;
            const int i_hi = (j + k + 1 < hi) ? (j + k + 1) : hi;  /* rows > j */
            int i = i_lo;
            for (; i + 4 <= i_hi; i += 4) {
                __m256d mh, ml; mf_rank1::load_dd4(&col[off + i], mh, ml);
                __m256d ph, pl; mf_rank1::dd_mul(mh, ml, bh, bl, ph, pl);
                __m256d rh, rl;
                mf_rank1::dd_add(_mm256_loadu_pd(yh + i), _mm256_loadu_pd(yl + i), ph, pl, rh, rl);
                _mm256_storeu_pd(yh + i, rh); _mm256_storeu_pd(yl + i, rl);
            }
            const T tmp{xh[j], xl[j]};
            for (; i < i_hi; ++i) { T yi{yh[i], yl[i]}; yi = yi + tmp * col[off + i]; yh[i] = yi.limbs[0]; yl[i] = yi.limbs[1]; }
            if (j >= lo && j < hi) {                      /* diagonal seed */
                if (nounit) { const T d = tmp * col[0]; yh[j] = d.limbs[0]; yl[j] = d.limbs[1]; }
                else { yh[j] = xh[j]; yl[j] = xl[j]; }
            }
        }
    }
}
#endif

/* Threaded in-place triangular band matvec. Each thread owns a disjoint output-
 * row range [lo,hi): it gathers y[lo,hi) reading x (never written here), a
 * barrier, then copies its own range back to x. NoTrans uses the contiguous
 * column-scatter above; Trans uses the row-gather (which reads its column
 * contiguously already). Unit-stride reads x directly — no serial input copy
 * (the old full x->xbuf memcpy was an O(n) Amdahl tax); only a strided x is
 * gathered to contiguous up front. Returns true if handled. */
__attribute__((noinline)) static bool mtbmv_omp(
    bool upper, bool trans, bool nounit, int n, int k,
    const T *a, std::size_t lda, T *x, int incx)
{
    if (n < MTBMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MTBMV_MAX_CPUS) nthreads = MTBMV_MAX_CPUS;

    if (incx < 0) x -= (std::ptrdiff_t)(n - 1) * incx;   /* x at logical 0 */

#ifdef MBLAS_SIMD_DD
    /* Both triangles thread the 4-wide SoA kernels: split x to SoA limb arrays
     * once, each thread fills its owned yh/yl rows (NoTrans column-scatter /
     * Trans row-gather), barrier, merge back. Disjoint row ownership + scalar
     * per-row column/d order keep it bit-exact. Alloc failure -> scalar below. */
    {
        const std::size_t np = ((std::size_t)n + 3) & ~(std::size_t)3;
        double *xh = static_cast<double *>(std::aligned_alloc(32, np * sizeof(double)));
        double *xl = static_cast<double *>(std::aligned_alloc(32, np * sizeof(double)));
        double *yh = static_cast<double *>(std::aligned_alloc(32, np * sizeof(double)));
        double *yl = static_cast<double *>(std::aligned_alloc(32, np * sizeof(double)));
        if (xh && xl && yh && yl) {
            for (int i = 0; i < n; ++i) {
                const T v = x[(std::ptrdiff_t)i * incx];
                xh[i] = v.limbs[0]; xl[i] = v.limbs[1];
            }
            for (std::size_t i = n; i < np; ++i) { xh[i] = 0.0; xl[i] = 0.0; }
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int lo = (int)((long long)n * tid / nthreads);
                int hi = (int)((long long)n * (tid + 1) / nthreads);
                if (!trans) mtbmv_colscatter_soa(upper, nounit, n, k, lo, hi, a, lda, xh, xl, yh, yl);
                else        mtbmv_rowgather_t_soa(upper, nounit, n, k, lo, hi, a, lda, xh, xl, yh, yl);
                #pragma omp barrier          /* all reads of x done before write-back */
                for (int i = lo; i < hi; ++i)
                    x[(std::ptrdiff_t)i * incx] = T{yh[i], yl[i]};
            }
            std::free(xh); std::free(xl); std::free(yh); std::free(yl);
            return true;
        }
        std::free(xh); std::free(xl); std::free(yh); std::free(yl);
        /* alloc failure -> scalar path below */
    }
#endif

    const T *xptr = x;
    T *xbuf = NULL;
    if (incx != 1) {
        xbuf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
        if (!xbuf) return false;
        for (int i = 0; i < n; ++i) xbuf[i] = x[(std::ptrdiff_t)i * incx];
        xptr = xbuf;
    }
    T *y = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
    if (!y) { std::free(xbuf); return false; }

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int lo = (int)((long long)n * tid / nthreads);
        int hi = (int)((long long)n * (tid + 1) / nthreads);
        if (!trans) mtbmv_colscatter(upper, nounit, n, k, lo, hi, a, lda, xptr, y);
        else        mtbmv_rowgather_t(upper, nounit, n, k, lo, hi, a, lda, xptr, y);
        #pragma omp barrier              /* all reads of x done before any write-back */
        if (incx == 1) for (int i = lo; i < hi; ++i) x[i] = y[i];
        else           for (int i = lo; i < hi; ++i) x[(std::ptrdiff_t)i * incx] = y[i];
    }
    std::free(y); std::free(xbuf);
    return true;
}
#endif

extern "C" void mtbmv_(
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
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
    const int nounit = (up(diag) != 'U');

    if (N == 0) return;

#ifdef _OPENMP
    if (N >= MTBMV_OMP_MIN && blas_omp_max_threads() > 1
        && mtbmv_omp(UPLO == 'U', TR != 'N', nounit != 0, N, K, a, lda, x, incx))
        return;
#endif

    T *xp = x;
    if (incx < 0) xp -= (std::ptrdiff_t)(N - 1) * incx;   /* x at logical 0 */

#ifdef MBLAS_SIMD_DD
    /* 4-wide SoA — NoTrans axpy-per-column, Trans row-gather. A strided x is
     * gathered into the SoA limb arrays and scattered back (xp is at logical 0). */
    if (TR == 'N'
        && mtbmv_notrans_soa(UPLO == 'U', nounit != 0, N, K, a, lda, xp, incx))
        return;
    if (TR == 'T'
        && mtbmv_trans_soa(UPLO == 'U', nounit != 0, N, K, a, lda, xp, incx))
        return;
#endif

    mtbmv_serial(UPLO == 'U', TR != 'N', nounit != 0,
                 N, K, a, lda, xp, incx);
}

#undef A_
