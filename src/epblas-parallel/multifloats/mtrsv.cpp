/* mtrsv — multifloats real DD triangular solve.
 * SIMD: pre-pack x to SoA scratch. Per i, scalar divide then SIMD
 * inner loop. TRANS='N' inner loop is an AXPY-into-x; TRANS='T'
 * inner loop is a dot-product reduction with horizontal-reduce. */

#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <multifloats.h>
#include "mf_util.h"
#include "mf_pred.h"
#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define MTRSV_OMP_MIN 256   /* below this, run the serial SIMD path */
#define MTRSV_BLK     128   /* diagonal-block size for the blocked solve */
#define MTRSV_MAX_CPUS 256
#endif

namespace mf = multifloats;
using T = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
const T zero_dd{0.0, 0.0};

#ifdef MBLAS_SIMD_DD
static inline __attribute__((always_inline)) void
soa_load4(const double *p, __m256d &hi, __m256d &lo)
{
    __m256d a01 = _mm256_loadu_pd(p);
    __m256d a23 = _mm256_loadu_pd(p + 4);
    __m256d t0 = _mm256_unpacklo_pd(a01, a23);
    __m256d t1 = _mm256_unpackhi_pd(a01, a23);
    hi = _mm256_permute4x64_pd(t0, 0xD8);
    lo = _mm256_permute4x64_pd(t1, 0xD8);
}
static inline __attribute__((always_inline)) T
hreduce_dd(__m256d s_h, __m256d s_l)
{
    __m256d sh_sw = _mm256_permute2f128_pd(s_h, s_h, 0x01);
    __m256d sl_sw = _mm256_permute2f128_pd(s_l, s_l, 0x01);
    __m256d p_h, p_l;
    simd_fast::add(s_h, s_l, sh_sw, sl_sw, p_h, p_l);
    __m256d ph_sw = _mm256_shuffle_pd(p_h, p_h, 0x5);
    __m256d pl_sw = _mm256_shuffle_pd(p_l, p_l, 0x5);
    __m256d r_h, r_l;
    simd_fast::add(p_h, p_l, ph_sw, pl_sw, r_h, r_l);
    double rh[4], rl[4];
    _mm256_storeu_pd(rh, r_h); _mm256_storeu_pd(rl, r_l);
    return T{rh[0], rl[0]};
}
static inline __attribute__((always_inline)) void
soa_store4(double *p, __m256d hi, __m256d lo)
{
    __m256d hp = _mm256_permute4x64_pd(hi, 0xD8);  /* [h0,h2,h1,h3] */
    __m256d lp = _mm256_permute4x64_pd(lo, 0xD8);  /* [l0,l2,l1,l3] */
    __m256d a01 = _mm256_unpacklo_pd(hp, lp);      /* [h0,l0,h1,l1] */
    __m256d a23 = _mm256_unpackhi_pd(hp, lp);      /* [h2,l2,h3,l3] */
    _mm256_storeu_pd(p,     a01);
    _mm256_storeu_pd(p + 4, a23);
}

/* Off-diagonal SIMD kernels for the blocked threaded solve.
 * msub:  x[k] -= xi * ai[k]  for k in [lo,hi)  (NoTrans axpy form).
 * dot :  returns sum_{k in [lo,hi)} ai[k] * x[k] (Trans dot form). */
static inline void
mtrsv_col_msub(T *x, const T *ai, T xi, int lo, int hi)
{
    double *xp = reinterpret_cast<double *>(x);
    const double *aip = reinterpret_cast<const double *>(ai);
    const __m256d xh = _mm256_set1_pd(xi.limbs[0]);
    const __m256d xl = _mm256_set1_pd(xi.limbs[1]);
    int k = lo;
    for (; k + 4 <= hi; k += 4) {
        __m256d ah, al, ch, cl;
        soa_load4(aip + 2 * k, ah, al);
        soa_load4(xp  + 2 * k, ch, cl);
        __m256d ph, pl;
        simd_fast::mul(xh, xl, ah, al, ph, pl);
        simd_fast::neg(ph, pl);
        __m256d rh, rl;
        simd_fast::add(ch, cl, ph, pl, rh, rl);
        soa_store4(xp + 2 * k, rh, rl);
    }
    for (; k < hi; ++k) x[k] = x[k] - xi * ai[k];
}
static inline T
mtrsv_dot_range(const T *ai, const T *x, int lo, int hi)
{
    const double *aip = reinterpret_cast<const double *>(ai);
    const double *xp  = reinterpret_cast<const double *>(x);
    __m256d sh = _mm256_setzero_pd(), sl = _mm256_setzero_pd();
    int k = lo;
    for (; k + 4 <= hi; k += 4) {
        __m256d ah, al, xih, xil;
        soa_load4(aip + 2 * k, ah, al);
        soa_load4(xp  + 2 * k, xih, xil);
        __m256d ph, pl;
        simd_fast::mul(ah, al, xih, xil, ph, pl);
        simd_fast::add(sh, sl, ph, pl, sh, sl);
    }
    T s = hreduce_dd(sh, sl);
    for (; k < hi; ++k) s = s + ai[k] * x[k];
    return s;
}
#endif
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

/* Bit-exact serial path (the SIMD-packed reference). Also reused as the
 * diagonal-block solver by the threaded path below. TR is already normalized
 * ('C' folded to 'T' by the caller). */
static void mtrsv_serial(char UPLO, char TR, bool nounit,
                         int N, const T *a, int lda, T *x, int incx)
{
    if (N == 0) return;

    if (incx == 1) {
#ifdef MBLAS_SIMD_DD
        const std::size_t N_pad = (static_cast<std::size_t>(N) + 3) & ~static_cast<std::size_t>(3);
        double *x_hi = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        double *x_lo = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        for (int i = 0; i < N; ++i) { x_hi[i] = x[i].limbs[0]; x_lo[i] = x[i].limbs[1]; }
        for (std::size_t i = static_cast<std::size_t>(N); i < N_pad; ++i) { x_hi[i] = 0.0; x_lo[i] = 0.0; }
        const __m256d zerov = _mm256_setzero_pd();

        if (TR == 'N') {
            if (UPLO == 'L') {
                for (int i = 0; i < N; ++i) {
                    T xi{x_hi[i], x_lo[i]};
                    if (eq0(xi)) continue;
                    if (nounit) xi = xi / A_(i, i);
                    x_hi[i] = xi.limbs[0]; x_lo[i] = xi.limbs[1];
                    const __m256d xih = _mm256_set1_pd(xi.limbs[0]);
                    const __m256d xil = _mm256_set1_pd(xi.limbs[1]);
                    const double *aip = reinterpret_cast<const double *>(&A_(0, i));
                    int k = i + 1;
                    for (; k < N && (k & 3) != 0; ++k) {
                        T xk{x_hi[k], x_lo[k]};
                        T aki{aip[2*k], aip[2*k+1]};
                        xk = xk - xi * aki;
                        x_hi[k] = xk.limbs[0]; x_lo[k] = xk.limbs[1];
                    }
                    for (; k + 3 < N; k += 4) {
                        __m256d a_h, a_l;
                        soa_load4(aip + 2 * k, a_h, a_l);
                        __m256d xh = _mm256_loadu_pd(x_hi + k);
                        __m256d xl = _mm256_loadu_pd(x_lo + k);
                        __m256d p_h, p_l;
                        simd_fast::mul(xih, xil, a_h, a_l, p_h, p_l);
                        simd_fast::neg(p_h, p_l);
                        __m256d nxh, nxl;
                        simd_fast::add(xh, xl, p_h, p_l, nxh, nxl);
                        _mm256_storeu_pd(x_hi + k, nxh);
                        _mm256_storeu_pd(x_lo + k, nxl);
                    }
                    for (; k < N; ++k) {
                        T xk{x_hi[k], x_lo[k]};
                        T aki{aip[2*k], aip[2*k+1]};
                        xk = xk - xi * aki;
                        x_hi[k] = xk.limbs[0]; x_lo[k] = xk.limbs[1];
                    }
                }
            } else {
                for (int i = N - 1; i >= 0; --i) {
                    T xi{x_hi[i], x_lo[i]};
                    if (eq0(xi)) continue;
                    if (nounit) xi = xi / A_(i, i);
                    x_hi[i] = xi.limbs[0]; x_lo[i] = xi.limbs[1];
                    const __m256d xih = _mm256_set1_pd(xi.limbs[0]);
                    const __m256d xil = _mm256_set1_pd(xi.limbs[1]);
                    const double *aip = reinterpret_cast<const double *>(&A_(0, i));
                    int k = 0;
                    for (; k + 3 < i; k += 4) {
                        __m256d a_h, a_l;
                        soa_load4(aip + 2 * k, a_h, a_l);
                        __m256d xh = _mm256_loadu_pd(x_hi + k);
                        __m256d xl = _mm256_loadu_pd(x_lo + k);
                        __m256d p_h, p_l;
                        simd_fast::mul(xih, xil, a_h, a_l, p_h, p_l);
                        simd_fast::neg(p_h, p_l);
                        __m256d nxh, nxl;
                        simd_fast::add(xh, xl, p_h, p_l, nxh, nxl);
                        _mm256_storeu_pd(x_hi + k, nxh);
                        _mm256_storeu_pd(x_lo + k, nxl);
                    }
                    for (; k < i; ++k) {
                        T xk{x_hi[k], x_lo[k]};
                        T aki{aip[2*k], aip[2*k+1]};
                        xk = xk - xi * aki;
                        x_hi[k] = xk.limbs[0]; x_lo[k] = xk.limbs[1];
                    }
                }
            }
        } else {  /* TRANS = 'T' */
            if (UPLO == 'L') {
                for (int i = N - 1; i >= 0; --i) {
                    const double *aip = reinterpret_cast<const double *>(&A_(0, i));
                    __m256d s_h = zerov, s_l = zerov;
                    T t{x_hi[i], x_lo[i]};
                    int k = i + 1;
                    for (; k < N && (k & 3) != 0; ++k) {
                        T xk{x_hi[k], x_lo[k]};
                        T aki{aip[2*k], aip[2*k+1]};
                        t = t - aki * xk;
                    }
                    for (; k + 3 < N; k += 4) {
                        __m256d a_h, a_l;
                        soa_load4(aip + 2 * k, a_h, a_l);
                        __m256d xh = _mm256_loadu_pd(x_hi + k);
                        __m256d xl = _mm256_loadu_pd(x_lo + k);
                        __m256d p_h, p_l;
                        simd_fast::mul(a_h, a_l, xh, xl, p_h, p_l);
                        __m256d nsh, nsl;
                        simd_fast::add(s_h, s_l, p_h, p_l, nsh, nsl);
                        s_h = nsh; s_l = nsl;
                    }
                    T s_red = hreduce_dd(s_h, s_l);
                    t = t - s_red;
                    for (; k < N; ++k) {
                        T xk{x_hi[k], x_lo[k]};
                        T aki{aip[2*k], aip[2*k+1]};
                        t = t - aki * xk;
                    }
                    if (nounit) t = t / A_(i, i);
                    x_hi[i] = t.limbs[0]; x_lo[i] = t.limbs[1];
                }
            } else {
                for (int i = 0; i < N; ++i) {
                    const double *aip = reinterpret_cast<const double *>(&A_(0, i));
                    __m256d s_h = zerov, s_l = zerov;
                    T t{x_hi[i], x_lo[i]};
                    int k = 0;
                    for (; k + 3 < i; k += 4) {
                        __m256d a_h, a_l;
                        soa_load4(aip + 2 * k, a_h, a_l);
                        __m256d xh = _mm256_loadu_pd(x_hi + k);
                        __m256d xl = _mm256_loadu_pd(x_lo + k);
                        __m256d p_h, p_l;
                        simd_fast::mul(a_h, a_l, xh, xl, p_h, p_l);
                        __m256d nsh, nsl;
                        simd_fast::add(s_h, s_l, p_h, p_l, nsh, nsl);
                        s_h = nsh; s_l = nsl;
                    }
                    T s_red = hreduce_dd(s_h, s_l);
                    t = t - s_red;
                    for (; k < i; ++k) {
                        T xk{x_hi[k], x_lo[k]};
                        T aki{aip[2*k], aip[2*k+1]};
                        t = t - aki * xk;
                    }
                    if (nounit) t = t / A_(i, i);
                    x_hi[i] = t.limbs[0]; x_lo[i] = t.limbs[1];
                }
            }
        }
        for (int i = 0; i < N; ++i) { x[i].limbs[0] = x_hi[i]; x[i].limbs[1] = x_lo[i]; }
        std::free(x_hi); std::free(x_lo);
#else
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (int i = 0; i < N; ++i) {
                    if (!eq0(x[i])) {
                        if (nounit) x[i] = x[i] / A_(i, i);
                        const T xi = x[i];
                        const T *ai = &A_(0, i);
                        for (int k = i + 1; k < N; ++k) x[k] = x[k] - xi * ai[k];
                    }
                }
            } else {
                for (int i = N - 1; i >= 0; --i) {
                    if (!eq0(x[i])) {
                        if (nounit) x[i] = x[i] / A_(i, i);
                        const T xi = x[i];
                        const T *ai = &A_(0, i);
                        for (int k = 0; k < i; ++k) x[k] = x[k] - xi * ai[k];
                    }
                }
            }
        } else {
            if (UPLO == 'L') {
                for (int i = N - 1; i >= 0; --i) {
                    T t = x[i];
                    const T *ai = &A_(0, i);
                    for (int k = i + 1; k < N; ++k) t = t - ai[k] * x[k];
                    if (nounit) t = t / ai[i];
                    x[i] = t;
                }
            } else {
                for (int i = 0; i < N; ++i) {
                    T t = x[i];
                    const T *ai = &A_(0, i);
                    for (int k = 0; k < i; ++k) t = t - ai[k] * x[k];
                    if (nounit) t = t / ai[i];
                    x[i] = t;
                }
            }
        }
#endif
    } else {
        /* Strided: gather x into contiguous scratch, run the incx==1 core (the
         * packed-SIMD solve above — ~4x faster per element than a strided scalar
         * walk), then scatter back. O(N) gather/scatter vs the O(N^2) solve, so
         * free past tiny N. The contiguous core is the bit-exact reference for
         * incx==1, so routing strided through it reproduces that exact result
         * (same SIMD grouping). Stack scratch for small N, heap past it; a
         * direct strided walk is the fallback if the heap alloc fails. */
        const std::ptrdiff_t n = N, sx = incx;
        const std::ptrdiff_t kx = (sx < 0) ? -(n - 1) * sx : 0;
        std::ptrdiff_t ix;
        T stackbuf[512];
        T *buf = (N <= 512) ? stackbuf
                            : static_cast<T *>(std::malloc(static_cast<std::size_t>(N) * sizeof(T)));
        if (buf) {
            std::ptrdiff_t ix = kx;
            for (std::ptrdiff_t i = 0; i < n; ++i) { buf[i] = x[ix]; ix += sx; }
            mtrsv_serial(UPLO, TR, nounit, N, a, lda, buf, 1);
            ix = kx;
            for (std::ptrdiff_t i = 0; i < n; ++i) { x[ix] = buf[i]; ix += sx; }
            if (N > 512) std::free(buf);
        } else if (TR == 'N') {
            if (UPLO == 'L') {
                for (std::ptrdiff_t i = 0; i < n; ++i) {
                    const std::ptrdiff_t ixi = kx + i * sx;
                    if (!eq0(x[ixi])) {
                        const T *ai = &A_(0, i);
                        if (nounit) x[ixi] = x[ixi] / ai[i];
                        const T xi = x[ixi];
                        ix = ixi;
                        for (std::ptrdiff_t k = i + 1; k < n; ++k) { ix += sx; x[ix] -= xi * ai[k]; }
                    }
                }
            } else {
                for (std::ptrdiff_t i = n - 1; i >= 0; --i) {
                    const std::ptrdiff_t ixi = kx + i * sx;
                    if (!eq0(x[ixi])) {
                        const T *ai = &A_(0, i);
                        if (nounit) x[ixi] = x[ixi] / ai[i];
                        const T xi = x[ixi];
                        ix = kx;
                        for (std::ptrdiff_t k = 0; k < i; ++k) { x[ix] -= xi * ai[k]; ix += sx; }
                    }
                }
            }
        } else {
            if (UPLO == 'L') {
                for (std::ptrdiff_t i = n - 1; i >= 0; --i) {
                    const std::ptrdiff_t ixi = kx + i * sx;
                    const T *ai = &A_(0, i);
                    T t = x[ixi];
                    ix = ixi;
                    for (std::ptrdiff_t k = i + 1; k < n; ++k) { ix += sx; t = t - ai[k] * x[ix]; }
                    if (nounit) t = t / ai[i];
                    x[ixi] = t;
                }
            } else {
                for (std::ptrdiff_t i = 0; i < n; ++i) {
                    const std::ptrdiff_t ixi = kx + i * sx;
                    const T *ai = &A_(0, i);
                    T t = x[ixi];
                    ix = kx;
                    for (std::ptrdiff_t k = 0; k < i; ++k) { t = t - ai[k] * x[ix]; ix += sx; }
                    if (nounit) t = t / ai[i];
                    x[ixi] = t;
                }
            }
        }
    }
}

#ifdef _OPENMP
/* Blocked threaded triangular solve, incx==1 only. The loop-carried dependence
 * is confined to small MTRSV_BLK diagonal blocks (solved serially via the
 * bit-exact mtrsv_serial); the bulk O(N^2) off-diagonal coupling is a
 * rectangular GEMV threaded over disjoint output rows. The serial fallback
 * stays bit-exact; this threaded path only matches it within DD fuzz tol
 * (the off-diagonal contribution is regrouped / scalar rather than SIMD).
 * Returns true if it handled the call. */
__attribute__((noinline)) static bool mtrsv_omp(
    char UPLO, char TR, bool nounit, int N, const T *a, int lda, T *x)
{
    if (N < MTRSV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MTRSV_MAX_CPUS) nthreads = MTRSV_MAX_CPUS;
    const bool lower = (UPLO == 'L');
    const bool trans = (TR != 'N');

    if (!trans) {
        /* NoTrans: axpy form. Solve a diagonal block, then propagate its solved
         * columns into the not-yet-solved rows (trailing for L, leading for U). */
        if (lower) {
            for (int j0 = 0; j0 < N; j0 += MTRSV_BLK) {
                int j1 = j0 + MTRSV_BLK; if (j1 > N) j1 = N;
                mtrsv_serial(UPLO, TR, nounit, j1 - j0, &A_(j0, j0), lda, x + j0, 1);
                if (j1 >= N) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    int tid = omp_get_thread_num();
                    int rlo = j1 + (int)((long long)(N - j1) * tid / nthreads);
                    int rhi = j1 + (int)((long long)(N - j1) * (tid + 1) / nthreads);
                    for (int i = j0; i < j1; ++i) {
                        const T xi = x[i];
                        if (eq0(xi)) continue;
                        const T *ai = &A_(0, i);
#ifdef MBLAS_SIMD_DD
                        mtrsv_col_msub(x, ai, xi, rlo, rhi);
#else
                        for (int k = rlo; k < rhi; ++k) x[k] = x[k] - xi * ai[k];
#endif
                    }
                }
            }
        } else {
            for (int j1 = N; j1 > 0; j1 -= MTRSV_BLK) {
                int j0 = j1 - MTRSV_BLK; if (j0 < 0) j0 = 0;
                mtrsv_serial(UPLO, TR, nounit, j1 - j0, &A_(j0, j0), lda, x + j0, 1);
                if (j0 <= 0) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    int tid = omp_get_thread_num();
                    int rlo = (int)((long long)j0 * tid / nthreads);
                    int rhi = (int)((long long)j0 * (tid + 1) / nthreads);
                    for (int i = j0; i < j1; ++i) {
                        const T xi = x[i];
                        if (eq0(xi)) continue;
                        const T *ai = &A_(0, i);
#ifdef MBLAS_SIMD_DD
                        mtrsv_col_msub(x, ai, xi, rlo, rhi);
#else
                        for (int k = rlo; k < rhi; ++k) x[k] = x[k] - xi * ai[k];
#endif
                    }
                }
            }
        }
    } else {
        /* Trans: dot form. Fold the already-solved out-of-block tail/head into
         * the block rows (threaded, disjoint rows), then solve the diagonal
         * block serially (it adds the within-block coupling + divides). */
        if (lower) {                                  /* backward, k > i */
            for (int j1 = N; j1 > 0; j1 -= MTRSV_BLK) {
                int j0 = j1 - MTRSV_BLK; if (j0 < 0) j0 = 0;
                if (j1 < N) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();
                        int ilo = j0 + (int)((long long)(j1 - j0) * tid / nthreads);
                        int ihi = j0 + (int)((long long)(j1 - j0) * (tid + 1) / nthreads);
                        for (int i = ilo; i < ihi; ++i) {
                            const T *ai = &A_(0, i);
#ifdef MBLAS_SIMD_DD
                            T s = mtrsv_dot_range(ai, x, j1, N);
#else
                            T s = zero_dd;
                            for (int k = j1; k < N; ++k) s = s + ai[k] * x[k];
#endif
                            x[i] = x[i] - s;
                        }
                    }
                }
                mtrsv_serial(UPLO, TR, nounit, j1 - j0, &A_(j0, j0), lda, x + j0, 1);
            }
        } else {                                      /* forward, k < i */
            for (int j0 = 0; j0 < N; j0 += MTRSV_BLK) {
                int j1 = j0 + MTRSV_BLK; if (j1 > N) j1 = N;
                if (j0 > 0) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();
                        int ilo = j0 + (int)((long long)(j1 - j0) * tid / nthreads);
                        int ihi = j0 + (int)((long long)(j1 - j0) * (tid + 1) / nthreads);
                        for (int i = ilo; i < ihi; ++i) {
                            const T *ai = &A_(0, i);
#ifdef MBLAS_SIMD_DD
                            T s = mtrsv_dot_range(ai, x, 0, j0);
#else
                            T s = zero_dd;
                            for (int k = 0; k < j0; ++k) s = s + ai[k] * x[k];
#endif
                            x[i] = x[i] - s;
                        }
                    }
                }
                mtrsv_serial(UPLO, TR, nounit, j1 - j0, &A_(j0, j0), lda, x + j0, 1);
            }
        }
    }
    return true;
}
#endif

extern "C" void mtrsv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *a, const int *lda_,
    T *x, const int *incx_,
    std::size_t uplo_len, std::size_t trans_len, std::size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const int N = *n_;
    const int lda = *lda_, incx = *incx_;
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
    const char DIAG = up(diag);
    const bool nounit = (DIAG != 'U');

    if (N == 0) return;

#ifdef _OPENMP
    if (incx == 1 && N >= MTRSV_OMP_MIN && blas_omp_max_threads() > 1
        && mtrsv_omp(UPLO, TR, nounit, N, a, lda, x))
        return;
#endif

    mtrsv_serial(UPLO, TR, nounit, N, a, lda, x, incx);
}

#undef A_
