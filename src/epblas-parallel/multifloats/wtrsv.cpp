/* wtrsv — multifloats complex DD triangular solve.
 * SIMD: pre-pack x to SoA scratch; per i, scalar divide then SIMD
 * inner loop using cdd_mul/cdd_add. TRANS='C' applies dd_neg to A.im
 * before cdd_mul to implement conj(A). */

#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <vector>
#include <multifloats.h>
#ifdef MBLAS_SIMD_DD
#include "mgemm_simd_kernel.h"
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define WTRSV_OMP_MIN 256   /* below this, run the serial SIMD path */
#define WTRSV_BLK     128   /* diagonal-block size for the blocked solve */
#define WTRSV_MAX_CPUS 256
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
const T zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };
inline bool cdd_iszero(const T &x) {
    return x.re.limbs[0] == 0.0 && x.re.limbs[1] == 0.0
        && x.im.limbs[0] == 0.0 && x.im.limbs[1] == 0.0;
}
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T csub(T const &a, T const &b) { return T{ a.re - b.re, a.im - b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }
inline T cdiv(T const &a, T const &b) {
    const R d = b.re * b.re + b.im * b.im;
    const R inv_d = R{1.0, 0.0} / d;
    return T{ (a.re * b.re + a.im * b.im) * inv_d,
              (a.im * b.re - a.re * b.im) * inv_d };
}

#ifdef MBLAS_SIMD_DD
static inline __attribute__((always_inline)) void
soa_load4_cdd(const double *p,
              __m256d &rh, __m256d &rl, __m256d &ih, __m256d &il)
{
    __m256d v0 = _mm256_loadu_pd(p +  0);
    __m256d v1 = _mm256_loadu_pd(p +  4);
    __m256d v2 = _mm256_loadu_pd(p +  8);
    __m256d v3 = _mm256_loadu_pd(p + 12);
    __m256d t0 = _mm256_unpacklo_pd(v0, v1);
    __m256d t1 = _mm256_unpackhi_pd(v0, v1);
    __m256d t2 = _mm256_unpacklo_pd(v2, v3);
    __m256d t3 = _mm256_unpackhi_pd(v2, v3);
    rh = _mm256_permute2f128_pd(t0, t2, 0x20);
    ih = _mm256_permute2f128_pd(t0, t2, 0x31);
    rl = _mm256_permute2f128_pd(t1, t3, 0x20);
    il = _mm256_permute2f128_pd(t1, t3, 0x31);
}
static inline __attribute__((always_inline)) T
hreduce_cdd(__m256d s_rh, __m256d s_rl, __m256d s_ih, __m256d s_il)
{
    __m256d srh_sw = _mm256_permute2f128_pd(s_rh, s_rh, 0x01);
    __m256d srl_sw = _mm256_permute2f128_pd(s_rl, s_rl, 0x01);
    __m256d sih_sw = _mm256_permute2f128_pd(s_ih, s_ih, 0x01);
    __m256d sil_sw = _mm256_permute2f128_pd(s_il, s_il, 0x01);
    __m256d p_rh, p_rl, p_ih, p_il;
    simd_dd::cdd_add(s_rh, s_rl, s_ih, s_il, srh_sw, srl_sw, sih_sw, sil_sw,
                     p_rh, p_rl, p_ih, p_il);
    __m256d prh_sw = _mm256_shuffle_pd(p_rh, p_rh, 0x5);
    __m256d prl_sw = _mm256_shuffle_pd(p_rl, p_rl, 0x5);
    __m256d pih_sw = _mm256_shuffle_pd(p_ih, p_ih, 0x5);
    __m256d pil_sw = _mm256_shuffle_pd(p_il, p_il, 0x5);
    __m256d r_rh, r_rl, r_ih, r_il;
    simd_dd::cdd_add(p_rh, p_rl, p_ih, p_il, prh_sw, prl_sw, pih_sw, pil_sw,
                     r_rh, r_rl, r_ih, r_il);
    double rh[4], rl[4], ih[4], il[4];
    _mm256_storeu_pd(rh, r_rh); _mm256_storeu_pd(rl, r_rl);
    _mm256_storeu_pd(ih, r_ih); _mm256_storeu_pd(il, r_il);
    return T{ R{rh[0], rl[0]}, R{ih[0], il[0]} };
}
#endif
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

/* Bit-exact serial path (the SIMD-packed reference). Also reused as the
 * diagonal-block solver by the threaded path below. */
static void wtrsv_serial(char UPLO, char TR, bool nounit,
                         int N, const T *a, int lda, T *x, int incx)
{
    if (N == 0) return;

    if (incx == 1) {
#ifdef MBLAS_SIMD_DD
        const std::size_t N_pad = (static_cast<std::size_t>(N) + 3) & ~static_cast<std::size_t>(3);
        double *x_rh = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        double *x_rl = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        double *x_ih = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        double *x_il = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        for (int i = 0; i < N; ++i) {
            x_rh[i] = x[i].re.limbs[0]; x_rl[i] = x[i].re.limbs[1];
            x_ih[i] = x[i].im.limbs[0]; x_il[i] = x[i].im.limbs[1];
        }
        for (std::size_t i = static_cast<std::size_t>(N); i < N_pad; ++i) {
            x_rh[i] = 0.0; x_rl[i] = 0.0; x_ih[i] = 0.0; x_il[i] = 0.0;
        }
        const __m256d zerov = _mm256_setzero_pd();

        auto load_x = [&](int k) -> T {
            return T{ R{x_rh[k], x_rl[k]}, R{x_ih[k], x_il[k]} };
        };
        auto store_x = [&](int k, const T &v) {
            x_rh[k] = v.re.limbs[0]; x_rl[k] = v.re.limbs[1];
            x_ih[k] = v.im.limbs[0]; x_il[k] = v.im.limbs[1];
        };

        if (TR == 'N') {
            auto do_axpy_range = [&](int i, int k_lo, int k_hi) {
                T xi = load_x(i);
                if (cdd_iszero(xi)) return;
                if (nounit) { xi = cdiv(xi, A_(i, i)); store_x(i, xi); }
                const __m256d xrh = _mm256_set1_pd(xi.re.limbs[0]);
                const __m256d xrl = _mm256_set1_pd(xi.re.limbs[1]);
                const __m256d xih = _mm256_set1_pd(xi.im.limbs[0]);
                const __m256d xil = _mm256_set1_pd(xi.im.limbs[1]);
                const double *aip = reinterpret_cast<const double *>(&A_(0, i));
                int k = k_lo;
                for (; k < k_hi && (k & 3) != 0; ++k) {
                    T aki{ R{aip[4*k], aip[4*k+1]}, R{aip[4*k+2], aip[4*k+3]} };
                    store_x(k, csub(load_x(k), cmul(xi, aki)));
                }
                for (; k + 3 < k_hi; k += 4) {
                    __m256d a_rh, a_rl, a_ih, a_il;
                    soa_load4_cdd(aip + 4 * k, a_rh, a_rl, a_ih, a_il);
                    __m256d xkrh = _mm256_loadu_pd(x_rh + k);
                    __m256d xkrl = _mm256_loadu_pd(x_rl + k);
                    __m256d xkih = _mm256_loadu_pd(x_ih + k);
                    __m256d xkil = _mm256_loadu_pd(x_il + k);
                    __m256d p_rh, p_rl, p_ih, p_il;
                    simd_dd::cdd_mul(xrh, xrl, xih, xil, a_rh, a_rl, a_ih, a_il,
                                     p_rh, p_rl, p_ih, p_il);
                    simd_dd::dd_neg(p_rh, p_rl);
                    simd_dd::dd_neg(p_ih, p_il);
                    __m256d nrh, nrl, nih, nil;
                    simd_dd::cdd_add(xkrh, xkrl, xkih, xkil, p_rh, p_rl, p_ih, p_il,
                                     nrh, nrl, nih, nil);
                    _mm256_storeu_pd(x_rh + k, nrh);
                    _mm256_storeu_pd(x_rl + k, nrl);
                    _mm256_storeu_pd(x_ih + k, nih);
                    _mm256_storeu_pd(x_il + k, nil);
                }
                for (; k < k_hi; ++k) {
                    T aki{ R{aip[4*k], aip[4*k+1]}, R{aip[4*k+2], aip[4*k+3]} };
                    store_x(k, csub(load_x(k), cmul(xi, aki)));
                }
            };
            if (UPLO == 'L') {
                for (int i = 0; i < N; ++i) do_axpy_range(i, i + 1, N);
            } else {
                for (int i = N - 1; i >= 0; --i) do_axpy_range(i, 0, i);
            }
        } else {
            const bool conj_a = (TR == 'C');
            auto do_dot_range = [&](int i, int k_lo, int k_hi) {
                const double *aip = reinterpret_cast<const double *>(&A_(0, i));
                __m256d s_rh = zerov, s_rl = zerov, s_ih = zerov, s_il = zerov;
                T t = load_x(i);
                int k = k_lo;
                for (; k < k_hi && (k & 3) != 0; ++k) {
                    T aki{ R{aip[4*k], aip[4*k+1]}, R{aip[4*k+2], aip[4*k+3]} };
                    if (conj_a) aki = cconj(aki);
                    t = csub(t, cmul(aki, load_x(k)));
                }
                for (; k + 3 < k_hi; k += 4) {
                    __m256d a_rh, a_rl, a_ih, a_il;
                    soa_load4_cdd(aip + 4 * k, a_rh, a_rl, a_ih, a_il);
                    if (conj_a) simd_dd::dd_neg(a_ih, a_il);
                    __m256d xkrh = _mm256_loadu_pd(x_rh + k);
                    __m256d xkrl = _mm256_loadu_pd(x_rl + k);
                    __m256d xkih = _mm256_loadu_pd(x_ih + k);
                    __m256d xkil = _mm256_loadu_pd(x_il + k);
                    __m256d p_rh, p_rl, p_ih, p_il;
                    simd_dd::cdd_mul(a_rh, a_rl, a_ih, a_il, xkrh, xkrl, xkih, xkil,
                                     p_rh, p_rl, p_ih, p_il);
                    __m256d nsrh, nsrl, nsih, nsil;
                    simd_dd::cdd_add(s_rh, s_rl, s_ih, s_il, p_rh, p_rl, p_ih, p_il,
                                     nsrh, nsrl, nsih, nsil);
                    s_rh = nsrh; s_rl = nsrl; s_ih = nsih; s_il = nsil;
                }
                T s_red = hreduce_cdd(s_rh, s_rl, s_ih, s_il);
                t = csub(t, s_red);
                for (; k < k_hi; ++k) {
                    T aki{ R{aip[4*k], aip[4*k+1]}, R{aip[4*k+2], aip[4*k+3]} };
                    if (conj_a) aki = cconj(aki);
                    t = csub(t, cmul(aki, load_x(k)));
                }
                if (nounit) {
                    T diag_v{ R{aip[4*i], aip[4*i+1]}, R{aip[4*i+2], aip[4*i+3]} };
                    if (conj_a) diag_v = cconj(diag_v);
                    t = cdiv(t, diag_v);
                }
                store_x(i, t);
            };
            if (UPLO == 'L') {
                for (int i = N - 1; i >= 0; --i) do_dot_range(i, i + 1, N);
            } else {
                for (int i = 0; i < N; ++i) do_dot_range(i, 0, i);
            }
        }
        for (int i = 0; i < N; ++i) {
            x[i].re.limbs[0] = x_rh[i]; x[i].re.limbs[1] = x_rl[i];
            x[i].im.limbs[0] = x_ih[i]; x[i].im.limbs[1] = x_il[i];
        }
        std::free(x_rh); std::free(x_rl); std::free(x_ih); std::free(x_il);
#else
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (int i = 0; i < N; ++i) {
                    if (!cdd_iszero(x[i])) {
                        if (nounit) x[i] = cdiv(x[i], A_(i, i));
                        const T xi = x[i];
                        const T *ai = &A_(0, i);
                        for (int k = i + 1; k < N; ++k) x[k] = csub(x[k], cmul(xi, ai[k]));
                    }
                }
            } else {
                for (int i = N - 1; i >= 0; --i) {
                    if (!cdd_iszero(x[i])) {
                        if (nounit) x[i] = cdiv(x[i], A_(i, i));
                        const T xi = x[i];
                        const T *ai = &A_(0, i);
                        for (int k = 0; k < i; ++k) x[k] = csub(x[k], cmul(xi, ai[k]));
                    }
                }
            }
        } else {
            const bool conj_a = (TR == 'C');
            if (UPLO == 'L') {
                for (int i = N - 1; i >= 0; --i) {
                    T t = x[i];
                    const T *ai = &A_(0, i);
                    if (conj_a) {
                        for (int k = i + 1; k < N; ++k) t = csub(t, cmul(cconj(ai[k]), x[k]));
                        if (nounit) t = cdiv(t, cconj(ai[i]));
                    } else {
                        for (int k = i + 1; k < N; ++k) t = csub(t, cmul(ai[k], x[k]));
                        if (nounit) t = cdiv(t, ai[i]);
                    }
                    x[i] = t;
                }
            } else {
                for (int i = 0; i < N; ++i) {
                    T t = x[i];
                    const T *ai = &A_(0, i);
                    if (conj_a) {
                        for (int k = 0; k < i; ++k) t = csub(t, cmul(cconj(ai[k]), x[k]));
                        if (nounit) t = cdiv(t, cconj(ai[i]));
                    } else {
                        for (int k = 0; k < i; ++k) t = csub(t, cmul(ai[k], x[k]));
                        if (nounit) t = cdiv(t, ai[i]);
                    }
                    x[i] = t;
                }
            }
        }
#endif
    } else {
        /* Strided: gather x to a contiguous scratch, run the SIMD incx==1 core,
         * scatter back. O(N) gather vs the O(N^2) strided scalar sweep. */
        T *xbase = (incx < 0) ? x - (std::ptrdiff_t)(N - 1) * incx : x;
        std::vector<T> xs(static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i) xs[i] = xbase[(std::ptrdiff_t)i * incx];
        wtrsv_serial(UPLO, TR, nounit, N, a, lda, xs.data(), 1);
        for (int i = 0; i < N; ++i) xbase[(std::ptrdiff_t)i * incx] = xs[i];
    }
}

#ifdef _OPENMP
/* Blocked threaded complex triangular solve, incx==1 only. Diagonal blocks
 * (MTRSV_BLK) are solved serially via the bit-exact wtrsv_serial; the bulk
 * off-diagonal coupling is a rectangular GEMV threaded over disjoint output
 * rows. Serial fallback stays bit-exact; threaded path matches within DD fuzz
 * tol. Returns true if it handled the call. */
__attribute__((noinline)) static bool wtrsv_omp(
    char UPLO, char TR, bool nounit, int N, const T *a, int lda, T *x)
{
    if (N < WTRSV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > WTRSV_MAX_CPUS) nthreads = WTRSV_MAX_CPUS;
    const bool lower = (UPLO == 'L');
    const bool trans = (TR != 'N');
    const bool conj_a = (TR == 'C');

    if (!trans) {
        /* NoTrans: axpy form. Solve a diagonal block, then propagate its solved
         * columns into the not-yet-solved rows (trailing for L, leading for U). */
        if (lower) {
            for (int j0 = 0; j0 < N; j0 += WTRSV_BLK) {
                int j1 = j0 + WTRSV_BLK; if (j1 > N) j1 = N;
                wtrsv_serial(UPLO, TR, nounit, j1 - j0, &A_(j0, j0), lda, x + j0, 1);
                if (j1 >= N) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    int tid = omp_get_thread_num();
                    int rlo = j1 + (int)((long long)(N - j1) * tid / nthreads);
                    int rhi = j1 + (int)((long long)(N - j1) * (tid + 1) / nthreads);
                    for (int i = j0; i < j1; ++i) {
                        const T xi = x[i];
                        if (cdd_iszero(xi)) continue;
                        const T *ai = &A_(0, i);
                        for (int k = rlo; k < rhi; ++k) x[k] = csub(x[k], cmul(xi, ai[k]));
                    }
                }
            }
        } else {
            for (int j1 = N; j1 > 0; j1 -= WTRSV_BLK) {
                int j0 = j1 - WTRSV_BLK; if (j0 < 0) j0 = 0;
                wtrsv_serial(UPLO, TR, nounit, j1 - j0, &A_(j0, j0), lda, x + j0, 1);
                if (j0 <= 0) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    int tid = omp_get_thread_num();
                    int rlo = (int)((long long)j0 * tid / nthreads);
                    int rhi = (int)((long long)j0 * (tid + 1) / nthreads);
                    for (int i = j0; i < j1; ++i) {
                        const T xi = x[i];
                        if (cdd_iszero(xi)) continue;
                        const T *ai = &A_(0, i);
                        for (int k = rlo; k < rhi; ++k) x[k] = csub(x[k], cmul(xi, ai[k]));
                    }
                }
            }
        }
    } else {
        /* Trans/ConjTrans: dot form. Fold the already-solved out-of-block
         * tail/head into the block rows (threaded, disjoint rows), then solve
         * the diagonal block serially (within-block coupling + divide). */
        if (lower) {                                  /* backward, k > i */
            for (int j1 = N; j1 > 0; j1 -= WTRSV_BLK) {
                int j0 = j1 - WTRSV_BLK; if (j0 < 0) j0 = 0;
                if (j1 < N) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();
                        int ilo = j0 + (int)((long long)(j1 - j0) * tid / nthreads);
                        int ihi = j0 + (int)((long long)(j1 - j0) * (tid + 1) / nthreads);
                        for (int i = ilo; i < ihi; ++i) {
                            const T *ai = &A_(0, i);
                            T s = zero_cdd;
                            for (int k = j1; k < N; ++k) {
                                const T e = conj_a ? cconj(ai[k]) : ai[k];
                                s = cadd(s, cmul(e, x[k]));
                            }
                            x[i] = csub(x[i], s);
                        }
                    }
                }
                wtrsv_serial(UPLO, TR, nounit, j1 - j0, &A_(j0, j0), lda, x + j0, 1);
            }
        } else {                                      /* forward, k < i */
            for (int j0 = 0; j0 < N; j0 += WTRSV_BLK) {
                int j1 = j0 + WTRSV_BLK; if (j1 > N) j1 = N;
                if (j0 > 0) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();
                        int ilo = j0 + (int)((long long)(j1 - j0) * tid / nthreads);
                        int ihi = j0 + (int)((long long)(j1 - j0) * (tid + 1) / nthreads);
                        for (int i = ilo; i < ihi; ++i) {
                            const T *ai = &A_(0, i);
                            T s = zero_cdd;
                            for (int k = 0; k < j0; ++k) {
                                const T e = conj_a ? cconj(ai[k]) : ai[k];
                                s = cadd(s, cmul(e, x[k]));
                            }
                            x[i] = csub(x[i], s);
                        }
                    }
                }
                wtrsv_serial(UPLO, TR, nounit, j1 - j0, &A_(j0, j0), lda, x + j0, 1);
            }
        }
    }
    return true;
}
#endif

extern "C" void wtrsv_(
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
    const char TR   = up(trans);
    const char DIAG = up(diag);
    const bool nounit = (DIAG != 'U');

    if (N == 0) return;

#ifdef _OPENMP
    if (incx == 1 && N >= WTRSV_OMP_MIN && blas_omp_max_threads() > 1
        && wtrsv_omp(UPLO, TR, nounit, N, a, lda, x))
        return;
#endif

    wtrsv_serial(UPLO, TR, nounit, N, a, lda, x, incx);
}

#undef A_
