/* mtrsv — multifloats real DD triangular solve.
 * SIMD: pre-pack x to SoA scratch. Per i, scalar divide then SIMD
 * inner loop. TRANS='N' inner loop is an AXPY-into-x; TRANS='T'
 * inner loop is a dot-product reduction with horizontal-reduce. */

#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <multifloats.h>
#ifdef MBLAS_SIMD_DD
#include "mgemm_simd_kernel.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
const T zero_dd{0.0, 0.0};
inline bool dd_iszero(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }

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
    simd_dd::dd_add(s_h, s_l, sh_sw, sl_sw, p_h, p_l);
    __m256d ph_sw = _mm256_shuffle_pd(p_h, p_h, 0x5);
    __m256d pl_sw = _mm256_shuffle_pd(p_l, p_l, 0x5);
    __m256d r_h, r_l;
    simd_dd::dd_add(p_h, p_l, ph_sw, pl_sw, r_h, r_l);
    double rh[4], rl[4];
    _mm256_storeu_pd(rh, r_h); _mm256_storeu_pd(rl, r_l);
    return T{rh[0], rl[0]};
}
#endif
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

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
                    if (dd_iszero(xi)) continue;
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
                        simd_dd::dd_mul(xih, xil, a_h, a_l, p_h, p_l);
                        simd_dd::dd_neg(p_h, p_l);
                        __m256d nxh, nxl;
                        simd_dd::dd_add(xh, xl, p_h, p_l, nxh, nxl);
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
                    if (dd_iszero(xi)) continue;
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
                        simd_dd::dd_mul(xih, xil, a_h, a_l, p_h, p_l);
                        simd_dd::dd_neg(p_h, p_l);
                        __m256d nxh, nxl;
                        simd_dd::dd_add(xh, xl, p_h, p_l, nxh, nxl);
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
                        simd_dd::dd_mul(a_h, a_l, xh, xl, p_h, p_l);
                        __m256d nsh, nsl;
                        simd_dd::dd_add(s_h, s_l, p_h, p_l, nsh, nsl);
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
                        simd_dd::dd_mul(a_h, a_l, xh, xl, p_h, p_l);
                        __m256d nsh, nsl;
                        simd_dd::dd_add(s_h, s_l, p_h, p_l, nsh, nsl);
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
                    if (!dd_iszero(x[i])) {
                        if (nounit) x[i] = x[i] / A_(i, i);
                        const T xi = x[i];
                        const T *ai = &A_(0, i);
                        for (int k = i + 1; k < N; ++k) x[k] = x[k] - xi * ai[k];
                    }
                }
            } else {
                for (int i = N - 1; i >= 0; --i) {
                    if (!dd_iszero(x[i])) {
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
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (int i = 0; i < N; ++i) {
                    const int ix = kx + i * incx;
                    if (!dd_iszero(x[ix])) {
                        if (nounit) x[ix] = x[ix] / A_(i, i);
                        const T xi = x[ix];
                        for (int k = i + 1; k < N; ++k) x[kx + k * incx] = x[kx + k * incx] - xi * A_(k, i);
                    }
                }
            } else {
                for (int i = N - 1; i >= 0; --i) {
                    const int ix = kx + i * incx;
                    if (!dd_iszero(x[ix])) {
                        if (nounit) x[ix] = x[ix] / A_(i, i);
                        const T xi = x[ix];
                        for (int k = 0; k < i; ++k) x[kx + k * incx] = x[kx + k * incx] - xi * A_(k, i);
                    }
                }
            }
        } else {
            if (UPLO == 'L') {
                for (int i = N - 1; i >= 0; --i) {
                    T t = x[kx + i * incx];
                    for (int k = i + 1; k < N; ++k) t = t - A_(k, i) * x[kx + k * incx];
                    if (nounit) t = t / A_(i, i);
                    x[kx + i * incx] = t;
                }
            } else {
                for (int i = 0; i < N; ++i) {
                    T t = x[kx + i * incx];
                    for (int k = 0; k < i; ++k) t = t - A_(k, i) * x[kx + k * incx];
                    if (nounit) t = t / A_(i, i);
                    x[kx + i * incx] = t;
                }
            }
        }
    }
}

#undef A_
