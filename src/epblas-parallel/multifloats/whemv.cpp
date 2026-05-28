/* whemv — multifloats Hermitian matrix-vector.
 * SIMD: same two-pass pattern as msymv with cdd_mul/cdd_add; Hermitian
 * uses conj(A[k,i]) for the temp2 accumulation, achieved by dd_neg on
 * the loaded A.im before the cdd_mul. Diagonal A[i,i] kept real. */

#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <multifloats.h>
#ifdef MBLAS_SIMD_DD
#include "mgemm_simd_kernel.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
const R rzero{0.0, 0.0};
const T zero_cdd{ rzero, rzero };
const T one_cdd { R{1.0, 0.0}, rzero };

inline bool dd_iszero(R x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool cdd_iszero(const T &x) { return dd_iszero(x.re) && dd_iszero(x.im); }
inline bool cdd_isone (const T &x) {
    return x.re.limbs[0] == 1.0 && x.re.limbs[1] == 0.0 && dd_iszero(x.im);
}
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }

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
/* Horizontal-reduce 4-lane complex DD to scalar T (lane 0). */
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

extern "C" void whemv_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *x, const int *incx_,
    const T *beta_,
    T *y, const int *incy_,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int lda = *lda_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);

    if (N == 0) return;

    if (!cdd_isone(beta)) {
        int iy = (incy < 0) ? -(N - 1) * incy : 0;
        for (int i = 0; i < N; ++i) {
            if (cdd_iszero(beta)) y[iy] = zero_cdd;
            else                  y[iy] = cmul(y[iy], beta);
            iy += incy;
        }
    }
    if (cdd_iszero(alpha)) return;

    if (incx == 1 && incy == 1) {
#ifdef MBLAS_SIMD_DD
        const std::size_t N_pad = (static_cast<std::size_t>(N) + 3) & ~static_cast<std::size_t>(3);
        double *x_rh = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        double *x_rl = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        double *x_ih = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        double *x_il = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        double *y_rh = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        double *y_rl = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        double *y_ih = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        double *y_il = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
        for (int i = 0; i < N; ++i) {
            x_rh[i] = x[i].re.limbs[0]; x_rl[i] = x[i].re.limbs[1];
            x_ih[i] = x[i].im.limbs[0]; x_il[i] = x[i].im.limbs[1];
            y_rh[i] = y[i].re.limbs[0]; y_rl[i] = y[i].re.limbs[1];
            y_ih[i] = y[i].im.limbs[0]; y_il[i] = y[i].im.limbs[1];
        }
        for (std::size_t i = static_cast<std::size_t>(N); i < N_pad; ++i) {
            x_rh[i] = 0.0; x_rl[i] = 0.0; x_ih[i] = 0.0; x_il[i] = 0.0;
            y_rh[i] = 0.0; y_rl[i] = 0.0; y_ih[i] = 0.0; y_il[i] = 0.0;
        }
        const __m256d zerov = _mm256_setzero_pd();

        auto run_inner = [&](int i, int k_lo, int k_hi) -> T {
            const T temp1 = cmul(alpha, x[i]);
            const __m256d t1rh = _mm256_set1_pd(temp1.re.limbs[0]);
            const __m256d t1rl = _mm256_set1_pd(temp1.re.limbs[1]);
            const __m256d t1ih = _mm256_set1_pd(temp1.im.limbs[0]);
            const __m256d t1il = _mm256_set1_pd(temp1.im.limbs[1]);
            const double *aip = reinterpret_cast<const double *>(&A_(0, i));
            __m256d s_rh = zerov, s_rl = zerov, s_ih = zerov, s_il = zerov;
            int k = k_lo;
            /* Align to 4-boundary for unit-aligned SIMD. */
            T temp2_sc = zero_cdd;
            for (; k < k_hi && (k & 3) != 0; ++k) {
                T aki{ R{aip[4*k], aip[4*k+1]}, R{aip[4*k+2], aip[4*k+3]} };
                T yk{ R{y_rh[k], y_rl[k]}, R{y_ih[k], y_il[k]} };
                yk = cadd(yk, cmul(temp1, aki));
                y_rh[k] = yk.re.limbs[0]; y_rl[k] = yk.re.limbs[1];
                y_ih[k] = yk.im.limbs[0]; y_il[k] = yk.im.limbs[1];
                T xk{ R{x_rh[k], x_rl[k]}, R{x_ih[k], x_il[k]} };
                temp2_sc = cadd(temp2_sc, cmul(cconj(aki), xk));
            }
            for (; k + 3 < k_hi; k += 4) {
                __m256d a_rh, a_rl, a_ih, a_il;
                soa_load4_cdd(aip + 4 * k, a_rh, a_rl, a_ih, a_il);
                __m256d yrh = _mm256_loadu_pd(y_rh + k);
                __m256d yrl = _mm256_loadu_pd(y_rl + k);
                __m256d yih = _mm256_loadu_pd(y_ih + k);
                __m256d yil = _mm256_loadu_pd(y_il + k);
                __m256d xrh = _mm256_loadu_pd(x_rh + k);
                __m256d xrl = _mm256_loadu_pd(x_rl + k);
                __m256d xih = _mm256_loadu_pd(x_ih + k);
                __m256d xil = _mm256_loadu_pd(x_il + k);
                /* y[k] += temp1 * A[k,i] */
                __m256d p_rh, p_rl, p_ih, p_il;
                simd_dd::cdd_mul(t1rh, t1rl, t1ih, t1il, a_rh, a_rl, a_ih, a_il,
                                 p_rh, p_rl, p_ih, p_il);
                __m256d nrh, nrl, nih, nil;
                simd_dd::cdd_add(yrh, yrl, yih, yil, p_rh, p_rl, p_ih, p_il,
                                 nrh, nrl, nih, nil);
                _mm256_storeu_pd(y_rh + k, nrh);
                _mm256_storeu_pd(y_rl + k, nrl);
                _mm256_storeu_pd(y_ih + k, nih);
                _mm256_storeu_pd(y_il + k, nil);
                /* temp2 += conj(A[k,i]) * x[k] */
                simd_dd::dd_neg(a_ih, a_il);
                __m256d q_rh, q_rl, q_ih, q_il;
                simd_dd::cdd_mul(a_rh, a_rl, a_ih, a_il, xrh, xrl, xih, xil,
                                 q_rh, q_rl, q_ih, q_il);
                __m256d nsrh, nsrl, nsih, nsil;
                simd_dd::cdd_add(s_rh, s_rl, s_ih, s_il, q_rh, q_rl, q_ih, q_il,
                                 nsrh, nsrl, nsih, nsil);
                s_rh = nsrh; s_rl = nsrl; s_ih = nsih; s_il = nsil;
            }
            T temp2 = hreduce_cdd(s_rh, s_rl, s_ih, s_il);
            temp2 = cadd(temp2, temp2_sc);
            for (; k < k_hi; ++k) {
                T aki{ R{aip[4*k], aip[4*k+1]}, R{aip[4*k+2], aip[4*k+3]} };
                T yk{ R{y_rh[k], y_rl[k]}, R{y_ih[k], y_il[k]} };
                yk = cadd(yk, cmul(temp1, aki));
                y_rh[k] = yk.re.limbs[0]; y_rl[k] = yk.re.limbs[1];
                y_ih[k] = yk.im.limbs[0]; y_il[k] = yk.im.limbs[1];
                T xk{ R{x_rh[k], x_rl[k]}, R{x_ih[k], x_il[k]} };
                temp2 = cadd(temp2, cmul(cconj(aki), xk));
            }
            return temp2;
        };

        if (UPLO == 'L') {
            for (int i = 0; i < N; ++i) {
                const T temp1 = cmul(alpha, x[i]);
                /* Diagonal contribution (A[i,i] real). */
                {
                    T aii_re{ A_(i, i).re, rzero };
                    T yi{ R{y_rh[i], y_rl[i]}, R{y_ih[i], y_il[i]} };
                    yi = cadd(yi, cmul(temp1, aii_re));
                    y_rh[i] = yi.re.limbs[0]; y_rl[i] = yi.re.limbs[1];
                    y_ih[i] = yi.im.limbs[0]; y_il[i] = yi.im.limbs[1];
                }
                T temp2 = run_inner(i, i + 1, N);
                T yi{ R{y_rh[i], y_rl[i]}, R{y_ih[i], y_il[i]} };
                yi = cadd(yi, cmul(alpha, temp2));
                y_rh[i] = yi.re.limbs[0]; y_rl[i] = yi.re.limbs[1];
                y_ih[i] = yi.im.limbs[0]; y_il[i] = yi.im.limbs[1];
            }
        } else {
            for (int i = 0; i < N; ++i) {
                const T temp1 = cmul(alpha, x[i]);
                T temp2 = run_inner(i, 0, i);
                T aii_re{ A_(i, i).re, rzero };
                T yi{ R{y_rh[i], y_rl[i]}, R{y_ih[i], y_il[i]} };
                yi = cadd(yi, cadd(cmul(temp1, aii_re), cmul(alpha, temp2)));
                y_rh[i] = yi.re.limbs[0]; y_rl[i] = yi.re.limbs[1];
                y_ih[i] = yi.im.limbs[0]; y_il[i] = yi.im.limbs[1];
            }
        }
        for (int i = 0; i < N; ++i) {
            y[i].re.limbs[0] = y_rh[i]; y[i].re.limbs[1] = y_rl[i];
            y[i].im.limbs[0] = y_ih[i]; y[i].im.limbs[1] = y_il[i];
        }
        std::free(x_rh); std::free(x_rl); std::free(x_ih); std::free(x_il);
        std::free(y_rh); std::free(y_rl); std::free(y_ih); std::free(y_il);
#else
        if (UPLO == 'L') {
            for (int i = 0; i < N; ++i) {
                const T temp1 = cmul(alpha, x[i]);
                T temp2 = zero_cdd;
                const T *ai = &A_(0, i);
                const T aii_re{ ai[i].re, rzero };
                y[i] = cadd(y[i], cmul(temp1, aii_re));
                for (int k = i + 1; k < N; ++k) {
                    y[k]  = cadd(y[k], cmul(temp1, ai[k]));
                    temp2 = cadd(temp2, cmul(cconj(ai[k]), x[k]));
                }
                y[i] = cadd(y[i], cmul(alpha, temp2));
            }
        } else {
            for (int i = 0; i < N; ++i) {
                const T temp1 = cmul(alpha, x[i]);
                T temp2 = zero_cdd;
                const T *ai = &A_(0, i);
                for (int k = 0; k < i; ++k) {
                    y[k]  = cadd(y[k], cmul(temp1, ai[k]));
                    temp2 = cadd(temp2, cmul(cconj(ai[k]), x[k]));
                }
                const T aii_re{ ai[i].re, rzero };
                y[i] = cadd(y[i], cadd(cmul(temp1, aii_re), cmul(alpha, temp2)));
            }
        }
#endif
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        if (UPLO == 'L') {
            for (int i = 0; i < N; ++i) {
                const T temp1 = cmul(alpha, x[kx + i * incx]);
                T temp2 = zero_cdd;
                const T aii_re{ A_(i, i).re, rzero };
                y[ky + i * incy] = cadd(y[ky + i * incy], cmul(temp1, aii_re));
                for (int k = i + 1; k < N; ++k) {
                    y[ky + k * incy] = cadd(y[ky + k * incy], cmul(temp1, A_(k, i)));
                    temp2 = cadd(temp2, cmul(cconj(A_(k, i)), x[kx + k * incx]));
                }
                y[ky + i * incy] = cadd(y[ky + i * incy], cmul(alpha, temp2));
            }
        } else {
            for (int i = 0; i < N; ++i) {
                const T temp1 = cmul(alpha, x[kx + i * incx]);
                T temp2 = zero_cdd;
                for (int k = 0; k < i; ++k) {
                    y[ky + k * incy] = cadd(y[ky + k * incy], cmul(temp1, A_(k, i)));
                    temp2 = cadd(temp2, cmul(cconj(A_(k, i)), x[kx + k * incx]));
                }
                const T aii_re{ A_(i, i).re, rzero };
                y[ky + i * incy] = cadd(y[ky + i * incy], cadd(cmul(temp1, aii_re), cmul(alpha, temp2)));
            }
        }
    }
}

#undef A_
