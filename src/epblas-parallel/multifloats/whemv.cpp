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
#if defined(_OPENMP) && defined(MBLAS_SIMD_DD)
#include <cstring>
#include <omp.h>
#include "../common/blas_omp.h"
#define WHEMV_OMP_MIN 256
#define WHEMV_MAX_CPUS 256
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

#ifdef MBLAS_SIMD_DD
namespace {
/* SIMD inner sweep for Hermitian column i over k in [k_lo,k_hi): the temp1-axpy
 * y[k] += temp1*A[k,i] folded into the SoA accumulator yacc, returning
 * temp2 = sum conj(A[k,i])*x[k]. Every yacc write is additive, so the same
 * instructions serve the serial path (yacc = shared y, in column order) and the
 * threaded path (yacc = private zero buffer, disjoint columns). */
static inline __attribute__((always_inline)) T
whemv_inner(int i, int k_lo, int k_hi, const T *a, std::size_t lda, T alpha,
            const double *x_rh, const double *x_rl,
            const double *x_ih, const double *x_il,
            double *yacc_rh, double *yacc_rl,
            double *yacc_ih, double *yacc_il)
{
    const T t1 = cmul(alpha, T{ R{x_rh[i], x_rl[i]}, R{x_ih[i], x_il[i]} });
    const __m256d t1rh = _mm256_set1_pd(t1.re.limbs[0]);
    const __m256d t1rl = _mm256_set1_pd(t1.re.limbs[1]);
    const __m256d t1ih = _mm256_set1_pd(t1.im.limbs[0]);
    const __m256d t1il = _mm256_set1_pd(t1.im.limbs[1]);
    const double *aip = reinterpret_cast<const double *>(&A_(0, i));
    const __m256d zerov = _mm256_setzero_pd();
    __m256d s_rh = zerov, s_rl = zerov, s_ih = zerov, s_il = zerov;
    int k = k_lo;
    T temp2_sc = zero_cdd;
    /* Align to 4-boundary for unit-aligned SIMD. */
    for (; k < k_hi && (k & 3) != 0; ++k) {
        T aki{ R{aip[4*k], aip[4*k+1]}, R{aip[4*k+2], aip[4*k+3]} };
        T yk{ R{yacc_rh[k], yacc_rl[k]}, R{yacc_ih[k], yacc_il[k]} };
        yk = cadd(yk, cmul(t1, aki));
        yacc_rh[k] = yk.re.limbs[0]; yacc_rl[k] = yk.re.limbs[1];
        yacc_ih[k] = yk.im.limbs[0]; yacc_il[k] = yk.im.limbs[1];
        T xk{ R{x_rh[k], x_rl[k]}, R{x_ih[k], x_il[k]} };
        temp2_sc = cadd(temp2_sc, cmul(cconj(aki), xk));
    }
    for (; k + 3 < k_hi; k += 4) {
        __m256d a_rh, a_rl, a_ih, a_il;
        soa_load4_cdd(aip + 4 * k, a_rh, a_rl, a_ih, a_il);
        __m256d yrh = _mm256_loadu_pd(yacc_rh + k);
        __m256d yrl = _mm256_loadu_pd(yacc_rl + k);
        __m256d yih = _mm256_loadu_pd(yacc_ih + k);
        __m256d yil = _mm256_loadu_pd(yacc_il + k);
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
        _mm256_storeu_pd(yacc_rh + k, nrh);
        _mm256_storeu_pd(yacc_rl + k, nrl);
        _mm256_storeu_pd(yacc_ih + k, nih);
        _mm256_storeu_pd(yacc_il + k, nil);
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
        T yk{ R{yacc_rh[k], yacc_rl[k]}, R{yacc_ih[k], yacc_il[k]} };
        yk = cadd(yk, cmul(t1, aki));
        yacc_rh[k] = yk.re.limbs[0]; yacc_rl[k] = yk.re.limbs[1];
        yacc_ih[k] = yk.im.limbs[0]; yacc_il[k] = yk.im.limbs[1];
        T xk{ R{x_rh[k], x_rl[k]}, R{x_ih[k], x_il[k]} };
        temp2 = cadd(temp2, cmul(cconj(aki), xk));
    }
    return temp2;
}

/* One Hermitian column i's contribution ADDED into the SoA accumulator yacc:
 * the off-diagonal axpy + temp2 (whemv_inner) plus the real diagonal and
 * alpha*temp2 folded into yacc[i]. Additive throughout → shared by serial
 * (column order, bit-identical to the prior inline body) and threaded
 * (private zero buffer, disjoint columns → within DD fuzz tol). */
static inline __attribute__((always_inline)) void
whemv_col(bool lower, int i, int N, const T *a, std::size_t lda, T alpha,
          const double *x_rh, const double *x_rl,
          const double *x_ih, const double *x_il,
          double *y_rh, double *y_rl, double *y_ih, double *y_il)
{
    const T temp1 = cmul(alpha, T{ R{x_rh[i], x_rl[i]}, R{x_ih[i], x_il[i]} });
    if (lower) {
        /* Diagonal contribution (A[i,i] real). */
        T aii_re{ A_(i, i).re, rzero };
        T yi{ R{y_rh[i], y_rl[i]}, R{y_ih[i], y_il[i]} };
        yi = cadd(yi, cmul(temp1, aii_re));
        y_rh[i] = yi.re.limbs[0]; y_rl[i] = yi.re.limbs[1];
        y_ih[i] = yi.im.limbs[0]; y_il[i] = yi.im.limbs[1];
        T temp2 = whemv_inner(i, i + 1, N, a, lda, alpha,
                              x_rh, x_rl, x_ih, x_il, y_rh, y_rl, y_ih, y_il);
        T yi2{ R{y_rh[i], y_rl[i]}, R{y_ih[i], y_il[i]} };
        yi2 = cadd(yi2, cmul(alpha, temp2));
        y_rh[i] = yi2.re.limbs[0]; y_rl[i] = yi2.re.limbs[1];
        y_ih[i] = yi2.im.limbs[0]; y_il[i] = yi2.im.limbs[1];
    } else {
        T temp2 = whemv_inner(i, 0, i, a, lda, alpha,
                              x_rh, x_rl, x_ih, x_il, y_rh, y_rl, y_ih, y_il);
        T aii_re{ A_(i, i).re, rzero };
        T yi{ R{y_rh[i], y_rl[i]}, R{y_ih[i], y_il[i]} };
        yi = cadd(yi, cadd(cmul(temp1, aii_re), cmul(alpha, temp2)));
        y_rh[i] = yi.re.limbs[0]; y_rl[i] = yi.re.limbs[1];
        y_ih[i] = yi.im.limbs[0]; y_il[i] = yi.im.limbs[1];
    }
}
}
#endif

#if defined(_OPENMP) && defined(MBLAS_SIMD_DD)
/* Threaded Hermitian matvec: private-y-accumulator. y_* enter holding beta*y.
 * Columns distributed cyclically (balances triangular work); each thread folds
 * its columns into a private zero buffer via whemv_col, then partials reduce
 * back onto y_*. Contiguous column access preserved. Returns true if handled. */
__attribute__((noinline)) static bool whemv_omp(
    bool lower, int N, const T *a, std::size_t lda, T alpha,
    const double *x_rh, const double *x_rl, const double *x_ih, const double *x_il,
    double *y_rh, double *y_rl, double *y_ih, double *y_il)
{
    int nthreads = blas_omp_max_threads();
    if (nthreads <= 1 || omp_in_parallel()) return false;
    if (nthreads > WHEMV_MAX_CPUS) nthreads = WHEMV_MAX_CPUS;

    const std::size_t N_pad = (static_cast<std::size_t>(N) + 3) & ~static_cast<std::size_t>(3);
    const std::size_t per = 4 * N_pad;
    double *pool = static_cast<double *>(
        std::aligned_alloc(32, static_cast<std::size_t>(nthreads) * per * sizeof(double)));
    if (!pool) return false;
    std::memset(pool, 0, static_cast<std::size_t>(nthreads) * per * sizeof(double));

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        double *p = pool + static_cast<std::size_t>(tid) * per;
        double *yp_rh = p, *yp_rl = p + N_pad, *yp_ih = p + 2 * N_pad, *yp_il = p + 3 * N_pad;
        for (int i = tid; i < N; i += nthreads)
            whemv_col(lower, i, N, a, lda, alpha,
                      x_rh, x_rl, x_ih, x_il, yp_rh, yp_rl, yp_ih, yp_il);
    }

    for (int t = 0; t < nthreads; ++t) {
        const double *p = pool + static_cast<std::size_t>(t) * per;
        const double *yp_rh = p, *yp_rl = p + N_pad, *yp_ih = p + 2 * N_pad, *yp_il = p + 3 * N_pad;
        for (int k = 0; k < N; ++k) {
            T yk{ R{y_rh[k], y_rl[k]}, R{y_ih[k], y_il[k]} };
            yk = cadd(yk, T{ R{yp_rh[k], yp_rl[k]}, R{yp_ih[k], yp_il[k]} });
            y_rh[k] = yk.re.limbs[0]; y_rl[k] = yk.re.limbs[1];
            y_ih[k] = yk.im.limbs[0]; y_il[k] = yk.im.limbs[1];
        }
    }
    std::free(pool);
    return true;
}
#endif

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

        bool done_omp = false;
#if defined(_OPENMP)
        if (N >= WHEMV_OMP_MIN && blas_omp_max_threads() > 1)
            done_omp = whemv_omp(UPLO == 'L', N, a, lda, alpha,
                                 x_rh, x_rl, x_ih, x_il, y_rh, y_rl, y_ih, y_il);
#endif
        if (!done_omp) {
            if (UPLO == 'L') {
                for (int i = 0; i < N; ++i)
                    whemv_col(true, i, N, a, lda, alpha,
                              x_rh, x_rl, x_ih, x_il, y_rh, y_rl, y_ih, y_il);
            } else {
                for (int i = 0; i < N; ++i)
                    whemv_col(false, i, N, a, lda, alpha,
                              x_rh, x_rl, x_ih, x_il, y_rh, y_rl, y_ih, y_il);
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
        /* Hoist the column base ai=&A_(0,i) (A is unit-stride in the row index k,
         * so A_(k,i)=ai[k]), load each ai[k] once for both the temp1-axpy and the
         * conj-dot, and walk the strided x/y by incremental ix/iy (no k*inc
         * multiply per element). Bit-identical to the macro form. */
        if (UPLO == 'L') {
            for (int i = 0; i < N; ++i) {
                const int ii = ky + i * incy;
                const T temp1 = cmul(alpha, x[kx + i * incx]);
                T temp2 = zero_cdd;
                const T *ai = &A_(0, i);
                const T aii_re{ ai[i].re, rzero };
                y[ii] = cadd(y[ii], cmul(temp1, aii_re));
                int ix = kx + (i + 1) * incx, iy = ky + (i + 1) * incy;
                for (int k = i + 1; k < N; ++k) {
                    const T aik = ai[k];
                    y[iy] = cadd(y[iy], cmul(temp1, aik));
                    temp2 = cadd(temp2, cmul(cconj(aik), x[ix]));
                    ix += incx; iy += incy;
                }
                y[ii] = cadd(y[ii], cmul(alpha, temp2));
            }
        } else {
            for (int i = 0; i < N; ++i) {
                const T temp1 = cmul(alpha, x[kx + i * incx]);
                T temp2 = zero_cdd;
                const T *ai = &A_(0, i);
                int ix = kx, iy = ky;
                for (int k = 0; k < i; ++k) {
                    const T aik = ai[k];
                    y[iy] = cadd(y[iy], cmul(temp1, aik));
                    temp2 = cadd(temp2, cmul(cconj(aik), x[ix]));
                    ix += incx; iy += incy;
                }
                const T aii_re{ ai[i].re, rzero };
                const int ii = ky + i * incy;
                y[ii] = cadd(y[ii], cadd(cmul(temp1, aii_re), cmul(alpha, temp2)));
            }
        }
    }
}

#undef A_
