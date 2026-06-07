/*
 * wgemv — multifloats complex DD general matrix-vector multiply.
 *
 * Both N and T/C paths use AVX2 SoA SIMD when MBLAS_SIMD_DD is on:
 * 4 SoA scratch buffers per vector (re_hi, re_lo, im_hi, im_lo);
 * inline 4-way 4×4 transpose to deinterleave A columns; SIMD cdd_mul
 * + cdd_add primitives from mgemm_simd_kernel.h.
 */

#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <multifloats.h>
#ifdef MBLAS_SIMD_DD
#include "mgemm_simd_kernel.h"
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {

#define WGEMV_OMP_MIN 64

inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}

const T zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };
const T one_cdd { R{1.0, 0.0}, R{0.0, 0.0} };

inline bool cdd_iszero(const T &x) {
    return x.re.limbs[0] == 0.0 && x.re.limbs[1] == 0.0
        && x.im.limbs[0] == 0.0 && x.im.limbs[1] == 0.0;
}
inline bool cdd_isone(const T &x) {
    return x.re.limbs[0] == 1.0 && x.re.limbs[1] == 0.0
        && x.im.limbs[0] == 0.0 && x.im.limbs[1] == 0.0;
}

inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }

#ifdef MBLAS_SIMD_DD
/* Deinterleave 4 consecutive AoS complex-DD elements (16 doubles)
 * into 4 SoA vectors via 4×4 transpose. */
static inline __attribute__((always_inline)) void
soa_load4_cdd(const double *p,
              __m256d &rh, __m256d &rl, __m256d &ih, __m256d &il)
{
    __m256d v0 = _mm256_loadu_pd(p +  0);   /* [r0h, r0l, i0h, i0l] */
    __m256d v1 = _mm256_loadu_pd(p +  4);   /* [r1h, r1l, i1h, i1l] */
    __m256d v2 = _mm256_loadu_pd(p +  8);   /* [r2h, r2l, i2h, i2l] */
    __m256d v3 = _mm256_loadu_pd(p + 12);   /* [r3h, r3l, i3h, i3l] */
    /* Per-lane unpack. */
    __m256d t0 = _mm256_unpacklo_pd(v0, v1); /* [r0h, r1h, i0h, i1h] */
    __m256d t1 = _mm256_unpackhi_pd(v0, v1); /* [r0l, r1l, i0l, i1l] */
    __m256d t2 = _mm256_unpacklo_pd(v2, v3); /* [r2h, r3h, i2h, i3h] */
    __m256d t3 = _mm256_unpackhi_pd(v2, v3); /* [r2l, r3l, i2l, i3l] */
    rh = _mm256_permute2f128_pd(t0, t2, 0x20); /* [r0h, r1h, r2h, r3h] */
    ih = _mm256_permute2f128_pd(t0, t2, 0x31); /* [i0h, i1h, i2h, i3h] */
    rl = _mm256_permute2f128_pd(t1, t3, 0x20); /* [r0l, r1l, r2l, r3l] */
    il = _mm256_permute2f128_pd(t1, t3, 0x31); /* [i0l, i1l, i2l, i3l] */
}

/* Inverse of soa_load4_cdd. */
static inline __attribute__((always_inline)) void
soa_store4_cdd(double *p,
               __m256d rh, __m256d rl, __m256d ih, __m256d il)
{
    __m256d t0 = _mm256_permute2f128_pd(rh, ih, 0x20); /* [r0h,r1h,i0h,i1h] */
    __m256d t2 = _mm256_permute2f128_pd(rh, ih, 0x31); /* [r2h,r3h,i2h,i3h] */
    __m256d t1 = _mm256_permute2f128_pd(rl, il, 0x20);
    __m256d t3 = _mm256_permute2f128_pd(rl, il, 0x31);
    __m256d v0 = _mm256_unpacklo_pd(t0, t1); /* [r0h,r0l,i0h,i0l] */
    __m256d v1 = _mm256_unpackhi_pd(t0, t1); /* [r1h,r1l,i1h,i1l] */
    __m256d v2 = _mm256_unpacklo_pd(t2, t3); /* [r2h,r2l,i2h,i2l] */
    __m256d v3 = _mm256_unpackhi_pd(t2, t3); /* [r3h,r3l,i3h,i3l] */
    _mm256_storeu_pd(p +  0, v0);
    _mm256_storeu_pd(p +  4, v1);
    _mm256_storeu_pd(p +  8, v2);
    _mm256_storeu_pd(p + 12, v3);
}
#endif

} /* namespace */

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

#ifdef _OPENMP
#define WGEMV_MAX_CPUS 256
/* Threaded strided NoTrans complex gemv (contiguous is SIMD-serial; strided has no
 * SIMD lever). Each thread owns a disjoint band of output rows [lo,hi) and runs the
 * serial column-outer scatter (j outer, i inner) restricted to its band: the matrix
 * read A_(i,j) stays contiguous in i (cache-friendly, unlike a per-row stride-lda
 * gather), and each y[i] accumulates in ascending-j order -> bit-identical to serial.
 * ax[j]=alpha*x[j] precomputed; x,y distinct (gemv forbids aliasing); NoTrans never
 * conjugates. */
static bool wgemv_n_omp(int M, int N, T alpha, const T *a, int lda,
                        const T *x, int incx, T *y, int incy)
{
    if (M < WGEMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > WGEMV_MAX_CPUS) nthreads = WGEMV_MAX_CPUS;

    T *ax = static_cast<T *>(std::malloc(static_cast<std::size_t>(N) * sizeof(T)));
    if (!ax) return false;
    const int ix0 = (incx < 0) ? -(N - 1) * incx : 0;
    for (int j = 0; j < N; ++j) ax[j] = cmul(alpha, x[ix0 + j * incx]);
    const std::ptrdiff_t iy0 = (incy < 0) ? -static_cast<std::ptrdiff_t>(M - 1) * incy : 0;

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        std::ptrdiff_t lo = (static_cast<std::ptrdiff_t>(M) * tid) / nthreads;
        std::ptrdiff_t hi = (static_cast<std::ptrdiff_t>(M) * (tid + 1)) / nthreads;
        for (int j = 0; j < N; ++j) {
            const T t = ax[j];
            if (cdd_iszero(t)) continue;
            const T *aj = &A_(0, j);
            for (std::ptrdiff_t i = lo; i < hi; ++i) {
                T *yi = &y[iy0 + i * incy];
                *yi = cadd(*yi, cmul(t, aj[i]));
            }
        }
    }
    std::free(ax);
    return true;
}

/* Threaded strided Trans/ConjTrans complex gemv. Output columns partition; strided x
 * gathered to contiguous; conj_a selects C (conjugate A) vs T. Bit-identical to the
 * serial strided gather (ascending-i). */
static bool wgemv_t_omp(int M, int N, T alpha, const T *a, int lda,
                        const T *x, int incx, T *y, int incy, int conj_a)
{
    if (N < WGEMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > WGEMV_MAX_CPUS) nthreads = WGEMV_MAX_CPUS;

    if (incx < 0) x -= static_cast<std::ptrdiff_t>(M - 1) * incx;
    if (incy < 0) y -= static_cast<std::ptrdiff_t>(N - 1) * incy;

    const T *xptr = x;
    T *xbuf = nullptr;
    if (incx != 1) {
        xbuf = static_cast<T *>(std::malloc(static_cast<std::size_t>(M) * sizeof(T)));
        if (!xbuf) return false;
        for (int i = 0; i < M; ++i) xbuf[i] = x[static_cast<std::ptrdiff_t>(i) * incx];
        xptr = xbuf;
    }

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        std::ptrdiff_t lo = (static_cast<std::ptrdiff_t>(N) * tid) / nthreads;
        std::ptrdiff_t hi = (static_cast<std::ptrdiff_t>(N) * (tid + 1)) / nthreads;
        for (std::ptrdiff_t j = lo; j < hi; ++j) {
            const T *col = &A_(0, j);
            T s = zero_cdd;
            if (conj_a) for (int i = 0; i < M; ++i) s = cadd(s, cmul(cconj(col[i]), xptr[i]));
            else        for (int i = 0; i < M; ++i) s = cadd(s, cmul(col[i], xptr[i]));
            y[j * incy] = cadd(y[j * incy], cmul(alpha, s));
        }
    }
    std::free(xbuf);
    return true;
}
#endif /* _OPENMP */

extern "C" void wgemv_(
    const char *trans,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *x, const int *incx_,
    const T *beta_,
    T *y, const int *incy_,
    std::size_t trans_len)
{
    (void)trans_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const char TR = up(trans);

    if (M == 0 || N == 0) return;

    const int leny = (TR == 'N') ? M : N;

    if (!cdd_isone(beta)) {
        int iy = (incy < 0) ? -(leny - 1) * incy : 0;
        for (int i = 0; i < leny; ++i) {
            if (cdd_iszero(beta)) y[iy] = zero_cdd;
            else                  y[iy] = cmul(y[iy], beta);
            iy += incy;
        }
    }
    if (cdd_iszero(alpha)) return;

    if (TR == 'N' && incx == 1 && incy == 1) {
#ifdef MBLAS_SIMD_DD
        /* SIMD N-path. Pack y to SoA (4 buffers: re_hi, re_lo, im_hi, im_lo). */
        const std::size_t M_pad = (static_cast<std::size_t>(M) + 3) & ~static_cast<std::size_t>(3);
        double *y_rh = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *y_rl = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *y_ih = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *y_il = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        for (int i = 0; i < M; ++i) {
            y_rh[i] = y[i].re.limbs[0];  y_rl[i] = y[i].re.limbs[1];
            y_ih[i] = y[i].im.limbs[0];  y_il[i] = y[i].im.limbs[1];
        }
        for (std::size_t i = static_cast<std::size_t>(M); i < M_pad; ++i) {
            y_rh[i] = 0.0; y_rl[i] = 0.0; y_ih[i] = 0.0; y_il[i] = 0.0;
        }
        for (int j = 0; j < N; ++j) {
            const T xj = x[j];
            if (cdd_iszero(xj)) continue;
            const T t = cmul(alpha, xj);
            const __m256d trh = _mm256_set1_pd(t.re.limbs[0]);
            const __m256d trl = _mm256_set1_pd(t.re.limbs[1]);
            const __m256d tih = _mm256_set1_pd(t.im.limbs[0]);
            const __m256d til = _mm256_set1_pd(t.im.limbs[1]);
            const double *aj = reinterpret_cast<const double *>(&A_(0, j));
            int i = 0;
            for (; i + 3 < M; i += 4) {
                __m256d a_rh, a_rl, a_ih, a_il;
                soa_load4_cdd(aj + 4 * i, a_rh, a_rl, a_ih, a_il);
                __m256d p_rh, p_rl, p_ih, p_il;
                simd_dd::cdd_mul(trh, trl, tih, til, a_rh, a_rl, a_ih, a_il,
                                 p_rh, p_rl, p_ih, p_il);
                __m256d yrh = _mm256_loadu_pd(y_rh + i);
                __m256d yrl = _mm256_loadu_pd(y_rl + i);
                __m256d yih = _mm256_loadu_pd(y_ih + i);
                __m256d yil = _mm256_loadu_pd(y_il + i);
                __m256d nrh, nrl, nih, nil;
                simd_dd::cdd_add(yrh, yrl, yih, yil, p_rh, p_rl, p_ih, p_il,
                                 nrh, nrl, nih, nil);
                _mm256_storeu_pd(y_rh + i, nrh);
                _mm256_storeu_pd(y_rl + i, nrl);
                _mm256_storeu_pd(y_ih + i, nih);
                _mm256_storeu_pd(y_il + i, nil);
            }
            const T *ajs = &A_(0, j);
            for (; i < M; ++i) {
                T yi = T{ R{y_rh[i], y_rl[i]}, R{y_ih[i], y_il[i]} };
                yi = cadd(yi, cmul(t, ajs[i]));
                y_rh[i] = yi.re.limbs[0]; y_rl[i] = yi.re.limbs[1];
                y_ih[i] = yi.im.limbs[0]; y_il[i] = yi.im.limbs[1];
            }
        }
        for (int i = 0; i < M; ++i) {
            y[i].re.limbs[0] = y_rh[i]; y[i].re.limbs[1] = y_rl[i];
            y[i].im.limbs[0] = y_ih[i]; y[i].im.limbs[1] = y_il[i];
        }
        std::free(y_rh); std::free(y_rl); std::free(y_ih); std::free(y_il);
#else
#ifdef _OPENMP
        const int use_omp = (M >= WGEMV_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel if(use_omp)
        {
            int tid = 0, nt = 1;
            if (use_omp) { tid = omp_get_thread_num(); nt = omp_get_num_threads(); }
            const int i_lo = (static_cast<long long>(M) * tid) / nt;
            const int i_hi = (static_cast<long long>(M) * (tid + 1)) / nt;
            for (int j = 0; j < N; ++j) {
                const T xj = x[j];
                if (!cdd_iszero(xj)) {
                    const T t = cmul(alpha, xj);
                    const T *aj = &A_(0, j);
                    for (int i = i_lo; i < i_hi; ++i) y[i] = cadd(y[i], cmul(t, aj[i]));
                }
            }
        }
#else
        for (int j = 0; j < N; ++j) {
            const T xj = x[j];
            if (!cdd_iszero(xj)) {
                const T t = cmul(alpha, xj);
                const T *aj = &A_(0, j);
                for (int i = 0; i < M; ++i) y[i] = cadd(y[i], cmul(t, aj[i]));
            }
        }
#endif
#endif
    } else if ((TR == 'T' || TR == 'C') && incx == 1 && incy == 1) {
        const int conj_a = (TR == 'C');
#ifdef MBLAS_SIMD_DD
        /* SIMD T/C-path: pre-pack x to SoA; 4-lane cdd_mul/cdd_add
         * accumulator over i for each j; horizontal-reduce. */
        const std::size_t M_pad = (static_cast<std::size_t>(M) + 3) & ~static_cast<std::size_t>(3);
        double *x_rh = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *x_rl = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *x_ih = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *x_il = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        for (int i = 0; i < M; ++i) {
            x_rh[i] = x[i].re.limbs[0]; x_rl[i] = x[i].re.limbs[1];
            x_ih[i] = x[i].im.limbs[0]; x_il[i] = x[i].im.limbs[1];
        }
        for (std::size_t i = static_cast<std::size_t>(M); i < M_pad; ++i) {
            x_rh[i] = 0.0; x_rl[i] = 0.0; x_ih[i] = 0.0; x_il[i] = 0.0;
        }
        const __m256d zerov = _mm256_setzero_pd();
        for (int j = 0; j < N; ++j) {
            const double *aj = reinterpret_cast<const double *>(&A_(0, j));
            __m256d s_rh = zerov, s_rl = zerov, s_ih = zerov, s_il = zerov;
            int i = 0;
            for (; i + 3 < M; i += 4) {
                __m256d a_rh, a_rl, a_ih, a_il;
                soa_load4_cdd(aj + 4 * i, a_rh, a_rl, a_ih, a_il);
                if (conj_a) simd_dd::dd_neg(a_ih, a_il);
                __m256d xrh = _mm256_loadu_pd(x_rh + i);
                __m256d xrl = _mm256_loadu_pd(x_rl + i);
                __m256d xih = _mm256_loadu_pd(x_ih + i);
                __m256d xil = _mm256_loadu_pd(x_il + i);
                __m256d p_rh, p_rl, p_ih, p_il;
                simd_dd::cdd_mul(a_rh, a_rl, a_ih, a_il, xrh, xrl, xih, xil,
                                 p_rh, p_rl, p_ih, p_il);
                __m256d nrh, nrl, nih, nil;
                simd_dd::cdd_add(s_rh, s_rl, s_ih, s_il, p_rh, p_rl, p_ih, p_il,
                                 nrh, nrl, nih, nil);
                s_rh = nrh; s_rl = nrl; s_ih = nih; s_il = nil;
            }
            /* Horizontal reduce 4-lane complex DD to scalar.
             * Stage 1: swap 128-bit halves and cdd_add. */
            __m256d srh_sw = _mm256_permute2f128_pd(s_rh, s_rh, 0x01);
            __m256d srl_sw = _mm256_permute2f128_pd(s_rl, s_rl, 0x01);
            __m256d sih_sw = _mm256_permute2f128_pd(s_ih, s_ih, 0x01);
            __m256d sil_sw = _mm256_permute2f128_pd(s_il, s_il, 0x01);
            __m256d p_rh, p_rl, p_ih, p_il;
            simd_dd::cdd_add(s_rh, s_rl, s_ih, s_il, srh_sw, srl_sw, sih_sw, sil_sw,
                             p_rh, p_rl, p_ih, p_il);
            /* Stage 2: shuffle within 128-bit lanes. */
            __m256d prh_sw = _mm256_shuffle_pd(p_rh, p_rh, 0x5);
            __m256d prl_sw = _mm256_shuffle_pd(p_rl, p_rl, 0x5);
            __m256d pih_sw = _mm256_shuffle_pd(p_ih, p_ih, 0x5);
            __m256d pil_sw = _mm256_shuffle_pd(p_il, p_il, 0x5);
            __m256d r_rh, r_rl, r_ih, r_il;
            simd_dd::cdd_add(p_rh, p_rl, p_ih, p_il, prh_sw, prl_sw, pih_sw, pil_sw,
                             r_rh, r_rl, r_ih, r_il);
            double red_rh[4], red_rl[4], red_ih[4], red_il[4];
            _mm256_storeu_pd(red_rh, r_rh); _mm256_storeu_pd(red_rl, r_rl);
            _mm256_storeu_pd(red_ih, r_ih); _mm256_storeu_pd(red_il, r_il);
            T s{ R{red_rh[0], red_rl[0]}, R{red_ih[0], red_il[0]} };
            const T *ajs = &A_(0, j);
            for (; i < M; ++i) {
                const T aij = conj_a ? cconj(ajs[i]) : ajs[i];
                s = cadd(s, cmul(aij, x[i]));
            }
            y[j] = cadd(y[j], cmul(alpha, s));
        }
        std::free(x_rh); std::free(x_rl); std::free(x_ih); std::free(x_il);
#else
#ifdef _OPENMP
        const int use_omp = (N >= WGEMV_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const T *aj = &A_(0, j);
            T s = zero_cdd;
            if (conj_a) {
                for (int i = 0; i < M; ++i) s = cadd(s, cmul(cconj(aj[i]), x[i]));
            } else {
                for (int i = 0; i < M; ++i) s = cadd(s, cmul(aj[i], x[i]));
            }
            y[j] = cadd(y[j], cmul(alpha, s));
        }
#endif
    } else {
        if (TR == 'N') {
#ifdef _OPENMP
            if (M >= WGEMV_OMP_MIN && blas_omp_max_threads() > 1
                && wgemv_n_omp(M, N, alpha, a, lda, x, incx, y, incy))
                return;
#endif
            int jx = (incx < 0) ? -(N - 1) * incx : 0;
            for (int j = 0; j < N; ++j) {
                const T xj = x[jx];
                if (!cdd_iszero(xj)) {
                    const T t = cmul(alpha, xj);
                    int iy = (incy < 0) ? -(M - 1) * incy : 0;
                    for (int i = 0; i < M; ++i) {
                        y[iy] = cadd(y[iy], cmul(t, A_(i, j)));
                        iy += incy;
                    }
                }
                jx += incx;
            }
        } else {
            const int conj_a = (TR == 'C');
#ifdef _OPENMP
            if (N >= WGEMV_OMP_MIN && blas_omp_max_threads() > 1
                && wgemv_t_omp(M, N, alpha, a, lda, x, incx, y, incy, conj_a))
                return;
#endif
            int jy = (incy < 0) ? -(N - 1) * incy : 0;
            for (int j = 0; j < N; ++j) {
                T s = zero_cdd;
                int ix = (incx < 0) ? -(M - 1) * incx : 0;
                for (int i = 0; i < M; ++i) {
                    const T aij = conj_a ? cconj(A_(i, j)) : A_(i, j);
                    s = cadd(s, cmul(aij, x[ix]));
                    ix += incx;
                }
                y[jy] = cadd(y[jy], cmul(alpha, s));
                jy += incy;
            }
        }
    }
}

#undef A_
