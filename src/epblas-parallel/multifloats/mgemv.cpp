/*
 * mgemv — multifloats real DD general matrix-vector multiply.
 *
 * N-path uses AVX2 SoA SIMD when MBLAS_SIMD_DD is on:
 *   - Pre-pack y into SoA scratch (y_hi[], y_lo[])
 *   - For each column j: broadcast tj.hi, tj.lo, run inner i loop
 *     in chunks of 4 with simd_dd::dd_mul + dd_add into SoA y
 *   - Re-pack SoA y back to AoS at end
 * For T-path the dot-product reduction is left scalar (horizontal
 * SIMD reductions on DD don't pay off at typical M).
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
using T = mf::float64x2;

namespace {

#define MGEMV_OMP_MIN 64

inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}

const T zero_dd{0.0, 0.0};
inline bool dd_iszero(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_isone (T x) { return x.limbs[0] == 1.0 && x.limbs[1] == 0.0; }

#ifdef MBLAS_SIMD_DD
/* Deinterleave 4 contiguous AoS DD elements at *p (8 doubles) into
 * SoA (hi, lo) vectors. p is `T*` cast to double*. */
static inline __attribute__((always_inline)) void
soa_load4(const double *p, __m256d &hi, __m256d &lo)
{
    __m256d a01 = _mm256_loadu_pd(p);       /* [h0, l0, h1, l1] */
    __m256d a23 = _mm256_loadu_pd(p + 4);   /* [h2, l2, h3, l3] */
    __m256d t0 = _mm256_unpacklo_pd(a01, a23);   /* [h0, h2, h1, h3] */
    __m256d t1 = _mm256_unpackhi_pd(a01, a23);   /* [l0, l2, l1, l3] */
    hi = _mm256_permute4x64_pd(t0, 0xD8);    /* [h0, h1, h2, h3] */
    lo = _mm256_permute4x64_pd(t1, 0xD8);    /* [l0, l1, l2, l3] */
}

/* Inverse of soa_load4: interleave hi/lo and store 8 doubles at *p. */
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
#endif

} /* namespace */

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

#ifdef _OPENMP
#define MGEMV_MAX_CPUS 256
/* Threaded strided NoTrans gemv (the contiguous path is SIMD-serial; strided has no
 * SIMD lever, so threading is the only one). Each thread owns a disjoint band of
 * output rows [lo,hi) and runs the serial COLUMN scatter restricted to that band:
 * column-outer keeps the matrix read contiguous in i (cache-friendly, unlike a
 * row-outer gather striding by lda), and accumulating y[i] += alpha*x[j]*A(i,j) in
 * ascending j reproduces the serial scatter's exact association -> bit-identical.
 *
 * Mirrors ob gemv_n_strided exactly: t=alpha*x[jx] is computed inline inside the
 * thread (redundant across threads but fully overlapped), and the y index advances
 * by an incremental iy += incy. The earlier form precomputed ax[]=alpha*x via a
 * malloc + a SERIAL prologue ahead of the parallel region; that fixed serial cost
 * does not shrink with threads, so it capped strided scaling at ~2.8x (vs ob ~3.35x)
 * and left par4/ob4 ~1.05-1.15 at N<=512. x,y distinct so disjoint row bands never
 * race; skip on dd_iszero(x[jx]) matches the serial strided path. */
static bool mgemv_n_omp(int M, int N, T alpha, const T *a, int lda,
                        const T *x, int incx, T *y, int incy)
{
    if (M < MGEMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MGEMV_MAX_CPUS) nthreads = MGEMV_MAX_CPUS;

    const std::ptrdiff_t ix0 = (incx < 0) ? -static_cast<std::ptrdiff_t>(N - 1) * incx : 0;
    const std::ptrdiff_t iy0 = (incy < 0) ? -static_cast<std::ptrdiff_t>(M - 1) * incy : 0;

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        std::ptrdiff_t lo = (static_cast<std::ptrdiff_t>(M) * tid) / nthreads;
        std::ptrdiff_t hi = (static_cast<std::ptrdiff_t>(M) * (tid + 1)) / nthreads;
        std::ptrdiff_t jx = ix0;
        for (int j = 0; j < N; ++j) {
            const T xj = x[jx];
            if (!dd_iszero(xj)) {
                const T t = alpha * xj;
                const T *aj = &A_(0, j);
                std::ptrdiff_t iy = iy0 + lo * incy;
                for (std::ptrdiff_t i = lo; i < hi; ++i) {
                    y[iy] = y[iy] + t * aj[i];
                    iy += incy;
                }
            }
            jx += incx;
        }
    }
    return true;
}

/* Threaded strided Trans gemv. Output columns partition (each y[j]=alpha*Σ_i A(i,j)*x[i]
 * disjoint). Strided x gathered to contiguous so the inner dot reads the column (and x)
 * unit-stride. Bit-identical to the serial strided gather (ascending-i). */
static bool mgemv_t_omp(int M, int N, T alpha, const T *a, int lda,
                        const T *x, int incx, T *y, int incy)
{
    if (N < MGEMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MGEMV_MAX_CPUS) nthreads = MGEMV_MAX_CPUS;

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
            T s = zero_dd;
            for (int i = 0; i < M; ++i) s = s + col[i] * xptr[i];
            y[j * incy] = y[j * incy] + alpha * s;
        }
    }
    std::free(xbuf);
    return true;
}
#endif /* _OPENMP */

extern "C" void mgemv_(
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
    char TR = up(trans);
    if (TR == 'C') TR = 'T';

    if (M == 0 || N == 0) return;

    const int leny = (TR == 'N') ? M : N;

    if (!dd_isone(beta)) {
        int iy = (incy < 0) ? -(leny - 1) * incy : 0;
        for (int i = 0; i < leny; ++i) {
            if (dd_iszero(beta)) y[iy] = zero_dd;
            else                 y[iy] = y[iy] * beta;
            iy += incy;
        }
    }
    if (dd_iszero(alpha)) return;

    if (TR == 'N' && incx == 1 && incy == 1) {
#ifdef MBLAS_SIMD_DD
        /* SIMD N-path. Allocate aligned SoA scratch for y; pack once,
         * run the SIMD inner loop in chunks of 4 over i for each j,
         * unpack once. */
        const std::size_t M_pad = (static_cast<std::size_t>(M) + 3) & ~static_cast<std::size_t>(3);
        double *y_hi = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *y_lo = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        /* Pack y → SoA. */
        for (int i = 0; i < M; ++i) {
            y_hi[i] = y[i].limbs[0];
            y_lo[i] = y[i].limbs[1];
        }
        for (std::size_t i = static_cast<std::size_t>(M); i < M_pad; ++i) {
            y_hi[i] = 0.0;  y_lo[i] = 0.0;
        }
        for (int j = 0; j < N; ++j) {
            const T xj = x[j];
            if (dd_iszero(xj)) continue;
            const T t = alpha * xj;
            const __m256d thi = _mm256_set1_pd(t.limbs[0]);
            const __m256d tlo = _mm256_set1_pd(t.limbs[1]);
            const double *aj = reinterpret_cast<const double *>(&A_(0, j));
            int i = 0;
            for (; i + 3 < M; i += 4) {
                __m256d a_hi, a_lo;
                soa_load4(aj + 2 * i, a_hi, a_lo);
                __m256d p_hi, p_lo;
                simd_dd::dd_mul(thi, tlo, a_hi, a_lo, p_hi, p_lo);
                __m256d yh = _mm256_loadu_pd(y_hi + i);
                __m256d yl = _mm256_loadu_pd(y_lo + i);
                __m256d nyh, nyl;
                simd_dd::dd_add(yh, yl, p_hi, p_lo, nyh, nyl);
                _mm256_storeu_pd(y_hi + i, nyh);
                _mm256_storeu_pd(y_lo + i, nyl);
            }
            /* Scalar tail. */
            const T *ajs = &A_(0, j);
            for (; i < M; ++i) {
                T yi = T{y_hi[i], y_lo[i]} + t * ajs[i];
                y_hi[i] = yi.limbs[0];
                y_lo[i] = yi.limbs[1];
            }
        }
        /* Unpack SoA → y. */
        for (int i = 0; i < M; ++i) {
            y[i].limbs[0] = y_hi[i];
            y[i].limbs[1] = y_lo[i];
        }
        std::free(y_hi);  std::free(y_lo);
#else
#ifdef _OPENMP
        const int use_omp = (M >= MGEMV_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel if(use_omp)
        {
            int tid = 0, nt = 1;
            if (use_omp) { tid = omp_get_thread_num(); nt = omp_get_num_threads(); }
            const int i_lo = (static_cast<long long>(M) * tid) / nt;
            const int i_hi = (static_cast<long long>(M) * (tid + 1)) / nt;
            for (int j = 0; j < N; ++j) {
                const T xj = x[j];
                if (!dd_iszero(xj)) {
                    const T t = alpha * xj;
                    const T *aj = &A_(0, j);
                    for (int i = i_lo; i < i_hi; ++i) y[i] = y[i] + t * aj[i];
                }
            }
        }
#else
        for (int j = 0; j < N; ++j) {
            const T xj = x[j];
            if (!dd_iszero(xj)) {
                const T t = alpha * xj;
                const T *aj = &A_(0, j);
                for (int i = 0; i < M; ++i) y[i] = y[i] + t * aj[i];
            }
        }
#endif
#endif
    } else if (TR != 'N' && incx == 1 && incy == 1) {
#ifdef MBLAS_SIMD_DD
        /* SIMD T-path: per j, 4-lane SoA dot-product accumulator over
         * i, then horizontal-reduce + alpha*s into y[j].
         * Pre-pack x to SoA once. */
        const std::size_t M_pad = (static_cast<std::size_t>(M) + 3) & ~static_cast<std::size_t>(3);
        double *x_hi = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *x_lo = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        for (int i = 0; i < M; ++i) {
            x_hi[i] = x[i].limbs[0];
            x_lo[i] = x[i].limbs[1];
        }
        for (std::size_t i = static_cast<std::size_t>(M); i < M_pad; ++i) {
            x_hi[i] = 0.0;  x_lo[i] = 0.0;
        }
        const __m256d zerov = _mm256_setzero_pd();
        /* Columns are independent dots over the shared read-only x SoA scratch;
         * thread over j (each y[j] disjoint, per-j SIMD reduction order unchanged
         * -> consistency unaffected). The SIMD T-path was otherwise serial, so
         * ob's threaded Trans overtook it; this restores ~3.7x scaling. */
#ifdef _OPENMP
        const int use_omp_t = (N >= MGEMV_OMP_MIN && blas_omp_max_threads() > 1
                               && !omp_in_parallel());
        #pragma omp parallel for if(use_omp_t) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const double *aj = reinterpret_cast<const double *>(&A_(0, j));
            __m256d s_h = zerov, s_l = zerov;
            int i = 0;
            for (; i + 3 < M; i += 4) {
                __m256d a_h, a_l;
                soa_load4(aj + 2 * i, a_h, a_l);
                __m256d xh = _mm256_loadu_pd(x_hi + i);
                __m256d xl = _mm256_loadu_pd(x_lo + i);
                __m256d p_h, p_l;
                simd_dd::dd_mul(a_h, a_l, xh, xl, p_h, p_l);
                __m256d nh, nl;
                simd_dd::dd_add(s_h, s_l, p_h, p_l, nh, nl);
                s_h = nh;  s_l = nl;
            }
            /* Horizontal reduce 4-lane DD accumulator to scalar DD.
             * Stage 1: swap 128-bit halves and dd_add. */
            __m256d sh_sw = _mm256_permute2f128_pd(s_h, s_h, 0x01);
            __m256d sl_sw = _mm256_permute2f128_pd(s_l, s_l, 0x01);
            __m256d p_h, p_l;
            simd_dd::dd_add(s_h, s_l, sh_sw, sl_sw, p_h, p_l);
            /* Stage 2: shuffle within each 128-bit lane and dd_add. */
            __m256d ph_sw = _mm256_shuffle_pd(p_h, p_h, 0x5);
            __m256d pl_sw = _mm256_shuffle_pd(p_l, p_l, 0x5);
            __m256d r_h, r_l;
            simd_dd::dd_add(p_h, p_l, ph_sw, pl_sw, r_h, r_l);
            double red_h[4], red_l[4];
            _mm256_storeu_pd(red_h, r_h);
            _mm256_storeu_pd(red_l, r_l);
            T s{red_h[0], red_l[0]};
            for (; i < M; ++i) s = s + (&A_(0, j))[i] * x[i];
            y[j] = y[j] + alpha * s;
        }
        std::free(x_hi);  std::free(x_lo);
#else
#ifdef _OPENMP
        const int use_omp = (N >= MGEMV_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const T *aj = &A_(0, j);
            T s = zero_dd;
            for (int i = 0; i < M; ++i) s = s + aj[i] * x[i];
            y[j] = y[j] + alpha * s;
        }
#endif
    } else {
        if (TR == 'N') {
#ifdef _OPENMP
            if (M >= MGEMV_OMP_MIN && blas_omp_max_threads() > 1
                && mgemv_n_omp(M, N, alpha, a, lda, x, incx, y, incy))
                return;
#endif
            int jx = (incx < 0) ? -(N - 1) * incx : 0;
            for (int j = 0; j < N; ++j) {
                const T xj = x[jx];
                if (!dd_iszero(xj)) {
                    const T t = alpha * xj;
                    int iy = (incy < 0) ? -(M - 1) * incy : 0;
                    for (int i = 0; i < M; ++i) {
                        y[iy] = y[iy] + t * A_(i, j);
                        iy += incy;
                    }
                }
                jx += incx;
            }
        } else {
#ifdef _OPENMP
            if (N >= MGEMV_OMP_MIN && blas_omp_max_threads() > 1
                && mgemv_t_omp(M, N, alpha, a, lda, x, incx, y, incy))
                return;
#endif
            int jy = (incy < 0) ? -(N - 1) * incy : 0;
            for (int j = 0; j < N; ++j) {
                T s = zero_dd;
                int ix = (incx < 0) ? -(M - 1) * incx : 0;
                for (int i = 0; i < M; ++i) {
                    s = s + A_(i, j) * x[ix];
                    ix += incx;
                }
                y[jy] = y[jy] + alpha * s;
                jy += incy;
            }
        }
    }
}

#undef A_
