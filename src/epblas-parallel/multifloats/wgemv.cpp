/*
 * wgemv — multifloats complex DD general matrix-vector multiply.
 *
 * Both N and T/C contiguous paths use AVX2 SoA SIMD when MBLAS_SIMD_DD is on:
 * 4 SoA scratch buffers per vector (re_hi, re_lo, im_hi, im_lo);
 * inline 4-way 4×4 transpose to deinterleave A columns; SIMD cmul
 * + cadd primitives from mf_simd_fast.h. Strided callers gather x/y to
 * contiguous scratch, run the SIMD core, scatter back (O(N) gather vs the old
 * O(N·M) scalar strided loop).
 */

#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <vector>
#include <multifloats.h>
#include "mf_kernels.h"
#include "mf_util.h"
#include "mf_pred.h"
#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;
using mf_pred::ceq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {

#define WGEMV_OMP_MIN 64


const T zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };


using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::cconj;

#ifdef MBLAS_SIMD_DD
using simd_exact::cload4;
#endif

} /* namespace */

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

/* Contiguous (unit-stride) NoTrans core: y += alpha * A * x, with x length N and
 * y length M (beta already applied by the caller). SIMD-serial when MBLAS_SIMD_DD
 * (already beats ob's threaded path); scalar fallback threads over output rows. */
#ifdef MBLAS_SIMD_DD
/* NoTrans SIMD core over the output-row slice [i_lo, i_hi): accumulates
 * alpha*A*x into the SoA y buffers for rows in the slice only. i_lo (and i_hi
 * for interior threads) are multiples of 4 so the 4-wide vector blocks never
 * straddle a slice boundary; only the final slice (i_hi == M) runs a tail. */
static inline void wgemv_n_simd_rows(std::ptrdiff_t i_lo, std::ptrdiff_t i_hi, std::ptrdiff_t N, T alpha,
                                     const T *a, std::size_t lda, const T *x,
                                     double *y_rh, double *y_rl,
                                     double *y_ih, double *y_il)
{
    for (std::ptrdiff_t j = 0; j < N; ++j) {
        const T xj = x[j];
        if (ceq0(xj)) continue;
        const T t = cmul(alpha, xj);
        const __m256d trh = _mm256_set1_pd(t.re.limbs[0]);
        const __m256d trl = _mm256_set1_pd(t.re.limbs[1]);
        const __m256d tih = _mm256_set1_pd(t.im.limbs[0]);
        const __m256d til = _mm256_set1_pd(t.im.limbs[1]);
        const double *aj = reinterpret_cast<const double *>(&A_(0, j));
        std::ptrdiff_t i = i_lo;
        for (; i + 3 < i_hi; i += 4) {
            __m256d a_rh, a_rl, a_ih, a_il;
            cload4(aj + 4 * i, a_rh, a_rl, a_ih, a_il);
            __m256d p_rh, p_rl, p_ih, p_il;
            simd_fast::cmul(trh, trl, tih, til, a_rh, a_rl, a_ih, a_il,
                             p_rh, p_rl, p_ih, p_il);
            __m256d yrh = _mm256_loadu_pd(y_rh + i);
            __m256d yrl = _mm256_loadu_pd(y_rl + i);
            __m256d yih = _mm256_loadu_pd(y_ih + i);
            __m256d yil = _mm256_loadu_pd(y_il + i);
            __m256d nrh, nrl, nih, nil;
            simd_fast::cadd(yrh, yrl, yih, yil, p_rh, p_rl, p_ih, p_il,
                             nrh, nrl, nih, nil);
            _mm256_storeu_pd(y_rh + i, nrh);
            _mm256_storeu_pd(y_rl + i, nrl);
            _mm256_storeu_pd(y_ih + i, nih);
            _mm256_storeu_pd(y_il + i, nil);
        }
        const T *ajs = &A_(0, j);
        for (; i < i_hi; ++i) {
            T yi = T{ R{y_rh[i], y_rl[i]}, R{y_ih[i], y_il[i]} };
            yi = cadd(yi, cmul(t, ajs[i]));
            y_rh[i] = yi.re.limbs[0]; y_rl[i] = yi.re.limbs[1];
            y_ih[i] = yi.im.limbs[0]; y_il[i] = yi.im.limbs[1];
        }
    }
}
#endif

static void wgemv_n_contig(std::ptrdiff_t M, std::ptrdiff_t N, T alpha, const T *a, std::size_t lda,
                           const T *x, T *y)
{
#ifdef MBLAS_SIMD_DD
    /* Pack y to SoA (4 buffers: re_hi, re_lo, im_hi, im_lo). */
    const std::size_t M_pad = (static_cast<std::size_t>(M) + 3) & ~static_cast<std::size_t>(3);
    double *y_rh = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
    double *y_rl = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
    double *y_ih = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
    double *y_il = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
    for (std::ptrdiff_t i = 0; i < M; ++i) {
        y_rh[i] = y[i].re.limbs[0];  y_rl[i] = y[i].re.limbs[1];
        y_ih[i] = y[i].im.limbs[0];  y_il[i] = y[i].im.limbs[1];
    }
    for (std::size_t i = static_cast<std::size_t>(M); i < M_pad; ++i) {
        y_rh[i] = 0.0; y_rl[i] = 0.0; y_ih[i] = 0.0; y_il[i] = 0.0;
    }
#ifdef _OPENMP
    const std::ptrdiff_t use_omp = (M >= WGEMV_OMP_MIN && blas_omp_available());
    #pragma omp parallel if(use_omp)
    {
        std::ptrdiff_t tid = 0, nt = 1;
        if (use_omp) { tid = omp_get_thread_num(); nt = omp_get_num_threads(); }
        /* Disjoint row slices, boundaries floored to a multiple of 4 so the
         * vector blocks stay within one thread; the last thread owns the tail. */
        const std::ptrdiff_t i_lo = blas_part_bound(M, tid, nt) & ~static_cast<std::ptrdiff_t>(3);
        const std::ptrdiff_t i_hi = (tid == nt - 1)
            ? M
            : (blas_part_bound(M, tid + 1, nt) & ~static_cast<std::ptrdiff_t>(3));
        wgemv_n_simd_rows(i_lo, i_hi, N, alpha, a, lda, x,
                          y_rh, y_rl, y_ih, y_il);
    }
#else
    wgemv_n_simd_rows(0, M, N, alpha, a, lda, x, y_rh, y_rl, y_ih, y_il);
#endif
    for (std::ptrdiff_t i = 0; i < M; ++i) {
        y[i].re.limbs[0] = y_rh[i]; y[i].re.limbs[1] = y_rl[i];
        y[i].im.limbs[0] = y_ih[i]; y[i].im.limbs[1] = y_il[i];
    }
    std::free(y_rh); std::free(y_rl); std::free(y_ih); std::free(y_il);
#else
#ifdef _OPENMP
    const std::ptrdiff_t use_omp = (M >= WGEMV_OMP_MIN && blas_omp_available());
    #pragma omp parallel if(use_omp)
    {
        std::ptrdiff_t tid = 0, nt = 1;
        if (use_omp) { tid = omp_get_thread_num(); nt = omp_get_num_threads(); }
        const std::ptrdiff_t i_lo = blas_part_bound(M, tid, nt);
        const std::ptrdiff_t i_hi = blas_part_bound(M, tid + 1, nt);
        for (std::ptrdiff_t j = 0; j < N; ++j) {
            const T xj = x[j];
            if (!ceq0(xj)) {
                const T t = cmul(alpha, xj);
                const T *aj = &A_(0, j);
                for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) y[i] = cadd(y[i], cmul(t, aj[i]));
            }
        }
    }
#else
    for (std::ptrdiff_t j = 0; j < N; ++j) {
        const T xj = x[j];
        if (!ceq0(xj)) {
            const T t = cmul(alpha, xj);
            const T *aj = &A_(0, j);
            for (std::ptrdiff_t i = 0; i < M; ++i) y[i] = cadd(y[i], cmul(t, aj[i]));
        }
    }
#endif
#endif
}

/* Contiguous (unit-stride) Trans/ConjTrans core: y += alpha * op(A) * x, with x
 * length M and y length N (beta already applied). conj_a selects C (conjugate A)
 * vs T. SIMD-serial when MBLAS_SIMD_DD; scalar fallback threads over output cols. */
static void wgemv_t_contig(std::ptrdiff_t M, std::ptrdiff_t N, T alpha, const T *a, std::size_t lda,
                           const T *x, T *y, std::ptrdiff_t conj_a)
{
#ifdef MBLAS_SIMD_DD
    /* Pre-pack x to SoA; 4-lane cmul/cadd accumulator over i for each j;
     * horizontal-reduce. */
    const std::size_t M_pad = (static_cast<std::size_t>(M) + 3) & ~static_cast<std::size_t>(3);
    double *x_rh = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
    double *x_rl = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
    double *x_ih = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
    double *x_il = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
    for (std::ptrdiff_t i = 0; i < M; ++i) {
        x_rh[i] = x[i].re.limbs[0]; x_rl[i] = x[i].re.limbs[1];
        x_ih[i] = x[i].im.limbs[0]; x_il[i] = x[i].im.limbs[1];
    }
    for (std::size_t i = static_cast<std::size_t>(M); i < M_pad; ++i) {
        x_rh[i] = 0.0; x_rl[i] = 0.0; x_ih[i] = 0.0; x_il[i] = 0.0;
    }
    const __m256d zerov = _mm256_setzero_pd();
    /* Each output column j is independent (reads shared x SoA, writes y[j]). */
#ifdef _OPENMP
    const std::ptrdiff_t use_omp = (N >= WGEMV_OMP_MIN && blas_omp_available());
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (std::ptrdiff_t j = 0; j < N; ++j) {
        const double *aj = reinterpret_cast<const double *>(&A_(0, j));
        __m256d s_rh = zerov, s_rl = zerov, s_ih = zerov, s_il = zerov;
        std::ptrdiff_t i = 0;
        for (; i + 3 < M; i += 4) {
            __m256d a_rh, a_rl, a_ih, a_il;
            cload4(aj + 4 * i, a_rh, a_rl, a_ih, a_il);
            if (conj_a) simd_fast::neg(a_ih, a_il);
            __m256d xrh = _mm256_loadu_pd(x_rh + i);
            __m256d xrl = _mm256_loadu_pd(x_rl + i);
            __m256d xih = _mm256_loadu_pd(x_ih + i);
            __m256d xil = _mm256_loadu_pd(x_il + i);
            __m256d p_rh, p_rl, p_ih, p_il;
            simd_fast::cmul(a_rh, a_rl, a_ih, a_il, xrh, xrl, xih, xil,
                             p_rh, p_rl, p_ih, p_il);
            __m256d nrh, nrl, nih, nil;
            simd_fast::cadd(s_rh, s_rl, s_ih, s_il, p_rh, p_rl, p_ih, p_il,
                             nrh, nrl, nih, nil);
            s_rh = nrh; s_rl = nrl; s_ih = nih; s_il = nil;
        }
        /* Horizontal reduce 4-lane complex DD to scalar.
         * Stage 1: swap 128-bit halves and cadd. */
        __m256d srh_sw = _mm256_permute2f128_pd(s_rh, s_rh, 0x01);
        __m256d srl_sw = _mm256_permute2f128_pd(s_rl, s_rl, 0x01);
        __m256d sih_sw = _mm256_permute2f128_pd(s_ih, s_ih, 0x01);
        __m256d sil_sw = _mm256_permute2f128_pd(s_il, s_il, 0x01);
        __m256d p_rh, p_rl, p_ih, p_il;
        simd_fast::cadd(s_rh, s_rl, s_ih, s_il, srh_sw, srl_sw, sih_sw, sil_sw,
                         p_rh, p_rl, p_ih, p_il);
        /* Stage 2: shuffle within 128-bit lanes. */
        __m256d prh_sw = _mm256_shuffle_pd(p_rh, p_rh, 0x5);
        __m256d prl_sw = _mm256_shuffle_pd(p_rl, p_rl, 0x5);
        __m256d pih_sw = _mm256_shuffle_pd(p_ih, p_ih, 0x5);
        __m256d pil_sw = _mm256_shuffle_pd(p_il, p_il, 0x5);
        __m256d r_rh, r_rl, r_ih, r_il;
        simd_fast::cadd(p_rh, p_rl, p_ih, p_il, prh_sw, prl_sw, pih_sw, pil_sw,
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
    const std::ptrdiff_t use_omp = (N >= WGEMV_OMP_MIN && blas_omp_available());
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (std::ptrdiff_t j = 0; j < N; ++j) {
        const T *aj = &A_(0, j);
        T s = zero_cdd;
        if (conj_a) {
            for (std::ptrdiff_t i = 0; i < M; ++i) s = cadd(s, cmul(cconj(aj[i]), x[i]));
        } else {
            for (std::ptrdiff_t i = 0; i < M; ++i) s = cadd(s, cmul(aj[i], x[i]));
        }
        y[j] = cadd(y[j], cmul(alpha, s));
    }
#endif
}

static void wgemv_core(
    char trans,
    std::ptrdiff_t M, std::ptrdiff_t N,
    const T *alpha_,
    const T *a, std::ptrdiff_t lda,
    const T *x, std::ptrdiff_t incx,
    const T *beta_,
    T *y, std::ptrdiff_t incy)
{
    const T alpha = *alpha_, beta = *beta_;
    const char TR = up(&trans);

    if (M == 0 || N == 0) return;

    const std::ptrdiff_t leny = (TR == 'N') ? M : N;

    mf_kernels::cscale_y(leny, beta, y, incy);
    if (ceq0(alpha)) return;

    const std::ptrdiff_t conj_a = (TR == 'C');

    if (TR == 'N') {
        if (incx == 1 && incy == 1) {
            wgemv_n_contig(M, N, alpha, a, lda, x, y);
            return;
        }
        /* Strided: gather x (len N) to contiguous, run the SIMD core on a
         * contiguous y scratch (already beta-applied), scatter back. */
        std::vector<T> xs(static_cast<std::size_t>(N)), ys(static_cast<std::size_t>(M));
        mf_kernels::gather_strided(N, x, incx, xs.data());
        mf_kernels::gather_strided(M, y, incy, ys.data());
        wgemv_n_contig(M, N, alpha, a, lda, xs.data(), ys.data());
        mf_kernels::scatter_strided(M, y, incy, ys.data());
    } else {
        if (incx == 1 && incy == 1) {
            wgemv_t_contig(M, N, alpha, a, lda, x, y, conj_a);
            return;
        }
        /* Strided: gather x (len M), contiguous y scratch (len N), scatter back. */
        std::vector<T> xs(static_cast<std::size_t>(M)), ys(static_cast<std::size_t>(N));
        mf_kernels::gather_strided(M, x, incx, xs.data());
        mf_kernels::gather_strided(N, y, incy, ys.data());
        wgemv_t_contig(M, N, alpha, a, lda, xs.data(), ys.data(), conj_a);
        mf_kernels::scatter_strided(N, y, incy, ys.data());
    }
}

extern "C" {
EPBLAS_FACADE_GEMV(wgemv, T)
}

#undef A_
