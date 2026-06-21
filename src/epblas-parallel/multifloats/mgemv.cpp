/*
 * mgemv — multifloats real DD general matrix-vector multiply.
 *
 * Both NoTrans and Trans run an AVX2 SoA-DD core on contiguous data:
 *   - NoTrans packs y to SoA scratch and threads over disjoint output-row
 *     bands, each band running the column scatter (SIMD mul + add).
 *   - Trans packs x to SoA scratch and threads over columns, each an
 *     independent SIMD dot reduced (hi/lo horizontal) into y[j].
 * Strided x/y are gathered to unit stride, run through the same cores, and
 * scattered back (O(M+N) gather vs O(M*N) work), so the SIMD+threaded path
 * serves every increment combination.
 */

#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <vector>
#include <multifloats.h>
#include "mf_util.h"
#include "mf_pred.h"
#include "mf_kernels.h"
#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using T = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
using mf_pred::eq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {

#define MGEMV_OMP_MIN 64
#ifdef _OPENMP
#define MGEMV_MAX_CPUS 256
#endif


const T zero_dd{0.0, 0.0};

#ifdef MBLAS_SIMD_DD
using simd_exact::load_dd4;
#endif

} /* namespace */

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

/* Contiguous NoTrans core: y += alpha*A*x, x len N, y len M (pre-beta'd).
 * Threaded over disjoint output-row bands [lo,hi) — each thread runs the column
 * scatter over only its rows (matrix read contiguous in i; ascending j ->
 * bit-exact vs serial). SIMD build packs y to SoA and uses the AVX2 DD kernel;
 * the scalar fallback runs the reference inner loop. */
static void mgemv_n_contig(int M, int N, T alpha, const T *a, std::size_t lda,
                           const T *x, T *y)
{
    int nt = 1;
#ifdef _OPENMP
    if (M >= MGEMV_OMP_MIN && blas_omp_available() && !omp_in_parallel()) {
        nt = blas_omp_max_threads();
        if (nt > MGEMV_MAX_CPUS) nt = MGEMV_MAX_CPUS;
    }
#endif
#ifdef MBLAS_SIMD_DD
    const std::size_t M_pad = (static_cast<std::size_t>(M) + 3) & ~static_cast<std::size_t>(3);
    double *y_hi = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
    double *y_lo = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
    for (int i = 0; i < M; ++i) { y_hi[i] = y[i].limbs[0]; y_lo[i] = y[i].limbs[1]; }
    for (std::size_t i = static_cast<std::size_t>(M); i < M_pad; ++i) { y_hi[i] = 0.0; y_lo[i] = 0.0; }
#ifdef _OPENMP
    #pragma omp parallel num_threads(nt)
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const int lo = (int)((std::ptrdiff_t)M * tid / nt);
        const int hi = (int)((std::ptrdiff_t)M * (tid + 1) / nt);
        for (int j = 0; j < N; ++j) {
            const T xj = x[j];
            if (eq0(xj)) continue;
            const T t = alpha * xj;
            const __m256d thi = _mm256_set1_pd(t.limbs[0]);
            const __m256d tlo = _mm256_set1_pd(t.limbs[1]);
            const double *aj = reinterpret_cast<const double *>(&A_(0, j));
            const T *ajs = &A_(0, j);
            int i = lo;
            for (; i + 3 < hi; i += 4) {
                __m256d a_hi, a_lo;
                load_dd4(aj + 2 * i, a_hi, a_lo);
                __m256d p_hi, p_lo;
                simd_fast::mul(thi, tlo, a_hi, a_lo, p_hi, p_lo);
                __m256d yh = _mm256_loadu_pd(y_hi + i);
                __m256d yl = _mm256_loadu_pd(y_lo + i);
                __m256d nyh, nyl;
                simd_fast::add(yh, yl, p_hi, p_lo, nyh, nyl);
                _mm256_storeu_pd(y_hi + i, nyh);
                _mm256_storeu_pd(y_lo + i, nyl);
            }
            for (; i < hi; ++i) {
                T yi = T{y_hi[i], y_lo[i]} + t * ajs[i];
                y_hi[i] = yi.limbs[0]; y_lo[i] = yi.limbs[1];
            }
        }
    }
    for (int i = 0; i < M; ++i) { y[i].limbs[0] = y_hi[i]; y[i].limbs[1] = y_lo[i]; }
    std::free(y_hi); std::free(y_lo);
#else
#ifdef _OPENMP
    #pragma omp parallel num_threads(nt)
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        const int lo = (int)((std::ptrdiff_t)M * tid / nt);
        const int hi = (int)((std::ptrdiff_t)M * (tid + 1) / nt);
        for (int j = 0; j < N; ++j) {
            const T xj = x[j];
            if (eq0(xj)) continue;
            const T t = alpha * xj;
            const T *aj = &A_(0, j);
            for (int i = lo; i < hi; ++i) y[i] = y[i] + t * aj[i];
        }
    }
#endif
}

/* Contiguous Trans core: y += alpha*A^T*x, x len M, y len N (pre-beta'd).
 * Columns are independent dots over the shared read-only x; thread over j
 * (disjoint y[j], per-j reduction order fixed). SIMD build runs a 4-lane SoA
 * DD accumulator + hi/lo horizontal reduce; scalar fallback a plain dot. */
static void mgemv_t_contig(int M, int N, T alpha, const T *a, std::size_t lda,
                           const T *x, T *y)
{
#ifdef _OPENMP
    const int use_omp = (N >= MGEMV_OMP_MIN && blas_omp_available()
                         && !omp_in_parallel());
#endif
#ifdef MBLAS_SIMD_DD
    const std::size_t M_pad = (static_cast<std::size_t>(M) + 3) & ~static_cast<std::size_t>(3);
    double *x_hi = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
    double *x_lo = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
    for (int i = 0; i < M; ++i) { x_hi[i] = x[i].limbs[0]; x_lo[i] = x[i].limbs[1]; }
    for (std::size_t i = static_cast<std::size_t>(M); i < M_pad; ++i) { x_hi[i] = 0.0; x_lo[i] = 0.0; }
    const __m256d zerov = _mm256_setzero_pd();
#ifdef _OPENMP
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (int j = 0; j < N; ++j) {
        const double *aj = reinterpret_cast<const double *>(&A_(0, j));
        __m256d s_h = zerov, s_l = zerov;
        int i = 0;
        for (; i + 3 < M; i += 4) {
            __m256d a_h, a_l;
            load_dd4(aj + 2 * i, a_h, a_l);
            __m256d xh = _mm256_loadu_pd(x_hi + i);
            __m256d xl = _mm256_loadu_pd(x_lo + i);
            __m256d p_h, p_l;
            simd_fast::mul(a_h, a_l, xh, xl, p_h, p_l);
            __m256d nh, nl;
            simd_fast::add(s_h, s_l, p_h, p_l, nh, nl);
            s_h = nh; s_l = nl;
        }
        /* Horizontal-reduce the 4-lane DD accumulator to scalar DD. */
        __m256d sh_sw = _mm256_permute2f128_pd(s_h, s_h, 0x01);
        __m256d sl_sw = _mm256_permute2f128_pd(s_l, s_l, 0x01);
        __m256d p_h, p_l;
        simd_fast::add(s_h, s_l, sh_sw, sl_sw, p_h, p_l);
        __m256d ph_sw = _mm256_shuffle_pd(p_h, p_h, 0x5);
        __m256d pl_sw = _mm256_shuffle_pd(p_l, p_l, 0x5);
        __m256d r_h, r_l;
        simd_fast::add(p_h, p_l, ph_sw, pl_sw, r_h, r_l);
        double red_h[4], red_l[4];
        _mm256_storeu_pd(red_h, r_h);
        _mm256_storeu_pd(red_l, r_l);
        T s{red_h[0], red_l[0]};
        const T *ajs = &A_(0, j);
        for (; i < M; ++i) s = s + ajs[i] * T{x_hi[i], x_lo[i]};
        y[j] = y[j] + alpha * s;
    }
    std::free(x_hi); std::free(x_lo);
#else
#ifdef _OPENMP
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (int j = 0; j < N; ++j) {
        const T *aj = &A_(0, j);
        T s = zero_dd;
        for (int i = 0; i < M; ++i) s = s + aj[i] * x[i];
        y[j] = y[j] + alpha * s;
    }
#endif
}

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
    const std::size_t lda = static_cast<std::size_t>(*lda_);
    const int incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
    const bool notrans = (TR == 'N');

    if (M == 0 || N == 0) return;

    const int leny = notrans ? M : N;
    const int lenx = notrans ? N : M;

    mf_kernels::scale_y(leny, beta, y, incy);
    if (eq0(alpha)) return;

    if (incx == 1 && incy == 1) {
        if (notrans) mgemv_n_contig(M, N, alpha, a, lda, x, y);
        else         mgemv_t_contig(M, N, alpha, a, lda, x, y);
        return;
    }

    /* Strided x,y: gather to unit stride (y already beta-applied), run the SIMD
     * core, scatter y back. Handles negative increments; O(M+N) gather. */
    std::vector<T> xs(static_cast<std::size_t>(lenx)), ys(static_cast<std::size_t>(leny));
    mf_kernels::gather_strided(lenx, x, incx, xs.data());
    mf_kernels::gather_strided(leny, y, incy, ys.data());
    if (notrans) mgemv_n_contig(M, N, alpha, a, lda, xs.data(), ys.data());
    else         mgemv_t_contig(M, N, alpha, a, lda, xs.data(), ys.data());
    mf_kernels::scatter_strided(leny, y, incy, ys.data());
}

#undef A_
