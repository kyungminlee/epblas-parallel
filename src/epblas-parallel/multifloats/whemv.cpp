/* whemv — multifloats Hermitian matrix-vector.
 * SIMD: same two-pass pattern as msymv with cmul/cadd; Hermitian
 * uses conj(A[k,i]) for the temp2 accumulation, achieved by neg on
 * the loaded A.im before the cmul. Diagonal A[i,i] kept real. */

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
#include "mf_dispatch.h"   /* MF_SIMD_TARGET + mf_have_avx2_fma() runtime gate */
#if defined(_OPENMP) && defined(MBLAS_SIMD_DD)
#include <cstring>
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#define WHEMV_OMP_MIN 256
#define WHEMV_MAX_CPUS 256
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using R = mf::float64x2;
using TC = mf::complex64x2;


/* zero/one predicates — see mf_pred.h */
using mf_pred::ceq0;
using mf_pred::ceq1;

using mf_util::up;  /* char flag uppercase — mf_util.h */
namespace {
const R rzero{0.0, 0.0};
using mf_pred::zero_cdd;   /* shared DD constants — mf_pred.h */

using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::cconj;

#ifdef MBLAS_SIMD_DD
using simd_exact::cload4;
/* Horizontal-reduce 4-lane complex DD to scalar T (lane 0). */
using simd_fast::chreduce;
#endif
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

#ifdef MBLAS_SIMD_DD
namespace {
/* AVX2+FMA under a possibly pre-Haswell baseline -march; compiled with the feature
 * enabled and reached only behind mf_have_avx2_fma() (via whemv_col at the gated
 * call sites). Plain static (not always_inline) so it is legally called from the
 * baseline whemv_col across the target mismatch. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")
/* SIMD inner sweep for Hermitian column i over k in [k_lo,k_hi): the temp1-axpy
 * y[k] += temp1*A[k,i] folded into the SoA accumulator yacc, returning
 * temp2 = sum conj(A[k,i])*x[k]. Every yacc write is additive, so the same
 * instructions serve the serial path (yacc = shared y, in column order) and the
 * threaded path (yacc = private zero buffer, disjoint columns). */
static TC
whemv_inner_simd(std::ptrdiff_t i, std::ptrdiff_t k_lo, std::ptrdiff_t k_hi, const TC *a, std::size_t lda, TC alpha,
            const double *x_rh, const double *x_rl,
            const double *x_ih, const double *x_il,
            double *yacc_rh, double *yacc_rl,
            double *yacc_ih, double *yacc_il)
{
    const TC t1 = cmul(alpha, TC{ R{x_rh[i], x_rl[i]}, R{x_ih[i], x_il[i]} });
    const __m256d t1rh = _mm256_set1_pd(t1.re.limbs[0]);
    const __m256d t1rl = _mm256_set1_pd(t1.re.limbs[1]);
    const __m256d t1ih = _mm256_set1_pd(t1.im.limbs[0]);
    const __m256d t1il = _mm256_set1_pd(t1.im.limbs[1]);
    const double *aip = reinterpret_cast<const double *>(&A_(0, i));
    const __m256d zerov = _mm256_setzero_pd();
    __m256d s_rh = zerov, s_rl = zerov, s_ih = zerov, s_il = zerov;
    std::ptrdiff_t k = k_lo;
    TC temp2_sc = zero_cdd;
    /* Align to 4-boundary for unit-aligned SIMD. */
    for (; k < k_hi && (k & 3) != 0; ++k) {
        TC aki{ R{aip[4*k], aip[4*k+1]}, R{aip[4*k+2], aip[4*k+3]} };
        TC yk{ R{yacc_rh[k], yacc_rl[k]}, R{yacc_ih[k], yacc_il[k]} };
        yk = cadd(yk, cmul(t1, aki));
        yacc_rh[k] = yk.re.limbs[0]; yacc_rl[k] = yk.re.limbs[1];
        yacc_ih[k] = yk.im.limbs[0]; yacc_il[k] = yk.im.limbs[1];
        TC xk{ R{x_rh[k], x_rl[k]}, R{x_ih[k], x_il[k]} };
        temp2_sc = cadd(temp2_sc, cmul(cconj(aki), xk));
    }
    for (; k + 3 < k_hi; k += 4) {
        __m256d a_rh, a_rl, a_ih, a_il;
        cload4(aip + 4 * k, a_rh, a_rl, a_ih, a_il);
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
        simd_fast::cmul(t1rh, t1rl, t1ih, t1il, a_rh, a_rl, a_ih, a_il,
                         p_rh, p_rl, p_ih, p_il);
        __m256d nrh, nrl, nih, nil;
        simd_fast::cadd(yrh, yrl, yih, yil, p_rh, p_rl, p_ih, p_il,
                         nrh, nrl, nih, nil);
        _mm256_storeu_pd(yacc_rh + k, nrh);
        _mm256_storeu_pd(yacc_rl + k, nrl);
        _mm256_storeu_pd(yacc_ih + k, nih);
        _mm256_storeu_pd(yacc_il + k, nil);
        /* temp2 += conj(A[k,i]) * x[k] */
        simd_fast::neg(a_ih, a_il);
        __m256d q_rh, q_rl, q_ih, q_il;
        simd_fast::cmul(a_rh, a_rl, a_ih, a_il, xrh, xrl, xih, xil,
                         q_rh, q_rl, q_ih, q_il);
        __m256d nsrh, nsrl, nsih, nsil;
        simd_fast::cadd(s_rh, s_rl, s_ih, s_il, q_rh, q_rl, q_ih, q_il,
                         nsrh, nsrl, nsih, nsil);
        s_rh = nsrh; s_rl = nsrl; s_ih = nsih; s_il = nsil;
    }
    TC temp2 = chreduce(s_rh, s_rl, s_ih, s_il);
    temp2 = cadd(temp2, temp2_sc);
    for (; k < k_hi; ++k) {
        TC aki{ R{aip[4*k], aip[4*k+1]}, R{aip[4*k+2], aip[4*k+3]} };
        TC yk{ R{yacc_rh[k], yacc_rl[k]}, R{yacc_ih[k], yacc_il[k]} };
        yk = cadd(yk, cmul(t1, aki));
        yacc_rh[k] = yk.re.limbs[0]; yacc_rl[k] = yk.re.limbs[1];
        yacc_ih[k] = yk.im.limbs[0]; yacc_il[k] = yk.im.limbs[1];
        TC xk{ R{x_rh[k], x_rl[k]}, R{x_ih[k], x_il[k]} };
        temp2 = cadd(temp2, cmul(cconj(aki), xk));
    }
    return temp2;
}
#pragma GCC pop_options

/* One Hermitian column i's contribution ADDED into the SoA accumulator yacc:
 * the off-diagonal axpy + temp2 (whemv_inner) plus the real diagonal and
 * alpha*temp2 folded into yacc[i]. Additive throughout → shared by serial
 * (column order, bit-identical to the prior inline body) and threaded
 * (private zero buffer, disjoint columns → within DD fuzz tol). */
static inline __attribute__((always_inline)) void
whemv_col(bool lower, std::ptrdiff_t i, std::ptrdiff_t n, const TC *a, std::size_t lda, TC alpha,
          const double *x_rh, const double *x_rl,
          const double *x_ih, const double *x_il,
          double *y_rh, double *y_rl, double *y_ih, double *y_il)
{
    const TC temp1 = cmul(alpha, TC{ R{x_rh[i], x_rl[i]}, R{x_ih[i], x_il[i]} });
    if (lower) {
        /* Diagonal contribution (A[i,i] real). */
        TC aii_re{ A_(i, i).re, rzero };
        TC yi{ R{y_rh[i], y_rl[i]}, R{y_ih[i], y_il[i]} };
        yi = cadd(yi, cmul(temp1, aii_re));
        y_rh[i] = yi.re.limbs[0]; y_rl[i] = yi.re.limbs[1];
        y_ih[i] = yi.im.limbs[0]; y_il[i] = yi.im.limbs[1];
        TC temp2 = whemv_inner_simd(i, i + 1, n, a, lda, alpha,
                              x_rh, x_rl, x_ih, x_il, y_rh, y_rl, y_ih, y_il);
        TC yi2{ R{y_rh[i], y_rl[i]}, R{y_ih[i], y_il[i]} };
        yi2 = cadd(yi2, cmul(alpha, temp2));
        y_rh[i] = yi2.re.limbs[0]; y_rl[i] = yi2.re.limbs[1];
        y_ih[i] = yi2.im.limbs[0]; y_il[i] = yi2.im.limbs[1];
    } else {
        TC temp2 = whemv_inner_simd(i, 0, i, a, lda, alpha,
                              x_rh, x_rl, x_ih, x_il, y_rh, y_rl, y_ih, y_il);
        TC aii_re{ A_(i, i).re, rzero };
        TC yi{ R{y_rh[i], y_rl[i]}, R{y_ih[i], y_il[i]} };
        yi = cadd(yi, cadd(cmul(temp1, aii_re), cmul(alpha, temp2)));
        y_rh[i] = yi.re.limbs[0]; y_rl[i] = yi.re.limbs[1];
        y_ih[i] = yi.im.limbs[0]; y_il[i] = yi.im.limbs[1];
    }
}
}
#endif

#if defined(_OPENMP) && defined(MBLAS_SIMD_DD)
/* Threaded Hermitian matvec. y_* enter holding beta*y. Columns are split into
 * area-balanced CONTIGUOUS slices (per-column work ~ (N-i) lower / ~i upper ->
 * tri_area_bounds, heavy_high=upper), each thread folding its slice into a
 * private zero buffer via whemv_col (contiguous A + localized row writes). A
 * BOUNDED reduction then sums just each thread's populated row window (lower:
 * col i writes rows [i,N) -> thread window [range[t],N); upper: rows [0,i] ->
 * [0,range[t+1])) onto y. Escapes the old cyclic scheme's O(nthreads*N) full
 * fold. Reorders the per-row sum vs serial -> within DD fuzz tol. Returns true
 * if handled. */
__attribute__((noinline)) static bool whemv_omp(
    bool lower, std::ptrdiff_t n, const TC *a, std::size_t lda, TC alpha,
    const double *x_rh, const double *x_rl, const double *x_ih, const double *x_il,
    double *y_rh, double *y_rl, double *y_ih, double *y_il)
{
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (!blas_omp_should_thread()) return false;
    if (nthreads > WHEMV_MAX_CPUS) nthreads = WHEMV_MAX_CPUS;

    std::ptrdiff_t range[WHEMV_MAX_CPUS + 1];
    std::ptrdiff_t ncpu = mf_omp::tri_area_bounds(n, nthreads, 3, 4, !lower,
                                       WHEMV_MAX_CPUS, range);
    if (ncpu <= 1) return false;

    const std::size_t N_pad = (static_cast<std::size_t>(n) + 3) & ~static_cast<std::size_t>(3);
    const std::size_t per = 4 * N_pad;
    double *pool = static_cast<double *>(
        std::aligned_alloc(32, static_cast<std::size_t>(ncpu) * per * sizeof(double)));
    if (!pool) return false;
    std::memset(pool, 0, static_cast<std::size_t>(ncpu) * per * sizeof(double));

    #pragma omp parallel for schedule(static, 1) num_threads(ncpu)
    for (std::ptrdiff_t tid = 0; tid < ncpu; ++tid)
    {
        double *p = pool + static_cast<std::size_t>(tid) * per;
        double *yp_rh = p, *yp_rl = p + N_pad, *yp_ih = p + 2 * N_pad, *yp_il = p + 3 * N_pad;
        for (std::ptrdiff_t i = (std::ptrdiff_t)range[tid]; i < (std::ptrdiff_t)range[tid + 1]; ++i)
            whemv_col(lower, i, n, a, lda, alpha,
                      x_rh, x_rl, x_ih, x_il, yp_rh, yp_rl, yp_ih, yp_il);
    }

    /* Bounded reduction: fold each thread's populated row window onto y. */
    for (std::ptrdiff_t t = 0; t < ncpu; ++t) {
        const double *p = pool + static_cast<std::size_t>(t) * per;
        const double *yp_rh = p, *yp_rl = p + N_pad, *yp_ih = p + 2 * N_pad, *yp_il = p + 3 * N_pad;
        std::ptrdiff_t k_from, k_to;
        mf_omp::tri_row_window(t, !lower, range, n, k_from, k_to);
        for (std::ptrdiff_t k = k_from; k < k_to; ++k) {
            TC yk{ R{y_rh[k], y_rl[k]}, R{y_ih[k], y_il[k]} };
            yk = cadd(yk, TC{ R{yp_rh[k], yp_rl[k]}, R{yp_ih[k], yp_il[k]} });
            y_rh[k] = yk.re.limbs[0]; y_rl[k] = yk.re.limbs[1];
            y_ih[k] = yk.im.limbs[0]; y_il[k] = yk.im.limbs[1];
        }
    }
    std::free(pool);
    return true;
}
#endif

/* Contiguous (unit-stride x,y) core: Hermitian matvec y += alpha*A*x, with y
 * already beta-applied. SIMD SoA path (+ threaded private-accumulator) when
 * built with MBLAS_SIMD_DD; faithful scalar column sweep otherwise. Strided
 * callers gather x,y to unit stride around this. */
static void whemv_contig(bool lower, std::ptrdiff_t n, const TC *a, std::size_t lda, TC alpha,
                         const TC *x, TC *y)
{
#ifdef MBLAS_SIMD_DD
    if (mf_have_avx2_fma()) {
    const std::size_t N_pad = (static_cast<std::size_t>(n) + 3) & ~static_cast<std::size_t>(3);
    double *x_rh = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
    double *x_rl = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
    double *x_ih = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
    double *x_il = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
    double *y_rh = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
    double *y_rl = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
    double *y_ih = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
    double *y_il = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        x_rh[i] = x[i].re.limbs[0]; x_rl[i] = x[i].re.limbs[1];
        x_ih[i] = x[i].im.limbs[0]; x_il[i] = x[i].im.limbs[1];
        y_rh[i] = y[i].re.limbs[0]; y_rl[i] = y[i].re.limbs[1];
        y_ih[i] = y[i].im.limbs[0]; y_il[i] = y[i].im.limbs[1];
    }
    for (std::size_t i = static_cast<std::size_t>(n); i < N_pad; ++i) {
        x_rh[i] = 0.0; x_rl[i] = 0.0; x_ih[i] = 0.0; x_il[i] = 0.0;
        y_rh[i] = 0.0; y_rl[i] = 0.0; y_ih[i] = 0.0; y_il[i] = 0.0;
    }

    bool done_omp = false;
#if defined(_OPENMP)
    if (n >= WHEMV_OMP_MIN && blas_omp_available())
        done_omp = whemv_omp(lower, n, a, lda, alpha,
                             x_rh, x_rl, x_ih, x_il, y_rh, y_rl, y_ih, y_il);
#endif
    if (!done_omp)
        for (std::ptrdiff_t i = 0; i < n; ++i)
            whemv_col(lower, i, n, a, lda, alpha,
                      x_rh, x_rl, x_ih, x_il, y_rh, y_rl, y_ih, y_il);

    for (std::ptrdiff_t i = 0; i < n; ++i) {
        y[i].re.limbs[0] = y_rh[i]; y[i].re.limbs[1] = y_rl[i];
        y[i].im.limbs[0] = y_ih[i]; y[i].im.limbs[1] = y_il[i];
    }
    std::free(x_rh); std::free(x_rl); std::free(x_ih); std::free(x_il);
    std::free(y_rh); std::free(y_rl); std::free(y_ih); std::free(y_il);
    return;
    }
#endif
    if (lower) {
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            const TC temp1 = cmul(alpha, x[i]);
            TC temp2 = zero_cdd;
            const TC *ai = &A_(0, i);
            const TC aii_re{ ai[i].re, rzero };
            y[i] = cadd(y[i], cmul(temp1, aii_re));
            for (std::ptrdiff_t k = i + 1; k < n; ++k) {
                y[k]  = cadd(y[k], cmul(temp1, ai[k]));
                temp2 = cadd(temp2, cmul(cconj(ai[k]), x[k]));
            }
            y[i] = cadd(y[i], cmul(alpha, temp2));
        }
    } else {
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            const TC temp1 = cmul(alpha, x[i]);
            TC temp2 = zero_cdd;
            const TC *ai = &A_(0, i);
            for (std::ptrdiff_t k = 0; k < i; ++k) {
                y[k]  = cadd(y[k], cmul(temp1, ai[k]));
                temp2 = cadd(temp2, cmul(cconj(ai[k]), x[k]));
            }
            const TC aii_re{ ai[i].re, rzero };
            y[i] = cadd(y[i], cadd(cmul(temp1, aii_re), cmul(alpha, temp2)));
        }
    }
}

static void whemv_core(
    char uplo,
    std::ptrdiff_t n,
    const TC *alpha_,
    const TC *a, std::ptrdiff_t lda,
    const TC *x, std::ptrdiff_t incx,
    const TC *beta_,
    TC *y, std::ptrdiff_t incy)
{
    const TC alpha = *alpha_, beta = *beta_;
    const char UPLO = up(&uplo);

    if (n == 0) return;

    mf_kernels::cscale_y(n, beta, y, incy);
    if (ceq0(alpha)) return;

    if (incx == 1 && incy == 1) {
        whemv_contig(UPLO == 'L', n, a, lda, alpha, x, y);
        return;
    }
    /* Strided x,y: gather to unit stride (y already beta-applied), run the SIMD
     * core, scatter y back. Handles negative increments; O(N) gather vs the old
     * O(N^2) strided sweep, and lets the strided case thread like contiguous. */
    std::vector<TC> xs(static_cast<std::size_t>(n)), ys(static_cast<std::size_t>(n));
    mf_kernels::gather_strided(n, x, incx, xs.data());
    mf_kernels::gather_strided(n, y, incy, ys.data());
    whemv_contig(UPLO == 'L', n, a, lda, alpha, xs.data(), ys.data());
    mf_kernels::scatter_strided(n, y, incy, ys.data());
}

extern "C" {
EPBLAS_FACADE_SYMV(whemv, TC)
}

#undef A_
