/* msymv — multifloats real DD symmetric matrix-vector.
 * SIMD: per outer i, the inner k loop simultaneously updates y[k]
 * (axpy with temp1) and accumulates temp2 = sum A[k,i]*x[k]. The
 * same A[k,i] feeds both ops — single load, two uses. SoA-pack y
 * and x at entry, unpack y on exit. Horizontal-reduce 4-lane temp2
 * to scalar per i. */

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
#if defined(_OPENMP) && defined(MBLAS_SIMD_DD)
#include <cstring>
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#define MSYMV_OMP_MIN 256
#define MSYMV_MAX_CPUS 256
#endif

namespace mf = multifloats;
using T = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
using mf_pred::eq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
const T zero_dd{0.0, 0.0};

#ifdef MBLAS_SIMD_DD
using simd_exact::load_dd4;
/* Horizontal-reduce a 4-lane DD vector pair to scalar DD (lane 0
 * of result holds the sum across all 4 lanes). */
using simd_fast::hreduce;
#endif
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

#ifdef MBLAS_SIMD_DD
namespace {
/* One symmetric column i's contribution ADDED into the SoA accumulator
 * yacc_hi/yacc_lo: the temp1-axpy over the off-diagonal run, the diagonal, and
 * alpha*temp2 (temp2 = sum A[k,i]*x[k]) folded into yacc[i]. Every write is
 * additive (yacc += ...), so the same instructions serve both the serial path
 * (yacc = the shared beta-scaled y, columns applied in order → bit-identical to
 * the prior inline body) and the threaded path (yacc = a private zero buffer,
 * a disjoint column subset; partials reduced afterwards → within DD fuzz tol). */
static inline __attribute__((always_inline)) void
msymv_col(bool lower, int i, int N, const T *a, std::size_t lda,
          const T *x, const double *x_hi, const double *x_lo,
          double *yacc_hi, double *yacc_lo, T alpha)
{
    const T temp1 = alpha * x[i];
    const __m256d t1h = _mm256_set1_pd(temp1.limbs[0]);
    const __m256d t1l = _mm256_set1_pd(temp1.limbs[1]);
    const T *ai = &A_(0, i);
    const double *aip = reinterpret_cast<const double *>(ai);
    const __m256d zerov = _mm256_setzero_pd();
    __m256d s_h = zerov, s_l = zerov;
    if (lower) {
        /* Scalar diagonal first. */
        T yi{yacc_hi[i], yacc_lo[i]};
        yi = yi + temp1 * ai[i];
        yacc_hi[i] = yi.limbs[0]; yacc_lo[i] = yi.limbs[1];
        int k = i + 1;
        /* Align to 4-element boundary at start. */
        for (; k < N && (k & 3) != 0; ++k) {
            T yk{yacc_hi[k], yacc_lo[k]};
            T aki = ai[k];
            yk = yk + temp1 * aki;
            yacc_hi[k] = yk.limbs[0]; yacc_lo[k] = yk.limbs[1];
            T xk{x_hi[k], x_lo[k]};
            T t2 = aki * xk;
            double red_h[4], red_l[4];
            _mm256_storeu_pd(red_h, s_h); _mm256_storeu_pd(red_l, s_l);
            T s{red_h[0], red_l[0]};
            s = s + t2;
            red_h[0] = s.limbs[0]; red_l[0] = s.limbs[1];
            s_h = _mm256_loadu_pd(red_h); s_l = _mm256_loadu_pd(red_l);
        }
        for (; k + 3 < N; k += 4) {
            __m256d a_h, a_l;
            load_dd4(aip + 2 * k, a_h, a_l);
            __m256d yh = _mm256_loadu_pd(yacc_hi + k);
            __m256d yl = _mm256_loadu_pd(yacc_lo + k);
            __m256d xh = _mm256_loadu_pd(x_hi + k);
            __m256d xl = _mm256_loadu_pd(x_lo + k);
            __m256d p1h, p1l;
            simd_fast::mul(t1h, t1l, a_h, a_l, p1h, p1l);
            __m256d nyh, nyl;
            simd_fast::add(yh, yl, p1h, p1l, nyh, nyl);
            _mm256_storeu_pd(yacc_hi + k, nyh);
            _mm256_storeu_pd(yacc_lo + k, nyl);
            __m256d p2h, p2l;
            simd_fast::mul(a_h, a_l, xh, xl, p2h, p2l);
            __m256d nsh, nsl;
            simd_fast::add(s_h, s_l, p2h, p2l, nsh, nsl);
            s_h = nsh; s_l = nsl;
        }
        T temp2 = hreduce(s_h, s_l);
        for (; k < N; ++k) {
            T yk{yacc_hi[k], yacc_lo[k]};
            T aki = ai[k];
            yk = yk + temp1 * aki;
            yacc_hi[k] = yk.limbs[0]; yacc_lo[k] = yk.limbs[1];
            temp2 = temp2 + aki * T{x_hi[k], x_lo[k]};
        }
        T yi2{yacc_hi[i], yacc_lo[i]};
        yi2 = yi2 + alpha * temp2;
        yacc_hi[i] = yi2.limbs[0]; yacc_lo[i] = yi2.limbs[1];
    } else {  /* UPLO == 'U', inner k = 0..i-1 (already 4-aligned at k=0) */
        int k = 0;
        for (; k + 3 < i; k += 4) {
            __m256d a_h, a_l;
            load_dd4(aip + 2 * k, a_h, a_l);
            __m256d yh = _mm256_loadu_pd(yacc_hi + k);
            __m256d yl = _mm256_loadu_pd(yacc_lo + k);
            __m256d xh = _mm256_loadu_pd(x_hi + k);
            __m256d xl = _mm256_loadu_pd(x_lo + k);
            __m256d p1h, p1l;
            simd_fast::mul(t1h, t1l, a_h, a_l, p1h, p1l);
            __m256d nyh, nyl;
            simd_fast::add(yh, yl, p1h, p1l, nyh, nyl);
            _mm256_storeu_pd(yacc_hi + k, nyh);
            _mm256_storeu_pd(yacc_lo + k, nyl);
            __m256d p2h, p2l;
            simd_fast::mul(a_h, a_l, xh, xl, p2h, p2l);
            __m256d nsh, nsl;
            simd_fast::add(s_h, s_l, p2h, p2l, nsh, nsl);
            s_h = nsh; s_l = nsl;
        }
        T temp2 = hreduce(s_h, s_l);
        for (; k < i; ++k) {
            T yk{yacc_hi[k], yacc_lo[k]};
            T aki = ai[k];
            yk = yk + temp1 * aki;
            yacc_hi[k] = yk.limbs[0]; yacc_lo[k] = yk.limbs[1];
            temp2 = temp2 + aki * T{x_hi[k], x_lo[k]};
        }
        T yi{yacc_hi[i], yacc_lo[i]};
        yi = yi + temp1 * ai[i] + alpha * temp2;
        yacc_hi[i] = yi.limbs[0]; yacc_lo[i] = yi.limbs[1];
    }
}
}
#endif

#if defined(_OPENMP) && defined(MBLAS_SIMD_DD)
/* Threaded symmetric matvec. y_hi/y_lo enter holding the beta-scaled y. Columns
 * are split into area-balanced CONTIGUOUS slices (per-column work ~ (N-i) lower /
 * ~i upper -> tri_area_bounds, heavy_high=upper), each thread folding its slice
 * into a private zero buffer via msymv_col (contiguous A + localized row writes).
 * A BOUNDED reduction then sums just each thread's populated row window (lower:
 * col i writes rows [i,N) -> thread window [range[t],N); upper: rows [0,i] ->
 * [0,range[t+1])) onto y. Escapes the old cyclic scheme's O(nthreads*N) full
 * fold. Reorders the per-row sum vs serial -> within DD fuzz tol. Returns true
 * if handled. */
__attribute__((noinline)) static bool msymv_omp(
    bool lower, int N, const T *a, std::size_t lda, const T *x,
    const double *x_hi, const double *x_lo,
    double *y_hi, double *y_lo, T alpha)
{
    int nthreads = blas_omp_max_threads();
    if (nthreads <= 1 || omp_in_parallel()) return false;
    if (nthreads > MSYMV_MAX_CPUS) nthreads = MSYMV_MAX_CPUS;

    std::ptrdiff_t range[MSYMV_MAX_CPUS + 1];
    int ncpu = mf_omp::tri_area_bounds(N, nthreads, 3, 4, !lower,
                                       MSYMV_MAX_CPUS, range);
    if (ncpu <= 1) return false;

    const std::size_t N_pad = (static_cast<std::size_t>(N) + 3) & ~static_cast<std::size_t>(3);
    const std::size_t per = 2 * N_pad;
    double *pool = static_cast<double *>(
        std::aligned_alloc(32, static_cast<std::size_t>(ncpu) * per * sizeof(double)));
    if (!pool) return false;
    std::memset(pool, 0, static_cast<std::size_t>(ncpu) * per * sizeof(double));

    #pragma omp parallel num_threads(ncpu)
    {
        int tid = omp_get_thread_num();
        double *yp_hi = pool + static_cast<std::size_t>(tid) * per;
        double *yp_lo = yp_hi + N_pad;
        for (int i = (int)range[tid]; i < (int)range[tid + 1]; ++i)
            msymv_col(lower, i, N, a, lda, x, x_hi, x_lo, yp_hi, yp_lo, alpha);
    }

    /* Bounded reduction: fold each thread's populated row window onto y. */
    for (int t = 0; t < ncpu; ++t) {
        const double *yp_hi = pool + static_cast<std::size_t>(t) * per;
        const double *yp_lo = yp_hi + N_pad;
        std::ptrdiff_t k_from, k_to;
        mf_omp::tri_row_window(t, !lower, range, N, k_from, k_to);
        for (std::ptrdiff_t k = k_from; k < k_to; ++k) {
            T yk{y_hi[k], y_lo[k]};
            yk = yk + T{yp_hi[k], yp_lo[k]};
            y_hi[k] = yk.limbs[0]; y_lo[k] = yk.limbs[1];
        }
    }
    std::free(pool);
    return true;
}
#endif

/* Contiguous (unit-stride) symmetric matvec core: y += alpha*A*x, y pre-beta'd.
 * SIMD build packs x/y to SoA and runs the threaded/serial msymv_col; the scalar
 * fallback build runs the reference column loops. Strided callers gather x/y to
 * contiguous scratch and scatter y back, so this single core serves every case. */
static void msymv_contig(bool lower, int N, const T *a, std::size_t lda,
                         const T *x, T *y, T alpha)
{
#ifdef MBLAS_SIMD_DD
    const std::size_t N_pad = (static_cast<std::size_t>(N) + 3) & ~static_cast<std::size_t>(3);
    double *x_hi = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
    double *x_lo = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
    double *y_hi = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
    double *y_lo = static_cast<double *>(std::aligned_alloc(32, N_pad * sizeof(double)));
    for (int i = 0; i < N; ++i) {
        x_hi[i] = x[i].limbs[0]; x_lo[i] = x[i].limbs[1];
        y_hi[i] = y[i].limbs[0]; y_lo[i] = y[i].limbs[1];
    }
    for (std::size_t i = static_cast<std::size_t>(N); i < N_pad; ++i) {
        x_hi[i] = 0.0; x_lo[i] = 0.0; y_hi[i] = 0.0; y_lo[i] = 0.0;
    }
    bool done_omp = false;
#if defined(_OPENMP)
    if (N >= MSYMV_OMP_MIN && blas_omp_available())
        done_omp = msymv_omp(lower, N, a, lda, x, x_hi, x_lo, y_hi, y_lo, alpha);
#endif
    if (!done_omp)
        for (int i = 0; i < N; ++i)
            msymv_col(lower, i, N, a, lda, x, x_hi, x_lo, y_hi, y_lo, alpha);
    for (int i = 0; i < N; ++i) {
        y[i].limbs[0] = y_hi[i]; y[i].limbs[1] = y_lo[i];
    }
    std::free(x_hi); std::free(x_lo); std::free(y_hi); std::free(y_lo);
#else
    if (lower) {
        for (int i = 0; i < N; ++i) {
            const T temp1 = alpha * x[i];
            T temp2 = zero_dd;
            const T *ai = &A_(0, i);
            y[i] = y[i] + temp1 * ai[i];
            for (int k = i + 1; k < N; ++k) {
                y[k]  = y[k] + temp1 * ai[k];
                temp2 = temp2 + ai[k] * x[k];
            }
            y[i] = y[i] + alpha * temp2;
        }
    } else {
        for (int i = 0; i < N; ++i) {
            const T temp1 = alpha * x[i];
            T temp2 = zero_dd;
            const T *ai = &A_(0, i);
            for (int k = 0; k < i; ++k) {
                y[k]  = y[k] + temp1 * ai[k];
                temp2 = temp2 + ai[k] * x[k];
            }
            y[i] = y[i] + temp1 * ai[i] + alpha * temp2;
        }
    }
#endif
}

extern "C" void msymv_(
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
    const std::size_t lda = static_cast<std::size_t>(*lda_);
    const int incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const bool lower = (up(uplo) == 'L');

    if (N == 0) return;

    mf_kernels::scale_y(N, beta, y, incy);
    if (eq0(alpha)) return;

    if (incx == 1 && incy == 1) {
        msymv_contig(lower, N, a, lda, x, y, alpha);
        return;
    }

    /* Strided x,y: gather to unit stride (y already beta-applied), run the SIMD
     * core, scatter y back. Handles negative increments; O(N) gather vs O(N^2). */
    std::vector<T> xs(static_cast<std::size_t>(N)), ys(static_cast<std::size_t>(N));
    mf_kernels::gather_strided(N, x, incx, xs.data());
    mf_kernels::gather_strided(N, y, incy, ys.data());
    msymv_contig(lower, N, a, lda, xs.data(), ys.data(), alpha);
    mf_kernels::scatter_strided(N, y, incy, ys.data());
}

#undef A_
