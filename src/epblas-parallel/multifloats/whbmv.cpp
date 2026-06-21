/* whbmv — multifloats complex DD Hermitian band matrix-vector multiply.
 *   y := alpha*A*x + beta*y, A Hermitian with K super-(sub-)diagonals.
 */

#include <cstddef>
#include <cctype>
#include <vector>
#include <multifloats.h>
#include "mf_util.h"
#include "mf_pred.h"
#include "mf_kernels.h"
#ifdef _OPENMP
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#define WHBMV_OMP_MIN 256
#define WHBMV_MAX_CPUS 256
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;
using mf_pred::ceq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
const R rzero{0.0, 0.0};
const T czero{ rzero, rzero };
using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::rcmul;
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

/* One Hermitian-band column j's contribution ADDED into accumulator yacc: the
 * off-diagonal SIMD AXPY (caxpy_add over the contiguous band run) + the real
 * diagonal + the SIMD conj-dot reflected term (cdot). Additive throughout, so
 * the same kernel serves the serial sweep (yacc = y, sequential j -> AXPY half
 * bit-exact) and the threaded path (yacc = a private zero buffer, disjoint
 * cyclic columns -> within DD fuzz tol). The conj-dot reorders its reduction
 * either way. */
static inline void whbmv_col_upper(int j, int K, const T *a, std::size_t lda,
                                   const T *x, T alpha, T *yacc) {
    const T t1 = cmul(alpha, x[j]);
    const int L = K - j;
    const int i_lo = (j - K > 0) ? (j - K) : 0;
    const int len = j - i_lo;
    mf_kernels::caxpy_add(len, &yacc[i_lo], &A_(L + i_lo, j), t1);
    const T t2 = mf_kernels::cdot(len, &A_(L + i_lo, j), &x[i_lo], true);
    yacc[j] = cadd(yacc[j], cadd(rcmul(A_(K, j).re, t1), cmul(alpha, t2)));
}

static inline void whbmv_col_lower(int j, int N, int K, const T *a, std::size_t lda,
                                   const T *x, T alpha, T *yacc) {
    const T t1 = cmul(alpha, x[j]);
    yacc[j] = cadd(yacc[j], rcmul(A_(0, j).re, t1));
    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
    const int len = i_hi - (j + 1);
    mf_kernels::caxpy_add(len, &yacc[j + 1], &A_(1, j), t1);
    const T t2 = mf_kernels::cdot(len, &A_(1, j), &x[j + 1], true);
    yacc[j] = cadd(yacc[j], cmul(alpha, t2));
}

#ifdef _OPENMP
/* Unit-stride threaded Hermitian band matvec (contiguous x,y; y already
 * beta-applied). Faithful port of the msbmv axpydot (the real symmetric-band
 * twin): EQUAL-WIDTH contiguous column partition (band work per column is uniform
 * ~2K entries, so an equal split balances load — a triangular sqrt split would
 * skew it), each thread folding the RAW Hermitian product A*x (alpha deferred)
 * into a private slot[N] via the same SIMD band kernels as the serial sweep
 * (caxpy_add scatter + conj cdot reflected term + real diagonal). A WINDOWED
 * bounded reduction then sums just each thread's band window ([range[t]-K, range
 * [t+1]) upper / [range[t], range[t+1]+K) lower) and alpha-scales into y —
 * adjacent windows overlap by K but each column's contribution lives in exactly
 * one slot, so the overlap-sum is correct. Escapes the old cyclic scheme's
 * O(nthreads*N) full fold. Reorders the per-row sum vs serial -> within DD fuzz
 * tol. Returns true if handled. */
__attribute__((noinline)) static bool whbmv_omp(
    bool upper, int N, int K, const T *a, std::size_t lda, const T *x, T alpha, T *y)
{
    int nthreads = blas_omp_max_threads();
    if (nthreads <= 1 || omp_in_parallel()) return false;
    if (nthreads > WHBMV_MAX_CPUS) nthreads = WHBMV_MAX_CPUS;

    std::ptrdiff_t range[WHBMV_MAX_CPUS + 1];
    int num_cpu = mf_omp::band_bounds(N, nthreads, 3, 4, WHBMV_MAX_CPUS, range);
    if (num_cpu <= 1) return false;

    T *buf = static_cast<T *>(std::calloc((std::size_t)num_cpu * N, sizeof(T)));
    if (!buf) return false;

    #pragma omp parallel num_threads(num_cpu)
    {
        int t = omp_get_thread_num();
        std::ptrdiff_t m_from = range[t];
        std::ptrdiff_t m_to   = range[t + 1];
        T *slot = buf + (std::size_t)t * N;
        if (upper) {
            for (std::ptrdiff_t j = m_from; j < m_to; ++j) {
                const T temp1 = x[j];                          /* alpha deferred */
                const int L = K - (int)j;
                const std::ptrdiff_t i_lo = (j - K > 0) ? (j - K) : 0;
                const int len = (int)(j - i_lo);
                const T *col = &A_(L + i_lo, j);               /* contiguous band run */
                T temp2 = czero;
                if (len > 0) {
                    mf_kernels::caxpy_add(len, &slot[i_lo], col, temp1);
                    temp2 = mf_kernels::cdot(len, col, &x[i_lo], true);
                }
                slot[j] = cadd(slot[j], cadd(rcmul(A_(K, j).re, temp1), temp2));
            }
        } else {
            for (std::ptrdiff_t j = m_from; j < m_to; ++j) {
                const T temp1 = x[j];
                slot[j] = cadd(slot[j], rcmul(A_(0, j).re, temp1));
                const std::ptrdiff_t i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                const int len = (int)(i_hi - (j + 1));
                if (len > 0) {
                    mf_kernels::caxpy_add(len, &slot[j + 1], &A_(1, j), temp1);
                    slot[j] = cadd(slot[j],
                                   mf_kernels::cdot(len, &A_(1, j), &x[j + 1], true));
                }
            }
        }
    }

    /* Windowed bounded reduction: each slot is touched only over a band window
     * around its column range; sum just those windows, alpha-scaled, into y. */
    for (int t = 0; t < num_cpu; ++t) {
        const T *slot = buf + (std::size_t)t * N;
        std::ptrdiff_t lo, hi;
        mf_omp::band_row_window(t, upper, range, N, K, lo, hi);
        for (std::ptrdiff_t i = lo; i < hi; ++i) y[i] = cadd(y[i], cmul(alpha, slot[i]));
    }
    std::free(buf);
    return true;
}
#endif

/* Contiguous (unit-stride x,y) core: Hermitian band matvec y += alpha*A*x, with
 * y already beta-applied. Threaded private-accumulator sweep when enabled, else
 * a serial SIMD column sweep. Strided callers gather x,y around this. */
static void whbmv_contig(bool upper, int N, int K, const T *a, std::size_t lda,
                         const T *x, T alpha, T *y)
{
#ifdef _OPENMP
    if (N >= WHBMV_OMP_MIN && blas_omp_max_threads() > 1
        && whbmv_omp(upper, N, K, a, lda, x, alpha, y))
        return;
#endif
    if (upper) for (int j = 0; j < N; ++j) whbmv_col_upper(j, K, a, lda, x, alpha, y);
    else       for (int j = 0; j < N; ++j) whbmv_col_lower(j, N, K, a, lda, x, alpha, y);
}

extern "C" void whbmv_(
    const char *uplo,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *x, const int *incx_,
    const T *beta_,
    T *y, const int *incy_,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);

    if (N == 0 || (ceq0(alpha) && ceq1(beta))) return;

    if (!ceq1(beta)) {
        int iy = (incy < 0) ? -(N - 1) * incy : 0;
        if (ceq0(beta)) for (int i = 0; i < N; ++i) { y[iy] = czero; iy += incy; }
        else                  for (int i = 0; i < N; ++i) { y[iy] = cmul(beta, y[iy]); iy += incy; }
    }
    if (ceq0(alpha)) return;

    if (incx == 1 && incy == 1) {
        whbmv_contig(UPLO == 'U', N, K, a, lda, x, alpha, y);
        return;
    }
    /* Strided x,y: gather to unit stride (y already beta-applied), run the SIMD
     * core, scatter y back. Handles negative increments; O(N) gather vs the old
     * O(N*K) strided sweep, and unifies the strided and contiguous paths. */
    const T *xbase = (incx < 0) ? x - (std::ptrdiff_t)(N - 1) * incx : x;
    T *ybase = (incy < 0) ? y - (std::ptrdiff_t)(N - 1) * incy : y;
    std::vector<T> xs(static_cast<std::size_t>(N)), ys(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        xs[i] = xbase[(std::ptrdiff_t)i * incx];
        ys[i] = ybase[(std::ptrdiff_t)i * incy];
    }
    whbmv_contig(UPLO == 'U', N, K, a, lda, xs.data(), alpha, ys.data());
    for (int i = 0; i < N; ++i) ybase[(std::ptrdiff_t)i * incy] = ys[i];
}

#undef A_
