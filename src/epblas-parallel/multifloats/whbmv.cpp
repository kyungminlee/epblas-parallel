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
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using R = mf::float64x2;
using TC = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;
using mf_pred::ceq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
const R rzero{0.0, 0.0};
const TC czero{ rzero, rzero };
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
static inline void whbmv_col_upper(std::ptrdiff_t j, std::ptrdiff_t k, const TC *a, std::size_t lda,
                                   const TC *x, TC alpha, TC *yacc) {
    const TC t1 = cmul(alpha, x[j]);
    const std::ptrdiff_t L = k - j;
    const std::ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
    const std::ptrdiff_t len = j - i_lo;
    mf_kernels::caxpy_add(len, &yacc[i_lo], &A_(L + i_lo, j), t1);
    const TC t2 = mf_kernels::cdot(len, &A_(L + i_lo, j), &x[i_lo], true);
    yacc[j] = cadd(yacc[j], cadd(rcmul(A_(k, j).re, t1), cmul(alpha, t2)));
}

static inline void whbmv_col_lower(std::ptrdiff_t j, std::ptrdiff_t n, std::ptrdiff_t k, const TC *a, std::size_t lda,
                                   const TC *x, TC alpha, TC *yacc) {
    const TC t1 = cmul(alpha, x[j]);
    yacc[j] = cadd(yacc[j], rcmul(A_(0, j).re, t1));
    const std::ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
    const std::ptrdiff_t len = i_hi - (j + 1);
    mf_kernels::caxpy_add(len, &yacc[j + 1], &A_(1, j), t1);
    const TC t2 = mf_kernels::cdot(len, &A_(1, j), &x[j + 1], true);
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
    bool upper, std::ptrdiff_t n, std::ptrdiff_t k, const TC *a, std::size_t lda, const TC *x, TC alpha, TC *y)
{
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (!blas_omp_should_thread()) return false;
    if (nthreads > WHBMV_MAX_CPUS) nthreads = WHBMV_MAX_CPUS;

    std::ptrdiff_t range[WHBMV_MAX_CPUS + 1];
    std::ptrdiff_t num_cpu = mf_omp::band_bounds(n, nthreads, 3, 4, WHBMV_MAX_CPUS, range);
    if (num_cpu <= 1) return false;

    TC *buf = static_cast<TC *>(std::calloc((std::size_t)num_cpu * n, sizeof(TC)));
    if (!buf) return false;

    #pragma omp parallel for schedule(static, 1) num_threads(num_cpu)
    for (std::ptrdiff_t t = 0; t < num_cpu; ++t)
    {
        std::ptrdiff_t m_from = range[t];
        std::ptrdiff_t m_to   = range[t + 1];
        TC *slot = buf + (std::size_t)t * n;
        if (upper) {
            for (std::ptrdiff_t j = m_from; j < m_to; ++j) {
                const TC temp1 = x[j];                          /* alpha deferred */
                const std::ptrdiff_t L = k - (std::ptrdiff_t)j;
                const std::ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                const std::ptrdiff_t len = (std::ptrdiff_t)(j - i_lo);
                const TC *col = &A_(L + i_lo, j);               /* contiguous band run */
                TC temp2 = czero;
                if (len > 0) {
                    mf_kernels::caxpy_add(len, &slot[i_lo], col, temp1);
                    temp2 = mf_kernels::cdot(len, col, &x[i_lo], true);
                }
                slot[j] = cadd(slot[j], cadd(rcmul(A_(k, j).re, temp1), temp2));
            }
        } else {
            for (std::ptrdiff_t j = m_from; j < m_to; ++j) {
                const TC temp1 = x[j];
                slot[j] = cadd(slot[j], rcmul(A_(0, j).re, temp1));
                const std::ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                const std::ptrdiff_t len = (std::ptrdiff_t)(i_hi - (j + 1));
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
    for (std::ptrdiff_t t = 0; t < num_cpu; ++t) {
        const TC *slot = buf + (std::size_t)t * n;
        std::ptrdiff_t lo, hi;
        mf_omp::band_row_window(t, upper, range, n, k, lo, hi);
        for (std::ptrdiff_t i = lo; i < hi; ++i) y[i] = cadd(y[i], cmul(alpha, slot[i]));
    }
    std::free(buf);
    return true;
}
#endif

/* Contiguous (unit-stride x,y) core: Hermitian band matvec y += alpha*A*x, with
 * y already beta-applied. Threaded private-accumulator sweep when enabled, else
 * a serial SIMD column sweep. Strided callers gather x,y around this. */
static void whbmv_contig(bool upper, std::ptrdiff_t n, std::ptrdiff_t k, const TC *a, std::size_t lda,
                         const TC *x, TC alpha, TC *y)
{
#ifdef _OPENMP
    if (n >= WHBMV_OMP_MIN && blas_omp_available()
        && whbmv_omp(upper, n, k, a, lda, x, alpha, y))
        return;
#endif
    if (upper) for (std::ptrdiff_t j = 0; j < n; ++j) whbmv_col_upper(j, k, a, lda, x, alpha, y);
    else       for (std::ptrdiff_t j = 0; j < n; ++j) whbmv_col_lower(j, n, k, a, lda, x, alpha, y);
}

static void whbmv_core(
    char uplo,
    std::ptrdiff_t n, std::ptrdiff_t k,
    const TC *alpha_,
    const TC *a, std::ptrdiff_t lda,
    const TC *x, std::ptrdiff_t incx,
    const TC *beta_,
    TC *y, std::ptrdiff_t incy)
{
    const TC alpha = *alpha_, beta = *beta_;
    const char UPLO = up(&uplo);

    if (n == 0 || (ceq0(alpha) && ceq1(beta))) return;

    mf_kernels::cscale_y(n, beta, y, incy);
    if (ceq0(alpha)) return;

    if (incx == 1 && incy == 1) {
        whbmv_contig(UPLO == 'U', n, k, a, lda, x, alpha, y);
        return;
    }
    /* Strided x,y: gather to unit stride (y already beta-applied), run the SIMD
     * core, scatter y back. Handles negative increments; O(N) gather vs the old
     * O(N*K) strided sweep, and unifies the strided and contiguous paths. */
    std::vector<TC> xs(static_cast<std::size_t>(n)), ys(static_cast<std::size_t>(n));
    mf_kernels::gather_strided(n, x, incx, xs.data());
    mf_kernels::gather_strided(n, y, incy, ys.data());
    whbmv_contig(UPLO == 'U', n, k, a, lda, xs.data(), alpha, ys.data());
    mf_kernels::scatter_strided(n, y, incy, ys.data());
}

extern "C" {
EPBLAS_FACADE_SBMV(whbmv, TC)
}

#undef A_
