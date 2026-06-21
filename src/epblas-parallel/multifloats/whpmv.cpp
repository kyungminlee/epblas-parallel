/* whpmv — multifloats complex DD Hermitian packed matrix-vector multiply. */

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
#define WHPMV_OMP_MIN 256
#define WHPMV_MAX_CPUS 256
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

/* One Hermitian-packed column j's contribution ADDED into accumulator yacc:
 * the off-diagonal SIMD AXPY (caxpy_add over the contiguous packed column) +
 * the real diagonal + the SIMD conj-dot reflected term (cdot). Additive
 * throughout, so the same kernel serves the serial sweep (yacc = y, sequential
 * j -> AXPY half bit-exact) and the threaded path (yacc = a private zero buffer,
 * disjoint cyclic columns -> within DD fuzz tol). The conj-dot reorders its
 * reduction either way. kk is the packed base of column j. */
inline void whpmv_col_upper(int j, const T *ap, const T *x, T alpha, T *yacc) {
    const std::size_t kk = static_cast<std::size_t>(j) * (j + 1) / 2;
    const T t1 = cmul(alpha, x[j]);
    mf_kernels::caxpy_add(j, yacc, &ap[kk], t1);
    const T t2 = mf_kernels::cdot(j, &ap[kk], x, true);
    yacc[j] = cadd(yacc[j], cadd(rcmul(ap[kk + j].re, t1), cmul(alpha, t2)));
}

inline void whpmv_col_lower(int j, int N, const T *ap, const T *x, T alpha, T *yacc) {
    const std::size_t kk =
        static_cast<std::size_t>(j) * N - static_cast<std::size_t>(j) * (j - 1) / 2;
    const T t1 = cmul(alpha, x[j]);
    yacc[j] = cadd(yacc[j], rcmul(ap[kk].re, t1));
    const int len = N - j - 1;
    mf_kernels::caxpy_add(len, &yacc[j + 1], &ap[kk + 1], t1);
    const T t2 = mf_kernels::cdot(len, &ap[kk + 1], &x[j + 1], true);
    yacc[j] = cadd(yacc[j], cmul(alpha, t2));
}
}

#ifdef _OPENMP
/* Unit-stride threaded Hermitian packed matvec (contiguous x,y; y already
 * beta-applied). Area-balanced COLUMN partition — a faithful port of the mspmv
 * axpydot (the real symmetric-packed twin): each thread owns a disjoint
 * contiguous column range and folds the RAW Hermitian product A*x (alpha
 * deferred) into a private slot[N] via the same SIMD column kernels as the serial
 * sweep (caxpy_add scatter + conj cdot reflected term + real diagonal). A BOUNDED
 * reduction then sums just each thread's populated row span and alpha-scales into
 * y — escaping the old cyclic scheme's O(nthreads*N) full fold that floored
 * par4/par1 at ~0.47. Reorders the per-row sum vs serial -> within DD fuzz tol.
 * Returns true if handled. */
__attribute__((noinline)) static bool whpmv_omp(
    bool upper, int N, const T *ap, const T *x, T alpha, T *y)
{
    int nthreads = blas_omp_max_threads();
    if (nthreads <= 1 || omp_in_parallel()) return false;
    if (nthreads > WHPMV_MAX_CPUS) nthreads = WHPMV_MAX_CPUS;

    std::ptrdiff_t range[WHPMV_MAX_CPUS + 1];
    /* per-column work ~j (upper) / ~(N-j) (lower) -> heavy_high=upper. mask3/min4
     * (complex slot = 32B, 4-wide width = cache-line aligned -> no false share). */
    int num_cpu = mf_omp::tri_area_bounds(N, nthreads, 3, 4, upper,
                                          WHPMV_MAX_CPUS, range);
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
                const T *aj = &ap[(std::size_t)j * (j + 1) / 2];
                T temp2 = czero;
                if (j > 0) {
                    mf_kernels::caxpy_add((int)j, &slot[0], aj, temp1);
                    temp2 = mf_kernels::cdot((int)j, aj, &x[0], true);
                }
                slot[j] = cadd(slot[j], cadd(rcmul(aj[j].re, temp1), temp2));
            }
        } else {
            for (std::ptrdiff_t j = m_from; j < m_to; ++j) {
                const T temp1 = x[j];
                const T *aj =
                    &ap[(std::size_t)j * N - (std::size_t)j * (j - 1) / 2];
                slot[j] = cadd(slot[j], rcmul(aj[0].re, temp1));
                const int len = N - (int)j - 1;
                if (len > 0) {
                    mf_kernels::caxpy_add(len, &slot[j + 1], &aj[1], temp1);
                    slot[j] = cadd(slot[j],
                                   mf_kernels::cdot(len, &aj[1], &x[j + 1], true));
                }
            }
        }
    }

    /* Bounded reduction: fold each thread's populated row window (alpha deferred
     * to here) straight onto y. */
    for (int t = 0; t < num_cpu; ++t) {
        const T *slot = buf + (std::size_t)t * N;
        std::ptrdiff_t from, to;
        mf_omp::tri_row_window(t, upper, range, N, from, to);
        for (std::ptrdiff_t k = from; k < to; ++k) y[k] = cadd(y[k], cmul(alpha, slot[k]));
    }
    std::free(buf);
    return true;
}
#endif

/* Contiguous (unit-stride x,y) core: Hermitian packed matvec y += alpha*A*x,
 * with y already beta-applied. Threaded private-accumulator sweep when enabled,
 * else a serial SIMD column sweep. Strided callers gather x,y around this. */
static void whpmv_contig(bool upper, int N, const T *ap, const T *x, T alpha, T *y)
{
#ifdef _OPENMP
    if (N >= WHPMV_OMP_MIN && blas_omp_available()
        && whpmv_omp(upper, N, ap, x, alpha, y))
        return;
#endif
    if (upper) for (int j = 0; j < N; ++j) whpmv_col_upper(j, ap, x, alpha, y);
    else       for (int j = 0; j < N; ++j) whpmv_col_lower(j, N, ap, x, alpha, y);
}

extern "C" void whpmv_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *ap,
    const T *x, const int *incx_,
    const T *beta_,
    T *y, const int *incy_,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);

    if (N == 0 || (ceq0(alpha) && ceq1(beta))) return;

    mf_kernels::cscale_y(N, beta, y, incy);
    if (ceq0(alpha)) return;

    if (incx == 1 && incy == 1) {
        whpmv_contig(UPLO == 'U', N, ap, x, alpha, y);
        return;
    }
    /* Strided x,y: gather to unit stride (y already beta-applied), run the SIMD
     * core, scatter y back. Handles negative increments; O(N) vs O(N^2) work. */
    const T *xbase = (incx < 0) ? x - (std::ptrdiff_t)(N - 1) * incx : x;
    T *ybase = (incy < 0) ? y - (std::ptrdiff_t)(N - 1) * incy : y;
    std::vector<T> xs(static_cast<std::size_t>(N)), ys(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        xs[i] = xbase[(std::ptrdiff_t)i * incx];
        ys[i] = ybase[(std::ptrdiff_t)i * incy];
    }
    whpmv_contig(UPLO == 'U', N, ap, xs.data(), alpha, ys.data());
    for (int i = 0; i < N; ++i) ybase[(std::ptrdiff_t)i * incy] = ys[i];
}
