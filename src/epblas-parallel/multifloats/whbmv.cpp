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
#include <cstring>
#include <omp.h>
#include "../common/blas_omp.h"
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
inline T cmul_r(T const &a, R const &r) { return T{ a.re * r, a.im * r }; }
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
    yacc[j] = cadd(yacc[j], cadd(cmul_r(t1, A_(K, j).re), cmul(alpha, t2)));
}

static inline void whbmv_col_lower(int j, int N, int K, const T *a, std::size_t lda,
                                   const T *x, T alpha, T *yacc) {
    const T t1 = cmul(alpha, x[j]);
    yacc[j] = cadd(yacc[j], cmul_r(t1, A_(0, j).re));
    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
    const int len = i_hi - (j + 1);
    mf_kernels::caxpy_add(len, &yacc[j + 1], &A_(1, j), t1);
    const T t2 = mf_kernels::cdot(len, &A_(1, j), &x[j + 1], true);
    yacc[j] = cadd(yacc[j], cmul(alpha, t2));
}

#ifdef _OPENMP
/* Threaded Hermitian band matvec (contiguous x,y; y already beta-applied).
 * Private-accumulator column sweep mirroring whemv/whpmv: each thread folds a
 * cyclic subset of columns into its own zero buffer via the shared column
 * kernel, then the partials reduce onto y. Reusing the fully-SIMD column kernel
 * keeps both halves vectorized (the old row-gather left the reflected band
 * neighbours scalar -> omp4 near parity). Returns true if handled. */
__attribute__((noinline)) static bool whbmv_omp(
    bool upper, int N, int K, const T *a, std::size_t lda, const T *x, T alpha, T *y)
{
    int nthreads = blas_omp_max_threads();
    if (nthreads <= 1 || omp_in_parallel()) return false;
    if (nthreads > WHBMV_MAX_CPUS) nthreads = WHBMV_MAX_CPUS;

    T *pool = static_cast<T *>(
        std::malloc(static_cast<std::size_t>(nthreads) * N * sizeof(T)));
    if (!pool) return false;
    std::memset(pool, 0, static_cast<std::size_t>(nthreads) * N * sizeof(T));

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        T *yp = pool + static_cast<std::size_t>(tid) * N;
        if (upper) for (int j = tid; j < N; j += nthreads) whbmv_col_upper(j, K, a, lda, x, alpha, yp);
        else       for (int j = tid; j < N; j += nthreads) whbmv_col_lower(j, N, K, a, lda, x, alpha, yp);
    }

    for (int t = 0; t < nthreads; ++t) {
        const T *yp = pool + static_cast<std::size_t>(t) * N;
        for (int k = 0; k < N; ++k) y[k] = cadd(y[k], yp[k]);
    }
    std::free(pool);
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
