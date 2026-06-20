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
#include <cstring>
#include <omp.h>
#include "../common/blas_omp.h"
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
inline T cmul_r(T const &a, R const &r) { return T{ a.re * r, a.im * r }; }

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
    yacc[j] = cadd(yacc[j], cadd(cmul_r(t1, ap[kk + j].re), cmul(alpha, t2)));
}

inline void whpmv_col_lower(int j, int N, const T *ap, const T *x, T alpha, T *yacc) {
    const std::size_t kk =
        static_cast<std::size_t>(j) * N - static_cast<std::size_t>(j) * (j - 1) / 2;
    const T t1 = cmul(alpha, x[j]);
    yacc[j] = cadd(yacc[j], cmul_r(t1, ap[kk].re));
    const int len = N - j - 1;
    mf_kernels::caxpy_add(len, &yacc[j + 1], &ap[kk + 1], t1);
    const T t2 = mf_kernels::cdot(len, &ap[kk + 1], &x[j + 1], true);
    yacc[j] = cadd(yacc[j], cmul(alpha, t2));
}
}

#ifdef _OPENMP
/* Threaded Hermitian packed matvec (contiguous x,y; y already beta-applied).
 * Private-accumulator column sweep mirroring whemv_omp: each thread folds a
 * cyclic subset of columns (balances the triangular work) into its own zero
 * buffer via the shared column kernel, then the partials reduce onto y. Reusing
 * the fully-SIMD column kernel keeps both halves vectorized (the old rowgather
 * left the reflected col-jump scalar -> omp4 stuck near parity). Returns true
 * if handled. */
__attribute__((noinline)) static bool whpmv_omp(
    bool upper, int N, const T *ap, const T *x, T alpha, T *y)
{
    int nthreads = blas_omp_max_threads();
    if (nthreads <= 1 || omp_in_parallel()) return false;
    if (nthreads > WHPMV_MAX_CPUS) nthreads = WHPMV_MAX_CPUS;

    T *pool = static_cast<T *>(
        std::malloc(static_cast<std::size_t>(nthreads) * N * sizeof(T)));
    if (!pool) return false;
    std::memset(pool, 0, static_cast<std::size_t>(nthreads) * N * sizeof(T));

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        T *yp = pool + static_cast<std::size_t>(tid) * N;
        if (upper) for (int j = tid; j < N; j += nthreads) whpmv_col_upper(j, ap, x, alpha, yp);
        else       for (int j = tid; j < N; j += nthreads) whpmv_col_lower(j, N, ap, x, alpha, yp);
    }

    for (int t = 0; t < nthreads; ++t) {
        const T *yp = pool + static_cast<std::size_t>(t) * N;
        for (int k = 0; k < N; ++k) y[k] = cadd(y[k], yp[k]);
    }
    std::free(pool);
    return true;
}
#endif

/* Contiguous (unit-stride x,y) core: Hermitian packed matvec y += alpha*A*x,
 * with y already beta-applied. Threaded private-accumulator sweep when enabled,
 * else a serial SIMD column sweep. Strided callers gather x,y around this. */
static void whpmv_contig(bool upper, int N, const T *ap, const T *x, T alpha, T *y)
{
#ifdef _OPENMP
    if (N >= WHPMV_OMP_MIN && blas_omp_max_threads() > 1
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

    if (!ceq1(beta)) {
        int iy = (incy < 0) ? -(N - 1) * incy : 0;
        if (ceq0(beta)) for (int i = 0; i < N; ++i) { y[iy] = czero; iy += incy; }
        else                  for (int i = 0; i < N; ++i) { y[iy] = cmul(beta, y[iy]); iy += incy; }
    }
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
