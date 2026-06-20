/* wher — multifloats Hermitian rank-1 update (alpha real, diag real). */

#include <cstddef>
#include <cctype>
#include <vector>
#include <multifloats.h>
#include "mf_util.h"
#include "mf_pred.h"
#include "mf_kernels.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
using mf_pred::ceq0;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
#define WHER_OMP_MIN 64
const R rzero{0.0, 0.0};
using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::cconj;
inline T rcmul(R const &r, T const &z) { return T{ r * z.re, r * z.im }; }
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

/* Contiguous (unit-stride x) core: Hermitian rank-1 A += alpha * x * conj(x)^T,
 * updating one triangle. The off-diagonal column run is a SIMD column-AXPY
 * (caxpy_add, bit-exact); the diagonal stays real. Columns disjoint -> OMP-over-j
 * race-free. Strided callers gather x to unit stride around this. */
static void wher_contig(char UPLO, int N, R alpha, T *a, std::size_t lda, const T *x)
{
#ifdef _OPENMP
    const int use_omp = (N >= WHER_OMP_MIN && blas_omp_max_threads() > 1);
    /* static,1: cyclic interleave balances the triangular column skew; mirrors
     * the yher twin. */
    #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
    for (int j = 0; j < N; ++j) {
        const T xj = x[j];
        if (!ceq0(xj)) {
            const T t = rcmul(alpha, cconj(xj));
            T *aj = &A_(0, j);
            if (UPLO == 'L') {
                mf_kernels::caxpy_add(N - (j + 1), &aj[j + 1], &x[j + 1], t);
            } else {
                mf_kernels::caxpy_add(j, &aj[0], &x[0], t);
            }
            /* Diagonal stays real. */
            T prod = cmul(t, x[j]);
            aj[j] = T{ aj[j].re + prod.re, rzero };
        }
    }
}

extern "C" void wher_(
    const char *uplo,
    const int *n_,
    const R *alpha_,
    const T *x, const int *incx_,
    T *a, const int *lda_,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, lda = *lda_;
    const R alpha = *alpha_;
    const char UPLO = up(uplo);

    if (N == 0 || eq0(alpha)) return;

    if (incx == 1) {
        wher_contig(UPLO, N, alpha, a, lda, x);
        return;
    }
    /* Strided x: gather to unit-stride scratch, run the SIMD core (A is
     * column-major/lda regardless of x's stride). */
    const T *xbase = (incx < 0) ? x - static_cast<std::ptrdiff_t>(N - 1) * incx : x;
    std::vector<T> xs(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) xs[i] = xbase[static_cast<std::ptrdiff_t>(i) * incx];
    wher_contig(UPLO, N, alpha, a, lda, xs.data());
}

#undef A_
