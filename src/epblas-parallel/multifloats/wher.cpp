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
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using R = mf::float64x2;
using TC = mf::complex64x2;


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
using mf_kernels::rcmul;
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

/* Contiguous (unit-stride x) core: Hermitian rank-1 A += alpha * x * conj(x)^T,
 * updating one triangle. The off-diagonal column run is a SIMD column-AXPY
 * (caxpy_add, bit-exact); the diagonal stays real. Columns disjoint -> OMP-over-j
 * race-free. Strided callers gather x to unit stride around this. */
static void wher_contig(char UPLO, std::ptrdiff_t n, R alpha, TC *a, std::size_t lda, const TC *x)
{
#ifdef _OPENMP
    const bool use_omp = (n >= WHER_OMP_MIN && blas_omp_available());
    /* static,1: cyclic interleave balances the triangular column skew; mirrors
     * the yher twin. */
    #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        const TC xj = x[j];
        if (!ceq0(xj)) {
            const TC t = rcmul(alpha, cconj(xj));
            TC *aj = &A_(0, j);
            if (UPLO == 'L') {
                mf_kernels::caxpy_add(n - (j + 1), &aj[j + 1], &x[j + 1], t);
            } else {
                mf_kernels::caxpy_add(j, &aj[0], &x[0], t);
            }
            /* Diagonal stays real. */
            TC prod = cmul(t, x[j]);
            aj[j] = TC{ aj[j].re + prod.re, rzero };
        }
    }
}

static void wher_core(
    char uplo,
    std::ptrdiff_t n,
    const R *alpha_,
    const TC *x, std::ptrdiff_t incx,
    TC *a, std::ptrdiff_t lda)
{
    const R alpha = *alpha_;
    const char UPLO = up(&uplo);

    if (n == 0 || eq0(alpha)) return;

    if (incx == 1) {
        wher_contig(UPLO, n, alpha, a, lda, x);
        return;
    }
    /* Strided x: gather to unit-stride scratch, run the SIMD core (A is
     * column-major/lda regardless of x's stride). */
    const TC *xbase = (incx < 0) ? x - static_cast<std::ptrdiff_t>(n - 1) * incx : x;
    std::vector<TC> xs(static_cast<std::size_t>(n));
    for (std::ptrdiff_t i = 0; i < n; ++i) xs[i] = xbase[static_cast<std::ptrdiff_t>(i) * incx];
    wher_contig(UPLO, n, alpha, a, lda, xs.data());
}

extern "C" {
EPBLAS_FACADE_SYR(wher, R, TC)
}

#undef A_
