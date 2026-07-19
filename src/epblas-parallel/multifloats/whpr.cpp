/* whpr — multifloats complex DD Hermitian packed rank-1 update.
 *   A := alpha*x*x^H + A, alpha real.
 *
 * Columns independent → OMP over j.
 */

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


/* zero/one predicates — see mf_pred.h */
using mf_pred::eq0;
using mf_pred::ceq0;

using mf_util::up;  /* char flag uppercase — mf_util.h */
namespace {
#define WHPR_OMP_MIN 64
const R rzero{0.0, 0.0};
using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::cconj;
using mf_kernels::rcmul;
}

/* Contiguous (unit-stride x) core: packed Hermitian rank-1 A += alpha*x*x^H.
 * Off-diagonal packed-column run is a SIMD column-AXPY (caxpy_add, bit-exact;
 * cmul commutes so t=alpha*conj(x[j]) factors out); diagonal forced real. */
static void whpr_contig(char UPLO, std::ptrdiff_t n, R alpha, TC *ap, const TC *x)
{
    if (UPLO == 'U') {
#ifdef _OPENMP
        const bool use_omp = (n >= WHPR_OMP_MIN && blas_omp_available());
        /* static,1: cyclic interleave balances the triangular packed-column skew;
         * complex DD rank-1 work per element dominates any false sharing. */
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (std::ptrdiff_t j = 0; j < n; ++j) {
            const std::ptrdiff_t kk = (j * (j + 1)) / 2;
            if (!ceq0(x[j])) {
                const TC tmp = rcmul(alpha, cconj(x[j]));
                mf_kernels::caxpy_add(j, &ap[kk], &x[0], tmp);
                const R new_re = ap[kk + j].re + cmul(x[j], tmp).re;
                ap[kk + j] = TC{ new_re, rzero };
            } else {
                ap[kk + j] = TC{ ap[kk + j].re, rzero };
            }
        }
    } else {
#ifdef _OPENMP
        const bool use_omp = (n >= WHPR_OMP_MIN && blas_omp_available());
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (std::ptrdiff_t j = 0; j < n; ++j) {
            const std::ptrdiff_t kk = j * n - (j * (j - 1)) / 2;
            if (!ceq0(x[j])) {
                const TC tmp = rcmul(alpha, cconj(x[j]));
                const R new_re = ap[kk].re + cmul(tmp, x[j]).re;
                ap[kk] = TC{ new_re, rzero };
                mf_kernels::caxpy_add(n - (j + 1), &ap[kk + 1], &x[j + 1], tmp);
            } else {
                ap[kk] = TC{ ap[kk].re, rzero };
            }
        }
    }
}

static void whpr_core(
    char uplo,
    std::ptrdiff_t n,
    const R *alpha_,
    const TC *x, std::ptrdiff_t incx,
    TC *ap)
{
    const R alpha = *alpha_;
    const char UPLO = up(&uplo);

    if (n == 0 || eq0(alpha)) return;

    if (incx == 1) {
        whpr_contig(UPLO, n, alpha, ap, x);
        return;
    }
    /* Strided x: gather to unit-stride scratch, run the SIMD core. */
    const TC *xbase = (incx < 0) ? x - static_cast<std::ptrdiff_t>(n - 1) * incx : x;
    std::vector<TC> xs(static_cast<std::size_t>(n));
    for (std::ptrdiff_t i = 0; i < n; ++i) xs[i] = xbase[static_cast<std::ptrdiff_t>(i) * incx];
    whpr_contig(UPLO, n, alpha, ap, xs.data());
}

extern "C" {
EPBLAS_FACADE_SPR(whpr, R, TC)
}
