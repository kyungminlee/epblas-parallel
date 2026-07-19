/* whpr2 — multifloats complex DD Hermitian packed rank-2 update.
 *   A := alpha*x*y^H + conj(alpha)*y*x^H + A
 *
 * Columns independent → OMP over j. The packed triangular output makes a
 * contiguous static block hand one thread the heavy triangle end (par caps
 * at ~2.3x on 4 cores); cyclic schedule(static,1) interleaves short and long
 * columns across the team, balancing the skew symmetrically for both UPLO
 * (mirrors the proven kind10 yhpr2). The off-diagonal packed-column run is a
 * SIMD rank-2 AXPY (mf_kernels::caxpy2_add, two rank-1 passes -> within DD fuzz
 * tol); the Hermitian diagonal is forced real.
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
using mf_pred::ceq0;

using mf_util::up;  /* char flag uppercase — mf_util.h */
namespace {
#define WHPR2_OMP_MIN 64
const R rzero{0.0, 0.0};
using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::cconj;

/* Per-column rank-2 update; off-diagonal packed run is a SIMD rank-2 AXPY (two
 * rank-1 passes). The Hermitian diagonal is forced real. */
inline void whpr2_col_upper(std::ptrdiff_t j, TC t1, TC t2, const TC *x, const TC *y, TC *ap) {
    TC *c = ap + static_cast<std::size_t>(j) * (j + 1) / 2;
    mf_kernels::caxpy2_add(j, c, x, t1, y, t2);
    const TC prod = cadd(cmul(x[j], t1), cmul(y[j], t2));
    c[j] = TC{ c[j].re + prod.re, rzero };
}

inline void whpr2_col_lower(std::ptrdiff_t j, std::ptrdiff_t n, TC t1, TC t2, const TC *x, const TC *y, TC *ap) {
    TC *c0 = ap + (static_cast<std::size_t>(j) * n - static_cast<std::size_t>(j) * (j - 1) / 2);
    mf_kernels::caxpy2_add(n - j - 1, c0 + 1, x + j + 1, t1, y + j + 1, t2);
    const TC prod = cadd(cmul(x[j], t1), cmul(y[j], t2));
    c0[0] = TC{ c0[0].re + prod.re, rzero };
}
}

static void whpr2_core(
    char uplo,
    std::ptrdiff_t n,
    const TC *alpha_,
    const TC *x, std::ptrdiff_t incx,
    const TC *y, std::ptrdiff_t incy,
    TC *ap)
{
    const TC alpha = *alpha_;
    const char UPLO = up(&uplo);

    if (n == 0 || ceq0(alpha)) return;

    /* Gather strided x,y into contiguous scratch once (O(N), handles negative
     * incx/incy) so the column kernel is always unit-stride -- ap is already
     * packed-contiguous, so only x,y need gathering. This unifies the strided
     * and contiguous paths and lets the strided case thread like the
     * contiguous one (mirrors the wher2 twin). */
    std::vector<TC> xg, yg;
    const TC *xp = x, *yp = y;
    if (incx != 1 || incy != 1) {
        xg.resize(n); yg.resize(n);
        std::ptrdiff_t ix = (incx < 0) ? -(std::ptrdiff_t)(n - 1) * incx : 0;
        std::ptrdiff_t iy = (incy < 0) ? -(std::ptrdiff_t)(n - 1) * incy : 0;
        for (std::ptrdiff_t j = 0; j < n; ++j) {
            xg[j] = x[ix]; ix += incx;
            yg[j] = y[iy]; iy += incy;
        }
        xp = xg.data(); yp = yg.data();
    }

#ifdef _OPENMP
    const bool use_omp = (n >= WHPR2_OMP_MIN && blas_omp_available());
#endif
    if (UPLO == 'U') {
#ifdef _OPENMP
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (std::ptrdiff_t j = 0; j < n; ++j) {
            if (!ceq0(xp[j]) || !ceq0(yp[j]))
                whpr2_col_upper(j, cmul(alpha, cconj(yp[j])),
                                cconj(cmul(alpha, xp[j])), xp, yp, ap);
            else {
                const std::size_t kk = static_cast<std::size_t>(j) * (j + 1) / 2;
                ap[kk + j] = TC{ ap[kk + j].re, rzero };
            }
        }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (std::ptrdiff_t j = 0; j < n; ++j) {
            if (!ceq0(xp[j]) || !ceq0(yp[j]))
                whpr2_col_lower(j, n, cmul(alpha, cconj(yp[j])),
                                cconj(cmul(alpha, xp[j])), xp, yp, ap);
            else {
                const std::size_t kk = static_cast<std::size_t>(j) * n
                                     - static_cast<std::size_t>(j) * (j - 1) / 2;
                ap[kk] = TC{ ap[kk].re, rzero };
            }
        }
    }
}

extern "C" {
EPBLAS_FACADE_SPR2(whpr2, TC)
}
