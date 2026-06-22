/* msyr — multifloats real DD symmetric rank-1 update.
 *   A := alpha*x*x^T + A
 *
 * x is gathered+split once into SoA limb arrays (xh/xl); this both makes
 * the strided (incx != 1) case unit-stride from then on and feeds the
 * 4-wide AVX2 SoA double-double axpy kernel (mf_kernels::dd_axpy) that is
 * bit-identical to the scalar operators. Full storage → column j is the
 * contiguous run &A_(0,j); OMP over columns (lda apart, no false sharing).
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
using T = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
#define MSYR_OMP_MIN 64
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

static void msyr_core(
    char uplo,
    std::ptrdiff_t N,
    const T *alpha_,
    const T *x, std::ptrdiff_t incx,
    T *a, std::ptrdiff_t lda)
{
    const T alpha = *alpha_;
    const char UPLO = up(&uplo);

    if (N == 0 || eq0(alpha.limbs[0], alpha.limbs[1])) return;

    /* Gather x in logical order 0..N-1 and split into SoA limbs. O(N); also
     * the strided->contiguous fix (only x is strided — A is full storage). */
    std::vector<double> xh(N), xl(N);
    {
        std::ptrdiff_t ix = (incx < 0) ? -(std::ptrdiff_t)(N - 1) * incx : 0;
        for (std::ptrdiff_t j = 0; j < N; ++j) {
            xh[j] = x[ix].limbs[0];
            xl[j] = x[ix].limbs[1];
            ix += incx;
        }
    }
    const double *xhp = xh.data();
    const double *xlp = xl.data();

#ifdef _OPENMP
    const bool use_omp = (N >= MSYR_OMP_MIN && blas_omp_available());
    /* static,1 cyclic interleave balances the triangular column skew (column
     * j writes j+1 (U) / N-j (L) elems); full storage → columns lda apart, no
     * false sharing. The hot inner loop is mf_kernels::dd_axpy, so the per-
     * column UPLO branch is negligible and the old serial-codegen concern
     * (the if()-clause outlining the inner loop) no longer applies. */
#endif
    if (UPLO == 'L') {
#ifdef _OPENMP
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (std::ptrdiff_t j = 0; j < N; ++j) {
            if (eq0(xhp[j], xlp[j])) continue;
            const T t = alpha * T{xhp[j], xlp[j]};
            mf_kernels::dd_axpy(N - j, xhp + j, xlp + j, t.limbs[0], t.limbs[1], &A_(j, j));
        }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (std::ptrdiff_t j = 0; j < N; ++j) {
            if (eq0(xhp[j], xlp[j])) continue;
            const T t = alpha * T{xhp[j], xlp[j]};
            mf_kernels::dd_axpy(j + 1, xhp, xlp, t.limbs[0], t.limbs[1], &A_(0, j));
        }
    }
}

extern "C" {
EPBLAS_FACADE_SYR(msyr, T, T)
}

#undef A_
