/* msyr2 — multifloats real DD symmetric rank-2 update.
 *   A := alpha*x*y^T + alpha*y*x^T + A
 *
 * x and y are gathered+split once into SoA limb arrays; this makes the
 * strided case unit-stride and feeds the fused 4-wide AVX2 SoA DD rank-2
 * axpy kernel (mf_kernels::dd_axpy2), bit-identical to the scalar operators
 * (the fused ap=(ap+x*t1)+y*t2 matches the reference left-associatively).
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

namespace mf = multifloats;
using T = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
#define MSYR2_OMP_MIN 64
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

extern "C" void msyr2_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *x, const int *incx_,
    const T *y, const int *incy_,
    T *a, const int *lda_,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, incy = *incy_, lda = *lda_;
    const T alpha = *alpha_;
    const char UPLO = up(uplo);

    if (N == 0 || eq0(alpha.limbs[0], alpha.limbs[1])) return;

    /* Gather x,y in logical order 0..N-1 and split into SoA limbs. O(N);
     * also the strided->contiguous fix (A is full storage / contiguous). */
    std::vector<double> xh(N), xl(N), yh(N), yl(N);
    {
        std::ptrdiff_t ix = (incx < 0) ? -(std::ptrdiff_t)(N - 1) * incx : 0;
        std::ptrdiff_t iy = (incy < 0) ? -(std::ptrdiff_t)(N - 1) * incy : 0;
        for (int j = 0; j < N; ++j) {
            xh[j] = x[ix].limbs[0]; xl[j] = x[ix].limbs[1]; ix += incx;
            yh[j] = y[iy].limbs[0]; yl[j] = y[iy].limbs[1]; iy += incy;
        }
    }
    const double *xhp = xh.data(), *xlp = xl.data();
    const double *yhp = yh.data(), *ylp = yl.data();

#ifdef _OPENMP
    const int use_omp = (N >= MSYR2_OMP_MIN && blas_omp_max_threads() > 1);
    /* static,1 balances the triangular column skew; full storage → columns
     * lda apart, no false sharing. Hot loop is mf_kernels::dd_axpy2. */
#endif
    if (UPLO == 'L') {
#ifdef _OPENMP
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (int j = 0; j < N; ++j) {
            if (eq0(xhp[j], xlp[j]) && eq0(yhp[j], ylp[j])) continue;
            const T tx = alpha * T{yhp[j], ylp[j]};   /* x-row scale = alpha*y[j] */
            const T ty = alpha * T{xhp[j], xlp[j]};   /* y-row scale = alpha*x[j] */
            mf_kernels::dd_axpy2(N - j, xhp + j, xlp + j, tx.limbs[0], tx.limbs[1],
                               yhp + j, ylp + j, ty.limbs[0], ty.limbs[1], &A_(j, j));
        }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (int j = 0; j < N; ++j) {
            if (eq0(xhp[j], xlp[j]) && eq0(yhp[j], ylp[j])) continue;
            const T tx = alpha * T{yhp[j], ylp[j]};
            const T ty = alpha * T{xhp[j], xlp[j]};
            mf_kernels::dd_axpy2(j + 1, xhp, xlp, tx.limbs[0], tx.limbs[1],
                               yhp, ylp, ty.limbs[0], ty.limbs[1], &A_(0, j));
        }
    }
}

#undef A_
