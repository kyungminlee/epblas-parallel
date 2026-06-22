/* mspr — multifloats real DD symmetric packed rank-1 update.
 *   A := alpha*x*x^T + A
 *
 * x is gathered+split once into SoA limb arrays (xh/xl); this both makes
 * the strided (incx != 1) case unit-stride from then on — killing the old
 * strided-lower gap to the reference — and feeds the 4-wide AVX2 SoA
 * double-double axpy kernel (mf_kernels::dd_axpy) that is bit-identical to
 * the scalar operators. OMP over columns j (independent in packed storage).
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
#define MSPR_OMP_MIN 64
}

extern "C" void mspr_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *x, const int *incx_,
    T *ap,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const std::ptrdiff_t N = *n_;
    const std::ptrdiff_t incx = *incx_;
    const T alpha = *alpha_;
    const char UPLO = up(uplo);

    if (N == 0 || eq0(alpha.limbs[0], alpha.limbs[1])) return;

    /* Gather x in logical order 0..N-1 and split into SoA limbs. O(N); also
     * the strided->contiguous fix (only x is strided — ap is always packed
     * contiguous). */
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
    const std::ptrdiff_t use_omp = (N >= MSPR_OMP_MIN && blas_omp_available());
    /* static,8: column j writes j+1 (U) / N-j (L) packed elems — a triangular
     * skew that plain static dumps onto one thread. Packed columns are
     * contiguous in ap, so cyclic static,1 would false-share cache lines on
     * this light real rank-1 write. Chunk-8 balances the skew while keeping
     * each thread's run local (mirrors the espr twin). */
#endif
    if (UPLO == 'U') {
#ifdef _OPENMP
        #pragma omp parallel for if(use_omp) schedule(static, 8)
#endif
        for (std::ptrdiff_t j = 0; j < N; ++j) {
            if (eq0(xhp[j], xlp[j])) continue;
            const T tmp = alpha * T{xhp[j], xlp[j]};
            const std::ptrdiff_t kk = (j * (j + 1)) / 2;
            mf_kernels::dd_axpy(j + 1, xhp, xlp, tmp.limbs[0], tmp.limbs[1], &ap[kk]);
        }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for if(use_omp) schedule(static, 8)
#endif
        for (std::ptrdiff_t j = 0; j < N; ++j) {
            if (eq0(xhp[j], xlp[j])) continue;
            const T tmp = alpha * T{xhp[j], xlp[j]};
            const std::ptrdiff_t kk = j * N - (j * (j - 1)) / 2;
            mf_kernels::dd_axpy(N - j, xhp + j, xlp + j, tmp.limbs[0], tmp.limbs[1], &ap[kk]);
        }
    }
}
