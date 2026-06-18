/* mspr2 — multifloats real DD symmetric packed rank-2 update.
 *   A := alpha*x*y^T + alpha*y*x^T + A
 *
 * x and y are gathered+split once into SoA limb arrays (the strided->
 * contiguous fix — ap is always packed contiguous) and fed to the fused
 * 4-wide AVX2 SoA DD rank-2 axpy kernel (mf_rank1::dd_axpy2), bit-identical
 * to the scalar operators. OMP over columns j (independent in packed storage).
 */

#include <cstddef>
#include <cctype>
#include <vector>
#include <multifloats.h>
#include "mf_rank1_simd.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
#define MSPR2_OMP_MIN 64
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
inline bool dd_iszero(double h, double l) { return h == 0.0 && l == 0.0; }
}

extern "C" void mspr2_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *x, const int *incx_,
    const T *y, const int *incy_,
    T *ap,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, incy = *incy_;
    const T alpha = *alpha_;
    const char UPLO = up(uplo);

    if (N == 0 || dd_iszero(alpha.limbs[0], alpha.limbs[1])) return;

    /* Gather x,y in logical order 0..N-1 and split into SoA limbs. O(N). */
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
    const int use_omp = (N >= MSPR2_OMP_MIN && blas_omp_max_threads() > 1);
    /* static,8: packed columns are contiguous in ap, so cyclic static,1 would
     * false-share cache lines; chunk-8 balances the triangular skew while
     * keeping each thread's run local (mirrors the mspr/espr2 twins). */
#endif
    if (UPLO == 'U') {
#ifdef _OPENMP
        #pragma omp parallel for if(use_omp) schedule(static, 8)
#endif
        for (int j = 0; j < N; ++j) {
            if (dd_iszero(xhp[j], xlp[j]) && dd_iszero(yhp[j], ylp[j])) continue;
            const T t1 = alpha * T{yhp[j], ylp[j]};
            const T t2 = alpha * T{xhp[j], xlp[j]};
            const int kk = (j * (j + 1)) / 2;
            mf_rank1::dd_axpy2(j + 1, xhp, xlp, t1.limbs[0], t1.limbs[1],
                               yhp, ylp, t2.limbs[0], t2.limbs[1], &ap[kk]);
        }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for if(use_omp) schedule(static, 8)
#endif
        for (int j = 0; j < N; ++j) {
            if (dd_iszero(xhp[j], xlp[j]) && dd_iszero(yhp[j], ylp[j])) continue;
            const T t1 = alpha * T{yhp[j], ylp[j]};
            const T t2 = alpha * T{xhp[j], xlp[j]};
            const int kk = j * N - (j * (j - 1)) / 2;
            mf_rank1::dd_axpy2(N - j, xhp + j, xlp + j, t1.limbs[0], t1.limbs[1],
                               yhp + j, ylp + j, t2.limbs[0], t2.limbs[1], &ap[kk]);
        }
    }
}
