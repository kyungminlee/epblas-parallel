/* mgbmv — multifloats real DD general band matrix-vector multiply.
 *   y := alpha*A*x + beta*y  or  y := alpha*A^T*x + beta*y
 * Band storage: A(i,j) at AB[(ku + i - j) + j*lda].
 *
 * Reference algorithm + OMP over j on T-path only (N-path writes overlap).
 */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
#define MGBMV_OMP_MIN 64
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
const T zero_dd{0.0, 0.0};
const T one_dd{1.0, 0.0};
inline bool dd_iszero(const T &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_isone (const T &x) { return x.limbs[0] == 1.0 && x.limbs[1] == 0.0; }
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

extern "C" void mgbmv_(
    const char *trans,
    const int *m_, const int *n_,
    const int *kl_, const int *ku_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *x, const int *incx_,
    const T *beta_,
    T *y, const int *incy_,
    std::size_t trans_len)
{
    (void)trans_len;
    const int M = *m_, N = *n_;
    const int KL = *kl_, KU = *ku_;
    const int lda = *lda_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    char TR = up(trans);
    if (TR == 'C') TR = 'T';

    if (M == 0 || N == 0 || (dd_iszero(alpha) && dd_isone(beta))) return;

    const int leny = (TR == 'N') ? M : N;
    const int lenx = (TR == 'N') ? N : M;

    if (!dd_isone(beta)) {
        int iy = (incy < 0) ? -(leny - 1) * incy : 0;
        if (dd_iszero(beta)) {
            for (int i = 0; i < leny; ++i) { y[iy] = zero_dd; iy += incy; }
        } else {
            for (int i = 0; i < leny; ++i) { y[iy] = beta * y[iy]; iy += incy; }
        }
    }
    if (dd_iszero(alpha)) return;

    if (TR == 'N' && incx == 1 && incy == 1) {
        for (int j = 0; j < N; ++j) {
            const T tmp = alpha * x[j];
            const int i_lo = (j - KU > 0) ? (j - KU) : 0;
            const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
            const int k = KU - j;
            for (int i = i_lo; i < i_hi; ++i) y[i] = y[i] + tmp * A_(k + i, j);
        }
    } else if (TR != 'N' && incx == 1 && incy == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= MGBMV_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            T s = zero_dd;
            const int i_lo = (j - KU > 0) ? (j - KU) : 0;
            const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
            const int k = KU - j;
            for (int i = i_lo; i < i_hi; ++i) s = s + A_(k + i, j) * x[i];
            y[j] = y[j] + alpha * s;
        }
    } else {
        int kx = (incx < 0) ? -(lenx - 1) * incx : 0;
        int ky = (incy < 0) ? -(leny - 1) * incy : 0;
        if (TR == 'N') {
            int jx = kx;
            for (int j = 0; j < N; ++j) {
                const T tmp = alpha * x[jx];
                int iy = ky;
                const int i_lo = (j - KU > 0) ? (j - KU) : 0;
                const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
                const int k = KU - j;
                for (int i = i_lo; i < i_hi; ++i) {
                    y[iy] = y[iy] + tmp * A_(k + i, j);
                    iy += incy;
                }
                jx += incx;
                if (j >= KU) ky += incy;
            }
        } else {
            int jy = ky;
            for (int j = 0; j < N; ++j) {
                T s = zero_dd;
                int ix = kx;
                const int i_lo = (j - KU > 0) ? (j - KU) : 0;
                const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
                const int k = KU - j;
                for (int i = i_lo; i < i_hi; ++i) {
                    s = s + A_(k + i, j) * x[ix];
                    ix += incx;
                }
                y[jy] = y[jy] + alpha * s;
                jy += incy;
                if (j >= KU) kx += incx;
            }
        }
    }
}

#undef A_
