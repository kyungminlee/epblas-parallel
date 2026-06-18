/* whpr2 — multifloats complex DD Hermitian packed rank-2 update.
 *   A := alpha*x*y^H + conj(alpha)*y*x^H + A
 *
 * Columns independent → OMP over j. The packed triangular output makes a
 * contiguous static block hand one thread the heavy triangle end (par caps
 * at ~2.3x on 4 cores); cyclic schedule(static,1) interleaves short and long
 * columns across the team, balancing the skew symmetrically for both UPLO
 * (mirrors the proven kind10 yhpr2). The off-diagonal packed-column run is a
 * SIMD rank-2 AXPY (mf_tri::caxpy2_add, two rank-1 passes -> within DD fuzz
 * tol); the Hermitian diagonal is forced real.
 */

#include <cstddef>
#include <cctype>
#include <vector>
#include <multifloats.h>
#include "mf_tri_simd.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
#define WHPR2_OMP_MIN 64
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
const R rzero{0.0, 0.0};
inline bool dd_iszero(const R &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool cdd_iszero(const T &x) { return dd_iszero(x.re) && dd_iszero(x.im); }
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }

/* Per-column rank-2 update; off-diagonal packed run is a SIMD rank-2 AXPY (two
 * rank-1 passes). The Hermitian diagonal is forced real. */
inline void whpr2_col_upper(int j, T t1, T t2, const T *x, const T *y, T *ap) {
    T *c = ap + static_cast<std::size_t>(j) * (j + 1) / 2;
    mf_tri::caxpy2_add(j, c, x, t1, y, t2);
    const T prod = cadd(cmul(x[j], t1), cmul(y[j], t2));
    c[j] = T{ c[j].re + prod.re, rzero };
}

inline void whpr2_col_lower(int j, int N, T t1, T t2, const T *x, const T *y, T *ap) {
    T *c0 = ap + (static_cast<std::size_t>(j) * N - static_cast<std::size_t>(j) * (j - 1) / 2);
    mf_tri::caxpy2_add(N - j - 1, c0 + 1, x + j + 1, t1, y + j + 1, t2);
    const T prod = cadd(cmul(x[j], t1), cmul(y[j], t2));
    c0[0] = T{ c0[0].re + prod.re, rzero };
}
}

extern "C" void whpr2_(
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

    if (N == 0 || cdd_iszero(alpha)) return;

    /* Gather strided x,y into contiguous scratch once (O(N), handles negative
     * incx/incy) so the column kernel is always unit-stride -- ap is already
     * packed-contiguous, so only x,y need gathering. This unifies the strided
     * and contiguous paths and lets the strided case thread like the
     * contiguous one (mirrors the wher2 twin). */
    std::vector<T> xg, yg;
    const T *xp = x, *yp = y;
    if (incx != 1 || incy != 1) {
        xg.resize(N); yg.resize(N);
        std::ptrdiff_t ix = (incx < 0) ? -(std::ptrdiff_t)(N - 1) * incx : 0;
        std::ptrdiff_t iy = (incy < 0) ? -(std::ptrdiff_t)(N - 1) * incy : 0;
        for (int j = 0; j < N; ++j) {
            xg[j] = x[ix]; ix += incx;
            yg[j] = y[iy]; iy += incy;
        }
        xp = xg.data(); yp = yg.data();
    }

#ifdef _OPENMP
    const int use_omp = (N >= WHPR2_OMP_MIN && blas_omp_max_threads() > 1);
#endif
    if (UPLO == 'U') {
#ifdef _OPENMP
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (int j = 0; j < N; ++j) {
            if (!cdd_iszero(xp[j]) || !cdd_iszero(yp[j]))
                whpr2_col_upper(j, cmul(alpha, cconj(yp[j])),
                                cconj(cmul(alpha, xp[j])), xp, yp, ap);
            else {
                const std::size_t kk = static_cast<std::size_t>(j) * (j + 1) / 2;
                ap[kk + j] = T{ ap[kk + j].re, rzero };
            }
        }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (int j = 0; j < N; ++j) {
            if (!cdd_iszero(xp[j]) || !cdd_iszero(yp[j]))
                whpr2_col_lower(j, N, cmul(alpha, cconj(yp[j])),
                                cconj(cmul(alpha, xp[j])), xp, yp, ap);
            else {
                const std::size_t kk = static_cast<std::size_t>(j) * N
                                     - static_cast<std::size_t>(j) * (j - 1) / 2;
                ap[kk] = T{ ap[kk].re, rzero };
            }
        }
    }
}
