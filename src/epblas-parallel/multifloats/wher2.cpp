/* wher2 — multifloats Hermitian rank-2 update (alpha complex, diag real).
 *   A := alpha*x*y^H + conj(alpha)*y*x^H + A
 *
 * Columns independent -> OMP over j. The triangular output makes a contiguous
 * static block hand one thread the heavy triangle end (par caps at ~2.3x on 4
 * cores); cyclic schedule(static,1) interleaves short and long columns across
 * the team, balancing the skew symmetrically for both UPLO (mirrors the proven
 * kind10 yher2/whpr2). The off-diagonal column run is a SIMD rank-2 AXPY
 * (mf_kernels::caxpy2_add, two rank-1 passes -> within DD fuzz tol); the diagonal
 * is forced real.
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
using T = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
#define WHER2_OMP_MIN 64
const R rzero{0.0, 0.0};
using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::cconj;

/* Per-column rank-2 update. aj = &A(0,j); off-diagonal run is a SIMD rank-2
 * AXPY (two rank-1 passes); the Hermitian diagonal is forced real. */
inline void wher2_col_upper(std::ptrdiff_t j, T t1, T t2, const T *x, const T *y, T *aj) {
    mf_kernels::caxpy2_add(j, aj, x, t1, y, t2);
    const T prod = cadd(cmul(x[j], t1), cmul(y[j], t2));
    aj[j] = T{ aj[j].re + prod.re, rzero };
}

inline void wher2_col_lower(std::ptrdiff_t j, std::ptrdiff_t N, T t1, T t2, const T *x, const T *y, T *aj) {
    const T prod = cadd(cmul(x[j], t1), cmul(y[j], t2));
    aj[j] = T{ aj[j].re + prod.re, rzero };
    mf_kernels::caxpy2_add(N - (j + 1), aj + j + 1, x + j + 1, t1, y + j + 1, t2);
}
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

static void wher2_core(
    char uplo,
    std::ptrdiff_t N,
    const T *alpha_,
    const T *x, std::ptrdiff_t incx,
    const T *y, std::ptrdiff_t incy,
    T *a, std::ptrdiff_t lda)
{
    const T alpha = *alpha_;
    const char UPLO = up(&uplo);

    if (N == 0 || ceq0(alpha)) return;

    /* Gather strided x,y into contiguous scratch once (O(N), handles negative
     * incx/incy) so the column kernel is always unit-stride -- this both feeds
     * the tight noinline helpers and lets the strided case thread like the
     * contiguous one (the per-element kx+i*incx recompute otherwise left the
     * dense strided-LOWER path ~2-3% behind ob and unthreaded). A is always
     * contiguous-by-column, so only x,y need gathering. */
    std::vector<T> xg, yg;
    const T *xp = x, *yp = y;
    if (incx != 1 || incy != 1) {
        xg.resize(N); yg.resize(N);
        std::ptrdiff_t ix = (incx < 0) ? -(std::ptrdiff_t)(N - 1) * incx : 0;
        std::ptrdiff_t iy = (incy < 0) ? -(std::ptrdiff_t)(N - 1) * incy : 0;
        for (std::ptrdiff_t j = 0; j < N; ++j) {
            xg[j] = x[ix]; ix += incx;
            yg[j] = y[iy]; iy += incy;
        }
        xp = xg.data(); yp = yg.data();
    }

#ifdef _OPENMP
    const bool use_omp = (N >= WHER2_OMP_MIN && blas_omp_available());
#endif
    if (UPLO == 'L') {
#ifdef _OPENMP
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (std::ptrdiff_t j = 0; j < N; ++j) {
            if (!ceq0(xp[j]) || !ceq0(yp[j]))
                wher2_col_lower(j, N, cmul(alpha, cconj(yp[j])),
                                cconj(cmul(alpha, xp[j])), xp, yp, &A_(0, j));
            else {
                T *aj = &A_(0, j);
                aj[j] = T{ aj[j].re, rzero };
            }
        }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (std::ptrdiff_t j = 0; j < N; ++j) {
            if (!ceq0(xp[j]) || !ceq0(yp[j]))
                wher2_col_upper(j, cmul(alpha, cconj(yp[j])),
                                cconj(cmul(alpha, xp[j])), xp, yp, &A_(0, j));
            else {
                T *aj = &A_(0, j);
                aj[j] = T{ aj[j].re, rzero };
            }
        }
    }
}

extern "C" {
EPBLAS_FACADE_SYR2(wher2, T)
}

#undef A_
