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

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
using mf_pred::ceq0;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
#define WHPR_OMP_MIN 64
const R rzero{0.0, 0.0};
using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::cconj;
inline T scale_r(R const &alpha, T const &b) { return T{ alpha * b.re, alpha * b.im }; }
}

/* Contiguous (unit-stride x) core: packed Hermitian rank-1 A += alpha*x*x^H.
 * Off-diagonal packed-column run is a SIMD column-AXPY (caxpy_add, bit-exact;
 * cmul commutes so t=alpha*conj(x[j]) factors out); diagonal forced real. */
static void whpr_contig(char UPLO, int N, R alpha, T *ap, const T *x)
{
    if (UPLO == 'U') {
#ifdef _OPENMP
        const int use_omp = (N >= WHPR_OMP_MIN && blas_omp_max_threads() > 1);
        /* static,1: cyclic interleave balances the triangular packed-column skew;
         * complex DD rank-1 work per element dominates any false sharing. */
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (int j = 0; j < N; ++j) {
            const int kk = (j * (j + 1)) / 2;
            if (!ceq0(x[j])) {
                const T tmp = scale_r(alpha, cconj(x[j]));
                mf_kernels::caxpy_add(j, &ap[kk], &x[0], tmp);
                const R new_re = ap[kk + j].re + cmul(x[j], tmp).re;
                ap[kk + j] = T{ new_re, rzero };
            } else {
                ap[kk + j] = T{ ap[kk + j].re, rzero };
            }
        }
    } else {
#ifdef _OPENMP
        const int use_omp = (N >= WHPR_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (int j = 0; j < N; ++j) {
            const int kk = j * N - (j * (j - 1)) / 2;
            if (!ceq0(x[j])) {
                const T tmp = scale_r(alpha, cconj(x[j]));
                const R new_re = ap[kk].re + cmul(tmp, x[j]).re;
                ap[kk] = T{ new_re, rzero };
                mf_kernels::caxpy_add(N - (j + 1), &ap[kk + 1], &x[j + 1], tmp);
            } else {
                ap[kk] = T{ ap[kk].re, rzero };
            }
        }
    }
}

extern "C" void whpr_(
    const char *uplo,
    const int *n_,
    const R *alpha_,
    const T *x, const int *incx_,
    T *ap,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_;
    const R alpha = *alpha_;
    const char UPLO = up(uplo);

    if (N == 0 || eq0(alpha)) return;

    if (incx == 1) {
        whpr_contig(UPLO, N, alpha, ap, x);
        return;
    }
    /* Strided x: gather to unit-stride scratch, run the SIMD core. */
    const T *xbase = (incx < 0) ? x - static_cast<std::ptrdiff_t>(N - 1) * incx : x;
    std::vector<T> xs(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) xs[i] = xbase[static_cast<std::ptrdiff_t>(i) * incx];
    whpr_contig(UPLO, N, alpha, ap, xs.data());
}
