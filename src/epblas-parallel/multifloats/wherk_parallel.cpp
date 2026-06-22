/*
 * wherk_ — multifloats complex (DD) Hermitian rank-k update, public Fortran
 * entry. THREADING ORCHESTRATION ONLY: all the math lives in wherk_serial.cpp
 * (SIMD + scalar diagonal kernels, block policy, the per-block worker, the
 * per-column triangle scaler and the diagonal-imaginary zeroer), shared
 * through wherk_kernel.h.
 *
 *   C := α·A·Aᴴ + β·C   (TR_c='N')      C := α·Aᴴ·A + β·C   (TR_c='C')
 *   alpha/beta REAL, A/C complex, the diagonal of C stays real.
 *
 * One `omp parallel for` over the jc BLOCK loop (schedule(dynamic,1)). Each
 * diagonal block owns a disjoint set of C columns, so its diagonal update and
 * trailing gemm write disjoint regions — race-free and bitwise-identical to
 * the serial sweep. The alpha==0 / K==0 early exit either zeroes the diagonal
 * imaginary parts (beta==1, serial) or fans the per-column triangle scale
 * (schedule(static)). Delegates to wherk_serial when nested.
 */
#include "wherk_kernel.h"
#include "../common/epblas_facade.h"
#include "mf_util.h"
#include "mf_pred.h"
#include <cstddef>
#include <cctype>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using TR = mf::float64x2;
using TC = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
using mf_pred::eq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
}  // namespace

static void wherk_core(
    char uplo, char trans,
    std::ptrdiff_t n, std::ptrdiff_t k,
    const TR *alpha_,
    const TC *a, std::ptrdiff_t lda,
    const TR *beta_,
    TC *c, std::ptrdiff_t ldc)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        wherk_serial(uplo, trans, n, k, alpha_, a, lda, beta_, c, ldc);
        return;
    }
#endif
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO = up(&uplo);
    const char TR_c = up(&trans);

    if (n == 0) return;

    if (eq0(alpha) || k == 0) {
        if (eq1(beta)) {
            for (std::ptrdiff_t j = 0; j < n; ++j) wherk_zero_diag_im(j, c, ldc);
            return;
        }
#ifdef _OPENMP
        const bool use_omp = (n >= WHERK_OMP_MIN && blas_omp_available());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (std::ptrdiff_t j = 0; j < n; ++j) wherk_scale_col(j, n, UPLO, beta, c, ldc);
        return;
    }

    const std::ptrdiff_t nb = wherk_block_nb();

#ifdef _OPENMP
    const bool use_omp = (n >= WHERK_OMP_MIN && blas_omp_available());
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (std::ptrdiff_t jc = 0; jc < n; jc += nb) {
        const std::ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
        wherk_block(jc, jb, n, k, UPLO, TR_c, alpha, beta, a, lda, c, ldc);
    }
}

extern "C" {
EPBLAS_FACADE_SYRK(wherk, TR, TC)
}
