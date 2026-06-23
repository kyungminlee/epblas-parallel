/*
 * msyr2k_ — multifloats real (DD) symmetric rank-2k update, public Fortran
 * entry. THREADING ORCHESTRATION ONLY: all the math lives in msyr2k_serial.cpp
 * (SIMD + scalar diagonal kernels, block policy, the per-block worker and the
 * per-column triangle scaler), shared through msyr2k_kernel.h.
 *
 *   C := alpha · (A · Bᵀ + B · Aᵀ) + beta · C        (TRANS='N')
 *   C := alpha · (Aᵀ · B + Bᵀ · A) + beta · C        (TRANS='T'/'C')
 *
 * One `omp parallel for` over the jc BLOCK loop (schedule(dynamic,1)). Each
 * diagonal block owns a disjoint set of C columns, so its diagonal update and
 * two trailing gemm calls write disjoint regions — race-free and bitwise-
 * identical to the serial sweep. The alpha==0 / K==0 early exit fans the
 * per-column triangle scale (schedule(static)). Delegates to msyr2k_serial
 * when nested.
 */
#include "msyr2k_kernel.h"
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


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
using mf_pred::eq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
}  // namespace

static void msyr2k_core(
    char uplo, char trans,
    std::ptrdiff_t n, std::ptrdiff_t k,
    const TR *alpha_,
    const TR *a, std::ptrdiff_t lda,
    const TR *b, std::ptrdiff_t ldb,
    const TR *beta_,
    TR *c, std::ptrdiff_t ldc)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        msyr2k_serial(uplo, trans, n, k, alpha_, a, lda, b, ldb, beta_,
                      c, ldc);
        return;
    }
#endif
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO = up(&uplo);
    char TRANS = up(&trans);
    if (TRANS == 'C') TRANS = 'T';
    (void)lda; (void)ldb;

    if (n == 0) return;

    if (eq0(alpha) || k == 0) {
        if (eq1(beta)) return;
#ifdef _OPENMP
        const bool use_omp = (n >= MSYR2K_OMP_MIN && blas_omp_available());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (std::ptrdiff_t j = 0; j < n; ++j) msyr2k_scale_col(j, n, UPLO, beta, c, ldc);
        return;
    }

    const std::ptrdiff_t nb = msyr2k_block_nb();

#ifdef _OPENMP
    const bool use_omp = (n >= MSYR2K_OMP_MIN && blas_omp_available());
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (std::ptrdiff_t jc = 0; jc < n; jc += nb) {
        const std::ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
        msyr2k_block(jc, jb, n, k, UPLO, TRANS, alpha, beta, a, lda, b, ldb, c, ldc);
    }
}

extern "C" {
EPBLAS_FACADE_SYR2K(msyr2k, TR, TR, TR)
}
