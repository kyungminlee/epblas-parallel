/*
 * wsyrk_ — multifloats complex (DD) symmetric rank-k update, public Fortran
 * entry. THREADING ORCHESTRATION ONLY: all the math lives in wsyrk_serial.cpp
 * (SIMD + scalar diagonal kernels, block policy, the per-block worker and the
 * per-column triangle scaler), shared through wsyrk_kernel.h.
 *
 *   C := α·A·Aᵀ + β·C   (TR='N')      C := α·Aᵀ·A + β·C   (TR='T')
 *
 * One `omp parallel for` over the jc BLOCK loop (schedule(dynamic,1)). Each
 * diagonal block owns a disjoint set of C columns, so its diagonal update and
 * trailing gemm write disjoint regions — race-free and bitwise-identical to
 * the serial sweep. The alpha==0 / K==0 early exit fans the per-column triangle
 * scale (schedule(static)). Delegates to wsyrk_serial when nested.
 */
#include "wsyrk_kernel.h"
#include "mf_util.h"
#include "mf_pred.h"
#include "../common/epblas_facade.h"
#include <cstddef>
#include <cctype>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;
using mf_pred::ceq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
}  // namespace

static void wsyrk_core(
    char uplo, char trans,
    std::ptrdiff_t N, std::ptrdiff_t K,
    const T *alpha_,
    const T *a, std::ptrdiff_t lda,
    const T *beta_,
    T *c, std::ptrdiff_t ldc)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        wsyrk_serial(uplo, trans, N, K, alpha_, a, lda, beta_, c, ldc);
        return;
    }
#endif
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(&uplo);
    const char TR = up(&trans);

    if (N == 0) return;

    if (ceq0(alpha) || K == 0) {
        if (ceq1(beta)) return;
#ifdef _OPENMP
        const bool use_omp = (N >= WSYRK_OMP_MIN && blas_omp_available());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (std::ptrdiff_t j = 0; j < N; ++j) wsyrk_scale_col(j, N, UPLO, beta, c, ldc);
        return;
    }

    const std::ptrdiff_t nb = wsyrk_block_nb();

    std::ptrdiff_t pw = nb;
#ifdef _OPENMP
    const std::ptrdiff_t nthreads = blas_omp_max_threads();
    const bool use_omp = (N >= WSYRK_OMP_MIN && nthreads > 1);
    /* Shrink the block step so the team gets ~2·nthreads panels at small N
     * (N=64, nb=32 -> 2 blocks -> idle threads). Triangular C output makes the
     * per-block work uneven, so oversubscribe for dynamic balance -> ppt=2. */
    if (use_omp) pw = (std::ptrdiff_t)blas_omp_panel_width(N, nthreads, nb, 2);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (std::ptrdiff_t jc = 0; jc < N; jc += pw) {
        const std::ptrdiff_t jb = (N - jc < pw) ? (N - jc) : pw;
        wsyrk_block(jc, jb, N, K, UPLO, TR, alpha, beta, a, lda, c, ldc);
    }
}

extern "C" {
EPBLAS_FACADE_SYRK(wsyrk, T, T)
}
