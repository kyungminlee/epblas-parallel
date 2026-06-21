/*
 * wsyr2k_ — multifloats complex (DD) symmetric rank-2k update, public Fortran
 * entry. THREADING ORCHESTRATION ONLY: all the math lives in wsyr2k_serial.cpp
 * (SIMD + scalar diagonal kernels, block policy, the per-block worker and the
 * per-column triangle scaler), shared through wsyr2k_kernel.h.
 *
 *   C := alpha · (A · Bᵀ + B · Aᵀ) + beta · C        (TR='N')
 *   C := alpha · (Aᵀ · B + Bᵀ · A) + beta · C        (TR='T'/'C')
 *
 * One `omp parallel for` over the jc BLOCK loop (schedule(dynamic,1)). Each
 * diagonal block owns a disjoint set of C columns, so its diagonal update and
 * two trailing gemm calls write disjoint regions — race-free and bitwise-
 * identical to the serial sweep. The alpha==0 / K==0 early exit fans the
 * per-column triangle scale (schedule(static)). Delegates to wsyr2k_serial
 * when nested.
 */
#include "wsyr2k_kernel.h"
#include "mf_util.h"
#include "mf_pred.h"
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

extern "C" void wsyr2k_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    std::size_t uplo_len, std::size_t trans_len)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        wsyr2k_serial(uplo, trans, n_, k_, alpha_, a, lda_, b, ldb_, beta_,
                      c, ldc_, uplo_len, trans_len);
        return;
    }
#endif
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
    (void)lda; (void)ldb;

    if (N == 0) return;

    if (ceq0(alpha) || K == 0) {
        if (ceq1(beta)) return;
#ifdef _OPENMP
        const bool use_omp = (N >= WSYR2K_OMP_MIN && blas_omp_available());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) wsyr2k_scale_col(j, N, UPLO, beta, c, ldc);
        return;
    }

    const int nb = wsyr2k_block_nb();

#ifdef _OPENMP
    const bool use_omp = (N >= WSYR2K_OMP_MIN && blas_omp_available());
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        wsyr2k_block(jc, jb, N, K, UPLO, TR, alpha, beta, a, lda, b, ldb, c, ldc);
    }
}
