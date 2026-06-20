/*
 * wherk_ — multifloats complex (DD) Hermitian rank-k update, public Fortran
 * entry. THREADING ORCHESTRATION ONLY: all the math lives in wherk_serial.cpp
 * (SIMD + scalar diagonal kernels, block policy, the per-block worker, the
 * per-column triangle scaler and the diagonal-imaginary zeroer), shared
 * through wherk_kernel.h.
 *
 *   C := α·A·Aᴴ + β·C   (TR='N')      C := α·Aᴴ·A + β·C   (TR='C')
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
using mf_pred::eq0;
using mf_pred::eq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
}  // namespace

extern "C" void wherk_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const R *alpha_,
    const T *a, const int *lda_,
    const R *beta_,
    T *c, const int *ldc_,
    std::size_t uplo_len, std::size_t trans_len)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        wherk_serial(uplo, trans, n_, k_, alpha_, a, lda_, beta_, c, ldc_,
                     uplo_len, trans_len);
        return;
    }
#endif
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldc = *ldc_;
    const R alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);
    const char TR_c = up(trans);

    if (N == 0) return;

    if (eq0(alpha) || K == 0) {
        if (eq1(beta)) {
            for (int j = 0; j < N; ++j) wherk_zero_diag_im(j, c, ldc);
            return;
        }
#ifdef _OPENMP
        const bool use_omp = (N >= WHERK_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) wherk_scale_col(j, N, UPLO, beta, c, ldc);
        return;
    }

    const int nb = wherk_block_nb();

#ifdef _OPENMP
    const bool use_omp = (N >= WHERK_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        wherk_block(jc, jb, N, K, UPLO, TR_c, alpha, beta, a, lda, c, ldc);
    }
}
