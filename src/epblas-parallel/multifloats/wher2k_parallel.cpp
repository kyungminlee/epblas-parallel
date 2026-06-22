/*
 * wher2k_ — multifloats complex (DD) Hermitian rank-2k update, public Fortran
 * entry. THREADING ORCHESTRATION ONLY: all the math lives in wher2k_serial.cpp
 * (SIMD + scalar diagonal kernels, block policy, the per-block worker, the
 * per-column triangle scaler and the diagonal-imaginary zeroer), shared
 * through wher2k_kernel.h.
 *
 *   C := alpha · A · Bᴴ + conj(alpha) · B · Aᴴ + beta · C  (TR='N')
 *   C := alpha · Aᴴ · B + conj(alpha) · Bᴴ · A + beta · C  (TR='C')
 *   alpha complex, beta real, A/B/C complex, the diagonal of C stays real.
 *
 * One `omp parallel for` over the jc BLOCK loop (schedule(dynamic,1)). Each
 * diagonal block owns a disjoint set of C columns, so its diagonal update and
 * two trailing gemm calls write disjoint regions — race-free and bitwise-
 * identical to the serial sweep. The alpha==0 / K==0 early exit either zeroes
 * the diagonal imaginary parts (beta==1, serial) or fans the per-column
 * triangle scale (schedule(static)). Delegates to wher2k_serial when nested.
 */
#include "wher2k_kernel.h"
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

extern "C" void wher2k_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const R *beta_,
    T *c, const int *ldc_,
    std::size_t uplo_len, std::size_t trans_len)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        wher2k_serial(uplo, trans, n_, k_, alpha_, a, lda_, b, ldb_, beta_,
                      c, ldc_, uplo_len, trans_len);
        return;
    }
#endif
    (void)uplo_len; (void)trans_len;
    const std::ptrdiff_t N = *n_, K = *k_;
    const std::ptrdiff_t lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_;
    const R beta  = *beta_;
    const char UPLO = up(uplo);
    const char TR_c = up(trans);
    (void)lda; (void)ldb;

    if (N == 0) return;

    if ((eq0(alpha.re) && eq0(alpha.im)) || K == 0) {
        if (eq1(beta)) {
            for (std::ptrdiff_t j = 0; j < N; ++j) wher2k_zero_diag_im(j, c, ldc);
            return;
        }
#ifdef _OPENMP
        const bool use_omp = (N >= WHER2K_OMP_MIN && blas_omp_available());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (std::ptrdiff_t j = 0; j < N; ++j) wher2k_scale_col(j, N, UPLO, beta, c, ldc);
        return;
    }

    const std::ptrdiff_t nb = wher2k_block_nb();

#ifdef _OPENMP
    const bool use_omp = (N >= WHER2K_OMP_MIN && blas_omp_available());
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (std::ptrdiff_t jc = 0; jc < N; jc += nb) {
        const std::ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
        wher2k_block(jc, jb, N, K, UPLO, TR_c, alpha, beta, a, lda, b, ldb, c, ldc);
    }
}
