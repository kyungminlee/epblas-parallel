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
#include <cstddef>
#include <cctype>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
inline bool dd_iszero(R x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_isone (R x) { return x.limbs[0] == 1.0 && x.limbs[1] == 0.0; }
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
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_;
    const R beta  = *beta_;
    const char UPLO = up(uplo);
    const char TR_c = up(trans);
    (void)lda; (void)ldb;

    if (N == 0) return;

    if ((dd_iszero(alpha.re) && dd_iszero(alpha.im)) || K == 0) {
        if (dd_isone(beta)) {
            for (int j = 0; j < N; ++j) wher2k_zero_diag_im(j, c, ldc);
            return;
        }
#ifdef _OPENMP
        const bool use_omp = (N >= WHER2K_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) wher2k_scale_col(j, N, UPLO, beta, c, ldc);
        return;
    }

    const int nb = wher2k_block_nb();

#ifdef _OPENMP
    const bool use_omp = (N >= WHER2K_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        wher2k_block(jc, jb, N, K, UPLO, TR_c, alpha, beta, a, lda, b, ldb, c, ldc);
    }
}
