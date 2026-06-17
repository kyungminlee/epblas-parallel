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
inline bool cdd_iszero(const T &x) {
    return x.re.limbs[0] == 0.0 && x.re.limbs[1] == 0.0
        && x.im.limbs[0] == 0.0 && x.im.limbs[1] == 0.0;
}
inline bool cdd_isone(const T &x) {
    return x.re.limbs[0] == 1.0 && x.re.limbs[1] == 0.0
        && x.im.limbs[0] == 0.0 && x.im.limbs[1] == 0.0;
}
}  // namespace

extern "C" void wsyrk_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *beta_,
    T *c, const int *ldc_,
    std::size_t uplo_len, std::size_t trans_len)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        wsyrk_serial(uplo, trans, n_, k_, alpha_, a, lda_, beta_, c, ldc_,
                     uplo_len, trans_len);
        return;
    }
#endif
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);
    const char TR = up(trans);

    if (N == 0) return;

    if (cdd_iszero(alpha) || K == 0) {
        if (cdd_isone(beta)) return;
#ifdef _OPENMP
        const bool use_omp = (N >= WSYRK_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) wsyrk_scale_col(j, N, UPLO, beta, c, ldc);
        return;
    }

    const int nb = wsyrk_block_nb();

    int pw = nb;
#ifdef _OPENMP
    const int nt = blas_omp_max_threads();
    const bool use_omp = (N >= WSYRK_OMP_MIN && nt > 1);
    /* Shrink the block step so the team gets ~2·nt panels at small N
     * (N=64, nb=32 -> 2 blocks -> idle threads). Triangular C output makes the
     * per-block work uneven, so oversubscribe for dynamic balance -> ppt=2. */
    if (use_omp) pw = (int)blas_omp_panel_width(N, nt, nb, 2);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (int jc = 0; jc < N; jc += pw) {
        const int jb = (N - jc < pw) ? (N - jc) : pw;
        wsyrk_block(jc, jb, N, K, UPLO, TR, alpha, beta, a, lda, c, ldc);
    }
}
