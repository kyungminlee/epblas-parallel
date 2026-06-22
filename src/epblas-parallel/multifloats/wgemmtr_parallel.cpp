/*
 * wgemmtr_ — multifloats complex DD (complex64x2) triangular GEMM update,
 * public Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * wgemmtr_serial.cpp (block policy, scalar diagonal triangle, per-block
 * cores), shared through wgemmtr_kernel.h. The off-diagonal rectangles run
 * through wgemm_serial.
 *
 *   C := alpha · op(A) · op(B) + beta · C   (only UPLO triangle of C)
 *
 * Each jc-block owns a disjoint column range, so fanning the jc loop across
 * an OpenMP team is race-free and bitwise-identical to the serial sweep.
 * Delegates to wgemmtr_serial when already inside a parallel region. Unlike
 * the real case, ta/tb are kept distinct (N/T/C) — T and C differ for complex.
 */
#include "wgemmtr_kernel.h"
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

extern "C" void wgemmtr_(
    const char *uplo, const char *transa, const char *transb,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    std::size_t uplo_len, std::size_t ta_len, std::size_t tb_len)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        wgemmtr_serial(uplo, transa, transb, n_, k_, alpha_, a, lda_,
                       b, ldb_, beta_, c, ldc_, uplo_len, ta_len, tb_len);
        return;
    }
#endif
    (void)uplo_len; (void)ta_len; (void)tb_len;
    const std::ptrdiff_t N = *n_, K = *k_;
    const std::ptrdiff_t lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const bool upper = (up(uplo) == 'U');
    const char ta = up(transa);
    const char tb = up(transb);
    (void)lda; (void)ldb;

    if (N <= 0) return;

    if (ceq0(alpha) || K == 0) {
        if (ceq1(beta)) return;
#ifdef _OPENMP
        const bool use_omp0 = (N >= WGEMMTR_OMP_MIN && blas_omp_available());
        #pragma omp parallel for if(use_omp0) schedule(static)
#endif
        for (std::ptrdiff_t j = 0; j < N; ++j)
            wgemmtr_beta_core(j, j + 1, N, upper, beta, c, ldc);
        return;
    }

    const std::ptrdiff_t nb = wgemmtr_block_nb();
#ifdef _OPENMP
    const bool use_omp = (N >= WGEMMTR_OMP_MIN && blas_omp_available());
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (std::ptrdiff_t jc = 0; jc < N; jc += nb) {
        const std::ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
        wgemmtr_block_core(jc, jb, N, K, alpha, beta,
                           a, lda, b, ldb, c, ldc, upper, ta, tb);
    }
}
