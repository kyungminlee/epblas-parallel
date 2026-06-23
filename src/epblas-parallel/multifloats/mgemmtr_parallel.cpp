/*
 * mgemmtr_ — multifloats real (DD / float64x2) triangular GEMM update,
 * public Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * mgemmtr_serial.cpp (block policy, scalar diagonal triangle, per-block
 * cores), shared through mgemmtr_kernel.h. The off-diagonal rectangles run
 * through mgemm_serial.
 *
 *   C := alpha · op(A) · op(B) + beta · C   (only UPLO triangle of C)
 *
 * Each jc-block owns a disjoint column range, so fanning the jc loop across
 * an OpenMP team is race-free and bitwise-identical to the serial sweep.
 * Delegates to mgemmtr_serial when already inside a parallel region.
 */
#include "mgemmtr_kernel.h"
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

static void mgemmtr_core(
    char uplo, char transa, char transb,
    std::ptrdiff_t n, std::ptrdiff_t k,
    const TR *alpha_,
    const TR *a, std::ptrdiff_t lda,
    const TR *b, std::ptrdiff_t ldb,
    const TR *beta_,
    TR *c, std::ptrdiff_t ldc)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        mgemmtr_serial(uplo, transa, transb, n, k, alpha_, a, lda,
                       b, ldb, beta_, c, ldc);
        return;
    }
#endif
    const TR alpha = *alpha_, beta = *beta_;
    const bool upper = (up(&uplo) == 'U');
    char ta = up(&transa); if (ta == 'C') ta = 'T';
    char tb = up(&transb); if (tb == 'C') tb = 'T';

    if (n <= 0) return;

    if (eq0(alpha) || k == 0) {
        if (eq1(beta)) return;
#ifdef _OPENMP
        const bool use_omp0 = (n >= MGEMMTR_OMP_MIN && blas_omp_available());
        #pragma omp parallel for if(use_omp0) schedule(static)
#endif
        for (std::ptrdiff_t j = 0; j < n; ++j)
            mgemmtr_beta_core(j, j + 1, n, upper, beta, c, ldc);
        return;
    }

    const std::ptrdiff_t nb = mgemmtr_block_nb();
#ifdef _OPENMP
    const bool use_omp = (n >= MGEMMTR_OMP_MIN && blas_omp_available());
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (std::ptrdiff_t jc = 0; jc < n; jc += nb) {
        const std::ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
        mgemmtr_block_core(jc, jb, n, k, alpha, beta,
                           a, lda, b, ldb, c, ldc, upper, ta, tb);
    }
}

extern "C" {
EPBLAS_FACADE_GEMMTR(mgemmtr, TR)
}
