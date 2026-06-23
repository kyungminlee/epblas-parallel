/*
 * whemm_ — multifloats complex (DD) Hermitian matrix multiply, public Fortran
 * entry. NOT symmetric (see wsymm). THREADING ORCHESTRATION ONLY: all the math
 * lives in whemm_serial.cpp (SIMD + scalar diagonal kernels, block policy, the
 * per-block workers, the per-column scaler), shared through whemm_kernel.h.
 *
 *   C := α·A·B + β·C   (SIDE='L', A complex Hermitian)
 *   C := α·B·A + β·C   (SIDE='R', A complex Hermitian)
 *
 * One `omp parallel for` over the block loop (schedule(dynamic,1)): SIDE='L'
 * over the row blocks ic, SIDE='R' over the column blocks jc. Each block writes
 * a disjoint slab of C (rows for L, columns for R) — race-free and bitwise-
 * identical to the serial sweep. The alpha==0 early exit fans the per-column
 * beta scale (schedule(static)). Delegates to whemm_serial when nested.
 */
#include "whemm_kernel.h"
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
using TC = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;
using mf_pred::ceq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
}  // namespace

static void whemm_core(
    char side, char uplo,
    std::ptrdiff_t m, std::ptrdiff_t n,
    const TC *alpha_,
    const TC *a, std::ptrdiff_t lda,
    const TC *b, std::ptrdiff_t ldb,
    const TC *beta_,
    TC *c, std::ptrdiff_t ldc)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        whemm_serial(side, uplo, m, n, alpha_, a, lda, b, ldb, beta_, c, ldc);
        return;
    }
#endif
    const TC alpha = *alpha_, beta = *beta_;
    const char SIDE = up(&side);
    const char UPLO = up(&uplo);
    (void)lda; (void)ldb;

    if (m == 0 || n == 0) return;

    if (ceq0(alpha)) {
        if (ceq1(beta)) return;
#ifdef _OPENMP
        const std::ptrdiff_t axis = (SIDE == 'L') ? m : n;
        const bool use_omp = (axis >= WHEMM_OMP_MIN && blas_omp_available());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (std::ptrdiff_t j = 0; j < n; ++j) whemm_scale_col(j, m, beta, c, ldc);
        return;
    }

    const std::ptrdiff_t nb = whemm_block_nb();

    if (SIDE == 'L') {
        std::ptrdiff_t pw = nb;
#ifdef _OPENMP
        const std::ptrdiff_t nthreads = blas_omp_max_threads();
        const bool use_omp = (m >= WHEMM_OMP_MIN && nthreads > 1);
        /* Shrink the block step so the team gets ~nthreads panels at small M
         * (M=64, nb=32 -> 2 blocks -> 2 idle threads of 4). Rectangular work
         * (each row block multiplies its rows against full A·B) -> ppt=1. */
        if (use_omp) pw = (std::ptrdiff_t)blas_omp_panel_width(m, nthreads, nb, 1);
        #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
        for (std::ptrdiff_t ic = 0; ic < m; ic += pw) {
            const std::ptrdiff_t ib = (m - ic < pw) ? (m - ic) : pw;
            whemm_block_L(ic, ib, m, n, UPLO, alpha, beta, a, lda, b, ldb, c, ldc);
        }
    } else {
        std::ptrdiff_t pw = nb;
#ifdef _OPENMP
        const std::ptrdiff_t nthreads = blas_omp_max_threads();
        const bool use_omp = (n >= WHEMM_OMP_MIN && nthreads > 1);
        if (use_omp) pw = (std::ptrdiff_t)blas_omp_panel_width(n, nthreads, nb, 1);
        #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
        for (std::ptrdiff_t jc = 0; jc < n; jc += pw) {
            const std::ptrdiff_t jb = (n - jc < pw) ? (n - jc) : pw;
            whemm_block_R(jc, jb, m, n, UPLO, alpha, beta, a, lda, b, ldb, c, ldc);
        }
    }
}

extern "C" {
EPBLAS_FACADE_SYMM(whemm, TC)
}
