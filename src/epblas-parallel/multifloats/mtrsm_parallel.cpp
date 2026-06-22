/*
 * mtrsm_ — multifloats real (double-double) triangular solve, public Fortran
 * entry. THREADING ORCHESTRATION ONLY: all the math lives in mtrsm_serial.cpp
 * (scalar + SIMD diagonal kernels, block policy, blocked chunk worker),
 * shared through mtrsm_kernel.h.
 *
 *   op(A) · X = α·B   (SIDE='L')      X · op(A) = α·B   (SIDE='R')
 *
 * One `omp parallel` per solve. Each thread takes a disjoint slice of the
 * free axis — B's columns (SIDE='L') or rows (SIDE='R') — and runs a
 * per-slice worker with private scratch; the partition is race-free and
 * bitwise-identical to the serial sweep. SIDE='R' rounds interior slice
 * boundaries to multiples of 4 so the SIMD 4-row chunks stay aligned; the
 * last thread absorbs the M&3 tail. Delegates to mtrsm_serial when nested.
 */
#include "mtrsm_kernel.h"
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
using T = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
}  // namespace

static void mtrsm_core(
    char side, char uplo, char transa, char diag,
    std::ptrdiff_t M, std::ptrdiff_t N,
    const T *alpha_,
    const T *a, std::ptrdiff_t lda,
    T *b, std::ptrdiff_t ldb)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        mtrsm_serial(side, uplo, transa, diag, M, N, alpha_, a, lda,
                     b, ldb);
        return;
    }
#endif
    const T alpha = *alpha_;
    const char SIDE = up(&side);
    const char UPLO = up(&uplo);
    char TR = up(&transa);
    if (TR == 'C') TR = 'T';
    const std::ptrdiff_t nounit = (up(&diag) != 'U');

    if (M == 0 || N == 0) return;

    if (eq0(alpha)) { mtrsm_zero_B(M, N, b, ldb); return; }

    if (SIDE == 'L') {
        const std::ptrdiff_t nb = mtrsm_block_nb();
        const std::ptrdiff_t use_blocked = (M >= 2 * nb);
#ifdef _OPENMP
        const std::ptrdiff_t use_omp = (N >= MTRSM_OMP_N_MIN && blas_omp_available());
        if (use_omp) {
            #pragma omp parallel
            {
                std::ptrdiff_t tid = omp_get_thread_num();
                std::ptrdiff_t nt  = omp_get_num_threads();
                std::ptrdiff_t js  = blas_part_bound(N, tid, nt);
                std::ptrdiff_t je  = blas_part_bound(N, tid + 1, nt);
                mtrsm_L_slice(UPLO, TR, use_blocked, js, je, M, nb, alpha,
                              a, lda, b, ldb, nounit);
            }
            return;
        }
#endif
        mtrsm_L_slice(UPLO, TR, use_blocked, 0, N, M, nb, alpha,
                      a, lda, b, ldb, nounit);
    } else {
        /* SIDE='R': partition over rows of B. Round interior boundaries to
         * multiples of 4 so the SIMD 4-row chunks stay aligned; the last
         * thread absorbs the M&3 tail. */
#ifdef _OPENMP
        const std::ptrdiff_t use_omp = (M >= MTRSM_OMP_N_MIN && blas_omp_available());
        #pragma omp parallel if(use_omp)
#endif
        {
            std::ptrdiff_t tid = 0, nt = 1;
#ifdef _OPENMP
            if (use_omp) { tid = omp_get_thread_num(); nt = omp_get_num_threads(); }
#endif
            std::ptrdiff_t i_lo = blas_part_bound(M, tid, nt);
            std::ptrdiff_t i_hi = blas_part_bound(M, tid + 1, nt);
            if (tid > 0)      i_lo &= ~3;
            if (tid < nt - 1) i_hi &= ~3;
            mtrsm_R_slice(UPLO, TR, i_lo, i_hi, N, alpha, a, lda, b, ldb, nounit);
        }
    }
}

extern "C" {
EPBLAS_FACADE_TRMM(mtrsm, T)
}
