/*
 * mtrmm_ — multifloats real (double-double) triangular multiply, public
 * Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * mtrmm_serial.cpp (scalar + SIMD diagonal kernels, block policy, blocked
 * chunk workers for both sides), shared through mtrmm_kernel.h.
 *
 *   B := α·op(A)·B   (SIDE='L')      B := α·B·op(A)   (SIDE='R')
 *
 * One `omp parallel` per multiply. Each thread takes a disjoint slice of the
 * free axis — B's columns (SIDE='L') or rows (SIDE='R') — and runs a
 * per-slice worker; the partition is race-free and bitwise-identical to the
 * serial sweep (each column/row of B is transformed independently). SIDE='R'
 * rounds interior slice boundaries to multiples of 4 so the SIMD 4-row chunks
 * stay aligned; the last thread absorbs the M&3 tail. Delegates to
 * mtrmm_serial when nested.
 */
#include "mtrmm_kernel.h"
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

static void mtrmm_core(
    char side, char uplo, char transa, char diag,
    std::ptrdiff_t M, std::ptrdiff_t N,
    const T *alpha_,
    const T *a, std::ptrdiff_t lda,
    T *b, std::ptrdiff_t ldb)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        mtrmm_serial(side, uplo, transa, diag, M, N, alpha_, a, lda,
                     b, ldb);
        return;
    }
#endif
    const T alpha = *alpha_;
    const char SIDE = up(&side);
    const char UPLO = up(&uplo);
    char TR = up(&transa);
    if (TR == 'C') TR = 'T';   /* real DD: conj-trans ≡ trans */
    const bool nounit = (up(&diag) != 'U');

    if (M == 0 || N == 0) return;

    if (eq0(alpha)) { mtrmm_zero_B(M, N, b, ldb); return; }

    const std::ptrdiff_t nb = mtrmm_block_nb();

    if (SIDE == 'L') {
        const std::ptrdiff_t use_blocked = (M >= 2 * nb);
#ifdef _OPENMP
        const bool use_omp = (N >= MTRMM_OMP_MIN && blas_omp_available());
        if (use_omp) {
            #pragma omp parallel
            {
                std::ptrdiff_t tid = omp_get_thread_num();
                std::ptrdiff_t nth  = omp_get_num_threads();
                std::ptrdiff_t js  = blas_part_bound(N, tid, nth);
                std::ptrdiff_t je  = blas_part_bound(N, tid + 1, nth);
                mtrmm_L_slice(UPLO, TR, use_blocked, js, je, M, nb, alpha,
                              a, lda, b, ldb, nounit);
            }
            return;
        }
#endif
        mtrmm_L_slice(UPLO, TR, use_blocked, 0, N, M, nb, alpha,
                      a, lda, b, ldb, nounit);
    } else {
        /* SIDE='R': partition over rows of B. Round interior boundaries to
         * multiples of 4 so the SIMD 4-row chunks stay aligned; the last
         * thread absorbs the M&3 tail. */
        const std::ptrdiff_t use_blocked = (N >= 2 * nb);
#ifdef _OPENMP
        const bool use_omp = (M >= MTRMM_OMP_MIN && blas_omp_available());
        #pragma omp parallel if(use_omp)
#endif
        {
            std::ptrdiff_t tid = 0, nth = 1;
#ifdef _OPENMP
            if (use_omp) { tid = omp_get_thread_num(); nth = omp_get_num_threads(); }
#endif
            std::ptrdiff_t i_lo = blas_part_bound(M, tid, nth);
            std::ptrdiff_t i_hi = blas_part_bound(M, tid + 1, nth);
            if (tid > 0)      i_lo &= ~3;
            if (tid < nth - 1) i_hi &= ~3;
            mtrmm_R_slice(UPLO, TR, use_blocked, i_lo, i_hi, N, nb, alpha,
                          a, lda, b, ldb, nounit);
        }
    }
}

extern "C" {
EPBLAS_FACADE_TRMM(mtrmm, T)
}
