/*
 * wtrmm_ — multifloats complex (complex64x2) triangular multiply, public
 * Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * wtrmm_serial.cpp (scalar + SIMD diagonal kernels, block policy, blocked
 * chunk workers for both sides), shared through wtrmm_kernel.h.
 *
 *   B := α·op(A)·B   (SIDE='L')      B := α·B·op(A)   (SIDE='R')
 *
 * One `omp parallel` per multiply. Each thread takes a disjoint slice of the
 * free axis — B's columns (SIDE='L') or rows (SIDE='R') — and runs a
 * per-slice worker; the partition is race-free and bitwise-identical to the
 * serial sweep (each column/row of B is transformed independently). SIDE='R'
 * rounds interior slice boundaries to multiples of 4 so the SIMD 4-row chunks
 * stay aligned; the last thread absorbs the M&3 tail. Delegates to
 * wtrmm_serial when nested.
 *
 * Unlike the real (mtrmm) twin, TRANSA is kept as 'N'/'T'/'C' DISTINCT — the
 * conjugate transpose differs from the plain transpose for complex.
 */
#include "wtrmm_kernel.h"
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
using TC = mf::complex64x2;


/* zero/one predicates — see mf_pred.h */
using mf_pred::ceq0;

using mf_util::up;  /* char flag uppercase — mf_util.h */

static void wtrmm_core(
    char side, char uplo, char transa, char diag,
    std::ptrdiff_t m, std::ptrdiff_t n,
    const TC *alpha_,
    const TC *a, std::ptrdiff_t lda,
    TC *b, std::ptrdiff_t ldb)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        wtrmm_serial(side, uplo, transa, diag, m, n, alpha_, a, lda,
                     b, ldb);
        return;
    }
#endif
    const TC alpha = *alpha_;
    const char SIDE = up(&side);
    const char UPLO = up(&uplo);
    const char TRANS = up(&transa);   /* complex: N/T/C kept distinct */
    const bool nounit = (up(&diag) != 'U');

    if (m == 0 || n == 0) return;

    if (ceq0(alpha)) { wtrmm_zero_B(m, n, b, ldb); return; }

    const std::ptrdiff_t nb = wtrmm_block_nb();

    if (SIDE == 'L') {
        const std::ptrdiff_t use_blocked = (m >= 2 * nb);
#ifdef _OPENMP
        const bool use_omp = (n >= WTRMM_OMP_MIN && blas_omp_available());
        if (use_omp) {
            #pragma omp parallel
            {
                std::ptrdiff_t tid = omp_get_thread_num();
                std::ptrdiff_t nth  = omp_get_num_threads();
                std::ptrdiff_t js  = blas_part_bound(n, tid, nth);
                std::ptrdiff_t je  = blas_part_bound(n, tid + 1, nth);
                wtrmm_L_slice(UPLO, TRANS, use_blocked, js, je, m, nb, alpha,
                              a, lda, b, ldb, nounit);
            }
            return;
        }
#endif
        wtrmm_L_slice(UPLO, TRANS, use_blocked, 0, n, m, nb, alpha,
                      a, lda, b, ldb, nounit);
    } else {
        /* SIDE='R': partition over rows of B. Round interior boundaries to
         * multiples of 4 so the SIMD 4-row chunks stay aligned; the last
         * thread absorbs the M&3 tail. */
        const std::ptrdiff_t use_blocked = (n >= 2 * nb);
#ifdef _OPENMP
        const bool use_omp = (m >= WTRMM_OMP_MIN && blas_omp_available());
        #pragma omp parallel if(use_omp)
#endif
        {
            std::ptrdiff_t tid = 0, nth = 1;
#ifdef _OPENMP
            if (use_omp) { tid = omp_get_thread_num(); nth = omp_get_num_threads(); }
#endif
            std::ptrdiff_t i_lo = blas_part_bound(m, tid, nth);
            std::ptrdiff_t i_hi = blas_part_bound(m, tid + 1, nth);
            if (tid > 0)      i_lo &= ~3;
            if (tid < nth - 1) i_hi &= ~3;
            wtrmm_R_slice(UPLO, TRANS, use_blocked, i_lo, i_hi, n, nb, alpha,
                          a, lda, b, ldb, nounit);
        }
    }
}

extern "C" {
EPBLAS_FACADE_TRMM(wtrmm, TC)
}
