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
#include <cstddef>
#include <cctype>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using T = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
}  // namespace

extern "C" void wtrmm_(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    T *b, const int *ldb_,
    std::size_t side_len, std::size_t uplo_len,
    std::size_t transa_len, std::size_t diag_len)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        wtrmm_serial(side, uplo, transa, diag, m_, n_, alpha_, a, lda_,
                     b, ldb_, side_len, uplo_len, transa_len, diag_len);
        return;
    }
#endif
    (void)side_len; (void)uplo_len; (void)transa_len; (void)diag_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_;
    const T alpha = *alpha_;
    const char SIDE = up(side);
    const char UPLO = up(uplo);
    const char TR = up(transa);   /* complex: N/T/C kept distinct */
    const int nounit = (up(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (ceq0(alpha)) { wtrmm_zero_B(M, N, b, ldb); return; }

    const int nb = wtrmm_block_nb();

    if (SIDE == 'L') {
        const int use_blocked = (M >= 2 * nb);
#ifdef _OPENMP
        const int use_omp = (N >= WTRMM_OMP_MIN && blas_omp_max_threads() > 1);
        if (use_omp) {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nt  = omp_get_num_threads();
                int js  = static_cast<int>((long long)N * tid / nt);
                int je  = static_cast<int>((long long)N * (tid + 1) / nt);
                wtrmm_L_slice(UPLO, TR, use_blocked, js, je, M, nb, alpha,
                              a, lda, b, ldb, nounit);
            }
            return;
        }
#endif
        wtrmm_L_slice(UPLO, TR, use_blocked, 0, N, M, nb, alpha,
                      a, lda, b, ldb, nounit);
    } else {
        /* SIDE='R': partition over rows of B. Round interior boundaries to
         * multiples of 4 so the SIMD 4-row chunks stay aligned; the last
         * thread absorbs the M&3 tail. */
        const int use_blocked = (N >= 2 * nb);
#ifdef _OPENMP
        const int use_omp = (M >= WTRMM_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel if(use_omp)
#endif
        {
            int tid = 0, nt = 1;
#ifdef _OPENMP
            if (use_omp) { tid = omp_get_thread_num(); nt = omp_get_num_threads(); }
#endif
            int i_lo = (int)((long long)M * tid / nt);
            int i_hi = (int)((long long)M * (tid + 1) / nt);
            if (tid > 0)      i_lo &= ~3;
            if (tid < nt - 1) i_hi &= ~3;
            wtrmm_R_slice(UPLO, TR, use_blocked, i_lo, i_hi, N, nb, alpha,
                          a, lda, b, ldb, nounit);
        }
    }
}
