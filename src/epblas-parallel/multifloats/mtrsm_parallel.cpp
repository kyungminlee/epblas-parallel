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
#include <cstddef>
#include <cctype>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
inline bool dd_iszero(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
}  // namespace

extern "C" void mtrsm_(
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
        mtrsm_serial(side, uplo, transa, diag, m_, n_, alpha_, a, lda_,
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
    char TR = up(transa);
    if (TR == 'C') TR = 'T';
    const int nounit = (up(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (dd_iszero(alpha)) { mtrsm_zero_B(M, N, b, ldb); return; }

    if (SIDE == 'L') {
        const int nb = mtrsm_block_nb();
        const int use_blocked = (M >= 2 * nb);
#ifdef _OPENMP
        const int use_omp = (N >= MTRSM_OMP_N_MIN && blas_omp_max_threads() > 1);
        if (use_omp) {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                int nt  = omp_get_num_threads();
                int js  = static_cast<int>((long long)N * tid / nt);
                int je  = static_cast<int>((long long)N * (tid + 1) / nt);
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
        const int use_omp = (M >= MTRSM_OMP_N_MIN && blas_omp_max_threads() > 1);
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
            mtrsm_R_slice(UPLO, TR, i_lo, i_hi, N, alpha, a, lda, b, ldb, nounit);
        }
    }
}
