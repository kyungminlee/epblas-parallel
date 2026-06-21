/*
 * msymm_ — multifloats real (DD) symmetric matrix multiply, public Fortran
 * entry. THREADING ORCHESTRATION ONLY: all the math lives in msymm_serial.cpp
 * (SIMD + scalar diagonal kernels, block policy, the per-block workers, the
 * per-column scaler), shared through msymm_kernel.h.
 *
 *   C := α·A·B + β·C   (SIDE='L', A symmetric)
 *   C := α·B·A + β·C   (SIDE='R', A symmetric)
 *
 * One `omp parallel for` over the block loop (schedule(dynamic,1)): SIDE='L'
 * over the row blocks ic, SIDE='R' over the column blocks jc. Each block writes
 * a disjoint slab of C (rows for L, columns for R) — race-free and bitwise-
 * identical to the serial sweep. The alpha==0 early exit fans the per-column
 * beta scale (schedule(static)). Delegates to msymm_serial when nested.
 */
#include "msymm_kernel.h"
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
using mf_pred::eq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
}  // namespace

extern "C" void msymm_(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    std::size_t side_len, std::size_t uplo_len)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        msymm_serial(side, uplo, m_, n_, alpha_, a, lda_, b, ldb_, beta_,
                     c, ldc_, side_len, uplo_len);
        return;
    }
#endif
    (void)side_len; (void)uplo_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char SIDE = up(side);
    const char UPLO = up(uplo);
    (void)lda; (void)ldb;

    if (M == 0 || N == 0) return;

    if (eq0(alpha)) {
        if (eq1(beta)) return;
#ifdef _OPENMP
        const int axis = (SIDE == 'L') ? M : N;
        const bool use_omp = (axis >= MSYMM_OMP_MIN && blas_omp_available());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) msymm_scale_col(j, M, beta, c, ldc);
        return;
    }

    const int nb = msymm_block_nb();

    if (SIDE == 'L') {
        int pw = nb;
#ifdef _OPENMP
        const int nt = blas_omp_max_threads();
        const bool use_omp = (M >= MSYMM_OMP_MIN && nt > 1);
        /* Shrink the block step so the team gets ~nt panels at small M
         * (M=64, nb=32 -> 2 blocks -> 2 idle threads of 4). Rectangular work
         * (each row block multiplies its rows against full A·B) -> ppt=1. */
        if (use_omp) pw = (int)blas_omp_panel_width(M, nt, nb, 1);
        #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
        for (int ic = 0; ic < M; ic += pw) {
            const int ib = (M - ic < pw) ? (M - ic) : pw;
            msymm_block_L(ic, ib, M, N, UPLO, alpha, beta, a, lda, b, ldb, c, ldc);
        }
    } else {
        int pw = nb;
#ifdef _OPENMP
        const int nt = blas_omp_max_threads();
        const bool use_omp = (N >= MSYMM_OMP_MIN && nt > 1);
        if (use_omp) pw = (int)blas_omp_panel_width(N, nt, nb, 1);
        #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
        for (int jc = 0; jc < N; jc += pw) {
            const int jb = (N - jc < pw) ? (N - jc) : pw;
            msymm_block_R(jc, jb, M, N, UPLO, alpha, beta, a, lda, b, ldb, c, ldc);
        }
    }
}
