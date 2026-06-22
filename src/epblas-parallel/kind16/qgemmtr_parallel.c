/*
 * qgemmtr_ — kind16 (REAL(KIND=16) / __float128) triangular GEMM update,
 * public Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * qgemmtr_serial.c (the per-column compute cores and trans decode), shared
 * through qgemmtr_kernel.h.
 *
 *   C := alpha · op(A) · op(B) + beta · C   (only UPLO triangle of C)
 *
 * libquadmath bound, so arithmetic dominates and OpenMP across the triangular
 * column range scales near-linearly. Each column j is independent and writes
 * only the UPLO triangle, so per-column static scheduling is bit-exact vs. the
 * serial sweep.
 *
 * Falls back to the serial cores when invoked from inside another parallel
 * region (via !omp_in_parallel() in use_omp — the same `if(use_omp)` loop then
 * runs serially) or below the QGEMMTR_OMP_MIN threshold.
 *
 * Fortran ABI: name lowercased + trailing underscore; scalars by pointer;
 * character args followed by hidden trailing size_t lengths; REAL(KIND=16)
 * ↔ __float128.
 */

#include "qgemmtr_kernel.h"
#include "../common/blas_char.h"
#include "../common/epblas_facade.h"
#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define QGEMMTR_OMP_MIN 32

typedef qgemmtr_T T;

static void qgemmtr_core(char uplo_c, char transa, char transb,
              ptrdiff_t N, ptrdiff_t K,
              const T *alpha_,
              const T *a, ptrdiff_t lda,
              const T *b, ptrdiff_t ldb,
              const T *beta_,
              T *c, ptrdiff_t ldc)
{
    const T alpha = *alpha_, beta = *beta_;
    const bool upper = (blas_up(uplo_c) == 'U');
    const char ta = qgemmtr_trans_code(&transa);
    const char tb = qgemmtr_trans_code(&transb);

    if (N <= 0) return;
    const T zero = 0.0Q, one = 1.0Q;

    if (alpha == zero || K == 0) {
        if (beta == one) return;
#ifdef _OPENMP
        const bool use_omp0 = (N >= QGEMMTR_OMP_MIN && blas_omp_should_thread());
        #pragma omp parallel for if(use_omp0) schedule(static, 1)
#endif
        for (ptrdiff_t j = 0; j < N; ++j)
            qgemmtr_beta_core(j, j + 1, N, upper, beta, c, ldc);
        return;
    }

#ifdef _OPENMP
    const bool use_omp = (N >= QGEMMTR_OMP_MIN && blas_omp_should_thread());
    #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
    for (ptrdiff_t j = 0; j < N; ++j)
        qgemmtr_compute_core(j, j + 1, N, upper, K, ta, tb,
                             alpha, beta, a, lda, b, ldb, c, ldc);
}

EPBLAS_FACADE_GEMMTR(qgemmtr, T)
