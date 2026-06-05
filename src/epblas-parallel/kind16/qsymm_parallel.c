/*
 * qsymm_ — kind16 (REAL(KIND=16) / __float128) symmetric matrix multiply,
 * public Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * qsymm_serial.c (the per-column compute core and uplo decode), shared
 * through qsymm_kernel.h.
 *
 *   C := alpha · A · B + beta · C          (SIDE='L', A is M×M sym)
 *   C := alpha · B · A + beta · C          (SIDE='R', A is N×N sym)
 *
 * UPLO selects which triangle of A is stored. The other half is the
 * reflection (A(i,k) = A(k,i)).
 *
 * Unblocked Netlib reference with omp-parallel-for over columns of C. Each
 * column j is independent, so a per-column static partition is bitwise-
 * identical to the serial sweep. Falls back to running serially when invoked
 * from inside another parallel region.
 *
 * Fortran ABI: name lowercased + trailing underscore; scalars by pointer;
 * character args followed by hidden trailing size_t lengths; REAL(KIND=16)
 * ↔ __float128.
 */

#include "qsymm_kernel.h"
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define QSYMM_OMP_MIN 32

typedef qsymm_T T;

void qsymm_(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t side_len, size_t uplo_len)
{
    (void)side_len; (void)uplo_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char SIDE = qsymm_uplo(side);
    const char UPLO = qsymm_uplo(uplo);

    if (M == 0 || N == 0) return;

    const T zero = 0.0Q, one = 1.0Q;

    if (alpha == zero) {
        if (beta == one) return;
#ifdef _OPENMP
        const int use_omp = (N >= QSYMM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            T *cj = c + (size_t)j * ldc;
            if (beta == zero) for (int i = 0; i < M; ++i) cj[i]  = zero;
            else              for (int i = 0; i < M; ++i) cj[i] *= beta;
        }
        return;
    }

#ifdef _OPENMP
    const int use_omp = (N >= QSYMM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (int j = 0; j < N; ++j)
        qsymm_core(j, j + 1, SIDE, UPLO, M, N, alpha, beta, a, lda, b, ldb, c, ldc);
}
