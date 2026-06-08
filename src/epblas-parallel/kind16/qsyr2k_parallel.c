/*
 * qsyr2k_ — kind16 (REAL(KIND=16) / __float128) symmetric rank-2k, public
 * Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * qsyr2k_serial.c (the per-column compute core and uplo decode), shared
 * through qsyr2k_kernel.h.
 *
 *   C := alpha · (A · Bᵀ + B · Aᵀ) + beta · C         (TRANS='N')
 *   C := alpha · (Aᵀ · B + Bᵀ · A) + beta · C         (TRANS='T'/'C')
 *
 * One omp-parallel-for over the columns of C — each column is independent,
 * so static per-column scheduling is race-free and bitwise-identical to the
 * serial path. libquadmath dispatch dominates the per-op time, so no blocking
 * or packing; coarse-grain parallelism over j scales nearly linearly. Falls
 * back to serial when invoked from inside another parallel region (the
 * `!omp_in_parallel()` term in use_omp makes the same `if(use_omp)` loop run
 * serially).
 *
 * Fortran ABI: name lowercased + trailing underscore; scalars by pointer;
 * character args followed by hidden trailing size_t lengths; REAL(KIND=16)
 * ↔ __float128.
 */

#include "qsyr2k_kernel.h"
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define QSYR2K_OMP_MIN 32

typedef qsyr2k_T T;

void qsyr2k_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict b, const int *ldb_,
    const T *beta_,
    T *restrict c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = qsyr2k_uplo(uplo);
    char TR = qsyr2k_uplo(trans);
    if (TR == 'C') TR = 'T';

    if (N == 0) return;

    const T zero = 0.0Q, one = 1.0Q;

    if (alpha == zero || K == 0) {
        if (beta == one) return;
#ifdef _OPENMP
        const int use_omp = (N >= QSYR2K_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const int i_lo = (UPLO == 'L') ? j : 0;
            const int i_hi = (UPLO == 'L') ? N : j + 1;
            T *cj = c + (size_t)j * ldc;
            if (beta == zero) for (int i = i_lo; i < i_hi; ++i) cj[i] = zero;
            else              for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;
        }
        return;
    }

#ifdef _OPENMP
    const int use_omp = (N >= QSYR2K_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
    #pragma omp parallel for if(use_omp) schedule(static)
    for (int j = 0; j < N; ++j)
        qsyr2k_core(j, j + 1, N, K, UPLO, TR, alpha, beta, a, lda, b, ldb, c, ldc);
#else
    qsyr2k_core(0, N, N, K, UPLO, TR, alpha, beta, a, lda, b, ldb, c, ldc);
#endif
}
