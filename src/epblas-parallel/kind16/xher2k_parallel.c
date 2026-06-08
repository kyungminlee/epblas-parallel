/*
 * xher2k_ — kind16 complex (__complex128) Hermitian rank-2k, public Fortran
 * entry. THREADING ORCHESTRATION ONLY: all the math lives in xher2k_serial.c
 * (the per-column compute core and uplo decode), shared through
 * xher2k_kernel.h.
 *
 *   C := alpha · A · Bᴴ + conj(alpha) · B · Aᴴ + beta · C  (TRANS='N')
 *   C := alpha · Aᴴ · B + conj(alpha) · Bᴴ · A + beta · C  (TRANS='C')
 *
 * alpha is COMPLEX, beta is REAL; the diagonal of C stays real (the
 * imaginary part is dropped at every touch inside the core). One
 * omp-parallel-for over the columns of C — each column is independent, so
 * static per-column scheduling is race-free and bitwise-identical to the
 * serial path. Falls back to serial when invoked from inside another parallel
 * region (the `!omp_in_parallel()` term in use_omp makes the same
 * `if(use_omp)` loop run serially).
 *
 * Fortran ABI: name lowercased + trailing underscore; scalars by pointer;
 * character args followed by hidden trailing size_t lengths.
 */

#include "xher2k_kernel.h"
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define XHER2K_OMP_MIN 32

typedef xher2k_TC TC;
typedef xher2k_TR TR;

void xher2k_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const TC *alpha_,
    const TC *restrict a, const int *lda_,
    const TC *restrict b, const int *ldb_,
    const TR *beta_,
    TC *restrict c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const TC alpha = *alpha_;
    const TR beta  = *beta_;
    const char UPLO = xher2k_uplo(uplo);
    const char TR_c = xher2k_uplo(trans);

    if (N == 0) return;

    const TR rzero = 0.0Q, rone = 1.0Q;
    const TC czero = 0.0Q + 0.0Qi;
    const TC alpha_conj = conjq(alpha);

    if (alpha == czero || K == 0) {
        if (beta == rone) {
            for (int j = 0; j < N; ++j) c[(size_t)j * ldc + j] = crealq(c[(size_t)j * ldc + j]);
            return;
        }
#ifdef _OPENMP
        const int use_omp = (N >= XHER2K_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const int i_lo = (UPLO == 'L') ? j : 0;
            const int i_hi = (UPLO == 'L') ? N : j + 1;
            TC *cj = c + (size_t)j * ldc;
            if (beta == rzero) {
                for (int i = i_lo; i < i_hi; ++i) cj[i] = czero;
            } else {
                for (int i = i_lo; i < i_hi; ++i) {
                    if (i == j) cj[i] = beta * crealq(cj[i]);
                    else        cj[i] = beta * cj[i];
                }
            }
        }
        return;
    }

#ifdef _OPENMP
    const int use_omp = (N >= XHER2K_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
    #pragma omp parallel for if(use_omp) schedule(static)
    for (int j = 0; j < N; ++j)
        xher2k_core(j, j + 1, N, K, UPLO, TR_c, alpha, alpha_conj, beta,
                    a, lda, b, ldb, c, ldc);
#else
    xher2k_core(0, N, N, K, UPLO, TR_c, alpha, alpha_conj, beta,
                a, lda, b, ldb, c, ldc);
#endif
}
