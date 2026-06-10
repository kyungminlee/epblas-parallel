/*
 * qsyrk_ — kind16 (REAL(KIND=16) / __float128) symmetric rank-k update,
 * public Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * qsyrk_serial.c (the per-column compute core and the uplo/trans decode),
 * shared through qsyrk_kernel.h.
 *
 *   C := alpha · A · Aᵀ + beta · C          (TRANS='N')
 *   C := alpha · Aᵀ · A + beta · C          (TRANS='T'/'C')
 *
 * where C is N×N symmetric; only the UPLO triangle is referenced / written.
 *
 * 1D column decomposition: each column j of C is independent (distinct UPLO
 * slice, read-only A), so a `schedule(static)` fan-out over columns is
 * bitwise-identical to the serial sweep. Falls back to running the loop
 * serially when invoked from inside another parallel region.
 *
 * Fortran ABI: name lowercased + trailing underscore; scalars by pointer;
 * character args followed by hidden trailing size_t lengths.
 */

#include "qsyrk_kernel.h"
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define QSYRK_OMP_MIN 32

typedef qsyrk_T T;

void qsyrk_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = qsyrk_uplo(uplo);
    const int TR = qsyrk_trans_code(trans, trans_len);

    if (N == 0) return;

    const T zero = 0.0Q, one = 1.0Q;

    /* alpha == 0 or K == 0 quick return — just scale the UPLO triangle of C. */
    if (alpha == zero || K == 0) {
        if (beta == one) return;
#ifdef _OPENMP
        const int use_omp = (N >= QSYRK_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
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

    /* schedule(static, 1): column j's UPLO slice is the triangular [i_lo,i_hi)
     * — work per column is ∝ (N-j) (L) or (j+1) (U). Plain contiguous static
     * dumps the heavy end of the triangle on one thread, capping omp4 at ~2.3×
     * (vs ob's ~4× blocked 2D partition). Cyclic static,1 interleaves heavy and
     * light columns across the team so each thread carries a balanced share —
     * lifts NoTrans scaling and closes the par4/ob4 NoTrans breaches. The serial
     * per-column kernel (qsyrk_core, netlib unblocked) is unchanged; only the
     * parallel orchestration differs. Bitwise-identical to the serial sweep. */
#ifdef _OPENMP
    const int use_omp = (N >= QSYRK_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
    #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
    for (int j = 0; j < N; ++j)
        qsyrk_core(j, j + 1, UPLO, TR, N, K, alpha, beta, a, lda, c, ldc);
}
