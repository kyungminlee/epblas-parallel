/*
 * xsyrk_ — kind16 complex (__complex128 / COMPLEX(KIND=16)) symmetric
 * rank-k update, public Fortran entry. THREADING ORCHESTRATION ONLY: all the
 * math lives in xsyrk_serial.c (the per-column compute core and the
 * uplo/trans decode), shared through xsyrk_kernel.h.
 *
 *   C := alpha · A · Aᵀ + beta · C          (TRANS='N')
 *   C := alpha · Aᵀ · A + beta · C          (TRANS='T')
 *
 * C is N×N symmetric (NOT Hermitian — no conjugate); only the UPLO triangle
 * is referenced/written. TRANS='C' is not standard for complex syrk (use
 * xherk for the Hermitian variant).
 *
 * 1D column decomposition: each column j of C is independent (distinct UPLO
 * slice, read-only A), so a `schedule(static)` fan-out over columns is
 * bitwise-identical to the serial sweep. Falls back to running the loop
 * serially when invoked from inside another parallel region.
 *
 * Fortran ABI: name lowercased + trailing underscore; scalars by pointer;
 * character args followed by hidden trailing size_t lengths.
 */

#include "xsyrk_kernel.h"
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define XSYRK_OMP_MIN 32

typedef xsyrk_T T;

void xsyrk_(
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
    const char UPLO = xsyrk_uplo(uplo);
    const char TR = xsyrk_trans(trans);

    if (N == 0) return;

    const T zero = 0.0Q + 0.0Qi, one = 1.0Q + 0.0Qi;

    if (alpha == zero || K == 0) {
        if (beta == one) return;
#ifdef _OPENMP
        const int use_omp = (N >= XSYRK_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
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
    const int use_omp = (N >= XSYRK_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (int j = 0; j < N; ++j)
        xsyrk_core(j, j + 1, UPLO, TR, N, K, alpha, beta, a, lda, c, ldc);
}
