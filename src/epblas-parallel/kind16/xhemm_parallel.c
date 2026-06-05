/*
 * xhemm_ — kind16 complex (`__complex128`) Hermitian matrix multiply, public
 * Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * xhemm_serial.c (the per-column compute core and uplo decode), shared
 * through xhemm_kernel.h.
 *
 *   C := alpha · A · B + beta · C          (SIDE='L', A is M×M Hermitian)
 *   C := alpha · B · A + beta · C          (SIDE='R', A is N×N Hermitian)
 *
 * For UPLO='L' the lower triangle is stored. Off-diagonal entries reflect via
 * the Hermitian property A(k,i) = conj(A(i,k)); the diagonal of A is real —
 * only its real part is read.
 *
 * Unblocked Netlib reference (matches Netlib ZHEMM column ordering) with
 * omp-parallel-for over columns of C. Each column j is independent, so a
 * per-column static partition is bitwise-identical to the serial sweep. Falls
 * back to running serially when invoked from inside another parallel region.
 *
 * Fortran ABI: name lowercased + trailing underscore; scalars by pointer;
 * character args followed by hidden trailing size_t lengths; COMPLEX(KIND=16)
 * ↔ __complex128.
 */

#include "xhemm_kernel.h"
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define XHEMM_OMP_MIN 32

typedef xhemm_T T;

static const T ZERO = 0.0Q + 0.0Qi;
static const T ONE  = 1.0Q + 0.0Qi;

void xhemm_(
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
    const char SIDE = xhemm_uplo(side);
    const char UPLO = xhemm_uplo(uplo);

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        if (beta == ONE) return;
#ifdef _OPENMP
        const int use_omp = (N >= XHEMM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            T *cj = c + (size_t)j * ldc;
            if (beta == ZERO) for (int i = 0; i < M; ++i) cj[i] = ZERO;
            else              for (int i = 0; i < M; ++i) cj[i] *= beta;
        }
        return;
    }

#ifdef _OPENMP
    const int use_omp = (N >= XHEMM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (int j = 0; j < N; ++j)
        xhemm_core(j, j + 1, SIDE, UPLO, M, N, alpha, beta, a, lda, b, ldb, c, ldc);
}
