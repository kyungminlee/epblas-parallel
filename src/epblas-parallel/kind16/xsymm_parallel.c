/*
 * xsymm_ — kind16 complex (`__complex128`) symmetric matrix multiply, public
 * Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * xsymm_serial.c (the per-column compute core and uplo decode), shared
 * through xsymm_kernel.h. NOT Hermitian — no conjugate; see xhemm.
 *
 *   C := alpha · A · B + beta · C          (SIDE='L', A is M×M sym)
 *   C := alpha · B · A + beta · C          (SIDE='R', A is N×N sym)
 *
 * Unblocked Netlib reference with omp-parallel-for over columns of C. Each
 * column j is independent, so a per-column static partition is bitwise-
 * identical to the serial sweep. Falls back to running serially when invoked
 * from inside another parallel region.
 *
 * Fortran ABI: name lowercased + trailing underscore; scalars by pointer;
 * character args followed by hidden trailing size_t lengths; COMPLEX(KIND=16)
 * ↔ __complex128.
 */

#include "xsymm_kernel.h"
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define XSYMM_OMP_MIN 32

typedef xsymm_T T;

static const T ZERO = 0.0Q + 0.0Qi;
static const T ONE  = 1.0Q + 0.0Qi;

void xsymm_(
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
    const char SIDE = xsymm_uplo(side);
    const char UPLO = xsymm_uplo(uplo);

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        if (beta == ONE) return;
#ifdef _OPENMP
        const int use_omp = (N >= XSYMM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
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
    const int use_omp = (N >= XSYMM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (int j = 0; j < N; ++j)
        xsymm_core(j, j + 1, SIDE, UPLO, M, N, alpha, beta, a, lda, b, ldb, c, ldc);
}
