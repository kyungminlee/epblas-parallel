/*
 * xherk_ — kind16 complex (__complex128) Hermitian rank-k update, public
 * Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * xherk_serial.c (the per-column compute core and the uplo/trans decode),
 * shared through xherk_kernel.h.
 *
 *   C := alpha · A · Aᴴ + beta · C          (TRANS='N')
 *   C := alpha · Aᴴ · A + beta · C          (TRANS='C')
 *
 * where C is N×N Hermitian; only the UPLO triangle is referenced / written
 * and its diagonal is forced real. alpha and beta are REAL (kind=16).
 *
 * 1D column decomposition: each column j of C is independent (distinct UPLO
 * slice, read-only A), so a `schedule(static)` fan-out over columns is
 * bitwise-identical to the serial sweep. Falls back to running the loop
 * serially when invoked from inside another parallel region.
 *
 * Fortran ABI: name lowercased + trailing underscore; scalars by pointer;
 * character args followed by hidden trailing size_t lengths.
 */

#include "xherk_kernel.h"
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define XHERK_OMP_MIN 32

typedef xherk_TC TC;
typedef xherk_TR TR;

void xherk_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const TR *alpha_,
    const TC *a, const int *lda_,
    const TR *beta_,
    TC *c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldc = *ldc_;
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO = xherk_uplo(uplo);
    const char TR_c = xherk_trans(trans);

    if (N == 0) return;

    const TR rzero = 0.0Q, rone = 1.0Q;
    const TC czero = 0.0Q + 0.0Qi;

    /* Quick return when only beta scaling is needed. */
    if (alpha == rzero || K == 0) {
        if (beta == rone) {
            /* Even for beta=1, ZHERK strictly zeros the imag part of
             * the diagonal so it stays real. */
            for (int j = 0; j < N; ++j) c[(size_t)j * ldc + j] = crealq(c[(size_t)j * ldc + j]);
            return;
        }
#ifdef _OPENMP
        const int use_omp = (N >= XHERK_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
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
    const int use_omp = (N >= XHERK_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (int j = 0; j < N; ++j)
        xherk_core(j, j + 1, UPLO, TR_c, N, K, alpha, beta, a, lda, c, ldc);
}
