/*
 * xgemmtr_ — kind16 (COMPLEX(KIND=16) / __complex128) triangular GEMM update,
 * public Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * xgemmtr_serial.c (the per-column compute cores and trans decode), shared
 * through xgemmtr_kernel.h.
 *
 *   C := alpha · op(A) · op(B) + beta · C   (only UPLO triangle of C)
 *
 * libquadmath bound, so arithmetic dominates and OpenMP across the triangular
 * column range scales near-linearly. Conjugation under 'C' is applied at
 * element access — one sign flip vs. ~100 cycles per libquadmath complex mul,
 * so the branch cost is noise. Each column j is independent and writes only
 * the UPLO triangle, so per-column static scheduling is bit-exact vs. the
 * serial sweep.
 *
 * Falls back to the serial cores when invoked from inside another parallel
 * region (via !omp_in_parallel() in use_omp — the same `if(use_omp)` loop then
 * runs serially) or below the XGEMMTR_OMP_MIN threshold.
 *
 * Fortran ABI: name lowercased + trailing underscore; scalars by pointer;
 * character args are bare char* — no hidden trailing lengths (see
 * common/epblas_facade.h; never re-add them); COMPLEX(KIND=16) ↔ __complex128.
 */

#include "xgemmtr_kernel.h"
#include "../common/blas_char.h"
#include "../common/epblas_facade.h"
#include <stddef.h>
#include <stdbool.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define XGEMMTR_OMP_MIN 32

typedef xgemmtr_TC TC;

static void xgemmtr_core(char uplo, char transa, char transb,
              ptrdiff_t n, ptrdiff_t k,
              const TC *alpha_,
              const TC *a, ptrdiff_t lda,
              const TC *b, ptrdiff_t ldb,
              const TC *beta_,
              TC *c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        xgemmtr_serial(uplo, transa, transb, n, k, alpha_, a, lda, b, ldb, beta_, c, ldc);
        return;
    }
#endif
    const TC alpha = *alpha_, beta = *beta_;
    const bool upper = (blas_up(uplo) == 'U');
    const char ta = blas_trans_complex(transa);
    const char tb = blas_trans_complex(transb);

    if (n <= 0) return;
    const TC zero = 0.0Q + 0.0Qi;
    const TC one  = 1.0Q + 0.0Qi;

    const bool conj_a = (ta == 'C');
    const bool conj_b = (tb == 'C');
    const bool trans_a = (ta != 'N');
    const bool trans_b = (tb != 'N');

    if (alpha == zero || k == 0) {
        if (beta == one) return;
#ifdef _OPENMP
        const bool use_omp0 = (n >= XGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp0) schedule(static, 1)
#endif
        for (ptrdiff_t j = 0; j < n; ++j)
            xgemmtr_beta_core(j, j + 1, n, upper, beta, c, ldc);
        return;
    }

#ifdef _OPENMP
    const bool use_omp = (n >= XGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
    for (ptrdiff_t j = 0; j < n; ++j)
        xgemmtr_compute_core(j, j + 1, n, upper, k,
                             trans_a, trans_b, conj_a, conj_b,
                             alpha, beta, a, lda, b, ldb, c, ldc);
}

EPBLAS_FACADE_GEMMTR(xgemmtr, TC)
