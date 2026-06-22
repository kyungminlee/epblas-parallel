/*
 * ygemm_ — kind10 complex GEMM (COMPLEX(KIND=10) / _Complex long double),
 * the public Fortran entry and threading-orchestration half of the ygemm
 * overlay (see ygemm_kernel.h for the split rationale; all the math lives
 * in ygemm_serial.c).
 *
 *   C := alpha * op(A) * op(B) + beta * C
 *
 * Parallel shape: one `omp parallel` partitions C's columns (the j axis)
 * across the team; each thread runs the serial orientation core on its
 * own contiguous column slice — embarrassingly parallel, scales linearly
 * until memory bandwidth saturates. The beta pre-pass runs once up front.
 *
 * Nesting guard: when ygemm_ is itself called from inside another
 * routine's parallel region (the complex L3 family — ytrsm, ytrmm, ysyrk,
 * … runs ygemm trailing updates inside its own `omp parallel`), it
 * delegates to ygemm_serial and opens no region of its own — opening a
 * nested team here trips the libgomp barrier wedge (memory
 * project-etrsm-omp4-wedge).
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "ygemm_kernel.h"
#include "../common/epblas_facade.h"

typedef ygemm_T T;

static const T zero = 0.0L + 0.0iL;

/* Threshold below which OMP parallel-for on the column axis isn't worth
 * the parallel-region setup. */
#define YGEMM_OMP_N_MIN 32

static ptrdiff_t trans_code(char c) {
    return blas_up(c);
}

/* Orientation selector — chosen once from (TRANSA, TRANSB), dispatched
 * per chunk so the omp boilerplate is written exactly once. */
enum ygemm_klass { Y_NN, Y_TN, Y_NT, Y_TT };

static inline void ygemm_dispatch(enum ygemm_klass klass, ptrdiff_t js, ptrdiff_t je,
                                  ptrdiff_t m, ptrdiff_t k, T alpha,
                                  const T *a, ptrdiff_t lda, const T *b, ptrdiff_t ldb,
                                  T *c, ptrdiff_t ldc, bool conj_a, bool conj_b)
{
    switch (klass) {
    case Y_NN: ygemm_nn_core(js, je, m, k, alpha, a, lda, b, ldb, c, ldc); break;
    case Y_TN: ygemm_tn_core(js, je, m, k, alpha, a, lda, b, ldb, c, ldc, conj_a); break;
    case Y_NT: ygemm_nt_core(js, je, m, k, alpha, a, lda, b, ldb, c, ldc, conj_b); break;
    case Y_TT: ygemm_tt_core(js, je, m, k, alpha, a, lda, b, ldb, c, ldc, conj_a, conj_b); break;
    }
}

static void ygemm_core(
    char transa, char transb,
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    const T *b, ptrdiff_t ldb,
    const T *beta_,
    T *c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        ygemm_serial(transa, transb, m, n, k, alpha_, a, lda,
                     b, ldb, beta_, c, ldc);
        return;
    }
#endif

    const T alpha = *alpha_, beta = *beta_;
    const char ta = trans_code(transa);
    const char tb = trans_code(transb);

    if (m <= 0 || n <= 0) return;

    ygemm_beta_prepass(m, n, beta, c, ldc);
    if (alpha == zero || k == 0) return;

    const bool conj_a = (ta == 'C');
    const bool conj_b = (tb == 'C');
    enum ygemm_klass klass;
    if (ta == 'N' && tb == 'N')                       klass = Y_NN;
    else if ((ta == 'T' || ta == 'C') && tb == 'N')   klass = Y_TN;
    else if (ta == 'N' && (tb == 'T' || tb == 'C'))   klass = Y_NT;
    else                                              klass = Y_TT;

#ifdef _OPENMP
    if (n >= YGEMM_OMP_N_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            ptrdiff_t tid = omp_get_thread_num();
            ptrdiff_t nth  = omp_get_num_threads();
            ptrdiff_t js  = blas_part_bound(n, tid, nth);
            ptrdiff_t je  = blas_part_bound(n, tid + 1, nth);
            ygemm_dispatch(klass, js, je, m, k, alpha, a, lda, b, ldb,
                           c, ldc, conj_a, conj_b);
        }
        return;
    }
#endif
    ygemm_dispatch(klass, 0, n, m, k, alpha, a, lda, b, ldb,
                   c, ldc, conj_a, conj_b);
}

EPBLAS_FACADE_GEMM(ygemm, T)
