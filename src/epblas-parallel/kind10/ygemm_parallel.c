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
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "ygemm_kernel.h"

typedef ygemm_T T;

static const T zero = 0.0L + 0.0iL;

/* Threshold below which OMP parallel-for on the column axis isn't worth
 * the parallel-region setup. */
#define YGEMM_OMP_N_MIN 32

static ptrdiff_t trans_code(const char *p, size_t len) {
    (void)len;
    return (char)toupper((unsigned char)*p);
}

/* Orientation selector — chosen once from (TRANSA, TRANSB), dispatched
 * per chunk so the omp boilerplate is written exactly once. */
enum ygemm_klass { Y_NN, Y_TN, Y_NT, Y_TT };

static inline void ygemm_dispatch(enum ygemm_klass klass, ptrdiff_t js, ptrdiff_t je,
                                  ptrdiff_t M, ptrdiff_t K, T alpha,
                                  const T *a, ptrdiff_t lda, const T *b, ptrdiff_t ldb,
                                  T *c, ptrdiff_t ldc, ptrdiff_t conj_a, ptrdiff_t conj_b)
{
    switch (klass) {
    case Y_NN: ygemm_nn_core(js, je, M, K, alpha, a, lda, b, ldb, c, ldc); break;
    case Y_TN: ygemm_tn_core(js, je, M, K, alpha, a, lda, b, ldb, c, ldc, conj_a); break;
    case Y_NT: ygemm_nt_core(js, je, M, K, alpha, a, lda, b, ldb, c, ldc, conj_b); break;
    case Y_TT: ygemm_tt_core(js, je, M, K, alpha, a, lda, b, ldb, c, ldc, conj_a, conj_b); break;
    }
}

void ygemm_(
    const char *transa, const char *transb,
    const int *m_, const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t transa_len, size_t transb_len)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        const ptrdiff_t m_pt = *m_, n_pt = *n_, k_pt = *k_, lda_pt = *lda_, ldb_pt = *ldb_, ldc_pt = *ldc_;
        ygemm_serial(transa, transb, &m_pt, &n_pt, &k_pt, alpha_, a, &lda_pt,
                     b, &ldb_pt, beta_, c, &ldc_pt, transa_len, transb_len);
        return;
    }
#endif

    const ptrdiff_t M = *m_, N = *n_, K = *k_;
    const ptrdiff_t lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const ptrdiff_t ta = trans_code(transa, transa_len);
    const ptrdiff_t tb = trans_code(transb, transb_len);

    if (M <= 0 || N <= 0) return;

    ygemm_beta_prepass(M, N, beta, c, ldc);
    if (alpha == zero || K == 0) return;

    const ptrdiff_t conj_a = (ta == 'C');
    const ptrdiff_t conj_b = (tb == 'C');
    enum ygemm_klass klass;
    if (ta == 'N' && tb == 'N')                       klass = Y_NN;
    else if ((ta == 'T' || ta == 'C') && tb == 'N')   klass = Y_TN;
    else if (ta == 'N' && (tb == 'T' || tb == 'C'))   klass = Y_NT;
    else                                              klass = Y_TT;

#ifdef _OPENMP
    if (N >= YGEMM_OMP_N_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            ptrdiff_t tid = omp_get_thread_num();
            ptrdiff_t nt  = omp_get_num_threads();
            ptrdiff_t js  = (ptrdiff_t)(((long long)N * tid) / nt);
            ptrdiff_t je  = (ptrdiff_t)(((long long)N * (tid + 1)) / nt);
            ygemm_dispatch(klass, js, je, M, K, alpha, a, lda, b, ldb,
                           c, ldc, conj_a, conj_b);
        }
        return;
    }
#endif
    ygemm_dispatch(klass, 0, N, M, K, alpha, a, lda, b, ldb,
                   c, ldc, conj_a, conj_b);
}
