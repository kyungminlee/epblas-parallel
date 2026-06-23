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
#include <stdbool.h>
#include <stdlib.h>
#include "../common/blas_char.h"
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "ygemm_kernel.h"
#include "../common/epblas_facade.h"

typedef ygemm_TC TC;

static const TC zero = 0.0L + 0.0iL;

/* Threshold below which OMP parallel-for on the column axis isn't worth
 * the parallel-region setup. */
#define YGEMM_OMP_N_MIN 32

/* Orientation selector — chosen once from (TRANSA, TRANSB), dispatched
 * per chunk so the omp boilerplate is written exactly once. */
enum ygemm_klass { Y_NN, Y_TN, Y_NT, Y_TT };

static inline void ygemm_dispatch(enum ygemm_klass klass, ptrdiff_t js, ptrdiff_t je,
                                  ptrdiff_t m, ptrdiff_t k, TC alpha,
                                  const TC *a, ptrdiff_t lda, const TC *b, ptrdiff_t ldb,
                                  TC *c, ptrdiff_t ldc, bool conj_a, bool conj_b)
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
        ygemm_serial(transa, transb, m, n, k, alpha_, a, lda,
                     b, ldb, beta_, c, ldc);
        return;
    }
#endif

    const TC alpha = *alpha_, beta = *beta_;
    const char ta = blas_trans_complex(transa);
    const char tb = blas_trans_complex(transb);

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
        /* TT under threading: op(B)=B^T is read row-strided (ldb) and the
         * dot re-reads each B row M times. With 4 threads at N≥128 the
         * scattered B traffic (a 256×256 complex(10) panel is 2 MB) makes
         * it bandwidth-bound — par loses to ob's packed B. Transpose B^T
         * into a contiguous K×N buffer ONCE (folding conj_b in), then run
         * it as a TN dot over contiguous Bt: same l-order, bit-identical,
         * and TN already threads well. Cheap O(N·K) vs the O(M·N·K) GEMM. */
        if (klass == Y_TT && k >= 8) {
            TC *bt = malloc((size_t)k * (size_t)n * sizeof(TC));
            if (bt) {
                #pragma omp parallel
                {
                    ptrdiff_t tid = omp_get_thread_num();
                    ptrdiff_t nth = omp_get_num_threads();
                    ptrdiff_t js  = blas_part_bound(n, tid, nth);
                    ptrdiff_t je  = blas_part_bound(n, tid + 1, nth);
                    for (ptrdiff_t j2 = js; j2 < je; ++j2) {
                        TC *btj = &bt[(size_t)j2 * k];
                        for (ptrdiff_t l = 0; l < k; ++l) {
                            const TC v = b[(size_t)l * ldb + j2];
                            btj[l] = conj_b ? ~v : v;
                        }
                    }
                    /* Each thread transposes and consumes only its own
                     * [js,je) columns of bt — no cross-thread sharing,
                     * so no barrier needed. */
                    ygemm_tn_core(js, je, m, k, alpha, a, lda, bt, k,
                                  c, ldc, conj_a);
                }
                free(bt);
                return;
            }
            /* malloc failed — fall through to the strided core. */
        }
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

EPBLAS_FACADE_GEMM(ygemm, TC)
