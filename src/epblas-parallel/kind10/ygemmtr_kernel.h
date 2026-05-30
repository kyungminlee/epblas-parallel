/*
 * ygemmtr_kernel.h — internal shared declarations for the kind10 complex
 * (COMPLEX(KIND=10) / _Complex long double) triangular GEMM-update overlay,
 * split across two translation units:
 *
 *   ygemmtr_serial.c   — all the math: the alpha==0 beta-only column scaler,
 *                        the per-column worker (op(A)·op(B) into one
 *                        triangle column of C, with the K-unrolled fast
 *                        path and the conj/trans variants), and the
 *                        pure-serial Fortran-ABI entry `ygemmtr_serial`.
 *                        No `#pragma omp`.
 *   ygemmtr_parallel.c — the public Fortran entry `ygemmtr_`: threading only
 *                        (one `omp parallel for schedule(static,1)` over the
 *                        columns j), with an `omp_in_parallel()` guard that
 *                        delegates to `ygemmtr_serial` when called from
 *                        inside another routine's parallel region.
 *
 * Work is partitioned by column j: the serial entry walks the columns in a
 * plain loop; the parallel driver hands the same per-column worker to the
 * team. This routine touches only C (no ygemm trailing calls), so the
 * nesting guard is purely about not opening a redundant nested team.
 */
#ifndef EPBLAS_PARALLEL_KIND10_YGEMMTR_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_YGEMMTR_KERNEL_H

#include <stddef.h>
#include <complex.h>

typedef _Complex long double ygemmtr_T;

/* C := beta*C over the triangle of columns [j_start, j_end) — the
 * alpha==0 / K==0 quick path. `upper` selects the stored triangle. */
void ygemmtr_beta_scale(int j_start, int j_end, int N, int upper,
                        ygemmtr_T beta, ygemmtr_T *c, int ldc);

/* One triangle column j of C := alpha·op(A)·op(B) + beta·C. The resolved
 * transpose/conjugate flags are passed in (computed once by the entry). */
void ygemmtr_col(int j, int N, int K, int upper,
                 ygemmtr_T alpha, ygemmtr_T beta,
                 const ygemmtr_T *a, int lda,
                 const ygemmtr_T *b, int ldb,
                 ygemmtr_T *c, int ldc,
                 int trans_a, int conj_a, int trans_b, int conj_b);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same signature as ygemmtr_. */
void ygemmtr_serial(const char *uplo, const char *transa, const char *transb,
                    const int *n_, const int *k_,
                    const ygemmtr_T *alpha_,
                    const ygemmtr_T *a, const int *lda_,
                    const ygemmtr_T *b, const int *ldb_,
                    const ygemmtr_T *beta_,
                    ygemmtr_T *c, const int *ldc_,
                    size_t uplo_len, size_t ta_len, size_t tb_len);

#endif /* EPBLAS_PARALLEL_KIND10_YGEMMTR_KERNEL_H */
