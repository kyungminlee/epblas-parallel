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

typedef _Complex long double ygemmtr_TC;

/* C := beta*C over the triangle of columns [j_start, j_end) — the
 * alpha==0 / K==0 quick path. `upper` selects the stored triangle. */
void ygemmtr_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t n, bool upper,
                        ygemmtr_TC beta, ygemmtr_TC *c, ptrdiff_t ldc);

/* One triangle column j of C := alpha·op(A)·op(B) + beta·C. The resolved
 * transpose/conjugate flags are passed in (computed once by the entry). */
void ygemmtr_col(ptrdiff_t j, ptrdiff_t n, ptrdiff_t k, bool upper,
                 ygemmtr_TC alpha, ygemmtr_TC beta,
                 const ygemmtr_TC *a, ptrdiff_t lda,
                 const ygemmtr_TC *b, ptrdiff_t ldb,
                 ygemmtr_TC *c, ptrdiff_t ldc,
                 bool trans_a, bool conj_a, bool trans_b, bool conj_b);

/* Pure-serial by-value core (no OpenMP). */
void ygemmtr_serial(char uplo, char transa, char transb,
                    ptrdiff_t n, ptrdiff_t k,
                    const ygemmtr_TC *alpha_,
                    const ygemmtr_TC *a, ptrdiff_t lda,
                    const ygemmtr_TC *b, ptrdiff_t ldb,
                    const ygemmtr_TC *beta_,
                    ygemmtr_TC *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND10_YGEMMTR_KERNEL_H */
