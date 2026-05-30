/*
 * ygemm_kernel.h — internal shared declarations for the kind10 complex
 * GEMM overlay (COMPLEX(KIND=10) / _Complex long double), split across
 * two translation units:
 *
 *   ygemm_serial.c   — all the math: the four orientation cores (one per
 *                      (TRANSA, TRANSB) class), the beta pre-pass, and the
 *                      pure-serial Fortran-ABI entry `ygemm_serial`. No
 *                      `#pragma omp`. Called directly by the complex L3
 *                      routines (ytrsm, ytrmm, ysyrk, … ) that run ygemm
 *                      trailing updates inside their OWN parallel region.
 *   ygemm_parallel.c — the public Fortran entry `ygemm_`: threading
 *                      orchestration only, with an `omp_in_parallel()`
 *                      guard that delegates to `ygemm_serial` when called
 *                      from inside another routine's parallel region
 *                      (opening a nested team there trips the libgomp
 *                      barrier wedge — see memory project-etrsm-omp4-wedge).
 *
 * The cores are range-parameterized over the column (j) axis of C —
 * [j_start, j_end). The serial entry calls them over the full range; the
 * parallel driver partitions the range across one team and calls the same
 * cores per chunk. ygemm has no packing/micro-kernel to share (a complex
 * long double tile over-pressures the x87 stack — see ygemm_serial.c), so
 * the cores ARE the math, mirroring the etrsm split rather than egemm's.
 *
 * `ygemm_serial` keeps the exact Fortran-ABI signature of ygemm_ so callers
 * already inside a parallel region swap the symbol name only.
 */
#ifndef EPBLAS_PARALLEL_KIND10_YGEMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_YGEMM_KERNEL_H

#include <stddef.h>
#include <complex.h>

typedef _Complex long double ygemm_T;

/* C := beta*C pre-pass over the full M×N tile (handles K==0 / alpha==0). */
void ygemm_beta_prepass(int M, int N, ygemm_T beta, ygemm_T *c, int ldc);

/* ── Orientation cores: serial work over columns [j_start, j_end) of C.
 *    One per (TRANSA, TRANSB) class; conjugation is a runtime flag. ── */

/* TA='N', TB='N': rank-1 update over l, K-unrolled by 2. */
void ygemm_nn_core(int j_start, int j_end, int M, int K, ygemm_T alpha,
                   const ygemm_T *a, int lda, const ygemm_T *b, int ldb,
                   ygemm_T *c, int ldc);

/* TA in {'T','C'}, TB='N': dot of A col i and B col j. */
void ygemm_tn_core(int j_start, int j_end, int M, int K, ygemm_T alpha,
                   const ygemm_T *a, int lda, const ygemm_T *b, int ldb,
                   ygemm_T *c, int ldc, int conj_a);

/* TA='N', TB in {'T','C'}: rank-1 update over l, K-unrolled by 2. */
void ygemm_nt_core(int j_start, int j_end, int M, int K, ygemm_T alpha,
                   const ygemm_T *a, int lda, const ygemm_T *b, int ldb,
                   ygemm_T *c, int ldc, int conj_b);

/* Both transposed: A col i × B row j, dot-product form. */
void ygemm_tt_core(int j_start, int j_end, int M, int K, ygemm_T alpha,
                   const ygemm_T *a, int lda, const ygemm_T *b, int ldb,
                   ygemm_T *c, int ldc, int conj_a, int conj_b);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same signature as ygemm_. */
void ygemm_serial(
    const char *transa, const char *transb,
    const int *m_, const int *n_, const int *k_,
    const ygemm_T *alpha_,
    const ygemm_T *a, const int *lda_,
    const ygemm_T *b, const int *ldb_,
    const ygemm_T *beta_,
    ygemm_T *c, const int *ldc_,
    size_t transa_len, size_t transb_len);

#endif /* EPBLAS_PARALLEL_KIND10_YGEMM_KERNEL_H */
