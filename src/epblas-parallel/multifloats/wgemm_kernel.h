/*
 * wgemm_kernel.h — internal kernel surface shared by the two translation
 * units the multifloats (complex double-double / complex64x2) wgemm overlay
 * is split across:
 *
 *   wgemm_serial.cpp    The pure single-thread complex GEMM (no OpenMP).
 *                       Owns ALL the math — scalar + AVX2 SIMD complex
 *                       packers, scalar + SIMD micro-kernels, block policy
 *                       — and the public `wgemm_serial` entry. Called
 *                       directly by the L3 routines that run wgemm trailing
 *                       updates inside their OWN parallel region, and by
 *                       wgemm_ as its serial / nesting delegate.
 *
 *   wgemm_parallel.cpp  The public Fortran entry `wgemm_` — threading
 *                       orchestration only. Delegates to wgemm_serial when
 *                       called from inside a parallel region; otherwise fans
 *                       the jc loop across a team.
 *
 * Leaf names are wgemm_-prefixed so they keep external linkage without
 * colliding with the other routines' packers/kernels in the same archive.
 */
#ifndef EPBLAS_PARALLEL_MULTIFLOATS_WGEMM_KERNEL_H
#define EPBLAS_PARALLEL_MULTIFLOATS_WGEMM_KERNEL_H

#include <cstddef>
#include <multifloats.h>

using wgemm_T = multifloats::complex64x2;

/* Normalize a Fortran trans char (N/T/C kept distinct for complex). */
int wgemm_trans_code(const char *p, std::size_t len);

/* Cache-block sizes (env-overridable MBLAS_MC/KC/NC). */
void wgemm_choose_blocks(int *MC, int *KC, int *NC);

/* Scalar (non-SIMD) panel packers + macro kernel. */
void wgemm_pack_A(const wgemm_T *A, int lda,
                  int ic, int pc, int ib, int pb, int ta, wgemm_T *Ap);
void wgemm_pack_B(const wgemm_T *B, int ldb,
                  int pc, int jc, int pb, int jb, int tb, wgemm_T *Bp);
void wgemm_inner_kernel(int ib, int jb, int pb, wgemm_T alpha,
                        const wgemm_T *Ap, const wgemm_T *Bp,
                        wgemm_T *C, int ldc);

#ifdef WBLAS_SIMD_DD
/* SIMD panel width W = simd_fast::NR * WGEMM_SIMD_NR_PAN (runtime accessor so
 * the parallel driver can size the SoA Bp pad without the SIMD headers). */
int wgemm_simd_pack_W(void);
/* Complex SoA pack of B (re/im hi/lo split) + templated AVX2 kernel. */
void wgemm_pack_B_soa_complex(const wgemm_T *B, int ldb,
                              int pc, int jc, int pb, int jb, int tb,
                              double *Bp_rh, double *Bp_rl,
                              double *Bp_ih, double *Bp_il);
void wgemm_inner_kernel_simd_complex(int ib, int jb, int pb, wgemm_T alpha,
                                     const wgemm_T *Ap,
                                     const double *Bp_rh, const double *Bp_rl,
                                     const double *Bp_ih, const double *Bp_il,
                                     wgemm_T *C, int ldc);
#endif /* WBLAS_SIMD_DD */

/* Pure single-thread GEMM. Same Fortran-ABI signature as wgemm_ — no OpenMP. */
extern "C" void wgemm_serial(
    const char *transa, const char *transb,
    const int *m_, const int *n_, const int *k_,
    const wgemm_T *alpha_,
    const wgemm_T *a, const int *lda_,
    const wgemm_T *b, const int *ldb_,
    const wgemm_T *beta_,
    wgemm_T *c, const int *ldc_,
    std::size_t transa_len, std::size_t transb_len);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_WGEMM_KERNEL_H */
