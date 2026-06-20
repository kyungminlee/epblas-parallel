/*
 * mgemm_kernel.h — internal kernel surface shared by the two translation
 * units the multifloats (double-double / float64x2) mgemm overlay is split
 * across:
 *
 *   mgemm_serial.cpp    The pure single-thread GEMM (no OpenMP). Owns ALL
 *                       the math — scalar + SIMD packers, scalar + SIMD
 *                       micro-kernels, block-size policy — and the public
 *                       `mgemm_serial` entry. Called directly by the L3
 *                       routines (msyrk/msymm/msyr2k/mtrmm/mtrsm, and the
 *                       w* twins) that run mgemm trailing updates inside
 *                       their OWN parallel region, and by mgemm_ as its
 *                       serial branch / nesting delegate.
 *
 *   mgemm_parallel.cpp  The public Fortran entry `mgemm_` — threading
 *                       orchestration only. Delegates to mgemm_serial when
 *                       called from inside a parallel region (the nested
 *                       case — previously handled implicitly by OpenMP's
 *                       default-off nested parallelism collapsing the inner
 *                       team to one thread; now an explicit serial call,
 *                       matching kind10/kind16 and avoiding team-of-1
 *                       churn); otherwise fans the jc loop across a team.
 *
 * Everything here is internal to the overlay. `mgemm_serial` keeps the exact
 * Fortran-ABI signature of mgemm_ so callers already inside a parallel region
 * swap the symbol name only. Leaf names are mgemm_-prefixed so they keep
 * external linkage without colliding with the other routines' packers/kernels
 * in the same archive.
 */
#ifndef EPBLAS_PARALLEL_MULTIFLOATS_MGEMM_KERNEL_H
#define EPBLAS_PARALLEL_MULTIFLOATS_MGEMM_KERNEL_H

#include <cstddef>
#include <multifloats.h>

using mgemm_T = multifloats::float64x2;

/* Normalize a Fortran trans char to a code ('C' ≡ 'T' for real input). */
int mgemm_trans_code(const char *p, std::size_t len);

/* Cache-block sizes (env-overridable MBLAS_MC/KC/NC). */
void mgemm_choose_blocks(int *MC, int *KC, int *NC);

/* Scalar (non-SIMD) panel packers + macro kernel. */
void mgemm_pack_A(const mgemm_T *A, int lda,
                  int ic, int pc, int ib, int pb, int ta, mgemm_T *Ap);
void mgemm_pack_B(const mgemm_T *B, int ldb,
                  int pc, int jc, int pb, int jb, int tb, mgemm_T *Bp);
void mgemm_inner_kernel(int ib, int jb, int pb, mgemm_T alpha,
                        const mgemm_T *Ap, const mgemm_T *Bp,
                        mgemm_T *C, int ldc);

#ifdef MBLAS_SIMD_DD
/* SIMD panel width W = simd_fast::NR * MGEMM_SIMD_NR_PAN (runtime accessor so
 * the parallel driver can size the SoA Bp pad without pulling in the SIMD
 * headers). */
int mgemm_simd_pack_W(void);
/* SoA pack of B (hi/lo split) + templated AVX2 micro-kernel wrapper. */
void mgemm_pack_B_soa(const mgemm_T *B, int ldb,
                      int pc, int jc, int pb, int jb, int tb,
                      double *Bp_hi, double *Bp_lo);
void mgemm_inner_kernel_simd(int ib, int jb, int pb, mgemm_T alpha,
                             const mgemm_T *Ap,
                             const double *Bp_hi, const double *Bp_lo,
                             mgemm_T *C, int ldc);
#endif /* MBLAS_SIMD_DD */

/* Pure single-thread GEMM. Same Fortran-ABI signature as mgemm_ — no OpenMP. */
extern "C" void mgemm_serial(
    const char *transa, const char *transb,
    const int *m_, const int *n_, const int *k_,
    const mgemm_T *alpha_,
    const mgemm_T *a, const int *lda_,
    const mgemm_T *b, const int *ldb_,
    const mgemm_T *beta_,
    mgemm_T *c, const int *ldc_,
    std::size_t transa_len, std::size_t transb_len);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_MGEMM_KERNEL_H */
