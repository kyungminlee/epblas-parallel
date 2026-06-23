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


/* Cache-block sizes (env-overridable MBLAS_MC/KC/NC). */
void mgemm_choose_blocks(std::ptrdiff_t *MC, std::ptrdiff_t *KC, std::ptrdiff_t *NC);

/* Scalar (non-SIMD) panel packers + macro kernel. */
void mgemm_pack_A(const mgemm_T *A, std::ptrdiff_t lda,
                  std::ptrdiff_t ic, std::ptrdiff_t pc, std::ptrdiff_t ib, std::ptrdiff_t pb, std::ptrdiff_t ta, mgemm_T *Ap);
void mgemm_pack_B(const mgemm_T *B, std::ptrdiff_t ldb,
                  std::ptrdiff_t pc, std::ptrdiff_t jc, std::ptrdiff_t pb, std::ptrdiff_t jb, std::ptrdiff_t tb, mgemm_T *Bp);
void mgemm_inner_kernel(std::ptrdiff_t ib, std::ptrdiff_t jb, std::ptrdiff_t pb, mgemm_T alpha,
                        const mgemm_T *Ap, const mgemm_T *Bp,
                        mgemm_T *C, std::ptrdiff_t ldc);

#ifdef MBLAS_SIMD_DD
/* SIMD panel width W = simd_fast::NR * MGEMM_SIMD_NR_PAN (runtime accessor so
 * the parallel driver can size the SoA Bp pad without pulling in the SIMD
 * headers). */
std::ptrdiff_t mgemm_simd_pack_W(void);
/* SoA pack of B (hi/lo split) + templated AVX2 micro-kernel wrapper. */
void mgemm_pack_B_soa(const mgemm_T *B, std::ptrdiff_t ldb,
                      std::ptrdiff_t pc, std::ptrdiff_t jc, std::ptrdiff_t pb, std::ptrdiff_t jb, std::ptrdiff_t tb,
                      double *Bp_hi, double *Bp_lo);
void mgemm_inner_kernel_simd(std::ptrdiff_t ib, std::ptrdiff_t jb, std::ptrdiff_t pb, mgemm_T alpha,
                             const mgemm_T *Ap,
                             const double *Bp_hi, const double *Bp_lo,
                             mgemm_T *C, std::ptrdiff_t ldc);
#endif /* MBLAS_SIMD_DD */

/* Pure single-thread GEMM — by-value ptrdiff_t core (NOT a Fortran entry).
 * Called directly by the L3 routines' trailing updates and by mgemm_core. */
extern "C" void mgemm_serial(
    char transa, char transb,
    std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
    const mgemm_T *alpha_,
    const mgemm_T *a, std::ptrdiff_t lda,
    const mgemm_T *b, std::ptrdiff_t ldb,
    const mgemm_T *beta_,
    mgemm_T *c, std::ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_MGEMM_KERNEL_H */
