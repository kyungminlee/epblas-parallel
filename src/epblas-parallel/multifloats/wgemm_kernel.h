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
std::ptrdiff_t wgemm_trans_code(const char *p, std::size_t len);

/* Cache-block sizes (env-overridable MBLAS_MC/KC/NC). */
void wgemm_choose_blocks(std::ptrdiff_t *MC, std::ptrdiff_t *KC, std::ptrdiff_t *NC);

/* Scalar (non-SIMD) panel packers + macro kernel. */
void wgemm_pack_A(const wgemm_T *A, std::ptrdiff_t lda,
                  std::ptrdiff_t ic, std::ptrdiff_t pc, std::ptrdiff_t ib, std::ptrdiff_t pb, std::ptrdiff_t ta, wgemm_T *Ap);
void wgemm_pack_B(const wgemm_T *B, std::ptrdiff_t ldb,
                  std::ptrdiff_t pc, std::ptrdiff_t jc, std::ptrdiff_t pb, std::ptrdiff_t jb, std::ptrdiff_t tb, wgemm_T *Bp);
void wgemm_inner_kernel(std::ptrdiff_t ib, std::ptrdiff_t jb, std::ptrdiff_t pb, wgemm_T alpha,
                        const wgemm_T *Ap, const wgemm_T *Bp,
                        wgemm_T *C, std::ptrdiff_t ldc);

#ifdef WBLAS_SIMD_DD
/* SIMD panel width W = simd_fast::NR * WGEMM_SIMD_NR_PAN (runtime accessor so
 * the parallel driver can size the SoA Bp pad without the SIMD headers). */
std::ptrdiff_t wgemm_simd_pack_W(void);
/* Complex SoA pack of B (re/im hi/lo split) + templated AVX2 kernel. */
void wgemm_pack_B_soa_complex(const wgemm_T *B, std::ptrdiff_t ldb,
                              std::ptrdiff_t pc, std::ptrdiff_t jc, std::ptrdiff_t pb, std::ptrdiff_t jb, std::ptrdiff_t tb,
                              double *Bp_rh, double *Bp_rl,
                              double *Bp_ih, double *Bp_il);
void wgemm_inner_kernel_simd_complex(std::ptrdiff_t ib, std::ptrdiff_t jb, std::ptrdiff_t pb, wgemm_T alpha,
                                     const wgemm_T *Ap,
                                     const double *Bp_rh, const double *Bp_rl,
                                     const double *Bp_ih, const double *Bp_il,
                                     wgemm_T *C, std::ptrdiff_t ldc);
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

/* ptrdiff_t-by-reference forwarder to the Fortran-ABI wgemm_serial entry. The
 * internal L3 callers carry block dimensions as ptrdiff_t; this marshals them to
 * the int* ABI in one place so call sites stay identical apart from the name.
 *
 * TODO(core/facade split): remove this forwarder once wgemm_serial is split into
 * a ptrdiff_t-by-value core and a thin Fortran-ABI facade. The internal callers
 * then call the core directly and these *_pd renames revert to wgemm_serial. */
static inline void wgemm_serial_pd(
    const char *transa, const char *transb,
    const std::ptrdiff_t *m_, const std::ptrdiff_t *n_, const std::ptrdiff_t *k_,
    const wgemm_T *alpha_,
    const wgemm_T *a, const std::ptrdiff_t *lda_,
    const wgemm_T *b, const std::ptrdiff_t *ldb_,
    const wgemm_T *beta_,
    wgemm_T *c, const std::ptrdiff_t *ldc_,
    std::size_t transa_len, std::size_t transb_len)
{
    const int m = (int)*m_, n = (int)*n_, k = (int)*k_;
    const int lda = (int)*lda_, ldb = (int)*ldb_, ldc = (int)*ldc_;
    wgemm_serial(transa, transb, &m, &n, &k, alpha_, a, &lda, b, &ldb,
                 beta_, c, &ldc, transa_len, transb_len);
}

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_WGEMM_KERNEL_H */
