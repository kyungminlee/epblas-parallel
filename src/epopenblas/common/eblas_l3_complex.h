/*
 * eblas_l3_complex.h — shared L3 kernel + packers for the kind10 complex path.
 *
 * Port source: OpenBLAS.
 *   - kernel/generic/zgemmkernel_2x2.c   → eblas_ygemm_kernel
 *                                          (NN path only; conjugation absorbed
 *                                           into the packers — see below)
 *   - kernel/generic/zgemm_ncopy_2.c     → eblas_ygemm_ncopy
 *   - kernel/generic/zgemm_tcopy_2.c     → eblas_ygemm_tcopy
 *   - kernel/generic/zgemm_beta.c        → eblas_ygemm_beta
 *
 * Shared across the L3 complex ports (ygemm, ysymm, ysyrk, ysyr2k, ytrmm,
 * ytrsm, ygemmtr, yhemm, yherk, yher2k — etrsm added). Each routine still owns its own
 * dispatch + level3 driver — the shared bits are only the per-tile
 * microkernel, the column-stride packers, and the beta pre-pass.
 *
 * Conjugation handling: OpenBLAS's ZGEMM kernel implements 4 conjugation
 * paths (NN / NR-NC / RN-CN / RR-CC) selected at compile time. For this
 * no-SIMD scalar port we collapse all 4 into one by having the packers
 * negate the imag float when their `conj` flag is set. The kernel only
 * implements the unconjugated complex-product form. This is bit-exact
 * (a sign flip is an exact IEEE-754 op) and keeps the K-loop branch-free.
 */
#ifndef EBLAS_L3_COMPLEX_H
#define EBLAS_L3_COMPLEX_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Register-tile dimensions (compile-time constants) ───────────── */
#define EBLAS_YGEMM_MR 2
#define EBLAS_YGEMM_NR 2

/* ── Cache-block defaults (overridable via getenv at routine entry) ─
 *
 * Sized so MC*KC of complex `long double` (32 B each on x86-64) fits
 * inside ~256 KB of L2:  64 * 256 * 32 B = 512 KB nominal (we run a
 * bit larger than the real path since complex doubles the per-element
 * size; the adaptive MC at small K rebalances).
 *
 * NC = 512 keeps KC*NC = 4 MB which lives in L3 across the KC-walk
 * (one OCOPY per (jc, pc) tile). */
#define EBLAS_YGEMM_GEMM_P  64
#define EBLAS_YGEMM_GEMM_Q  256
#define EBLAS_YGEMM_GEMM_R  512

/* ── Microkernel: NN-only 2x2 complex outer-product over K ────────────
 *
 * Compute C += alpha * Ap * Bp for one (bm, bn) tile of C, all dims in
 * complex elements. ldc is in complex elements; C is `long double *`
 * (each complex element = 2 long doubles, interleaved re,im).
 *
 * Ap layout (TCOPY of normal A or NCOPY of trans A, both shapes):
 *   per kernel i-slice (one MR=2 row strip of C), the data is bk
 *   K-rows × 4 long doubles, with each K-row = [r0_re, r0_im, r1_re,
 *   r1_im] for the 2 M-rows of the strip. Conjugation, if any, was
 *   absorbed at pack time by negating the im parts.
 *
 * Bp layout (NCOPY of normal B or TCOPY of trans B):
 *   per kernel j-panel (one NR=2 col strip of C), bk K-rows × 4 long
 *   doubles, with each K-row = [c0_re, c0_im, c1_re, c1_im].
 *
 * alpha is split into (alphar, alphai) — the complex multiply is fused
 * with the C accumulate at the bottom of each tile.
 */
void eblas_ygemm_kernel(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        long double alphar, long double alphai,
                        const long double *Ap,
                        const long double *Bp,
                        long double *C, ptrdiff_t ldc);

/* ── ncopy: pack 2 source cols per panel ────────────────────────────
 *
 * Used as OCOPY for normal B (m=K-dim of B, n=N-dim) and as ICOPY
 * for trans/conj-trans A (m=K-dim of op(A), n=M-dim of op(A)).
 *
 * `lda` is in complex elements (the packer doubles internally for
 * float-stride accesses).
 *
 * When `conj != 0`, the imag float of each complex element is negated
 * as it is written into `b` — this handles op = 'R' (conj-no-trans)
 * via ncopy and op = 'C' (conj-trans) via the ICOPY path.
 */
void eblas_ygemm_ncopy(ptrdiff_t m, ptrdiff_t n,
                       int conj,
                       const long double *a, ptrdiff_t lda,
                       long double *b);

/* ── tcopy: pack 2-source-col K-strips (transposed view) ────────────
 *
 * Used as ICOPY for normal A and as OCOPY for trans B.
 *
 * Same `conj` semantics and lda convention as eblas_ygemm_ncopy.
 */
void eblas_ygemm_tcopy(ptrdiff_t m, ptrdiff_t n,
                       int conj,
                       const long double *a, ptrdiff_t lda,
                       long double *b);

/* ── Beta pre-pass: C := beta * C with complex beta ──────────────── */
void eblas_ygemm_beta(ptrdiff_t m, ptrdiff_t n,
                      long double beta_r, long double beta_i,
                      long double *c, ptrdiff_t ldc);

/* ── Env-var block-size overrides (lazy, idempotent) ──────────────── */
void eblas_ygemm_blocks(int *mc, int *kc, int *nc);

/* ── SYMM-aware packers (complex) ────────────────────────────────────
 *
 * Port source: OpenBLAS kernel/generic/zsymm_ucopy_2.c / zsymm_lcopy_2.c.
 *
 * Read an (m × n) slab from a complex-symmetric matrix `a` (col-major,
 * leading dimension lda in COMPLEX elements) and emit it in the
 * MR=2 / NR=2 strip-packed format expected by the shared microkernel.
 *
 * `posX`, `posY` carry the same meaning as in the real path —
 * see eblas_l3_real.h. For SYMM (NOT hemm) there is no conjugation,
 * so this pair of packers does NOT take a `conj` flag; the imag float
 * is copied through unchanged.
 *
 * lda is in COMPLEX elements; the implementation doubles internally.
 */
void eblas_ysymm_ucopy(ptrdiff_t m, ptrdiff_t n,
                       const long double *a, ptrdiff_t lda,
                       ptrdiff_t posX, ptrdiff_t posY,
                       long double *b);

void eblas_ysymm_lcopy(ptrdiff_t m, ptrdiff_t n,
                       const long double *a, ptrdiff_t lda,
                       ptrdiff_t posX, ptrdiff_t posY,
                       long double *b);

/* ── HEMM-aware packers (complex Hermitian) ──────────────────────────
 *
 * Port source: OpenBLAS kernel/generic/zhemm_utcopy_2.c / zhemm_ltcopy_2.c.
 *
 * Same call signature and posX/posY semantics as the SYMM packers, but
 * the imag float is negated on the reflected-across-diagonal half (the
 * Hermitian conjugate) and zeroed on the diagonal itself (Hermitian
 * diagonals are real by definition; the input's diagonal imag is
 * discarded per the LAPACK ZHEMM contract).
 *
 * MR=2 and NR=2 in our kernel, so the upstream `_2` files cover both
 * the inside-copy (SIDE=L, ICOPY) and outside-copy (SIDE=R, OCOPY)
 * roles via the same function — exactly the upstream convention where
 * the Makefile maps HEMM_IUTCOPY and HEMM_OUTCOPY to the same source.
 */
void eblas_yhemm_ucopy(ptrdiff_t m, ptrdiff_t n,
                       const long double *a, ptrdiff_t lda,
                       ptrdiff_t posX, ptrdiff_t posY,
                       long double *b);

void eblas_yhemm_lcopy(ptrdiff_t m, ptrdiff_t n,
                       const long double *a, ptrdiff_t lda,
                       ptrdiff_t posX, ptrdiff_t posY,
                       long double *b);

/* OCOPY variants — used for the SIDE=R role (packing the Hermitian
 * matrix as the RIGHT factor). Differ from the IC variants only in the
 * imag-sign branch decisions: in IC use (SIDE=L), posX/posY mean
 * (row/col) of A and `offset > 0` puts us in the unstored half (need
 * conj-of-symmetric-mirror); in OC use (SIDE=R), posX/posY mean
 * (col/row) of A and `offset > 0` puts us in the stored half (no
 * conj). The diagonal write (imag = 0) is unchanged.
 *
 * Upstream OpenBLAS only ships one file (zhemm_utcopy / zhemm_ltcopy)
 * and gets away with using it for both roles by compiling zhemm_k.c
 * with -DNC for RSIDE — that switches the kernel to GEMM_KERNEL_R,
 * which conjugates Bp once more during the multiply, cancelling the
 * extra conjugation the IC-style packer baked in. Our shared kernel
 * is NN-only (matching ygemm/ysymm — conjugation absorbed at pack
 * time), so we provide the OC variants directly instead of growing a
 * GEMM_KERNEL_R analogue just for HEMM.
 */
void eblas_yhemm_ucopy_oc(ptrdiff_t m, ptrdiff_t n,
                          const long double *a, ptrdiff_t lda,
                          ptrdiff_t posX, ptrdiff_t posY,
                          long double *b);

void eblas_yhemm_lcopy_oc(ptrdiff_t m, ptrdiff_t n,
                          const long double *a, ptrdiff_t lda,
                          ptrdiff_t posX, ptrdiff_t posY,
                          long double *b);

/* ── Triangular beta pre-pass (complex) ──────────────────────────────
 *
 * Apply C := beta * C on the UPLO triangle only; off-UPLO triangle is
 * untouched (SYRK contract). Mirrors OpenBLAS driver/level3/syrk_k.c
 * `syrk_beta`. `c` is interleaved (re, im); ldc in complex elements.
 */
void eblas_ysyrk_beta_u(ptrdiff_t n,
                        long double beta_r, long double beta_i,
                        long double *c, ptrdiff_t ldc);
void eblas_ysyrk_beta_l(ptrdiff_t n,
                        long double beta_r, long double beta_i,
                        long double *c, ptrdiff_t ldc);

/* ── SYRK kernel: GEMM with diagonal-aware writeback (complex) ───────
 *
 * Port of OpenBLAS driver/level3/syrk_kernel.c (complex variant). See
 * the real header twin for the offset/strip semantics. `c` is `long
 * double *` (interleaved re,im); ldc, k, m, n in complex/element
 * counts.
 *
 * Complex SYRK has no conjugation (that's HERK — separate routine).
 * The pack functions used for this kernel must NOT set the `conj` flag.
 */
void eblas_ysyrk_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                          long double alphar, long double alphai,
                          const long double *Ap,
                          const long double *Bp,
                          long double *c, ptrdiff_t ldc,
                          ptrdiff_t offset);
void eblas_ysyrk_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                          long double alphar, long double alphai,
                          const long double *Ap,
                          const long double *Bp,
                          long double *c, ptrdiff_t ldc,
                          ptrdiff_t offset);

/* ── HERK triangular beta pre-pass (complex Hermitian C) ─────────────
 *
 * Apply C := beta * C on the UPLO triangle (beta is REAL for HERK) and
 * force the diagonal imaginary part to zero — the Hermitian C contract
 * requires real diagonal on output even when beta == 1. Mirrors
 * OpenBLAS driver/level3/zherk_beta.c. Differs from ysyrk_beta_{u,l}
 * by: (a) real beta only, no imag mix; (b) unconditional diag imag = 0.
 */
void eblas_yherk_beta_u(ptrdiff_t n, long double beta_r,
                        long double *c, ptrdiff_t ldc);
void eblas_yherk_beta_l(ptrdiff_t n, long double beta_r,
                        long double *c, ptrdiff_t ldc);

/* ── HERK kernel: SYRK kernel with diagonal-imag clamp ───────────────
 *
 * Port of OpenBLAS driver/level3/zherk_kernel.c. Identical to
 * eblas_ysyrk_kernel_{u,l} *except* that the diagonal element of each
 * NR×NR diagonal tile sets imag := 0 (not +=). Hermitian C diagonal
 * must remain real.
 *
 * alpha is REAL for HERK (the upstream kernel takes only alpha_r and
 * passes ZERO as alpha_i to GEMM_KERNEL). We mirror that ABI.
 *
 * Conjugation: HERK TRANS='N' uses GEMM_KERNEL_R upstream (conjugates
 * Bp); TRANS='C' uses GEMM_KERNEL_L (conjugates Ap). Our shared kernel
 * is NN-only, so the caller absorbs the conjugation at pack time —
 * pack Bp with conj=1 for TRANS='N', pack Ap with conj=1 for TRANS='C'.
 */
void eblas_yherk_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                          long double alphar,
                          const long double *Ap,
                          const long double *Bp,
                          long double *c, ptrdiff_t ldc,
                          ptrdiff_t offset);
void eblas_yherk_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                          long double alphar,
                          const long double *Ap,
                          const long double *Bp,
                          long double *c, ptrdiff_t ldc,
                          ptrdiff_t offset);

/* ── SYR2K kernel: two-pass diagonal-aware GEMM (complex) ────────────
 *
 * Port of OpenBLAS driver/level3/syr2k_kernel.c (complex variant). See
 * the real header twin for the offset/strip/flag semantics. Complex
 * SYR2K has no conjugation (that's HER2K — task #66). The pack
 * functions used for this kernel must NOT set the `conj` flag.
 */
void eblas_ysyr2k_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           long double alphar, long double alphai,
                           const long double *Ap,
                           const long double *Bp,
                           long double *c, ptrdiff_t ldc,
                           ptrdiff_t offset, int flag);
void eblas_ysyr2k_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           long double alphar, long double alphai,
                           const long double *Ap,
                           const long double *Bp,
                           long double *c, ptrdiff_t ldc,
                           ptrdiff_t offset, int flag);

/* ── HER2K kernel: SYR2K kernel with Hermitian diagonal writeback ────
 *
 * Port of OpenBLAS driver/level3/zher2k_kernel.c. Structural twin of
 * eblas_ysyr2k_kernel_{u,l}; differs only in the diagonal NR×NR sub-
 * block writeback:
 *
 *   SYR2K:  C[i,j].re += subbuf[i,j].re + subbuf[j,i].re
 *           C[i,j].im += subbuf[i,j].im + subbuf[j,i].im
 *
 *   HER2K:  C[i,j].re += subbuf[i,j].re + subbuf[j,i].re      (same)
 *           if i != j:  C[i,j].im += subbuf[i,j].im - subbuf[j,i].im
 *           else:       C[i,j].im  = 0                          (Hermitian)
 *
 * Why: the subbuf trick computes only α·A·B^H (one kernel call) and
 * mirrors its conjugate-transpose-partner ᾱ·B·A^H from subbuf[j,i],
 * giving the full α·A·B^H + ᾱ·B·A^H diagonal contribution. The imag
 * subtraction undoes the j↔i conjugation; the explicit imag=0 on the
 * actual diagonal enforces the Hermitian C contract.
 *
 * Two-pass calling convention (caller, per HER2K driver):
 *   pass 1: kernel(Ap=A, Bp=B, alpha = (αr,  αi), flag=1)
 *   pass 2: kernel(Ap=B, Bp=A, alpha = (αr, -αi), flag=0)
 *
 * Conjugation: TRANS='N' uses GEMM_KERNEL_R upstream (conj Bp);
 * TRANS='C' uses GEMM_KERNEL_L (conj Ap). The shared NN kernel absorbs
 * the conjugation via the packer `conj` flag — see yher2k.c for the
 * per-trans pack-time conj assignment.
 */
void eblas_yher2k_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           long double alphar, long double alphai,
                           const long double *Ap,
                           const long double *Bp,
                           long double *c, ptrdiff_t ldc,
                           ptrdiff_t offset, int flag);
void eblas_yher2k_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           long double alphar, long double alphai,
                           const long double *Ap,
                           const long double *Bp,
                           long double *c, ptrdiff_t ldc,
                           ptrdiff_t offset, int flag);

/* ── TRMM A-side triangular packers (complex) ────────────────────────
 *
 * Faithful ports of OpenBLAS kernel/generic/ztrmm_{ut,un,lt,ln}copy_2.c
 * with compile-time UNIT replaced by runtime `unit` and a `conj` flag
 * added to absorb conjugation at pack time (matching ygemm/ysymm
 * pattern). When conj is set, the imag float is negated as written.
 * lda is in COMPLEX elements; doubled internally.
 */
void eblas_ytrmm_iutcopy(ptrdiff_t m, ptrdiff_t n,
                         const long double *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY,
                         long double *b, int unit, int conj);
void eblas_ytrmm_iuncopy(ptrdiff_t m, ptrdiff_t n,
                         const long double *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY,
                         long double *b, int unit, int conj);
void eblas_ytrmm_iltcopy(ptrdiff_t m, ptrdiff_t n,
                         const long double *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY,
                         long double *b, int unit, int conj);
void eblas_ytrmm_ilncopy(ptrdiff_t m, ptrdiff_t n,
                         const long double *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY,
                         long double *b, int unit, int conj);

/* ── TRMM diagonal-aware microkernel (complex) ───────────────────────
 *
 * Faithful port of OpenBLAS kernel/generic/ztrmmkernel_2x2.c (NN-only
 * path — conjugation absorbed by the packers, same pattern as
 * eblas_ygemm_kernel). LEFT and TRANSA macros converted to runtime
 * `left` and `trans` flags. C := alpha * ba * bb (overwrite), per-
 * element 2 long doubles (interleaved re,im).
 */
void eblas_ytrmm_kernel(int left, int trans,
                        ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        long double alphar, long double alphai,
                        const long double *ba,
                        const long double *bb,
                        long double *C, ptrdiff_t ldc,
                        ptrdiff_t offset);

/* ── GEMM kernel, overwrite variant (complex) ────────────────────────
 *
 * C := alpha * Ap * Bp (no accumulate). Implemented as: zero the
 * complex tile, then call eblas_ygemm_kernel.
 */
void eblas_ygemm_kernel_store(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                              long double alphar, long double alphai,
                              const long double *Ap,
                              const long double *Bp,
                              long double *C, ptrdiff_t ldc);

/* ── TRSM A-side triangular packers (complex) ────────────────────────
 *
 * Faithful ports of OpenBLAS kernel/generic/ztrsm_{ut,un,lt,ln}copy_2.c.
 * lda is in COMPLEX(KIND=10) elements; the implementation doubles it
 * internally to step floats. The diagonal block uses the `compinv`
 * formula (Smith's reciprocal — guards against overflow when |ai| >
 * |ar|) baked from the upstream common.h. `conj` is the runtime
 * conjugation flag: when set, the imag float of every written element
 * is sign-flipped (matches the ygemm/ytrmm pattern of absorbing
 * conjugation at pack time so the kernel runs only the NN form).
 *
 * Effect of conj on `compinv`: conjugating A first, then inverting,
 * equals inverting then conjugating (both bit-exact transformations on
 * the imag-sign). The simplest implementation runs `compinv` on the
 * already-conjugated inputs.
 */
void eblas_ytrsm_iutcopy(ptrdiff_t m, ptrdiff_t n,
                         const long double *a, ptrdiff_t lda,
                         ptrdiff_t offset, long double *b,
                         int unit, int conj);
void eblas_ytrsm_iuncopy(ptrdiff_t m, ptrdiff_t n,
                         const long double *a, ptrdiff_t lda,
                         ptrdiff_t offset, long double *b,
                         int unit, int conj);
void eblas_ytrsm_iltcopy(ptrdiff_t m, ptrdiff_t n,
                         const long double *a, ptrdiff_t lda,
                         ptrdiff_t offset, long double *b,
                         int unit, int conj);
void eblas_ytrsm_ilncopy(ptrdiff_t m, ptrdiff_t n,
                         const long double *a, ptrdiff_t lda,
                         ptrdiff_t offset, long double *b,
                         int unit, int conj);

/* ── TRSM diagonal-aware microkernel (complex) ───────────────────────
 *
 * Faithful port of OpenBLAS kernel/generic/trsm_kernel_{LN,LT,RN,RT}.c
 * (Z-variant — uses the complex form of solve()). 4 variants collapsed
 * into one function dispatching on runtime `left`/`trans`. The
 * complex-product form inside `solve` matches the unconjugated branch
 * of the upstream macro (ZTRSM with C-suffix variants are not used —
 * conjugation is absorbed at pack time by the `conj` flag, so the
 * kernel only runs the unconjugated complex multiply).
 *
 * Per-element width = 2 long doubles (interleaved re,im); ldc, k, m,
 * n in complex element counts.
 */
void eblas_ytrsm_kernel(int left, int trans,
                        ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        const long double *ba,
                        const long double *bb,
                        long double *C, ptrdiff_t ldc,
                        ptrdiff_t offset);

#ifdef __cplusplus
}
#endif

#endif /* EBLAS_L3_COMPLEX_H */
