/*
 * xl3_complex.h — shared L3 kernel + packers for the kind16 complex path,
 * PARALLEL OVERLAY COPY. Faithful transcription of the epblas-openblas
 * substrate src/epblas-openblas/common/qblas_l3_complex.h; the par overlay
 * keeps its own copy (par and ob never co-link), mirroring the kind10
 * etri_kernel.c convention. Symbol names are unchanged from the ob source.
 *
 * Port source: OpenBLAS.
 *   - kernel/generic/zgemmkernel_2x2.c   → qblas_xgemm_kernel
 *                                          (NN path only; conjugation absorbed
 *                                           into the packers — see below)
 *   - kernel/generic/zgemm_ncopy_2.c     → qblas_xgemm_ncopy
 *   - kernel/generic/zgemm_tcopy_2.c     → qblas_xgemm_tcopy
 *   - kernel/generic/zgemm_beta.c        → qblas_xgemm_beta
 *
 * Shared across the L3 complex ports (xgemm, xsymm, xsyrk, xsyr2k, xtrmm,
 * xtrsm, xgemmtr, xhemm, xherk, xher2k — qtrsm added). Each routine still owns its own
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
#ifndef EPBLAS_PARALLEL_KIND16_XL3_COMPLEX_H
#define EPBLAS_PARALLEL_KIND16_XL3_COMPLEX_H

#include <stddef.h>
#include <quadmath.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Register-tile dimensions (compile-time constants) ───────────── */
#define QBLAS_XGEMM_MR 2
#define QBLAS_XGEMM_NR 2

/* ── Cache-block sizes (fixed compile-time constants) ───────────────
 *
 * Sized so MC*KC of complex `__float128` (32 B each on x86-64) fits
 * inside ~256 KB of L2:  64 * 256 * 32 B = 512 KB nominal (we run a
 * bit larger than the real path since complex doubles the per-element
 * size; the adaptive MC at small K rebalances).
 *
 * NC = 512 keeps KC*NC = 4 MB which lives in L3 across the KC-walk
 * (one OCOPY per (jc, pc) tile). */
#define QBLAS_XGEMM_GEMM_P  64
#define QBLAS_XGEMM_GEMM_Q  256
#define QBLAS_XGEMM_GEMM_R  512

/* ── Microkernel: NN-only 2x2 complex outer-product over K ────────────
 *
 * Compute C += alpha * Ap * Bp for one (bm, bn) tile of C, all dims in
 * complex elements. ldc is in complex elements; C is `__float128 *`
 * (each complex element = 2 __float128s, interleaved re,im).
 *
 * Ap layout (TCOPY of normal A or NCOPY of trans A, both shapes):
 *   per kernel i-slice (one MR=2 row strip of C), the data is bk
 *   K-rows × 4 __float128s, with each K-row = [r0_re, r0_im, r1_re,
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
void qblas_xgemm_kernel(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        __float128 alphar, __float128 alphai,
                        const __float128 *Ap,
                        const __float128 *Bp,
                        __float128 *C, ptrdiff_t ldc);

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
void qblas_xgemm_ncopy(ptrdiff_t m, ptrdiff_t n,
                       bool conj,
                       const __float128 *a, ptrdiff_t lda,
                       __float128 *b);

/* ── tcopy: pack 2-source-col K-strips (transposed view) ────────────
 *
 * Used as ICOPY for normal A and as OCOPY for trans B.
 *
 * Same `conj` semantics and lda convention as qblas_xgemm_ncopy.
 */
void qblas_xgemm_tcopy(ptrdiff_t m, ptrdiff_t n,
                       bool conj,
                       const __float128 *a, ptrdiff_t lda,
                       __float128 *b);

/* ── Beta pre-pass: C := beta * C with complex beta ──────────────── */
void qblas_xgemm_beta(ptrdiff_t m, ptrdiff_t n,
                      __float128 beta_r, __float128 beta_i,
                      __float128 *c, ptrdiff_t ldc);

/* ── Env-var block-size overrides (lazy, idempotent) ──────────────── */
void qblas_xgemm_blocks(ptrdiff_t *mc, ptrdiff_t *kc, ptrdiff_t *nc);

/* ── SYMM-aware packers (complex) ────────────────────────────────────
 *
 * Port source: OpenBLAS kernel/generic/zsymm_ucopy_2.c / zsymm_lcopy_2.c.
 *
 * Read an (m × n) slab from a complex-symmetric matrix `a` (col-major,
 * leading dimension lda in COMPLEX elements) and emit it in the
 * MR=2 / NR=2 strip-packed format expected by the shared microkernel.
 *
 * `posX`, `posY` carry the same meaning as in the real path —
 * see qblas_l3_real.h. For SYMM (NOT hemm) there is no conjugation,
 * so this pair of packers does NOT take a `conj` flag; the imag float
 * is copied through unchanged.
 *
 * lda is in COMPLEX elements; the implementation doubles internally.
 */
void qblas_xsymm_ucopy(ptrdiff_t m, ptrdiff_t n,
                       const __float128 *a, ptrdiff_t lda,
                       ptrdiff_t posX, ptrdiff_t posY,
                       __float128 *b);

void qblas_xsymm_lcopy(ptrdiff_t m, ptrdiff_t n,
                       const __float128 *a, ptrdiff_t lda,
                       ptrdiff_t posX, ptrdiff_t posY,
                       __float128 *b);

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
void qblas_xhemm_ucopy(ptrdiff_t m, ptrdiff_t n,
                       const __float128 *a, ptrdiff_t lda,
                       ptrdiff_t posX, ptrdiff_t posY,
                       __float128 *b);

void qblas_xhemm_lcopy(ptrdiff_t m, ptrdiff_t n,
                       const __float128 *a, ptrdiff_t lda,
                       ptrdiff_t posX, ptrdiff_t posY,
                       __float128 *b);

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
 * is NN-only (matching xgemm/xsymm — conjugation absorbed at pack
 * time), so we provide the OC variants directly instead of growing a
 * GEMM_KERNEL_R analogue just for HEMM.
 */
void qblas_xhemm_ucopy_oc(ptrdiff_t m, ptrdiff_t n,
                          const __float128 *a, ptrdiff_t lda,
                          ptrdiff_t posX, ptrdiff_t posY,
                          __float128 *b);

void qblas_xhemm_lcopy_oc(ptrdiff_t m, ptrdiff_t n,
                          const __float128 *a, ptrdiff_t lda,
                          ptrdiff_t posX, ptrdiff_t posY,
                          __float128 *b);

/* ── SYR2K / HER2K substrate (complex rank-2k packed-GEMM port) ──────
 *
 * Beta pre-passes scale the UPLO triangle of C in place: the SYR2K one
 * takes a COMPLEX beta; the HER2K one a REAL beta and additionally
 * forces the diagonal imag = 0 (Hermitian C contract). The two-pass
 * diagonal-aware kernels mirror the real esyr2k twin — see xsyr2k.c /
 * xher2k.c for the calling convention (pass 1 flag=1 folds the
 * diagonal block via the symmetric/Hermitian mirror, pass 2 flag=0
 * adds the off-diagonal strips). Port source: OpenBLAS
 * driver/level3/{syr2k,zher2k}_kernel.c + {syrk,zherk}_beta.c.
 */
void qblas_xsyrk_beta_u(ptrdiff_t n, __float128 br, __float128 bi,
                        __float128 *c, ptrdiff_t ldc);
void qblas_xsyrk_beta_l(ptrdiff_t n, __float128 br, __float128 bi,
                        __float128 *c, ptrdiff_t ldc);
void qblas_xherk_beta_u(ptrdiff_t n, __float128 br,
                        __float128 *c, ptrdiff_t ldc);
void qblas_xherk_beta_l(ptrdiff_t n, __float128 br,
                        __float128 *c, ptrdiff_t ldc);

void qblas_xsyr2k_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           __float128 alphar, __float128 alphai,
                           const __float128 *a, const __float128 *b,
                           __float128 *c, ptrdiff_t ldc,
                           ptrdiff_t offset, bool flag);
void qblas_xsyr2k_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           __float128 alphar, __float128 alphai,
                           const __float128 *a, const __float128 *b,
                           __float128 *c, ptrdiff_t ldc,
                           ptrdiff_t offset, bool flag);
void qblas_xher2k_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           __float128 alphar, __float128 alphai,
                           const __float128 *a, const __float128 *b,
                           __float128 *c, ptrdiff_t ldc,
                           ptrdiff_t offset, bool flag);
void qblas_xher2k_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           __float128 alphar, __float128 alphai,
                           const __float128 *a, const __float128 *b,
                           __float128 *c, ptrdiff_t ldc,
                           ptrdiff_t offset, bool flag);

#ifdef __cplusplus
}
#endif

#endif /* EPBLAS_PARALLEL_KIND16_XL3_COMPLEX_H */
