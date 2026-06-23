/*
 * xgemm_serial — kind16 complex GEMM (COMPLEX(KIND=16), __complex128),
 * single-thread. This TU owns ALL of the xgemm math: the block plan, the
 * B-panel packer, the per-M-slab level3 worker, and the pure-serial
 * Fortran-ABI entry `xgemm_serial_`. xgemm_parallel.c only orchestrates
 * threads over these same pieces.
 *
 * Strategy: faithful port of the OpenBLAS GotoBLAS blocking nest (the ob
 * clone src/epblas-openblas/kind16/xgemm.c). NC × KC × MC three-level
 * blocking; ICOPY(A)/OCOPY(B) packing into MR×NR panels; a 2×2 register
 * microkernel (qblas_xgemm_kernel) whose four independent accumulator chains
 * overlap libquadmath soft-float call latency. Conjugation is absorbed into
 * the packers (imag sign flip), so the kernel is NN-only.
 *
 * Fortran ABI (xgemm_serial_ mirrors xgemm_ exactly):
 *   - scalars by pointer; complex scalar = __complex128 (re, im)
 *   - character args followed by hidden trailing `size_t` lengths
 *   - COMPLEX(KIND=16) ↔ __complex128; lda/ldb/ldc in complex elements
 * The substrate works on __float128* (interleaved re,im), reached by
 * reinterpreting the __complex128* a/b/c.
 */

#include "xgemm_kernel.h"
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include "xl3_complex.h"
#include <ctype.h>
#include <stddef.h>
#include <stdlib.h>

typedef __float128 R;

#define MR QBLAS_XGEMM_MR
#define NR QBLAS_XGEMM_NR

char xgemm_trans_code(char c) {
    return blas_up(c);
}

static bool op_is_conj(char c)  { return (c == 'C' || c == 'R') ? 1 : 0; }
static bool op_is_trans(char c) { return (c == 'T' || c == 'C') ? 1 : 0; }

/* ── Block plan (mirrors ob xgemm.c lines 136-155) ──────────────── */
void xgemm_make_plan(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, char ta, char tb, xgemm_plan_t *p)
{
    (void)m; (void)n;
    ptrdiff_t MC0, KC, NC;
    qblas_xgemm_blocks(&MC0, &KC, &NC);

    /* Adaptive MC for small K, sized to keep Ap inside L2.
     * Complex __float128 is 2 * sizeof(__float128) = 32 B/element. */
    ptrdiff_t MC = MC0;
    if (k <= KC) {
        const long L2_TARGET_BYTES = 256L * 1024L;
        long target_mc = L2_TARGET_BYTES / ((long)k * (long)(2 * sizeof(R)));
        if (target_mc > MC) {
            if (target_mc > 4L * MC0) target_mc = 4L * MC0;
            MC = blas_round_up((ptrdiff_t)target_mc, MR);
            if (MC < MC0) MC = MC0;
        }
    }

    p->MC = MC; p->KC = KC; p->NC = NC;
    p->ap_bytes = (size_t)blas_round_up(MC, MR) * (size_t)KC * 2 * sizeof(R);
    p->bp_bytes = (size_t)KC * (size_t)blas_round_up(NC, NR) * 2 * sizeof(R);
    p->conj_a = op_is_conj(ta);  p->trans_a = op_is_trans(ta);
    p->conj_b = op_is_conj(tb);  p->trans_b = op_is_trans(tb);
}

/* OCOPY(B): source offsets follow OpenBLAS's OCOPY_OPERATION —
 *   normal-B: NCOPY at b + (ls + js*ldb)*COMPSIZE
 *   trans-B:  TCOPY at b + (js + ls*ldb)*COMPSIZE  */
void xgemm_pack_B(const xgemm_plan_t *p,
                  const R *b, ptrdiff_t ldb,
                  ptrdiff_t js, ptrdiff_t ls, ptrdiff_t pb, ptrdiff_t jb,
                  R *Bp)
{
    if (!p->trans_b) {
        qblas_xgemm_ncopy(pb, jb, p->conj_b,
                          &b[((size_t)js * ldb + ls) * 2], ldb, Bp);
    } else {
        qblas_xgemm_tcopy(pb, jb, p->conj_b,
                          &b[((size_t)ls * ldb + js) * 2], ldb, Bp);
    }
}

/* One (m_lo..m_hi)×(js..) slab. ICOPY(A) source offsets follow
 * OpenBLAS's ICOPY_OPERATION —
 *   normal-A: TCOPY at A + (is + ls*lda)*COMPSIZE
 *   trans-A:  NCOPY at A + (ls + is*lda)*COMPSIZE  */
void xgemm_level3_slab(ptrdiff_t m_lo, ptrdiff_t m_hi, const xgemm_plan_t *p,
                       R alphar, R alphai,
                       const R *A, ptrdiff_t lda, R *Ap,
                       const R *Bp,
                       ptrdiff_t js, ptrdiff_t ls, ptrdiff_t pb, ptrdiff_t jb,
                       R *C, ptrdiff_t ldc)
{
    const ptrdiff_t MC = p->MC;
    for (ptrdiff_t is = m_lo; is < m_hi; is += MC) {
        ptrdiff_t min_i = (m_hi - is < MC) ? (m_hi - is) : MC;

        if (!p->trans_a) {
            qblas_xgemm_tcopy(pb, min_i, p->conj_a,
                              &A[((size_t)ls * lda + is) * 2], lda, Ap);
        } else {
            qblas_xgemm_ncopy(pb, min_i, p->conj_a,
                              &A[((size_t)is * lda + ls) * 2], lda, Ap);
        }

        qblas_xgemm_kernel(min_i, jb, pb, alphar, alphai,
                           Ap, Bp,
                           &C[((size_t)js * ldc + is) * 2], ldc);
    }
}

/* ── Single-thread by-value entry ─────────────────────────────── */
void xgemm_serial(
    char transa, char transb,
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
    const xgemm_TC *alpha_,
    const xgemm_TC *a, ptrdiff_t lda,
    const xgemm_TC *b, ptrdiff_t ldb,
    const xgemm_TC *beta_,
    xgemm_TC *c, ptrdiff_t ldc)
{
    const R alphar = __real__ *alpha_, alphai = __imag__ *alpha_;
    const R beta_r = __real__ *beta_,  beta_i = __imag__ *beta_;
    const char ta = xgemm_trans_code(transa);
    const char tb = xgemm_trans_code(transb);

    if (m <= 0 || n <= 0) return;

    const R *A = (const R *)a;
    const R *B = (const R *)b;
    R *C = (R *)c;

    qblas_xgemm_beta(m, n, beta_r, beta_i, C, ldc);
    if (k == 0 || (alphar == 0.0Q && alphai == 0.0Q)) return;

    xgemm_plan_t p;
    xgemm_make_plan(m, n, k, ta, tb, &p);

    R *Ap = aligned_alloc(64, (p.ap_bytes + 63) & ~(size_t)63);
    if (!Ap) return;
    R *Bp = aligned_alloc(64, (p.bp_bytes + 63) & ~(size_t)63);
    if (!Bp) { free(Ap); return; }

    for (ptrdiff_t js = 0; js < n; js += p.NC) {
        ptrdiff_t jb = (n - js < p.NC) ? (n - js) : p.NC;
        for (ptrdiff_t ls = 0; ls < k; ls += p.KC) {
            ptrdiff_t pb = (k - ls < p.KC) ? (k - ls) : p.KC;
            xgemm_pack_B(&p, B, ldb, js, ls, pb, jb, Bp);
            xgemm_level3_slab(0, m, &p, alphar, alphai,
                              A, lda, Ap, Bp, js, ls, pb, jb, C, ldc);
        }
    }

    free(Bp);
    free(Ap);
}
