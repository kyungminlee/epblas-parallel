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
 * microkernel (qblas_ygemm_kernel) whose four independent accumulator chains
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
#include "xl3_complex.h"
#include <ctype.h>
#include <stddef.h>
#include <stdlib.h>

typedef __float128 R;

#define MR QBLAS_YGEMM_MR
#define NR QBLAS_YGEMM_NR

int xgemm_trans_code(const char *p, size_t len) {
    (void)len;
    return (char)toupper((unsigned char)*p);
}

static int op_is_conj(int c)  { return (c == 'C' || c == 'R') ? 1 : 0; }
static int op_is_trans(int c) { return (c == 'T' || c == 'C') ? 1 : 0; }
static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }

/* ── Block plan (mirrors ob xgemm.c lines 136-155) ──────────────── */
void xgemm_make_plan(int M, int N, int K, int ta, int tb, xgemm_plan_t *p)
{
    (void)N;
    int MC0, KC, NC;
    qblas_ygemm_blocks(&MC0, &KC, &NC);

    /* Adaptive MC for small K, sized to keep Ap inside L2.
     * Complex __float128 is 2 * sizeof(__float128) = 32 B/element. */
    int MC = MC0;
    if (K <= KC) {
        const long L2_TARGET_BYTES = 256L * 1024L;
        long target_mc = L2_TARGET_BYTES / ((long)K * (long)(2 * sizeof(R)));
        if (target_mc > MC) {
            if (target_mc > 4L * MC0) target_mc = 4L * MC0;
            MC = round_up((int)target_mc, MR);
            if (MC < MC0) MC = MC0;
        }
    }

    p->MC = MC; p->KC = KC; p->NC = NC;
    p->ap_bytes = (size_t)round_up(MC, MR) * (size_t)KC * 2 * sizeof(R);
    p->bp_bytes = (size_t)KC * (size_t)round_up(NC, NR) * 2 * sizeof(R);
    p->conj_a = op_is_conj(ta);  p->trans_a = op_is_trans(ta);
    p->conj_b = op_is_conj(tb);  p->trans_b = op_is_trans(tb);
}

/* OCOPY(B): source offsets follow OpenBLAS's OCOPY_OPERATION —
 *   normal-B: NCOPY at b + (ls + js*ldb)*COMPSIZE
 *   trans-B:  TCOPY at b + (js + ls*ldb)*COMPSIZE  */
void xgemm_pack_B(const xgemm_plan_t *p,
                  const R *b, int ldb,
                  int js, int ls, int pb, int jb,
                  R *Bp)
{
    if (!p->trans_b) {
        qblas_ygemm_ncopy(pb, jb, p->conj_b,
                          &b[((size_t)js * ldb + ls) * 2], ldb, Bp);
    } else {
        qblas_ygemm_tcopy(pb, jb, p->conj_b,
                          &b[((size_t)ls * ldb + js) * 2], ldb, Bp);
    }
}

/* One (m_lo..m_hi)×(js..) slab. ICOPY(A) source offsets follow
 * OpenBLAS's ICOPY_OPERATION —
 *   normal-A: TCOPY at A + (is + ls*lda)*COMPSIZE
 *   trans-A:  NCOPY at A + (ls + is*lda)*COMPSIZE  */
void xgemm_level3_slab(int m_lo, int m_hi, const xgemm_plan_t *p,
                       R alphar, R alphai,
                       const R *A, int lda, R *Ap,
                       const R *Bp,
                       int js, int ls, int pb, int jb,
                       R *C, int ldc)
{
    const int MC = p->MC;
    for (int is = m_lo; is < m_hi; is += MC) {
        int min_i = (m_hi - is < MC) ? (m_hi - is) : MC;

        if (!p->trans_a) {
            qblas_ygemm_tcopy(pb, min_i, p->conj_a,
                              &A[((size_t)ls * lda + is) * 2], lda, Ap);
        } else {
            qblas_ygemm_ncopy(pb, min_i, p->conj_a,
                              &A[((size_t)is * lda + ls) * 2], lda, Ap);
        }

        qblas_ygemm_kernel(min_i, jb, pb, alphar, alphai,
                           Ap, Bp,
                           &C[((size_t)js * ldc + is) * 2], ldc);
    }
}

/* ── Single-thread entry (int Fortran ABI) ────────────────────── */
void xgemm_serial_(
    const char *transa, const char *transb,
    const int *m_, const int *n_, const int *k_,
    const xgemm_T *alpha_,
    const xgemm_T *a, const int *lda_,
    const xgemm_T *b, const int *ldb_,
    const xgemm_T *beta_,
    xgemm_T *c, const int *ldc_,
    size_t transa_len, size_t transb_len)
{
    const int M = *m_, N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const R alphar = __real__ *alpha_, alphai = __imag__ *alpha_;
    const R beta_r = __real__ *beta_,  beta_i = __imag__ *beta_;
    const int ta = xgemm_trans_code(transa, transa_len);
    const int tb = xgemm_trans_code(transb, transb_len);

    if (M <= 0 || N <= 0) return;

    const R *A = (const R *)a;
    const R *B = (const R *)b;
    R *C = (R *)c;

    qblas_ygemm_beta((ptrdiff_t)M, (ptrdiff_t)N, beta_r, beta_i, C, (ptrdiff_t)ldc);
    if (K == 0 || (alphar == 0.0Q && alphai == 0.0Q)) return;

    xgemm_plan_t p;
    xgemm_make_plan(M, N, K, ta, tb, &p);

    R *Ap = aligned_alloc(64, (p.ap_bytes + 63) & ~(size_t)63);
    if (!Ap) return;
    R *Bp = aligned_alloc(64, (p.bp_bytes + 63) & ~(size_t)63);
    if (!Bp) { free(Ap); return; }

    for (int js = 0; js < N; js += p.NC) {
        int jb = (N - js < p.NC) ? (N - js) : p.NC;
        for (int ls = 0; ls < K; ls += p.KC) {
            int pb = (K - ls < p.KC) ? (K - ls) : p.KC;
            xgemm_pack_B(&p, B, ldb, js, ls, pb, jb, Bp);
            xgemm_level3_slab(0, M, &p, alphar, alphai,
                              A, lda, Ap, Bp, js, ls, pb, jb, C, ldc);
        }
    }

    free(Bp);
    free(Ap);
}
