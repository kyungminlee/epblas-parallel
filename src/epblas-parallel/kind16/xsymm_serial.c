/*
 * xsymm_serial — kind16 complex SYMM (COMPLEX(KIND=16) / __complex128),
 * single-thread. This TU owns ALL of the xsymm math: the block plan, the
 * B-panel packer, the per-M-slab level3 worker, and the pure-serial
 * Fortran-ABI entry `xsymm_serial_`. xsymm_parallel.c only orchestrates
 * threads over these same pieces.
 *
 * Strategy: faithful port of the OpenBLAS GotoBLAS SYMM driver (the ob
 * clone src/epblas-openblas/kind16/xsymm.c) over the shared packed
 * substrate (xl3_complex.c). NOT Hermitian — the SYMM packers copy the
 * imag float through unchanged (no conjugation). For SIDE=L the symmetric
 * matrix A is the ICOPY factor and the regular B is the OCOPY factor; for
 * SIDE=R the roles swap (A_eff = b, B_eff = a) and the symmetric matrix is
 * packed on the B side.
 *
 * Fortran ABI: scalars by pointer; complex scalar = __complex128 (re, im);
 * character args followed by hidden trailing size_t lengths; lda/ldb/ldc
 * in complex elements. The substrate works on __float128* (interleaved
 * re,im), reached by reinterpreting the __complex128* a/b/c.
 */

#include "xsymm_kernel.h"
#include "xl3_complex.h"
#include <ctype.h>
#include <stddef.h>
#include <stdlib.h>

typedef __float128 R;

#define MR QBLAS_YGEMM_MR
#define NR QBLAS_YGEMM_NR

static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }

/* ── Block plan (mirrors ob xsymm.c lines 85-101) ───────────────── */
void xsymm_make_plan(int M, int N, int side, int uplo, xsymm_plan_t *p)
{
    const int K = (side == 'L') ? M : N;
    int MC0, KC, NC;
    qblas_ygemm_blocks(&MC0, &KC, &NC);

    /* Adaptive MC for small K, sized to keep Ap inside L2.
     * Complex __float128 is 2 * sizeof(__float128) = 32 B/element. */
    int MC = MC0;
    if (K > 0 && K <= KC) {
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
    p->side = side; p->uplo = uplo; p->K = K;
}

/* OCOPY(B). SIDE=L: regular factor → standard NCOPY (conj=0). SIDE=R: the
 * symmetric matrix → SYMM copy reading the reflected half at (js, ls). */
void xsymm_pack_B(const xsymm_plan_t *p,
                  const R *B_eff, int ldb_eff,
                  int js, int ls, int pb, int jb,
                  R *Bp)
{
    if (p->side == 'L') {
        qblas_ygemm_ncopy(pb, jb, 0,
                          &B_eff[((size_t)js * ldb_eff + ls) * 2], ldb_eff, Bp);
    } else if (p->uplo == 'U') {
        qblas_ysymm_ucopy(pb, jb, B_eff, ldb_eff, js, ls, Bp);
    } else {
        qblas_ysymm_lcopy(pb, jb, B_eff, ldb_eff, js, ls, Bp);
    }
}

/* One (m_lo..m_hi)×(js..) slab. SIDE=L: A is the symmetric matrix → SYMM
 * copy. SIDE=R: A is the regular factor → standard TCOPY (conj=0). */
void xsymm_level3_slab(int m_lo, int m_hi, const xsymm_plan_t *p,
                       R alphar, R alphai,
                       const R *A_eff, int lda_eff, R *Ap,
                       const R *Bp,
                       int js, int ls, int pb, int jb,
                       R *C, int ldc)
{
    const int MC = p->MC;
    for (int is = m_lo; is < m_hi; is += MC) {
        int min_i = (m_hi - is < MC) ? (m_hi - is) : MC;

        if (p->side == 'L') {
            if (p->uplo == 'U')
                qblas_ysymm_ucopy(pb, min_i, A_eff, lda_eff, is, ls, Ap);
            else
                qblas_ysymm_lcopy(pb, min_i, A_eff, lda_eff, is, ls, Ap);
        } else {
            qblas_ygemm_tcopy(pb, min_i, 0,
                              &A_eff[((size_t)ls * lda_eff + is) * 2], lda_eff, Ap);
        }

        qblas_ygemm_kernel(min_i, jb, pb, alphar, alphai,
                           Ap, Bp,
                           &C[((size_t)js * ldc + is) * 2], ldc);
    }
}

/* ── Single-thread entry (int Fortran ABI) ────────────────────── */
void xsymm_serial_(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const xsymm_T *alpha_,
    const xsymm_T *a, const int *lda_,
    const xsymm_T *b, const int *ldb_,
    const xsymm_T *beta_,
    xsymm_T *c, const int *ldc_,
    size_t side_len, size_t uplo_len)
{
    (void)side_len; (void)uplo_len;
    const int M = *m_, N = *n_;
    const R alphar = __real__ *alpha_, alphai = __imag__ *alpha_;
    const R beta_r = __real__ *beta_,  beta_i = __imag__ *beta_;
    const int sd = (char)toupper((unsigned char)*side);
    const int up = (char)toupper((unsigned char)*uplo);
    const int ldc = *ldc_;

    if (M <= 0 || N <= 0) return;

    R *C = (R *)c;
    qblas_ygemm_beta((ptrdiff_t)M, (ptrdiff_t)N, beta_r, beta_i, C, (ptrdiff_t)ldc);
    if (alphar == 0.0Q && alphai == 0.0Q) return;

    const R *A_eff = (const R *)((sd == 'L') ? a : b);
    const R *B_eff = (const R *)((sd == 'L') ? b : a);
    const int lda_eff = (sd == 'L') ? *lda_ : *ldb_;
    const int ldb_eff = (sd == 'L') ? *ldb_ : *lda_;

    xsymm_plan_t p;
    xsymm_make_plan(M, N, sd, up, &p);
    if (p.K == 0) return;

    R *Ap = aligned_alloc(64, (p.ap_bytes + 63) & ~(size_t)63);
    if (!Ap) return;
    R *Bp = aligned_alloc(64, (p.bp_bytes + 63) & ~(size_t)63);
    if (!Bp) { free(Ap); return; }

    for (int js = 0; js < N; js += p.NC) {
        int jb = (N - js < p.NC) ? (N - js) : p.NC;
        for (int ls = 0; ls < p.K; ls += p.KC) {
            int pb = (p.K - ls < p.KC) ? (p.K - ls) : p.KC;
            xsymm_pack_B(&p, B_eff, ldb_eff, js, ls, pb, jb, Bp);
            xsymm_level3_slab(0, M, &p, alphar, alphai,
                              A_eff, lda_eff, Ap, Bp, js, ls, pb, jb, C, ldc);
        }
    }

    free(Bp);
    free(Ap);
}
