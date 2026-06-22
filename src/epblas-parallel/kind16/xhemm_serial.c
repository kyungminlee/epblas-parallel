/*
 * xhemm_serial — kind16 complex HEMM (COMPLEX(KIND=16) / __complex128),
 * single-thread. This TU owns ALL of the xhemm math: the block plan, the
 * B-panel packer, the per-M-slab level3 worker, and the pure-serial
 * by-value entry `xhemm_serial`. xhemm_parallel.c only orchestrates
 * threads over these same pieces.
 *
 * Strategy: faithful port of the OpenBLAS GotoBLAS HEMM driver (the ob
 * clone src/epblas-openblas/kind16/xhemm.c) over the shared packed
 * substrate (xl3_complex.c). HERMITIAN — the HEMM packers negate the imag
 * float on the reflected half and zero it on the diagonal. For SIDE=L the
 * Hermitian matrix A is the ICOPY factor (HEMM ucopy/lcopy) and the regular
 * B is the OCOPY factor (standard ncopy, conj=0); for SIDE=R the roles swap
 * (A_eff = b, B_eff = a) and the Hermitian matrix is packed on the B side
 * with the _oc variants (the (posX,posY) reinterpret as (col,row)).
 *
 * ABI: dims/leading-dims by value as ptrdiff_t; complex scalar =
 * __complex128 (re, im) forwarded by pointer; side/uplo decoded chars by
 * value; lda/ldb/ldc in complex elements. The substrate works on
 * __float128* (interleaved re,im), reached by reinterpreting the
 * __complex128* a/b/c.
 */

#include "xhemm_kernel.h"
#include "../common/blas_char.h"
#include "xl3_complex.h"
#include <ctype.h>
#include <stddef.h>
#include <stdlib.h>

typedef __float128 R;

#define MR QBLAS_YGEMM_MR
#define NR QBLAS_YGEMM_NR

static ptrdiff_t round_up(ptrdiff_t v, ptrdiff_t m) { return ((v + m - 1) / m) * m; }

/* ── Block plan (mirrors ob xhemm.c lines 92-107) ───────────────── */
void xhemm_make_plan(ptrdiff_t M, ptrdiff_t N, char side, char uplo, xhemm_plan_t *p)
{
    const ptrdiff_t K = (side == 'L') ? M : N;
    ptrdiff_t MC0, KC, NC;
    qblas_ygemm_blocks(&MC0, &KC, &NC);

    ptrdiff_t MC = MC0;
    if (K > 0 && K <= KC) {
        const long L2_TARGET_BYTES = 256L * 1024L;
        long target_mc = L2_TARGET_BYTES / ((long)K * (long)(2 * sizeof(R)));
        if (target_mc > MC) {
            if (target_mc > 4L * MC0) target_mc = 4L * MC0;
            MC = round_up((ptrdiff_t)target_mc, MR);
            if (MC < MC0) MC = MC0;
        }
    }

    p->MC = MC; p->KC = KC; p->NC = NC;
    p->ap_bytes = (size_t)round_up(MC, MR) * (size_t)KC * 2 * sizeof(R);
    p->bp_bytes = (size_t)KC * (size_t)round_up(NC, NR) * 2 * sizeof(R);
    p->side = side; p->uplo = uplo; p->K = K;
}

/* OCOPY(B). SIDE=L: regular factor → standard NCOPY (conj=0). SIDE=R: the
 * Hermitian matrix → HEMM _oc copy reading the reflected half at (js, ls). */
void xhemm_pack_B(const xhemm_plan_t *p,
                  const R *B_eff, ptrdiff_t ldb_eff,
                  ptrdiff_t js, ptrdiff_t ls, ptrdiff_t pb, ptrdiff_t jb,
                  R *Bp)
{
    if (p->side == 'L') {
        qblas_ygemm_ncopy(pb, jb, 0,
                          &B_eff[((size_t)js * ldb_eff + ls) * 2], ldb_eff, Bp);
    } else if (p->uplo == 'U') {
        qblas_yhemm_ucopy_oc(pb, jb, B_eff, ldb_eff, js, ls, Bp);
    } else {
        qblas_yhemm_lcopy_oc(pb, jb, B_eff, ldb_eff, js, ls, Bp);
    }
}

/* One (m_lo..m_hi)×(js..) slab. SIDE=L: A is the Hermitian matrix → HEMM
 * copy. SIDE=R: A is the regular factor → standard TCOPY (conj=0). */
void xhemm_level3_slab(ptrdiff_t m_lo, ptrdiff_t m_hi, const xhemm_plan_t *p,
                       R alphar, R alphai,
                       const R *A_eff, ptrdiff_t lda_eff, R *Ap,
                       const R *Bp,
                       ptrdiff_t js, ptrdiff_t ls, ptrdiff_t pb, ptrdiff_t jb,
                       R *C, ptrdiff_t ldc)
{
    const ptrdiff_t MC = p->MC;
    for (ptrdiff_t is = m_lo; is < m_hi; is += MC) {
        ptrdiff_t min_i = (m_hi - is < MC) ? (m_hi - is) : MC;

        if (p->side == 'L') {
            if (p->uplo == 'U')
                qblas_yhemm_ucopy(pb, min_i, A_eff, lda_eff, is, ls, Ap);
            else
                qblas_yhemm_lcopy(pb, min_i, A_eff, lda_eff, is, ls, Ap);
        } else {
            qblas_ygemm_tcopy(pb, min_i, 0,
                              &A_eff[((size_t)ls * lda_eff + is) * 2], lda_eff, Ap);
        }

        qblas_ygemm_kernel(min_i, jb, pb, alphar, alphai,
                           Ap, Bp,
                           &C[((size_t)js * ldc + is) * 2], ldc);
    }
}

/* ── Single-thread entry (by-value ptrdiff_t core ABI) ────────── */
void xhemm_serial(
    char side, char uplo,
    ptrdiff_t M, ptrdiff_t N,
    const xhemm_T *alpha_,
    const xhemm_T *a, ptrdiff_t lda,
    const xhemm_T *b, ptrdiff_t ldb,
    const xhemm_T *beta_,
    xhemm_T *c, ptrdiff_t ldc)
{
    const R alphar = __real__ *alpha_, alphai = __imag__ *alpha_;
    const R beta_r = __real__ *beta_,  beta_i = __imag__ *beta_;
    const char sd = blas_up(side);
    const char up = blas_up(uplo);

    if (M <= 0 || N <= 0) return;

    R *C = (R *)c;
    qblas_ygemm_beta(M, N, beta_r, beta_i, C, ldc);
    if (alphar == 0.0Q && alphai == 0.0Q) return;

    const R *A_eff = (const R *)((sd == 'L') ? a : b);
    const R *B_eff = (const R *)((sd == 'L') ? b : a);
    const ptrdiff_t lda_eff = (sd == 'L') ? lda : ldb;
    const ptrdiff_t ldb_eff = (sd == 'L') ? ldb : lda;

    xhemm_plan_t p;
    xhemm_make_plan(M, N, sd, up, &p);
    if (p.K == 0) return;

    R *Ap = aligned_alloc(64, (p.ap_bytes + 63) & ~(size_t)63);
    if (!Ap) return;
    R *Bp = aligned_alloc(64, (p.bp_bytes + 63) & ~(size_t)63);
    if (!Bp) { free(Ap); return; }

    for (ptrdiff_t js = 0; js < N; js += p.NC) {
        ptrdiff_t jb = (N - js < p.NC) ? (N - js) : p.NC;
        for (ptrdiff_t ls = 0; ls < p.K; ls += p.KC) {
            ptrdiff_t pb = (p.K - ls < p.KC) ? (p.K - ls) : p.KC;
            xhemm_pack_B(&p, B_eff, ldb_eff, js, ls, pb, jb, Bp);
            xhemm_level3_slab(0, M, &p, alphar, alphai,
                              A_eff, lda_eff, Ap, Bp, js, ls, pb, jb, C, ldc);
        }
    }

    free(Bp);
    free(Ap);
}
