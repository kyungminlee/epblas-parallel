/*
 * xsymm_serial — kind16 complex SYMM (COMPLEX(KIND=16) / __complex128),
 * single-thread. This TU owns ALL of the xsymm math: the block plan, the
 * B-panel packer, the per-M-slab level3 worker, and the pure-serial
 * by-value entry `xsymm_serial`. xsymm_parallel.c only orchestrates
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
 * ABI: dims/leading-dims by value as ptrdiff_t; complex scalar =
 * __complex128 (re, im) forwarded by pointer; side/uplo decoded chars by
 * value; lda/ldb/ldc in complex elements. The substrate works on
 * __float128* (interleaved re,im), reached by reinterpreting the
 * __complex128* a/b/c.
 */

#include "xsymm_kernel.h"
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include "xl3_complex.h"
#include <ctype.h>
#include <stddef.h>
#include <stdlib.h>

typedef __float128 R;

#define MR QBLAS_XGEMM_MR
#define NR QBLAS_XGEMM_NR


/* ── Block plan (mirrors ob xsymm.c lines 85-101) ───────────────── */
void xsymm_make_plan(ptrdiff_t m, ptrdiff_t n, char side, char uplo, xsymm_plan_t *p)
{
    const ptrdiff_t k = (side == 'L') ? m : n;
    ptrdiff_t MC0, KC, NC;
    qblas_xgemm_blocks(&MC0, &KC, &NC);

    /* Adaptive MC for small K, sized to keep Ap inside L2.
     * Complex __float128 is 2 * sizeof(__float128) = 32 B/element. */
    ptrdiff_t MC = MC0;
    if (k > 0 && k <= KC) {
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
    p->side = side; p->uplo = uplo; p->k = k;
}

/* OCOPY(B). SIDE=L: regular factor → standard NCOPY (conj=0). SIDE=R: the
 * symmetric matrix → SYMM copy reading the reflected half at (js, ls). */
void xsymm_pack_B(const xsymm_plan_t *p,
                  const R *B_eff, ptrdiff_t ldb_eff,
                  ptrdiff_t js, ptrdiff_t ls, ptrdiff_t pb, ptrdiff_t jb,
                  R *Bp)
{
    if (p->side == 'L') {
        qblas_xgemm_ncopy(pb, jb, 0,
                          &B_eff[((size_t)js * ldb_eff + ls) * 2], ldb_eff, Bp);
    } else if (p->uplo == 'U') {
        qblas_xsymm_ucopy(pb, jb, B_eff, ldb_eff, js, ls, Bp);
    } else {
        qblas_xsymm_lcopy(pb, jb, B_eff, ldb_eff, js, ls, Bp);
    }
}

/* One (m_lo..m_hi)×(js..) slab. SIDE=L: A is the symmetric matrix → SYMM
 * copy. SIDE=R: A is the regular factor → standard TCOPY (conj=0). */
void xsymm_level3_slab(ptrdiff_t m_lo, ptrdiff_t m_hi, const xsymm_plan_t *p,
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
                qblas_xsymm_ucopy(pb, min_i, A_eff, lda_eff, is, ls, Ap);
            else
                qblas_xsymm_lcopy(pb, min_i, A_eff, lda_eff, is, ls, Ap);
        } else {
            qblas_xgemm_tcopy(pb, min_i, 0,
                              &A_eff[((size_t)ls * lda_eff + is) * 2], lda_eff, Ap);
        }

        qblas_xgemm_kernel(min_i, jb, pb, alphar, alphai,
                           Ap, Bp,
                           &C[((size_t)js * ldc + is) * 2], ldc);
    }
}

/* ── Single-thread entry (by-value ptrdiff_t core ABI) ────────── */
void xsymm_serial(
    char side, char uplo,
    ptrdiff_t m, ptrdiff_t n,
    const xsymm_TC *alpha_,
    const xsymm_TC *a, ptrdiff_t lda,
    const xsymm_TC *b, ptrdiff_t ldb,
    const xsymm_TC *beta_,
    xsymm_TC *c, ptrdiff_t ldc)
{
    const R alphar = __real__ *alpha_, alphai = __imag__ *alpha_;
    const R beta_r = __real__ *beta_,  beta_i = __imag__ *beta_;
    const char sd = blas_up(side);
    const char up = blas_up(uplo);

    if (m <= 0 || n <= 0) return;

    R *C = (R *)c;
    qblas_xgemm_beta(m, n, beta_r, beta_i, C, ldc);
    if (alphar == 0.0Q && alphai == 0.0Q) return;

    const R *A_eff = (const R *)((sd == 'L') ? a : b);
    const R *B_eff = (const R *)((sd == 'L') ? b : a);
    const ptrdiff_t lda_eff = (sd == 'L') ? lda : ldb;
    const ptrdiff_t ldb_eff = (sd == 'L') ? ldb : lda;

    xsymm_plan_t p;
    xsymm_make_plan(m, n, sd, up, &p);
    if (p.k == 0) return;

    R *Ap = aligned_alloc(64, (p.ap_bytes + 63) & ~(size_t)63);
    if (!Ap) return;
    R *Bp = aligned_alloc(64, (p.bp_bytes + 63) & ~(size_t)63);
    if (!Bp) { free(Ap); return; }

    for (ptrdiff_t js = 0; js < n; js += p.NC) {
        ptrdiff_t jb = (n - js < p.NC) ? (n - js) : p.NC;
        for (ptrdiff_t ls = 0; ls < p.k; ls += p.KC) {
            ptrdiff_t pb = (p.k - ls < p.KC) ? (p.k - ls) : p.KC;
            xsymm_pack_B(&p, B_eff, ldb_eff, js, ls, pb, jb, Bp);
            xsymm_level3_slab(0, m, &p, alphar, alphai,
                              A_eff, lda_eff, Ap, Bp, js, ls, pb, jb, C, ldc);
        }
    }

    free(Bp);
    free(Ap);
}
