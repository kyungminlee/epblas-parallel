/*
 * qblas_l3_real.c — shared L3 microkernel + packers (kind16 real).
 *
 * Port source: OpenBLAS.
 *   - kernel/generic/gemmkernel_2x2.c   ← qblas_egemm_kernel
 *   - kernel/generic/gemm_ncopy_2.c     ← qblas_egemm_ncopy
 *   - kernel/generic/gemm_tcopy_2.c     ← qblas_egemm_tcopy
 *   - kernel/generic/gemm_beta.c        ← qblas_egemm_beta
 *
 * See qblas_l3_real.h for the layout convention. The microkernel is
 * a literal 2x2 outer-product over the K-dim with the K-loop unrolled
 * by 4 inside, exactly like OpenBLAS's gemmkernel_2x2.c — the only
 * change is dropping the bf16 conversion macros (TO_F32 / TO_OUTPUT /
 * C_TO_F32) and the TRMM-offset branch (each L3 routine builds its own
 * TRMM driver and reaches into this kernel only for the GEMM body).
 */

#include "qblas_l3_real.h"
#include <quadmath.h>
#include <stdlib.h>

typedef __float128 T;

#define MR QBLAS_EGEMM_MR
#define NR QBLAS_EGEMM_NR


/* ── Microkernel: 2x2 outer-product over K ───────────────────────── */
void qblas_egemm_kernel(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        T alpha,
                        const T *Ap,
                        const T *Bp,
                        T *C, ptrdiff_t ldc)
{
    /* Walk B in NR=2-col panels. */
    const T *ptrba_base = Ap;
    const T *ptrbb = Bp;
    T *Cj = C;

    for (ptrdiff_t j = 0; j < bn / NR; ++j) {
        T *C0 = Cj;
        T *C1 = C0 + ldc;
        const T *ptrba = ptrba_base;

        /* MR=2 row panels (full 2x2 tiles). */
        for (ptrdiff_t i = 0; i < bm / MR; ++i) {
            const T *ptrbb_loc = ptrbb;
            T r0 = 0, r1 = 0, r2 = 0, r3 = 0;

            /* K-loop unrolled by 4 (same shape as OpenBLAS gen kernel). */
            ptrdiff_t k4 = bk / 4;
            for (ptrdiff_t k = 0; k < k4; ++k) {
                T a0 = ptrba[0], a1 = ptrba[1];
                T b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
                r0 += a0 * b0; r1 += a1 * b0;
                r2 += a0 * b1; r3 += a1 * b1;
                a0 = ptrba[2]; a1 = ptrba[3];
                b0 = ptrbb_loc[2]; b1 = ptrbb_loc[3];
                r0 += a0 * b0; r1 += a1 * b0;
                r2 += a0 * b1; r3 += a1 * b1;
                a0 = ptrba[4]; a1 = ptrba[5];
                b0 = ptrbb_loc[4]; b1 = ptrbb_loc[5];
                r0 += a0 * b0; r1 += a1 * b0;
                r2 += a0 * b1; r3 += a1 * b1;
                a0 = ptrba[6]; a1 = ptrba[7];
                b0 = ptrbb_loc[6]; b1 = ptrbb_loc[7];
                r0 += a0 * b0; r1 += a1 * b0;
                r2 += a0 * b1; r3 += a1 * b1;
                ptrba += 8;
                ptrbb_loc += 8;
            }
            for (ptrdiff_t k = 0; k < (bk & 3); ++k) {
                T a0 = ptrba[0], a1 = ptrba[1];
                T b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
                r0 += a0 * b0; r1 += a1 * b0;
                r2 += a0 * b1; r3 += a1 * b1;
                ptrba += 2;
                ptrbb_loc += 2;
            }

            C0[0] += alpha * r0;
            C0[1] += alpha * r1;
            C1[0] += alpha * r2;
            C1[1] += alpha * r3;
            C0 += 2;
            C1 += 2;
        }

        /* bm & 1 — single-row tail (mr=1). */
        for (ptrdiff_t i = 0; i < (bm & 1); ++i) {
            const T *ptrbb_loc = ptrbb;
            T r0 = 0, r1 = 0;
            for (ptrdiff_t k = 0; k < bk; ++k) {
                T a0 = ptrba[0];
                T b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
                r0 += a0 * b0;
                r1 += a0 * b1;
                ptrba += 1;
                ptrbb_loc += 2;
            }
            C0[0] += alpha * r0;
            C1[0] += alpha * r1;
            C0 += 1;
            C1 += 1;
        }

        ptrbb += bk * 2;       /* advance to next 2-col B panel */
        Cj += 2 * ldc;
    }

    /* bn & 1 — single-col tail (nr=1). */
    for (ptrdiff_t j = 0; j < (bn & 1); ++j) {
        T *C0 = Cj;
        const T *ptrba = ptrba_base;

        for (ptrdiff_t i = 0; i < bm / MR; ++i) {
            const T *ptrbb_loc = ptrbb;
            T r0 = 0, r1 = 0;
            for (ptrdiff_t k = 0; k < bk; ++k) {
                T a0 = ptrba[0], a1 = ptrba[1];
                T b0 = ptrbb_loc[0];
                r0 += a0 * b0;
                r1 += a1 * b0;
                ptrba += 2;
                ptrbb_loc += 1;
            }
            C0[0] += alpha * r0;
            C0[1] += alpha * r1;
            C0 += 2;
        }
        for (ptrdiff_t i = 0; i < (bm & 1); ++i) {
            const T *ptrbb_loc = ptrbb;
            T r0 = 0;
            for (ptrdiff_t k = 0; k < bk; ++k) {
                r0 += ptrba[0] * ptrbb_loc[0];
                ptrba += 1;
                ptrbb_loc += 1;
            }
            C0[0] += alpha * r0;
            C0 += 1;
        }
        ptrbb += bk;           /* advance over single-col B panel */
        Cj += ldc;
    }
}


/* ── ncopy: pack 2 columns per panel ──────────────────────────────── *
 *
 * Faithful translation of OpenBLAS gemm_ncopy_2.c:
 * - j-loop over (n >> 1) 2-col panels, each interleaving
 *   { a[i,j0], a[i,j1] } for i in 0..m, with 4x unroll on the m-walk.
 * - trailing (n & 1) single-col panel writes m elements unchanged.
 */
void qblas_egemm_ncopy(ptrdiff_t m, ptrdiff_t n,
                       const T *a, ptrdiff_t lda,
                       T *b)
{
    const T *a_off = a;
    T *b_off = b;
    ptrdiff_t j = n >> 1;

    while (j > 0) {
        const T *a_off1 = a_off;
        const T *a_off2 = a_off + lda;
        a_off += 2 * lda;

        ptrdiff_t i = m >> 2;
        while (i > 0) {
            b_off[0] = a_off1[0]; b_off[1] = a_off2[0];
            b_off[2] = a_off1[1]; b_off[3] = a_off2[1];
            b_off[4] = a_off1[2]; b_off[5] = a_off2[2];
            b_off[6] = a_off1[3]; b_off[7] = a_off2[3];
            a_off1 += 4;
            a_off2 += 4;
            b_off += 8;
            --i;
        }
        for (i = m & 3; i > 0; --i) {
            b_off[0] = a_off1[0];
            b_off[1] = a_off2[0];
            ++a_off1;
            ++a_off2;
            b_off += 2;
        }
        --j;
    }

    if (n & 1) {
        ptrdiff_t i = m >> 3;
        while (i > 0) {
            b_off[0] = a_off[0]; b_off[1] = a_off[1];
            b_off[2] = a_off[2]; b_off[3] = a_off[3];
            b_off[4] = a_off[4]; b_off[5] = a_off[5];
            b_off[6] = a_off[6]; b_off[7] = a_off[7];
            a_off += 8;
            b_off += 8;
            --i;
        }
        for (i = m & 7; i > 0; --i) {
            *b_off++ = *a_off++;
        }
    }
}


/* ── tcopy: faithful port of OpenBLAS kernel/generic/gemm_tcopy_2.c ──
 *
 * Called for ICOPY of normal A (NN/NT/...) and OCOPY of transposed B
 * (NT/TT/...). Conceptually:
 *
 *   - "rows" of the input (`m`) become 2-element K-strips of the
 *     output (a_offset2 = a + lda, so the +1 step walks one COLUMN
 *     of the source matrix). For ICOPY of normal A, m == K-dim of
 *     the panel and n == M-dim; the kernel then sees 2-row-A-per-
 *     panel as Ap = [A(0,k0), A(1,k0), A(0,k1), A(1,k1), ...].
 *
 *   - within each m-strip, j = n>>1 sub-panels of 2 input "cols"
 *     each (which in source-coordinates is 2 elements along the
 *     +1 stride — i.e., 2 source ROWS for col-major input).
 *
 * Trailing single-element strip for (m & 1) and (n & 1).
 */
void qblas_egemm_tcopy(ptrdiff_t m, ptrdiff_t n,
                       const T *a, ptrdiff_t lda,
                       T *b)
{
    const T *a_off = a;
    T *b_off = b;
    T *b_off2 = b + m * (n & ~(ptrdiff_t)1);

    ptrdiff_t i = m >> 1;
    while (i > 0) {
        const T *a_off1 = a_off;
        const T *a_off2 = a_off + lda;
        a_off += 2 * lda;

        T *b_off1 = b_off;
        b_off += 4;

        ptrdiff_t j = n >> 1;
        while (j > 0) {
            b_off1[0] = a_off1[0];
            b_off1[1] = a_off1[1];
            b_off1[2] = a_off2[0];
            b_off1[3] = a_off2[1];
            a_off1 += 2;
            a_off2 += 2;
            b_off1 += m * 2;
            --j;
        }
        if (n & 1) {
            b_off2[0] = a_off1[0];
            b_off2[1] = a_off2[0];
            b_off2 += 2;
        }
        --i;
    }

    if (m & 1) {
        ptrdiff_t j = n >> 1;
        while (j > 0) {
            b_off[0] = a_off[0];
            b_off[1] = a_off[1];
            a_off += 2;
            b_off += m * 2;
            --j;
        }
        if (n & 1) {
            b_off2[0] = a_off[0];
        }
    }
}


/* ── Beta pre-pass ──────────────────────────────────────────────────
 *
 * Mirrors OpenBLAS gemm_beta.c (driver/level3/gemm_beta.c — the
 * fallback C used when there's no asm GEMM_BETA kernel for the arch):
 * beta == 0 zeros C, beta == 1 returns, otherwise scales each column.
 */
void qblas_egemm_beta(ptrdiff_t m, ptrdiff_t n,
                      T beta,
                      T *c, ptrdiff_t ldc)
{
    if (beta == 1.0Q) return;

    for (ptrdiff_t j = 0; j < n; ++j) {
        T *cj = c + j * ldc;
        if (beta == 0.0Q) {
            for (ptrdiff_t i = 0; i < m; ++i) cj[i] = 0;
        } else {
            for (ptrdiff_t i = 0; i < m; ++i) cj[i] *= beta;
        }
    }
}


/* ── Env-var block-size lookup (lazy, idempotent) ───────────────── */
static int g_mc = 0, g_kc = 0, g_nc = 0;

static int env_int(const char *name, int dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    int v = atoi(s);
    return v > 0 ? v : dflt;
}

void qblas_egemm_blocks(int *mc, int *kc, int *nc) {
    if (!g_mc) {
        g_mc = env_int("QBLAS_MC", QBLAS_EGEMM_GEMM_P);
        g_kc = env_int("QBLAS_KC", QBLAS_EGEMM_GEMM_Q);
        g_nc = env_int("QBLAS_NC", QBLAS_EGEMM_GEMM_R);
    }
    *mc = g_mc; *kc = g_kc; *nc = g_nc;
}


/* ── SYMM-aware packers (real) ───────────────────────────────────────
 *
 * Faithful ports of OpenBLAS kernel/generic/symm_{u,l}copy_2.c. See
 * the header for the role of (posX, posY) and the per-iteration
 * mirror/direct switch.
 *
 * The single function serves both ICOPY (SIDE=L, posX walks M, posY
 * walks K) and OCOPY (SIDE=R, posX walks N, posY walks K) — the
 * symmetric A doesn't care which axis is "row" vs "col" of A, so
 * the same code reads the upper/lower triangle correctly in both
 * roles.
 */
void qblas_esymm_ucopy(ptrdiff_t m, ptrdiff_t n,
                       const T *a, ptrdiff_t lda,
                       ptrdiff_t posX, ptrdiff_t posY,
                       T *b)
{
    ptrdiff_t i, js, offset;
    T data01, data02;
    const T *ao1, *ao2;

    js = n >> 1;
    while (js > 0) {
        offset = posX - posY;

        if (offset >  0) ao1 = a + posY + (posX + 0) * lda;
        else             ao1 = a + (posX + 0) + posY * lda;
        if (offset > -1) ao2 = a + posY + (posX + 1) * lda;
        else             ao2 = a + (posX + 1) + posY * lda;

        i = m;
        while (i > 0) {
            data01 = ao1[0];
            data02 = ao2[0];

            if (offset >   0) ao1 += 1;   else ao1 += lda;
            if (offset >  -1) ao2 += 1;   else ao2 += lda;

            b[0] = data01;
            b[1] = data02;
            b += 2;

            offset--;
            i--;
        }
        posX += 2;
        js--;
    }

    if (n & 1) {
        offset = posX - posY;

        if (offset > 0) ao1 = a + posY + (posX + 0) * lda;
        else            ao1 = a + (posX + 0) + posY * lda;

        i = m;
        while (i > 0) {
            data01 = ao1[0];
            if (offset > 0) ao1 += 1; else ao1 += lda;
            b[0] = data01;
            b += 1;
            offset--;
            i--;
        }
    }
}


void qblas_esymm_lcopy(ptrdiff_t m, ptrdiff_t n,
                       const T *a, ptrdiff_t lda,
                       ptrdiff_t posX, ptrdiff_t posY,
                       T *b)
{
    ptrdiff_t i, js, offset;
    T data01, data02;
    const T *ao1, *ao2;

    js = n >> 1;
    while (js > 0) {
        offset = posX - posY;

        if (offset >  0) ao1 = a + (posX + 0) + posY * lda;
        else             ao1 = a + posY + (posX + 0) * lda;
        if (offset > -1) ao2 = a + (posX + 1) + posY * lda;
        else             ao2 = a + posY + (posX + 1) * lda;

        i = m;
        while (i > 0) {
            data01 = ao1[0];
            data02 = ao2[0];

            if (offset >   0) ao1 += lda; else ao1 += 1;
            if (offset >  -1) ao2 += lda; else ao2 += 1;

            b[0] = data01;
            b[1] = data02;
            b += 2;

            offset--;
            i--;
        }
        posX += 2;
        js--;
    }

    if (n & 1) {
        offset = posX - posY;

        if (offset > 0) ao1 = a + (posX + 0) + posY * lda;
        else            ao1 = a + posY + (posX + 0) * lda;

        i = m;
        while (i > 0) {
            data01 = ao1[0];
            if (offset > 0) ao1 += lda; else ao1 += 1;
            b[0] = data01;
            b += 1;
            offset--;
            i--;
        }
    }
}


/* ── Triangular beta pre-pass (real) ─────────────────────────────────
 *
 * Port of OpenBLAS driver/level3/syrk_k.c's `syrk_beta` (lines 52–99,
 * specialized to full-matrix range). The off-UPLO triangle is left
 * untouched — that's the property the SYRK fuzz tests check via
 * sentinels in the wrong triangle.
 */
void qblas_esyrk_beta_u(ptrdiff_t n, T beta, T *c, ptrdiff_t ldc) {
    if (beta == 1.0Q) return;
    if (beta == 0.0Q) {
        for (ptrdiff_t j = 0; j < n; ++j) {
            T *cj = c + j * ldc;
            for (ptrdiff_t i = 0; i <= j; ++i) cj[i] = 0;
        }
    } else {
        for (ptrdiff_t j = 0; j < n; ++j) {
            T *cj = c + j * ldc;
            for (ptrdiff_t i = 0; i <= j; ++i) cj[i] *= beta;
        }
    }
}

void qblas_esyrk_beta_l(ptrdiff_t n, T beta, T *c, ptrdiff_t ldc) {
    if (beta == 1.0Q) return;
    if (beta == 0.0Q) {
        for (ptrdiff_t j = 0; j < n; ++j) {
            T *cj = c + j * ldc;
            for (ptrdiff_t i = j; i < n; ++i) cj[i] = 0;
        }
    } else {
        for (ptrdiff_t j = 0; j < n; ++j) {
            T *cj = c + j * ldc;
            for (ptrdiff_t i = j; i < n; ++i) cj[i] *= beta;
        }
    }
}


/* ── SYRK kernel: GEMM with diagonal-aware writeback (real) ──────────
 *
 * Faithful port of OpenBLAS driver/level3/syrk_kernel.c. See the header
 * for the offset convention and the strip layout of Ap/Bp.
 *
 * For the diagonal block we GEMM into a small subbuffer (zeroed first
 * — qblas_egemm_kernel does C += ..., not C := ...), then merge only
 * the upper/lower triangle of the subbuffer into C. Subbuffer is
 * NR × NR which is at most 2×2 for this MR=NR=2 build; we size it as
 * NR*(NR+1) = 6 to match OpenBLAS's safety pad.
 */
void qblas_esyrk_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                          T alpha,
                          const T *a, const T *b,
                          T *c, ptrdiff_t ldc, ptrdiff_t offset)
{
    T subbuf[NR * (NR + 1)];

    if (m + offset < 0) {
        qblas_egemm_kernel(m, n, k, alpha, a, b, c, ldc);
        return;
    }
    if (n < offset) {
        return;
    }
    if (offset > 0) {
        b += offset * k;
        c += offset * ldc;
        n -= offset;
        offset = 0;
        if (n <= 0) return;
    }
    if (n > m + offset) {
        qblas_egemm_kernel(m, n - m - offset, k, alpha,
                           a, b + (m + offset) * k,
                           c + (m + offset) * ldc, ldc);
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        qblas_egemm_kernel(-offset, n, k, alpha, a, b, c, ldc);
        a -= offset * k;
        c -= offset;
        m += offset;
        offset = 0;
        if (m <= 0) return;
    }

    /* Diagonal walk in NR-step blocks. offset == 0, m == n here. */
    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t mm = loop;
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;
        (void)mm;

        /* Strict-upper portion: rows 0..loop-1 × cols loop..loop+nn-1. */
        if (loop > 0) {
            qblas_egemm_kernel(loop, nn, k, alpha,
                               a, b + loop * k, c + loop * ldc, ldc);
        }

        /* Diagonal block via subbuffer. */
        for (ptrdiff_t z = 0; z < nn * nn; ++z) subbuf[z] = 0;
        qblas_egemm_kernel(nn, nn, k, alpha,
                           a + loop * k, b + loop * k, subbuf, nn);

        T *cc = c + loop + loop * ldc;
        const T *ss = subbuf;
        for (ptrdiff_t j = 0; j < nn; ++j) {
            for (ptrdiff_t i = 0; i <= j; ++i) cc[i] += ss[i];
            ss += nn;
            cc += ldc;
        }
    }
}

void qblas_esyrk_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                          T alpha,
                          const T *a, const T *b,
                          T *c, ptrdiff_t ldc, ptrdiff_t offset)
{
    T subbuf[NR * (NR + 1)];

    if (m + offset < 0) {
        return;
    }
    if (n < offset) {
        qblas_egemm_kernel(m, n, k, alpha, a, b, c, ldc);
        return;
    }
    if (offset > 0) {
        qblas_egemm_kernel(m, offset, k, alpha, a, b, c, ldc);
        b += offset * k;
        c += offset * ldc;
        n -= offset;
        offset = 0;
        if (n <= 0) return;
    }
    if (n > m + offset) {
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        a -= offset * k;
        c -= offset;
        m += offset;
        offset = 0;
        if (m <= 0) return;
    }
    if (m > n - offset) {
        qblas_egemm_kernel(m - n + offset, n, k, alpha,
                           a + (n - offset) * k, b,
                           c + (n - offset), ldc);
        m = n + offset;
        if (m <= 0) return;
    }

    /* Diagonal walk in NR-step blocks. offset == 0, m == n here. */
    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t mm = loop;
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        /* Diagonal block via subbuffer. */
        for (ptrdiff_t z = 0; z < nn * nn; ++z) subbuf[z] = 0;
        qblas_egemm_kernel(nn, nn, k, alpha,
                           a + loop * k, b + loop * k, subbuf, nn);

        T *cc = c + loop + loop * ldc;
        const T *ss = subbuf;
        for (ptrdiff_t j = 0; j < nn; ++j) {
            for (ptrdiff_t i = j; i < nn; ++i) cc[i] += ss[i];
            ss += nn;
            cc += ldc;
        }

        /* Strict-lower portion: rows loop+nn..m-1 × cols loop..loop+nn-1. */
        if (m > mm + nn) {
            qblas_egemm_kernel(m - mm - nn, nn, k, alpha,
                               a + (mm + nn) * k, b + loop * k,
                               c + (mm + nn) + loop * ldc, ldc);
        }
    }
}


/* ── SYR2K kernel: two-pass diagonal-aware GEMM (real) ───────────────
 *
 * Faithful port of OpenBLAS driver/level3/syr2k_kernel.c. The strip
 * regions write `alpha * Ap * Bp` exactly like SYRK — the caller invokes
 * this kernel twice per (is, js) tile, once with (Ap=A-pack, Bp=B-pack)
 * and once with (Ap=B-pack, Bp=A-pack); the two strip writes accumulate
 * to give `alpha * (A*B^T + B*A^T)` over the kept triangle.
 *
 * The diagonal NR×NR block uses a single subbuffer-GEMM call gated by
 * `flag`: when flag != 0 the merge writes `subbuf[i,j] + subbuf[j,i]`
 * into C[i,j] (kept triangle), which captures both contributions in
 * one shot via the symmetric property. Pass 2 uses flag == 0 to skip
 * the redundant write.
 */
void qblas_esyr2k_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           T alpha,
                           const T *a, const T *b,
                           T *c, ptrdiff_t ldc, ptrdiff_t offset, int flag)
{
    T subbuf[NR * (NR + 1)];

    if (m + offset < 0) {
        qblas_egemm_kernel(m, n, k, alpha, a, b, c, ldc);
        return;
    }
    if (n < offset) {
        return;
    }
    if (offset > 0) {
        b += offset * k;
        c += offset * ldc;
        n -= offset;
        offset = 0;
        if (n <= 0) return;
    }
    if (n > m + offset) {
        qblas_egemm_kernel(m, n - m - offset, k, alpha,
                           a, b + (m + offset) * k,
                           c + (m + offset) * ldc, ldc);
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        qblas_egemm_kernel(-offset, n, k, alpha, a, b, c, ldc);
        a -= offset * k;
        c -= offset;
        m += offset;
        offset = 0;
        if (m <= 0) return;
    }

    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t mm = loop;
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;
        (void)mm;

        /* Strict-upper portion: rows 0..loop-1 × cols loop..loop+nn-1. */
        if (loop > 0) {
            qblas_egemm_kernel(loop, nn, k, alpha,
                               a, b + loop * k, c + loop * ldc, ldc);
        }

        if (flag) {
            /* Diagonal NR×NR block: GEMM into subbuf, merge symmetric. */
            for (ptrdiff_t z = 0; z < nn * nn; ++z) subbuf[z] = 0;
            qblas_egemm_kernel(nn, nn, k, alpha,
                               a + loop * k, b + loop * k, subbuf, nn);

            T *cc = c + loop + loop * ldc;
            for (ptrdiff_t j = 0; j < nn; ++j) {
                for (ptrdiff_t i = 0; i <= j; ++i) {
                    cc[i + j * ldc] += subbuf[i + j * nn]
                                     + subbuf[j + i * nn];
                }
            }
        }
    }
}

void qblas_esyr2k_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           T alpha,
                           const T *a, const T *b,
                           T *c, ptrdiff_t ldc, ptrdiff_t offset, int flag)
{
    T subbuf[NR * (NR + 1)];

    if (m + offset < 0) {
        return;
    }
    if (n < offset) {
        qblas_egemm_kernel(m, n, k, alpha, a, b, c, ldc);
        return;
    }
    if (offset > 0) {
        qblas_egemm_kernel(m, offset, k, alpha, a, b, c, ldc);
        b += offset * k;
        c += offset * ldc;
        n -= offset;
        offset = 0;
        if (n <= 0) return;
    }
    if (n > m + offset) {
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        a -= offset * k;
        c -= offset;
        m += offset;
        offset = 0;
        if (m <= 0) return;
    }
    if (m > n - offset) {
        qblas_egemm_kernel(m - n + offset, n, k, alpha,
                           a + (n - offset) * k, b,
                           c + (n - offset), ldc);
        m = n + offset;
        if (m <= 0) return;
    }

    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t mm = loop;
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        if (flag) {
            for (ptrdiff_t z = 0; z < nn * nn; ++z) subbuf[z] = 0;
            qblas_egemm_kernel(nn, nn, k, alpha,
                               a + loop * k, b + loop * k, subbuf, nn);

            T *cc = c + loop + loop * ldc;
            for (ptrdiff_t j = 0; j < nn; ++j) {
                for (ptrdiff_t i = j; i < nn; ++i) {
                    cc[i + j * ldc] += subbuf[i + j * nn]
                                     + subbuf[j + i * nn];
                }
            }
        }

        /* Strict-lower portion: rows loop+nn..m-1 × cols loop..loop+nn-1. */
        if (m > mm + nn) {
            qblas_egemm_kernel(m - mm - nn, nn, k, alpha,
                               a + (mm + nn) * k, b + loop * k,
                               c + (mm + nn) + loop * ldc, ldc);
        }
    }
}


/* ── GEMM kernel, overwrite variant ──────────────────────────────────
 *
 * C := alpha * Ap * Bp (overwrite). Used inside the TRMM L3 driver for
 * the off-diagonal sub-tiles where OpenBLAS calls GEMM_KERNEL with
 * beta=ZERO (interface/trsm.c sets args->beta = alpha; the kernel sees
 * dp1=1 as alpha and ZERO as beta, i.e. overwrite-with-product). Our
 * shared qblas_egemm_kernel does C += ..., so we zero the tile first.
 */
void qblas_egemm_kernel_store(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                              T alpha,
                              const T *Ap,
                              const T *Bp,
                              T *C, ptrdiff_t ldc)
{
    for (ptrdiff_t j = 0; j < bn; ++j) {
        T *cj = C + j * ldc;
        for (ptrdiff_t i = 0; i < bm; ++i) cj[i] = 0;
    }
    qblas_egemm_kernel(bm, bn, bk, alpha, Ap, Bp, C, ldc);
}


/* ── TRMM A-side triangular packers (real) ───────────────────────────
 *
 * Faithful ports of OpenBLAS kernel/generic/trmm_{ut,un,lt,ln}copy_2.c
 * with the compile-time UNIT macro replaced by a runtime `unit` flag.
 * Otherwise line-for-line — `ONE` / `ZERO` → `1.0Q` / `0.0Q`, the
 * function shape preserves (m, n, a, lda, posX, posY, b) plus `unit`.
 */

void qblas_etrmm_iutcopy(ptrdiff_t m, ptrdiff_t n,
                         const T *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY,
                         T *b, int unit)
{
    ptrdiff_t i, js;
    ptrdiff_t X;
    T data01, data02, data03, data04;
    const T *ao1, *ao2;

    js = (n >> 1);

    if (js > 0) {
        do {
            X = posX;

            if (posX <= posY) {
                ao1 = a + posX + (posY + 0) * lda;
                ao2 = a + posX + (posY + 1) * lda;
            } else {
                ao1 = a + posY + (posX + 0) * lda;
                ao2 = a + posY + (posX + 1) * lda;
            }

            i = (m >> 1);
            if (i > 0) {
                do {
                    if (X < posY) {
                        ao1 += 2;
                        ao2 += 2;
                        b   += 4;
                    } else if (X > posY) {
                        data01 = *(ao1 + 0);
                        data02 = *(ao1 + 1);
                        data03 = *(ao2 + 0);
                        data04 = *(ao2 + 1);

                        b[0] = data01;
                        b[1] = data02;
                        b[2] = data03;
                        b[3] = data04;

                        ao1 += 2 * lda;
                        ao2 += 2 * lda;
                        b += 4;
                    } else {
                        if (unit) {
                            data03 = *(ao2 + 0);
                            b[0] = 1.0Q;
                            b[1] = 0.0Q;
                            b[2] = data03;
                            b[3] = 1.0Q;
                        } else {
                            data01 = *(ao1 + 0);
                            data03 = *(ao2 + 0);
                            data04 = *(ao2 + 1);
                            b[0] = data01;
                            b[1] = 0.0Q;
                            b[2] = data03;
                            b[3] = data04;
                        }

                        ao1 += 2 * lda;
                        ao2 += 2 * lda;
                        b += 4;
                    }

                    X += 2;
                    i--;
                } while (i > 0);
            }

            if (m & 1) {
                if (X < posY) {
                    ao1 += 1;
                    ao2 += 1;
                    b += 2;
                } else if (X > posY) {
                    data01 = *(ao1 + 0);
                    data02 = *(ao1 + 1);
                    b[0] = data01;
                    b[1] = data02;
                    ao1 += lda;
                    b += 2;
                } else {
                    if (unit) {
                        b[0] = 1.0Q;
                        b[1] = 0.0Q;
                    } else {
                        data01 = *(ao1 + 0);
                        b[0] = data01;
                        b[1] = 0.0Q;
                    }
                    ao1 += lda;
                    b += 2;
                }
            }

            posY += 2;
            js--;
        } while (js > 0);
    }

    if (n & 1) {
        X = posX;

        if (posX <= posY) {
            ao1 = a + posX + (posY + 0) * lda;
        } else {
            ao1 = a + posY + (posX + 0) * lda;
        }

        i = m;
        if (m > 0) {
            do {
                if (X < posY) {
                    b += 1;
                    ao1 += 1;
                } else if (X > posY) {
                    data01 = *(ao1 + 0);
                    b[0] = data01;
                    b += 1;
                    ao1 += lda;
                } else {
                    if (unit) {
                        b[0] = 1.0Q;
                    } else {
                        data01 = *(ao1 + 0);
                        b[0] = data01;
                    }
                    b += 1;
                    ao1 += lda;
                }

                X += 1;
                i--;
            } while (i > 0);
        }
    }
}


void qblas_etrmm_iuncopy(ptrdiff_t m, ptrdiff_t n,
                         const T *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY,
                         T *b, int unit)
{
    ptrdiff_t i, js;
    ptrdiff_t X;
    T data01, data02, data03, data04;
    const T *ao1, *ao2;

    js = (n >> 1);

    if (js > 0) {
        do {
            X = posX;

            if (posX <= posY) {
                ao1 = a + posX + (posY + 0) * lda;
                ao2 = a + posX + (posY + 1) * lda;
            } else {
                ao1 = a + posY + (posX + 0) * lda;
                ao2 = a + posY + (posX + 1) * lda;
            }

            i = (m >> 1);
            if (i > 0) {
                do {
                    if (X < posY) {
                        data01 = *(ao1 + 0);
                        data02 = *(ao1 + 1);
                        data03 = *(ao2 + 0);
                        data04 = *(ao2 + 1);

                        b[0] = data01;
                        b[1] = data03;
                        b[2] = data02;
                        b[3] = data04;

                        ao1 += 2;
                        ao2 += 2;
                        b += 4;
                    } else if (X > posY) {
                        ao1 += 2 * lda;
                        ao2 += 2 * lda;
                        b += 4;
                    } else {
                        if (unit) {
                            data03 = *(ao2 + 0);
                            b[0] = 1.0Q;
                            b[1] = data03;
                            b[2] = 0.0Q;
                            b[3] = 1.0Q;
                        } else {
                            data01 = *(ao1 + 0);
                            data03 = *(ao2 + 0);
                            data04 = *(ao2 + 1);
                            b[0] = data01;
                            b[1] = data03;
                            b[2] = 0.0Q;
                            b[3] = data04;
                        }

                        ao1 += 2 * lda;
                        ao2 += 2 * lda;
                        b += 4;
                    }

                    X += 2;
                    i--;
                } while (i > 0);
            }

            if (m & 1) {
                if (X < posY) {
                    data01 = *(ao1 + 0);
                    data03 = *(ao2 + 0);
                    b[0] = data01;
                    b[1] = data03;
                    b += 2;
                } else if (X > posY) {
                    b += 2;
                } else {
                    if (unit) {
                        data03 = *(ao2 + 0);
                        b[0] = 1.0Q;
                        b[1] = data03;
                    } else {
                        data01 = *(ao1 + 0);
                        data03 = *(ao2 + 0);
                        b[0] = data01;
                        b[1] = data03;
                    }
                    b += 2;
                }
            }

            posY += 2;
            js--;
        } while (js > 0);
    }

    if (n & 1) {
        X = posX;

        if (posX <= posY) {
            ao1 = a + posX + (posY + 0) * lda;
        } else {
            ao1 = a + posY + (posX + 0) * lda;
        }

        i = m;
        if (m > 0) {
            do {
                if (X < posY) {
                    data01 = *(ao1 + 0);
                    b[0] = data01;
                    ao1 += 1;
                    b += 1;
                } else if (X > posY) {
                    ao1 += lda;
                    b += 1;
                } else {
                    if (unit) {
                        b[0] = 1.0Q;
                    } else {
                        data01 = *(ao1 + 0);
                        b[0] = data01;
                    }
                    b += 1;
                    ao1 += lda;
                }

                X += 1;
                i--;
            } while (i > 0);
        }
    }
}


void qblas_etrmm_iltcopy(ptrdiff_t m, ptrdiff_t n,
                         const T *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY,
                         T *b, int unit)
{
    ptrdiff_t i, js;
    ptrdiff_t X;
    T data01, data02, data03, data04;
    const T *ao1, *ao2;

    js = (n >> 1);

    if (js > 0) {
        do {
            X = posX;

            if (posX <= posY) {
                ao1 = a + posY + (posX + 0) * lda;
                ao2 = a + posY + (posX + 1) * lda;
            } else {
                ao1 = a + posX + (posY + 0) * lda;
                ao2 = a + posX + (posY + 1) * lda;
            }

            i = (m >> 1);
            if (i > 0) {
                do {
                    if (X > posY) {
                        ao1 += 2;
                        ao2 += 2;
                        b += 4;
                    } else if (X < posY) {
                        data01 = *(ao1 + 0);
                        data02 = *(ao1 + 1);
                        data03 = *(ao2 + 0);
                        data04 = *(ao2 + 1);

                        b[0] = data01;
                        b[1] = data02;
                        b[2] = data03;
                        b[3] = data04;

                        ao1 += 2 * lda;
                        ao2 += 2 * lda;
                        b += 4;
                    } else {
                        if (unit) {
                            data02 = *(ao1 + 1);
                            b[0] = 1.0Q;
                            b[1] = data02;
                            b[2] = 0.0Q;
                            b[3] = 1.0Q;
                        } else {
                            data01 = *(ao1 + 0);
                            data02 = *(ao1 + 1);
                            data04 = *(ao2 + 1);
                            b[0] = data01;
                            b[1] = data02;
                            b[2] = 0.0Q;
                            b[3] = data04;
                        }

                        ao1 += 2;
                        ao2 += 2;
                        b += 4;
                    }

                    X += 2;
                    i--;
                } while (i > 0);
            }

            if (m & 1) {
                if (X > posY) {
                    ao1 += 1;
                    ao2 += 1;
                    b += 2;
                } else if (X < posY) {
                    data01 = *(ao1 + 0);
                    data02 = *(ao1 + 1);
                    b[0] = data01;
                    b[1] = data02;
                    ao1 += lda;
                    b += 2;
                } else {
                    if (unit) {
                        data02 = *(ao1 + 1);
                        b[0] = 1.0Q;
                        b[1] = data02;
                    } else {
                        data01 = *(ao1 + 0);
                        data02 = *(ao1 + 1);
                        b[0] = data01;
                        b[1] = data02;
                    }
                    ao1 += 2;
                    b += 2;
                }
            }

            posY += 2;
            js--;
        } while (js > 0);
    }

    if (n & 1) {
        X = posX;

        if (posX <= posY) {
            ao1 = a + posY + (posX + 0) * lda;
        } else {
            ao1 = a + posX + (posY + 0) * lda;
        }

        i = m;
        if (i > 0) {
            do {
                if (X > posY) {
                    ao1 += 1;
                    b += 1;
                } else if (X < posY) {
                    data01 = *(ao1 + 0);
                    b[0] = data01;
                    ao1 += lda;
                    b += 1;
                } else {
                    if (unit) {
                        b[0] = 1.0Q;
                    } else {
                        data01 = *(ao1 + 0);
                        b[0] = data01;
                    }
                    b += 1;
                    ao1 += 1;
                }

                X++;
                i--;
            } while (i > 0);
        }
    }
}


void qblas_etrmm_ilncopy(ptrdiff_t m, ptrdiff_t n,
                         const T *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY,
                         T *b, int unit)
{
    ptrdiff_t i, js;
    ptrdiff_t X;
    T data01, data02, data03, data04;
    const T *ao1, *ao2;

    js = (n >> 1);

    if (js > 0) {
        do {
            X = posX;

            if (posX <= posY) {
                ao1 = a + posY + (posX + 0) * lda;
                ao2 = a + posY + (posX + 1) * lda;
            } else {
                ao1 = a + posX + (posY + 0) * lda;
                ao2 = a + posX + (posY + 1) * lda;
            }

            i = (m >> 1);
            if (i > 0) {
                do {
                    if (X > posY) {
                        data01 = *(ao1 + 0);
                        data02 = *(ao1 + 1);
                        data03 = *(ao2 + 0);
                        data04 = *(ao2 + 1);

                        b[0] = data01;
                        b[1] = data03;
                        b[2] = data02;
                        b[3] = data04;

                        ao1 += 2;
                        ao2 += 2;
                        b += 4;
                    } else if (X < posY) {
                        ao1 += 2 * lda;
                        ao2 += 2 * lda;
                        b += 4;
                    } else {
                        if (unit) {
                            data02 = *(ao1 + 1);
                            b[0] = 1.0Q;
                            b[1] = 0.0Q;
                            b[2] = data02;
                            b[3] = 1.0Q;
                        } else {
                            data01 = *(ao1 + 0);
                            data02 = *(ao1 + 1);
                            data04 = *(ao2 + 1);
                            b[0] = data01;
                            b[1] = 0.0Q;
                            b[2] = data02;
                            b[3] = data04;
                        }
                        ao1 += 2;
                        ao2 += 2;
                        b += 4;
                    }

                    X += 2;
                    i--;
                } while (i > 0);
            }

            if (m & 1) {
                if (X > posY) {
                    data01 = *(ao1 + 0);
                    data03 = *(ao2 + 0);
                    b[0] = data01;
                    b[1] = data03;
                    b += 2;
                } else if (X < posY) {
                    b += 2;
                } else {
                    if (unit) {
                        data03 = *(ao2 + 0);
                        b[0] = 1.0Q;
                        b[1] = data03;
                    } else {
                        data01 = *(ao1 + 0);
                        data03 = *(ao2 + 0);
                        b[0] = data01;
                        b[1] = data03;
                    }
                    b += 2;
                }
            }

            posY += 2;
            js--;
        } while (js > 0);
    }

    if (n & 1) {
        X = posX;

        if (posX <= posY) {
            ao1 = a + posY + (posX + 0) * lda;
        } else {
            ao1 = a + posX + (posY + 0) * lda;
        }

        i = m;
        if (i > 0) {
            do {
                if (X > posY) {
                    data01 = *(ao1 + 0);
                    b[0] = data01;
                    ao1 += 1;
                    b += 1;
                } else if (X < posY) {
                    ao1 += lda;
                    b += 1;
                } else {
                    if (unit) {
                        b[0] = 1.0Q;
                    } else {
                        data01 = *(ao1 + 0);
                        b[0] = data01;
                    }
                    b += 1;
                    ao1 += 1;
                }

                X++;
                i--;
            } while (i > 0);
        }
    }
}


/* ── TRMM diagonal-aware microkernel (real) ──────────────────────────
 *
 * Faithful port of OpenBLAS kernel/generic/trmmkernel_2x2.c with
 * compile-time LEFT and TRANSA macros converted to runtime `left` and
 * `trans` flags. The 4 (left, trans) combinations correspond to
 * OpenBLAS's TRMM_KERNEL_{LN,LT,RN,RT}; collapsing to one function with
 * runtime branches is branch-predictor friendly since left/trans are
 * loop-invariant per call. C := alpha * ba * bb (overwrite semantics —
 * matches OpenBLAS's GEMM_KERNEL(beta=0) convention inside the TRMM
 * driver).
 *
 * Local notation matches the source: `off`, `temp`, `res0..res3`,
 * `load0..load7`. Pointer-arithmetic formulas (`ptrba += off*2`,
 * `temp = bk - off`, etc.) are gated on the (left, trans) flags
 * with the same boolean structure as the OpenBLAS preprocessor logic.
 */
void qblas_etrmm_kernel(int left, int trans,
                        ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        T alpha,
                        const T *ba, const T *bb,
                        T *C, ptrdiff_t ldc,
                        ptrdiff_t offset)
{
    ptrdiff_t i, j, k;
    T *C0, *C1;
    const T *ptrba, *ptrbb;
    T res0, res1, res2, res3;
    T load0, load1, load2, load3, load4, load5, load6, load7;
    ptrdiff_t off, temp;

    /* if defined(TRMMKERNEL) && !defined(LEFT) → off = -offset, else 0 */
    if (!left) off = -offset;
    else       off = 0;

    for (j = 0; j < bn / 2; j += 1) {
        C0 = C;
        C1 = C0 + ldc;

        if (left) off = offset;

        ptrba = ba;
        for (i = 0; i < bm / 2; i += 1) {
            /* if (LEFT && TRANSA) || (!LEFT && !TRANSA): ptrbb = bb;
               else                                       ptrba += off*2; ptrbb = bb + off*2; */
            if ((left && trans) || (!left && !trans)) {
                ptrbb = bb;
            } else {
                ptrba += off * 2;
                ptrbb = bb + off * 2;
            }
            res0 = 0; res1 = 0; res2 = 0; res3 = 0;

            /* temp formula: see source lines 38-45.
               (LEFT && !TRANSA) || (!LEFT && TRANSA): temp = bk - off
               else (LEFT && TRANSA || !LEFT && !TRANSA): temp = off + 2 */
            if ((left && !trans) || (!left && trans))
                temp = bk - off;
            else
                temp = off + 2;

            for (k = 0; k < temp / 4; k += 1) {
                load0 = ptrba[2*0+0];
                load1 = ptrbb[2*0+0];
                res0 = res0 + load0 * load1;
                load2 = ptrba[2*0+1];
                res1 = res1 + load2 * load1;
                load3 = ptrbb[2*0+1];
                res2 = res2 + load0 * load3;
                res3 = res3 + load2 * load3;
                load4 = ptrba[2*1+0];
                load5 = ptrbb[2*1+0];
                res0 = res0 + load4 * load5;
                load6 = ptrba[2*1+1];
                res1 = res1 + load6 * load5;
                load7 = ptrbb[2*1+1];
                res2 = res2 + load4 * load7;
                res3 = res3 + load6 * load7;
                load0 = ptrba[2*2+0];
                load1 = ptrbb[2*2+0];
                res0 = res0 + load0 * load1;
                load2 = ptrba[2*2+1];
                res1 = res1 + load2 * load1;
                load3 = ptrbb[2*2+1];
                res2 = res2 + load0 * load3;
                res3 = res3 + load2 * load3;
                load4 = ptrba[2*3+0];
                load5 = ptrbb[2*3+0];
                res0 = res0 + load4 * load5;
                load6 = ptrba[2*3+1];
                res1 = res1 + load6 * load5;
                load7 = ptrbb[2*3+1];
                res2 = res2 + load4 * load7;
                res3 = res3 + load6 * load7;
                ptrba = ptrba + 8;
                ptrbb = ptrbb + 8;
            }
            for (k = 0; k < (temp & 3); k += 1) {
                load0 = ptrba[2*0+0];
                load1 = ptrbb[2*0+0];
                res0 = res0 + load0 * load1;
                load2 = ptrba[2*0+1];
                res1 = res1 + load2 * load1;
                load3 = ptrbb[2*0+1];
                res2 = res2 + load0 * load3;
                res3 = res3 + load2 * load3;
                ptrba = ptrba + 2;
                ptrbb = ptrbb + 2;
            }
            res0 = res0 * alpha; C0[0] = res0;
            res1 = res1 * alpha; C0[1] = res1;
            res2 = res2 * alpha; C1[0] = res2;
            res3 = res3 * alpha; C1[1] = res3;

            /* if (LEFT && TRANSA) || (!LEFT && !TRANSA): advance ptrba/ptrbb */
            if ((left && trans) || (!left && !trans)) {
                temp = bk - off;
                temp -= 2;
                ptrba += temp * 2;
                ptrbb += temp * 2;
            }
            if (left) off += 2;

            C0 = C0 + 2;
            C1 = C1 + 2;
        }

        /* bm & 1 — single-row tail */
        for (i = 0; i < (bm & 1); i += 1) {
            if ((left && trans) || (!left && !trans)) {
                ptrbb = bb;
            } else {
                ptrba += off;
                ptrbb = bb + off * 2;
            }
            res0 = 0; res1 = 0;

            if ((left && !trans) || (!left && trans))
                temp = bk - off;
            else if (left)
                temp = off + 1;
            else
                temp = off + 2;

            for (k = 0; k < temp; k += 1) {
                load0 = ptrba[0+0];
                load1 = ptrbb[2*0+0];
                res0 = res0 + load0 * load1;
                load2 = ptrbb[2*0+1];
                res1 = res1 + load0 * load2;
                ptrba = ptrba + 1;
                ptrbb = ptrbb + 2;
            }
            res0 = res0 * alpha; C0[0] = res0;
            res1 = res1 * alpha; C1[0] = res1;

            if ((left && trans) || (!left && !trans)) {
                temp = bk - off;
                if (left) temp -= 1;
                else      temp -= 2;
                ptrba += temp;
                ptrbb += temp * 2;
            }
            if (left) off += 1;

            C0 = C0 + 1;
            C1 = C1 + 1;
        }

        if (!left) off += 2;

        k = (bk << 1);
        bb = bb + k;
        i = (ldc << 1);
        C = C + i;
    }

    /* bn & 1 — single-col tail */
    for (j = 0; j < (bn & 1); j += 1) {
        C0 = C;
        if (left) off = offset;
        ptrba = ba;
        for (i = 0; i < bm / 2; i += 1) {
            if ((left && trans) || (!left && !trans)) {
                ptrbb = bb;
            } else {
                ptrba += off * 2;
                ptrbb = bb + off;
            }
            res0 = 0; res1 = 0;

            if ((left && !trans) || (!left && trans))
                temp = bk - off;
            else if (left)
                temp = off + 2;
            else
                temp = off + 1;

            for (k = 0; k < temp; k += 1) {
                load0 = ptrba[2*0+0];
                load1 = ptrbb[0+0];
                res0 = res0 + load0 * load1;
                load2 = ptrba[2*0+1];
                res1 = res1 + load2 * load1;
                ptrba = ptrba + 2;
                ptrbb = ptrbb + 1;
            }
            res0 = res0 * alpha; C0[0] = res0;
            res1 = res1 * alpha; C0[1] = res1;

            if ((left && trans) || (!left && !trans)) {
                temp = bk - off;
                if (left) temp -= 2;
                else      temp -= 1;
                ptrba += temp * 2;
                ptrbb += temp;
            }
            if (left) off += 2;

            C0 = C0 + 2;
        }
        for (i = 0; i < (bm & 1); i += 1) {
            if ((left && trans) || (!left && !trans)) {
                ptrbb = bb;
            } else {
                ptrba += off;
                ptrbb = bb + off;
            }
            res0 = 0;

            if ((left && !trans) || (!left && trans))
                temp = bk - off;
            else if (left)
                temp = off + 1;
            else
                temp = off + 1;

            for (k = 0; k < temp; k += 1) {
                load0 = ptrba[0+0];
                load1 = ptrbb[0+0];
                res0 = res0 + load0 * load1;
                ptrba = ptrba + 1;
                ptrbb = ptrbb + 1;
            }
            res0 = res0 * alpha; C0[0] = res0;

            if ((left && trans) || (!left && !trans)) {
                temp = bk - off;
                temp -= 1;
                ptrba += temp;
                ptrbb += temp;
            }
            if (left) off += 1;

            C0 = C0 + 1;
        }
        if (!left) off += 1;
        k = bk;  /* (bk<<0) */
        bb = bb + k;
        C = C + ldc;
    }
}


/* ── TRSM A-side triangular packers (real) ───────────────────────────
 *
 * Faithful ports of OpenBLAS kernel/generic/trsm_{ut,un,lt,ln}copy_2.c.
 * Diagonal positions get `1.0Q / a[diag]` (or 1.0Q when unit); strict-
 * stored-triangle data is written through; the other side of the
 * register-tile is left as-is (kernel never reads it).
 *
 * `offset` carries the OpenBLAS BLASLONG offset arg: it is the
 * (ii - jj) shift used by the L3 driver to position the diagonal
 * relative to the packer's local 2-step ii sweep.
 */

void qblas_etrsm_ilncopy(ptrdiff_t m, ptrdiff_t n,
                         const T *a, ptrdiff_t lda,
                         ptrdiff_t offset, T *b, int unit)
{
    ptrdiff_t i, ii, j, jj;
    T data01 = 0.0Q, data02, data03 = 0.0Q, data04 = 0.0Q;
    const T *a1, *a2;

    jj = offset;

    j = (n >> 1);
    while (j > 0) {
        a1 = a + 0 * lda;
        a2 = a + 1 * lda;

        i = (m >> 1);
        ii = 0;
        while (i > 0) {
            if (ii == jj) {
                if (!unit) {
                    data01 = *(a1 + 0);
                }
                data02 = *(a1 + 1);
                if (!unit) {
                    data04 = *(a2 + 1);
                }
                b[0] = unit ? 1.0Q : (1.0Q / data01);
                b[2] = data02;
                b[3] = unit ? 1.0Q : (1.0Q / data04);
            }
            if (ii > jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                data03 = *(a2 + 0);
                data04 = *(a2 + 1);
                b[0] = data01;
                b[1] = data03;
                b[2] = data02;
                b[3] = data04;
            }
            a1 += 2;
            a2 += 2;
            b += 4;
            i--;
            ii += 2;
        }

        if (m & 1) {
            if (ii == jj) {
                if (!unit) data01 = *(a1 + 0);
                b[0] = unit ? 1.0Q : (1.0Q / data01);
            }
            if (ii > jj) {
                data01 = *(a1 + 0);
                data02 = *(a2 + 0);
                b[0] = data01;
                b[1] = data02;
            }
            b += 2;
        }

        a += 2 * lda;
        jj += 2;
        j--;
    }

    if (n & 1) {
        a1 = a + 0 * lda;
        i = m;
        ii = 0;
        while (i > 0) {
            if (ii == jj) {
                if (!unit) data01 = *(a1 + 0);
                b[0] = unit ? 1.0Q : (1.0Q / data01);
            }
            if (ii > jj) {
                data01 = *(a1 + 0);
                b[0] = data01;
            }
            a1 += 1;
            b += 1;
            i--;
            ii += 1;
        }
    }
}


void qblas_etrsm_iltcopy(ptrdiff_t m, ptrdiff_t n,
                         const T *a, ptrdiff_t lda,
                         ptrdiff_t offset, T *b, int unit)
{
    ptrdiff_t i, ii, j, jj;
    T data01 = 0.0Q, data02 = 0.0Q, data03 = 0.0Q, data04 = 0.0Q;
    const T *a1, *a2;

    jj = offset;

    j = (n >> 1);
    while (j > 0) {
        a1 = a + 0 * lda;
        a2 = a + 1 * lda;

        i = (m >> 1);
        ii = 0;
        while (i > 0) {
            if (ii == jj) {
                if (!unit) data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                if (!unit) data04 = *(a2 + 1);
                b[0] = unit ? 1.0Q : (1.0Q / data01);
                b[1] = data02;
                b[3] = unit ? 1.0Q : (1.0Q / data04);
            }
            if (ii < jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                data03 = *(a2 + 0);
                data04 = *(a2 + 1);
                b[0] = data01;
                b[1] = data02;
                b[2] = data03;
                b[3] = data04;
            }
            a1 += 2 * lda;
            a2 += 2 * lda;
            b += 4;
            i--;
            ii += 2;
        }

        if (m & 1) {
            if (ii == jj) {
                if (!unit) data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                b[0] = unit ? 1.0Q : (1.0Q / data01);
                b[1] = data02;
            }
            if (ii < jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                b[0] = data01;
                b[1] = data02;
            }
            b += 2;
        }

        a += 2;
        jj += 2;
        j--;
    }

    if (n & 1) {
        a1 = a + 0 * lda;
        i = m;
        ii = 0;
        while (i > 0) {
            if (ii == jj) {
                if (!unit) data01 = *(a1 + 0);
                b[0] = unit ? 1.0Q : (1.0Q / data01);
            }
            if (ii < jj) {
                data01 = *(a1 + 0);
                b[0] = data01;
            }
            a1 += 1 * lda;
            b += 1;
            i--;
            ii += 1;
        }
    }
}


void qblas_etrsm_iuncopy(ptrdiff_t m, ptrdiff_t n,
                         const T *a, ptrdiff_t lda,
                         ptrdiff_t offset, T *b, int unit)
{
    ptrdiff_t i, ii, j, jj;
    T data01 = 0.0Q, data02, data03 = 0.0Q, data04 = 0.0Q;
    const T *a1, *a2;

    jj = offset;

    j = (n >> 1);
    while (j > 0) {
        a1 = a + 0 * lda;
        a2 = a + 1 * lda;

        i = (m >> 1);
        ii = 0;
        while (i > 0) {
            if (ii == jj) {
                if (!unit) data01 = *(a1 + 0);
                data03 = *(a2 + 0);
                if (!unit) data04 = *(a2 + 1);
                b[0] = unit ? 1.0Q : (1.0Q / data01);
                b[1] = data03;
                b[3] = unit ? 1.0Q : (1.0Q / data04);
            }
            if (ii < jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                data03 = *(a2 + 0);
                data04 = *(a2 + 1);
                b[0] = data01;
                b[1] = data03;
                b[2] = data02;
                b[3] = data04;
            }
            a1 += 2;
            a2 += 2;
            b += 4;
            i--;
            ii += 2;
        }

        if (m & 1) {
            if (ii == jj) {
                if (!unit) data01 = *(a1 + 0);
                data02 = *(a2 + 0);
                b[0] = unit ? 1.0Q : (1.0Q / data01);
                b[1] = data02;
            }
            if (ii < jj) {
                data01 = *(a1 + 0);
                data02 = *(a2 + 0);
                b[0] = data01;
                b[1] = data02;
            }
            b += 2;
        }

        a += 2 * lda;
        jj += 2;
        j--;
    }

    if (n & 1) {
        a1 = a + 0 * lda;
        i = m;
        ii = 0;
        while (i > 0) {
            if (ii == jj) {
                if (!unit) data01 = *(a1 + 0);
                b[0] = unit ? 1.0Q : (1.0Q / data01);
            }
            if (ii < jj) {
                data01 = *(a1 + 0);
                b[0] = data01;
            }
            a1 += 1;
            b += 1;
            i--;
            ii += 1;
        }
    }
}


void qblas_etrsm_iutcopy(ptrdiff_t m, ptrdiff_t n,
                         const T *a, ptrdiff_t lda,
                         ptrdiff_t offset, T *b, int unit)
{
    ptrdiff_t i, ii, j, jj;
    T data01 = 0.0Q, data02 = 0.0Q, data03 = 0.0Q, data04 = 0.0Q;
    const T *a1, *a2;

    jj = offset;

    j = (n >> 1);
    while (j > 0) {
        a1 = a + 0 * lda;
        a2 = a + 1 * lda;

        i = (m >> 1);
        ii = 0;
        while (i > 0) {
            if (ii == jj) {
                if (!unit) data01 = *(a1 + 0);
                data03 = *(a2 + 0);
                if (!unit) data04 = *(a2 + 1);
                b[0] = unit ? 1.0Q : (1.0Q / data01);
                b[2] = data03;
                b[3] = unit ? 1.0Q : (1.0Q / data04);
            }
            if (ii > jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                data03 = *(a2 + 0);
                data04 = *(a2 + 1);
                b[0] = data01;
                b[1] = data02;
                b[2] = data03;
                b[3] = data04;
            }
            a1 += 2 * lda;
            a2 += 2 * lda;
            b += 4;
            i--;
            ii += 2;
        }

        if (m & 1) {
            if (ii == jj) {
                if (!unit) data01 = *(a1 + 0);
                b[0] = unit ? 1.0Q : (1.0Q / data01);
            }
            if (ii > jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                b[0] = data01;
                b[1] = data02;
            }
            b += 2;
        }

        a += 2;
        jj += 2;
        j--;
    }

    if (n & 1) {
        a1 = a + 0 * lda;
        i = m;
        ii = 0;
        while (i > 0) {
            if (ii == jj) {
                if (!unit) data01 = *(a1 + 0);
                b[0] = unit ? 1.0Q : (1.0Q / data01);
            }
            if (ii > jj) {
                data01 = *(a1 + 0);
                b[0] = data01;
            }
            a1 += 1 * lda;
            b += 1;
            i--;
            ii += 1;
        }
    }
}


/* ── TRSM diagonal-aware microkernel (real) ──────────────────────────
 *
 * Faithful ports of OpenBLAS kernel/generic/trsm_kernel_{LN,LT,RN,RT}.c.
 * The 4 variants are dispatched via runtime (left, trans). Each variant
 * has its own `solve()` function (forward/backward; row-major-of-tile
 * vs col-major-of-tile) and its own outer (i, j) walk over the
 * register-tile grid.
 *
 * Notes:
 *   - dm1 = -1.0Q is the kernel's effective alpha (OpenBLAS uses a
 *     `static const FLOAT dm1 = -1.`); user alpha is applied as a
 *     B *= alpha pre-pass at the driver level.
 *   - GEMM_KERNEL is invoked as `kernel(MR, NR, k-kk, dm1, aa, b, cc, ldc)`.
 *     Our shared `qblas_egemm_kernel` has `C += alpha*A*B` semantics
 *     which lines up exactly when called with `alpha = dm1 = -1`
 *     (subtracts the trailing GEMM result from the partially-solved
 *     tile, mirroring OpenBLAS's GEMM_KERNEL_N(beta=ZERO) ... no wait,
 *     the upstream GEMM_KERNEL signature for the TRSM-internal call is
 *     `GEMM_KERNEL(m, n, k, alpha=dm1, [ZERO,] aa, b, cc, ldc)` — that
 *     IS the +=alpha variant. Match.).
 */

/* LN: solve walks i = m-1 down → 0; A holds strict-lower-triangle + invdiag */
static inline void solve_LN(ptrdiff_t m, ptrdiff_t n,
                            T *a, T *b, T *c, ptrdiff_t ldc)
{
    T aa, bb;
    ptrdiff_t i, j, k;
    a += (m - 1) * m;
    b += (m - 1) * n;
    for (i = m - 1; i >= 0; i--) {
        aa = *(a + i);
        for (j = 0; j < n; j++) {
            bb = *(c + i + j * ldc);
            bb *= aa;
            *b = bb;
            *(c + i + j * ldc) = bb;
            b++;
            for (k = 0; k < i; k++) {
                *(c + k + j * ldc) -= bb * *(a + k);
            }
        }
        a -= m;
        b -= 2 * n;
    }
}

/* LT: solve walks i = 0 up → m-1; A holds strict-upper-triangle + invdiag */
static inline void solve_LT(ptrdiff_t m, ptrdiff_t n,
                            T *a, T *b, T *c, ptrdiff_t ldc)
{
    T aa, bb;
    ptrdiff_t i, j, k;
    for (i = 0; i < m; i++) {
        aa = *(a + i);
        for (j = 0; j < n; j++) {
            bb = *(c + i + j * ldc);
            bb *= aa;
            *b = bb;
            *(c + i + j * ldc) = bb;
            b++;
            for (k = i + 1; k < m; k++) {
                *(c + k + j * ldc) -= bb * *(a + k);
            }
        }
        a += m;
    }
}

/* RN: solve walks i = 0 up → n-1; A holds strict-lower + invdiag (n×n) */
static inline void solve_RN(ptrdiff_t m, ptrdiff_t n,
                            T *a, T *b, T *c, ptrdiff_t ldc)
{
    T aa, bb;
    ptrdiff_t i, j, k;
    for (i = 0; i < n; i++) {
        bb = *(b + i);
        for (j = 0; j < m; j++) {
            aa = *(c + j + i * ldc);
            aa *= bb;
            *a = aa;
            *(c + j + i * ldc) = aa;
            a++;
            for (k = i + 1; k < n; k++) {
                *(c + j + k * ldc) -= aa * *(b + k);
            }
        }
        b += n;
    }
}

/* RT: solve walks i = n-1 down → 0; A holds strict-upper + invdiag */
static inline void solve_RT(ptrdiff_t m, ptrdiff_t n,
                            T *a, T *b, T *c, ptrdiff_t ldc)
{
    T aa, bb;
    ptrdiff_t i, j, k;
    a += (n - 1) * m;
    b += (n - 1) * n;
    for (i = n - 1; i >= 0; i--) {
        bb = *(b + i);
        for (j = 0; j < m; j++) {
            aa = *(c + j + i * ldc);
            aa *= bb;
            *a = aa;
            *(c + j + i * ldc) = aa;
            a++;
            for (k = 0; k < i; k++) {
                *(c + j + k * ldc) -= aa * *(b + k);
            }
        }
        b -= n;
        a -= 2 * m;
    }
}


/* The driver. UR / UN = unroll values; we use MR=2, NR=2 throughout, so
 * the OpenBLAS register-tile loop devolves to a single full-MR/full-NR
 * iteration plus possible single-row/col tails per (i,j) block.
 */
void qblas_etrsm_kernel(int left, int trans,
                        ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        const T *ba, const T *bb,
                        T *C, ptrdiff_t ldc,
                        ptrdiff_t offset)
{
    const T dm1 = -1.0Q;
    const ptrdiff_t UR = MR;   /* register tile rows */
    const ptrdiff_t UN = NR;   /* register tile cols */

    /* Cast away const for solve() in-place writes into the local pack
     * buffers — `ba` and `bb` are caller-owned per-thread scratch, so
     * mutating them is fine. */
    T *a_buf = (T *)ba;
    T *b_buf = (T *)bb;

    if (left && !trans) {
        /* trsm_kernel_LN.c — solve walks down rows. */
        ptrdiff_t i, j;
        T *aa, *cc;
        ptrdiff_t kk;

        j = (bn / UN);
        while (j > 0) {
            kk = bm + offset;

            /* m & (UR-1) tail (UR=2 → bm & 1) handled first, in a loop
             * that splits the tail into 1-row chunks. */
            if (bm & (UR - 1)) {
                for (i = 1; i < UR; i *= 2) {
                    if (bm & i) {
                        aa = a_buf + ((bm & ~(i - 1)) - i) * bk;
                        cc = C + ((bm & ~(i - 1)) - i);
                        if (bk - kk > 0) {
                            qblas_egemm_kernel(i, UN, bk - kk, dm1,
                                               aa + i * kk,
                                               b_buf + UN * kk, cc, ldc);
                        }
                        solve_LN(i, UN,
                                 aa + (kk - i) * i,
                                 b_buf + (kk - i) * UN,
                                 cc, ldc);
                        kk -= i;
                    }
                }
            }

            i = (bm / UR);
            if (i > 0) {
                aa = a_buf + ((bm & ~(UR - 1)) - UR) * bk;
                cc = C + ((bm & ~(UR - 1)) - UR);
                do {
                    if (bk - kk > 0) {
                        qblas_egemm_kernel(UR, UN, bk - kk, dm1,
                                           aa + UR * kk,
                                           b_buf + UN * kk, cc, ldc);
                    }
                    solve_LN(UR, UN,
                             aa + (kk - UR) * UR,
                             b_buf + (kk - UR) * UN,
                             cc, ldc);
                    aa -= UR * bk;
                    cc -= UR;
                    kk -= UR;
                    i--;
                } while (i > 0);
            }

            b_buf += UN * bk;
            C += UN * ldc;
            j--;
        }

        if (bn & (UN - 1)) {
            j = (UN >> 1);
            while (j > 0) {
                if (bn & j) {
                    kk = bm + offset;
                    if (bm & (UR - 1)) {
                        for (i = 1; i < UR; i *= 2) {
                            if (bm & i) {
                                aa = a_buf + ((bm & ~(i - 1)) - i) * bk;
                                cc = C + ((bm & ~(i - 1)) - i);
                                if (bk - kk > 0) {
                                    qblas_egemm_kernel(i, j, bk - kk, dm1,
                                                       aa + i * kk,
                                                       b_buf + j * kk, cc, ldc);
                                }
                                solve_LN(i, j,
                                         aa + (kk - i) * i,
                                         b_buf + (kk - i) * j,
                                         cc, ldc);
                                kk -= i;
                            }
                        }
                    }
                    i = (bm / UR);
                    if (i > 0) {
                        aa = a_buf + ((bm & ~(UR - 1)) - UR) * bk;
                        cc = C + ((bm & ~(UR - 1)) - UR);
                        do {
                            if (bk - kk > 0) {
                                qblas_egemm_kernel(UR, j, bk - kk, dm1,
                                                   aa + UR * kk,
                                                   b_buf + j * kk, cc, ldc);
                            }
                            solve_LN(UR, j,
                                     aa + (kk - UR) * UR,
                                     b_buf + (kk - UR) * j,
                                     cc, ldc);
                            aa -= UR * bk;
                            cc -= UR;
                            kk -= UR;
                            i--;
                        } while (i > 0);
                    }
                    b_buf += j * bk;
                    C += j * ldc;
                }
                j >>= 1;
            }
        }
    } else if (left && trans) {
        /* trsm_kernel_LT.c — solve walks up rows. */
        ptrdiff_t i, j;
        T *aa, *cc;
        ptrdiff_t kk;

        j = (bn / UN);
        while (j > 0) {
            kk = offset;
            aa = a_buf;
            cc = C;

            i = (bm / UR);
            while (i > 0) {
                if (kk > 0) {
                    qblas_egemm_kernel(UR, UN, kk, dm1, aa, b_buf, cc, ldc);
                }
                solve_LT(UR, UN,
                         aa + kk * UR,
                         b_buf + kk * UN,
                         cc, ldc);
                aa += UR * bk;
                cc += UR;
                kk += UR;
                i--;
            }

            if (bm & (UR - 1)) {
                i = (UR >> 1);
                while (i > 0) {
                    if (bm & i) {
                        if (kk > 0) {
                            qblas_egemm_kernel(i, UN, kk, dm1, aa, b_buf, cc, ldc);
                        }
                        solve_LT(i, UN,
                                 aa + kk * i,
                                 b_buf + kk * UN,
                                 cc, ldc);
                        aa += i * bk;
                        cc += i;
                        kk += i;
                    }
                    i >>= 1;
                }
            }

            b_buf += UN * bk;
            C += UN * ldc;
            j--;
        }

        if (bn & (UN - 1)) {
            j = (UN >> 1);
            while (j > 0) {
                if (bn & j) {
                    kk = offset;
                    aa = a_buf;
                    cc = C;
                    i = (bm / UR);
                    while (i > 0) {
                        if (kk > 0) {
                            qblas_egemm_kernel(UR, j, kk, dm1, aa, b_buf, cc, ldc);
                        }
                        solve_LT(UR, j,
                                 aa + kk * UR,
                                 b_buf + kk * j,
                                 cc, ldc);
                        aa += UR * bk;
                        cc += UR;
                        kk += UR;
                        i--;
                    }
                    if (bm & (UR - 1)) {
                        i = (UR >> 1);
                        while (i > 0) {
                            if (bm & i) {
                                if (kk > 0) {
                                    qblas_egemm_kernel(i, j, kk, dm1, aa, b_buf, cc, ldc);
                                }
                                solve_LT(i, j,
                                         aa + kk * i,
                                         b_buf + kk * j,
                                         cc, ldc);
                                aa += i * bk;
                                cc += i;
                                kk += i;
                            }
                            i >>= 1;
                        }
                    }
                    b_buf += j * bk;
                    C += j * ldc;
                }
                j >>= 1;
            }
        }
    } else if (!left && !trans) {
        /* trsm_kernel_RN.c — solve walks up cols (i = 0 → n-1). */
        ptrdiff_t i, j;
        T *aa, *cc;
        ptrdiff_t kk;

        kk = -offset;
        j = (bn / UN);
        while (j > 0) {
            aa = a_buf;
            cc = C;
            i = (bm / UR);
            if (i > 0) {
                do {
                    if (kk > 0) {
                        qblas_egemm_kernel(UR, UN, kk, dm1, aa, b_buf, cc, ldc);
                    }
                    solve_RN(UR, UN,
                             aa + kk * UR,
                             b_buf + kk * UN,
                             cc, ldc);
                    aa += UR * bk;
                    cc += UR;
                    i--;
                } while (i > 0);
            }

            if (bm & (UR - 1)) {
                i = (UR >> 1);
                while (i > 0) {
                    if (bm & i) {
                        if (kk > 0) {
                            qblas_egemm_kernel(i, UN, kk, dm1, aa, b_buf, cc, ldc);
                        }
                        solve_RN(i, UN,
                                 aa + kk * i,
                                 b_buf + kk * UN,
                                 cc, ldc);
                        aa += i * bk;
                        cc += i;
                    }
                    i >>= 1;
                }
            }

            kk += UN;
            b_buf += UN * bk;
            C += UN * ldc;
            j--;
        }

        if (bn & (UN - 1)) {
            j = (UN >> 1);
            while (j > 0) {
                if (bn & j) {
                    aa = a_buf;
                    cc = C;
                    i = (bm / UR);
                    while (i > 0) {
                        if (kk > 0) {
                            qblas_egemm_kernel(UR, j, kk, dm1, aa, b_buf, cc, ldc);
                        }
                        solve_RN(UR, j,
                                 aa + kk * UR,
                                 b_buf + kk * j,
                                 cc, ldc);
                        aa += UR * bk;
                        cc += UR;
                        i--;
                    }
                    if (bm & (UR - 1)) {
                        i = (UR >> 1);
                        while (i > 0) {
                            if (bm & i) {
                                if (kk > 0) {
                                    qblas_egemm_kernel(i, j, kk, dm1, aa, b_buf, cc, ldc);
                                }
                                solve_RN(i, j,
                                         aa + kk * i,
                                         b_buf + kk * j,
                                         cc, ldc);
                                aa += i * bk;
                                cc += i;
                            }
                            i >>= 1;
                        }
                    }
                    b_buf += j * bk;
                    C += j * ldc;
                    kk += j;
                }
                j >>= 1;
            }
        }
    } else {
        /* trsm_kernel_RT.c — solve walks down cols (i = n-1 → 0). */
        ptrdiff_t i, j;
        T *aa, *cc;
        ptrdiff_t kk;

        kk = bn - offset;
        C += bn * ldc;
        b_buf += bn * bk;

        if (bn & (UN - 1)) {
            j = 1;
            while (j < UN) {
                if (bn & j) {
                    aa = a_buf;
                    b_buf -= j * bk;
                    C -= j * ldc;
                    cc = C;

                    i = (bm / UR);
                    if (i > 0) {
                        do {
                            if (bk - kk > 0) {
                                qblas_egemm_kernel(UR, j, bk - kk, dm1,
                                                   aa + UR * kk,
                                                   b_buf + j * kk, cc, ldc);
                            }
                            solve_RT(UR, j,
                                     aa + (kk - j) * UR,
                                     b_buf + (kk - j) * j,
                                     cc, ldc);
                            aa += UR * bk;
                            cc += UR;
                            i--;
                        } while (i > 0);
                    }
                    if (bm & (UR - 1)) {
                        i = (UR >> 1);
                        do {
                            if (bm & i) {
                                if (bk - kk > 0) {
                                    qblas_egemm_kernel(i, j, bk - kk, dm1,
                                                       aa + i * kk,
                                                       b_buf + j * kk, cc, ldc);
                                }
                                solve_RT(i, j,
                                         aa + (kk - j) * i,
                                         b_buf + (kk - j) * j,
                                         cc, ldc);
                                aa += i * bk;
                                cc += i;
                            }
                            i >>= 1;
                        } while (i > 0);
                    }
                    kk -= j;
                }
                j <<= 1;
            }
        }

        j = (bn / UN);
        if (j > 0) {
            do {
                aa = a_buf;
                b_buf -= UN * bk;
                C -= UN * ldc;
                cc = C;

                i = (bm / UR);
                if (i > 0) {
                    do {
                        if (bk - kk > 0) {
                            qblas_egemm_kernel(UR, UN, bk - kk, dm1,
                                               aa + UR * kk,
                                               b_buf + UN * kk, cc, ldc);
                        }
                        solve_RT(UR, UN,
                                 aa + (kk - UN) * UR,
                                 b_buf + (kk - UN) * UN,
                                 cc, ldc);
                        aa += UR * bk;
                        cc += UR;
                        i--;
                    } while (i > 0);
                }
                if (bm & (UR - 1)) {
                    i = (UR >> 1);
                    do {
                        if (bm & i) {
                            if (bk - kk > 0) {
                                qblas_egemm_kernel(i, UN, bk - kk, dm1,
                                                   aa + i * kk,
                                                   b_buf + UN * kk, cc, ldc);
                            }
                            solve_RT(i, UN,
                                     aa + (kk - UN) * i,
                                     b_buf + (kk - UN) * UN,
                                     cc, ldc);
                            aa += i * bk;
                            cc += i;
                        }
                        i >>= 1;
                    } while (i > 0);
                }
                kk -= UN;
                j--;
            } while (j > 0);
        }
    }
}
