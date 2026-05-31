/*
 * etrsm_kernel — kind10 (REAL(KIND=10) / long double) L3 substrate for the
 * etrsm overlay: a private MR=NR=2 GEMM micro-kernel + its ncopy/tcopy
 * packers, the four diagonal solve directions, and the diagonal-aware TRSM
 * micro-kernel. Faithful ports of OpenBLAS kernel/generic/gemm_*_2.c and
 * trsm_kernel_{LN,LT,RN,RT}.c (the same code the openblas overlay carries
 * in eblas_l3_real.c).
 *
 * These are NOT par's egemm primitives: the TRSM solve and its trailing
 * GEMM share one packed diagonal-block buffer at MR/NR granularity, and
 * OpenBLAS's contiguous-odd-tail packing convention is baked into the
 * packer ↔ solve ↔ kernel triad. par's egemm zero-pads odd tails at
 * stride MR, so its kernel reads those bytes differently — hence this
 * self-consistent ob-convention copy. Layout-AGNOSTIC block-size policy
 * is still shared (the band drivers call egemm_choose_blocks). See
 * etrsm_kernel.h for the full rationale.
 *
 *   - etrsm_gemm_kernel: C += alpha·Ap·Bp over one packed (bm,bn,bk) tile.
 *   - etrsm_{n,t}copy:   pack a plain (non-triangular) A/B slab.
 *   - solve_{LN,LT,RN,RT}: in-pack triangular solve of one register tile.
 *   - etrsm_solve_kernel: dispatch over (left, trans); interleaves
 *     trailing GEMM (alpha = dm1 = -1) with the per-tile solve.
 */

#include <stddef.h>

#include "etrsm_kernel.h"

typedef etrsm_T T;

/* Register-tile dims — must match par's egemm (the packed layout dims). */
#define MR 2
#define NR 2

void etrsm_gemm_kernel(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
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

void etrsm_ncopy(ptrdiff_t m, ptrdiff_t n,
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

void etrsm_tcopy(ptrdiff_t m, ptrdiff_t n,
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

void etrsm_solve_kernel(int left, int trans,
                        ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        const T *ba, const T *bb,
                        T *C, ptrdiff_t ldc,
                        ptrdiff_t offset)
{
    const T dm1 = -1.0L;
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
                            etrsm_gemm_kernel(i, UN, bk - kk, dm1,
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
                        etrsm_gemm_kernel(UR, UN, bk - kk, dm1,
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
                                    etrsm_gemm_kernel(i, j, bk - kk, dm1,
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
                                etrsm_gemm_kernel(UR, j, bk - kk, dm1,
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
                    etrsm_gemm_kernel(UR, UN, kk, dm1, aa, b_buf, cc, ldc);
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
                            etrsm_gemm_kernel(i, UN, kk, dm1, aa, b_buf, cc, ldc);
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
                            etrsm_gemm_kernel(UR, j, kk, dm1, aa, b_buf, cc, ldc);
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
                                    etrsm_gemm_kernel(i, j, kk, dm1, aa, b_buf, cc, ldc);
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
                        etrsm_gemm_kernel(UR, UN, kk, dm1, aa, b_buf, cc, ldc);
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
                            etrsm_gemm_kernel(i, UN, kk, dm1, aa, b_buf, cc, ldc);
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
                            etrsm_gemm_kernel(UR, j, kk, dm1, aa, b_buf, cc, ldc);
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
                                    etrsm_gemm_kernel(i, j, kk, dm1, aa, b_buf, cc, ldc);
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
                                etrsm_gemm_kernel(UR, j, bk - kk, dm1,
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
                                    etrsm_gemm_kernel(i, j, bk - kk, dm1,
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
                            etrsm_gemm_kernel(UR, UN, bk - kk, dm1,
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
                                etrsm_gemm_kernel(i, UN, bk - kk, dm1,
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
