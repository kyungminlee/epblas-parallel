/*
 * etrsm_kernel — kind10 (REAL(KIND=10) / long double) diagonal-solve half of
 * the etrsm overlay: the four diagonal solve directions and the
 * diagonal-aware TRSM micro-kernel. Faithful port of OpenBLAS
 * kernel/generic/trsm_kernel_{LN,LT,RN,RT}.c.
 *
 * The shared ob-convention GEMM substrate (micro-kernel + ncopy/tcopy) lives
 * in etri_kernel.c — the TRSM solve and its trailing GEMM share one packed
 * diagonal-block buffer at MR/NR granularity, so they must use the same
 * contiguous-odd-tail convention; see etri_kernel.h for why that is NOT
 * par's egemm. This TU pairs that substrate with the solve.
 *
 *   - solve_{LN,LT,RN,RT}: in-pack triangular solve of one register tile.
 *   - etrsm_solve_kernel: dispatch over (left, trans); interleaves the
 *     trailing GEMM (etri_gemm_kernel_msub, the alpha = -1 / C -= A·B
 *     negate-specialized kernel) with the per-tile solve.
 */

#include <stddef.h>
#include <stdbool.h>

#include "etrsm_kernel.h"
#include "etri_kernel.h"   /* etri_gemm_kernel substrate */

typedef etrsm_TR TR;

/* Register-tile dims — must match the packed layout dims. */
#define MR 2
#define NR 2

/* Same-TU instantiation of the C -= Ap·Bp trailing-update kernel. The solve
 * below issues it thousands of times per call with tiny (MR,NR,kk) shapes —
 * at that grain the extern etri_gemm_kernel_msub costs real caller-side
 * spill/reload (GCC must assume full caller-saved clobber across the TU
 * boundary) plus the generic body's m/n dispatch. A static local copy gives
 * GCC IPA-RA and bm/bn constprop clones, which is exactly what the openblas
 * overlay gets from keeping its solve and GEMM in one TU (its interior calls
 * compile to eblas_egemm_kernel.constprop.0). noinline mirrors that shape:
 * a real call to a compact specialized kernel, not 16 inlined copies.
 * Bit-identical math — same body, same instantiation as the extern one. */
static __attribute__((noinline)) void
etrsm_gemm_msub(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                const TR *Ap, const TR *Bp,
                TR *C, ptrdiff_t ldc)
{
    etri_gemm_body(bm, bn, bk, -1.0L, 1, Ap, Bp, C, ldc);
}


static inline void solve_LN(ptrdiff_t m, ptrdiff_t n,
                            TR *a, TR *b, TR *c, ptrdiff_t ldc)
{
    TR aa, bb;
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
                            TR *a, TR *b, TR *c, ptrdiff_t ldc)
{
    TR aa, bb;
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
                            TR *a, TR *b, TR *c, ptrdiff_t ldc)
{
    TR aa, bb;
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
                            TR *a, TR *b, TR *c, ptrdiff_t ldc)
{
    TR aa, bb;
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

void etrsm_solve_kernel(bool left, bool trans,
                        ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        const TR *ba, const TR *bb,
                        TR *C, ptrdiff_t ldc,
                        ptrdiff_t offset)
{
    const ptrdiff_t UR = MR;   /* register tile rows */
    const ptrdiff_t UN = NR;   /* register tile cols */

    /* Cast away const for solve() in-place writes into the local pack
     * buffers — `ba` and `bb` are caller-owned per-thread scratch, so
     * mutating them is fine. */
    TR *a_buf = (TR *)ba;
    TR *b_buf = (TR *)bb;

    if (left && !trans) {
        /* trsm_kernel_LN.c — solve walks down rows. */
        ptrdiff_t i, j;
        TR *aa, *cc;
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
                            etrsm_gemm_msub(i, UN, bk - kk,
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
                        etrsm_gemm_msub(UR, UN, bk - kk,
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
                                    etrsm_gemm_msub(i, j, bk - kk,
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
                                etrsm_gemm_msub(UR, j, bk - kk,
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
        TR *aa, *cc;
        ptrdiff_t kk;

        j = (bn / UN);
        while (j > 0) {
            kk = offset;
            aa = a_buf;
            cc = C;

            i = (bm / UR);
            while (i > 0) {
                if (kk > 0) {
                    etrsm_gemm_msub(UR, UN, kk, aa, b_buf, cc, ldc);
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
                            etrsm_gemm_msub(i, UN, kk, aa, b_buf, cc, ldc);
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
                            etrsm_gemm_msub(UR, j, kk, aa, b_buf, cc, ldc);
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
                                    etrsm_gemm_msub(i, j, kk, aa, b_buf, cc, ldc);
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
        TR *aa, *cc;
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
                        etrsm_gemm_msub(UR, UN, kk, aa, b_buf, cc, ldc);
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
                            etrsm_gemm_msub(i, UN, kk, aa, b_buf, cc, ldc);
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
                            etrsm_gemm_msub(UR, j, kk, aa, b_buf, cc, ldc);
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
                                    etrsm_gemm_msub(i, j, kk, aa, b_buf, cc, ldc);
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
        TR *aa, *cc;
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
                                etrsm_gemm_msub(UR, j, bk - kk,
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
                                    etrsm_gemm_msub(i, j, bk - kk,
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
                            etrsm_gemm_msub(UR, UN, bk - kk,
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
                                etrsm_gemm_msub(i, UN, bk - kk,
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
