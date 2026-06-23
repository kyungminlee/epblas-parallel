/*
 * etrsm_pack — kind10 (REAL(KIND=10) / long double) diagonal-inverting
 * A-packers for the etrsm overlay's L3 pack-and-conquer path.
 *
 * Faithful ports of OpenBLAS kernel/generic/trsm_{lt,ut,ln,un}copy_2.c
 * (sibling of the symm packers in the openblas overlay's eblas_l3_real.c).
 * Each packer copies an m×n slab of the triangular A into the packed
 * buffer `b` in the SAME 2-row/2-col register-tile layout the etrsm
 * micro-kernel consumes, with two pieces of TRSM-specific work folded in:
 *
 *   - the DIAGONAL register-block (ii == jj along the packed walk) stores
 *     the reciprocal of each diagonal entry (1/A[k,k], or 1.0 for a unit
 *     diagonal) so the kernel's solve() multiplies instead of divides;
 *   - only the relevant triangle is read — the strict-upper (iu*) or
 *     strict-lower (il*) half plus the diagonal block — so the packer
 *     never touches the unreferenced triangle of A.
 *
 * `offset` positions the diagonal within the slab (start_is - ls etc.);
 * `unit` is nonzero for a unit-triangular A. The packed layout (and hence
 * these packers) is shared with the openblas overlay because both micro-
 * kernels are MR=NR=2 ports of the same OpenBLAS generic kernel — see
 * etrsm_serial.c for the substrate that consumes this output.
 *
 * Exported (not static): the SIDE='L'/'R' band drivers in etrsm_serial.c
 * dispatch into these by (uplo, trans, walk-direction).
 */

#include <stddef.h>
#include <stdbool.h>

#include "etrsm_kernel.h"

typedef etrsm_TR TR;

/* ── !UPPER, !TRANS (forward) / OLNCOPY ─────────────────────────────── */
void etrsm_ilncopy(ptrdiff_t m, ptrdiff_t n,
                   const TR *a, ptrdiff_t lda,
                   ptrdiff_t offset, TR *b, bool unit)
{
    ptrdiff_t i, ii, j, jj;
    TR data01 = 0.0L, data02, data03 = 0.0L, data04 = 0.0L;
    const TR *a1, *a2;

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
                b[0] = unit ? 1.0L : (1.0L / data01);
                b[2] = data02;
                b[3] = unit ? 1.0L : (1.0L / data04);
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
                b[0] = unit ? 1.0L : (1.0L / data01);
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
                b[0] = unit ? 1.0L : (1.0L / data01);
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

/* ── !UPPER, TRANS (forward) / OLTCOPY ──────────────────────────────── */
void etrsm_iltcopy(ptrdiff_t m, ptrdiff_t n,
                   const TR *a, ptrdiff_t lda,
                   ptrdiff_t offset, TR *b, bool unit)
{
    ptrdiff_t i, ii, j, jj;
    TR data01 = 0.0L, data02 = 0.0L, data03 = 0.0L, data04 = 0.0L;
    const TR *a1, *a2;

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
                b[0] = unit ? 1.0L : (1.0L / data01);
                b[1] = data02;
                b[3] = unit ? 1.0L : (1.0L / data04);
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
                b[0] = unit ? 1.0L : (1.0L / data01);
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
                b[0] = unit ? 1.0L : (1.0L / data01);
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

/* ── UPPER, TRANS (backward) / IUNCOPY ──────────────────────────────── */
void etrsm_iuncopy(ptrdiff_t m, ptrdiff_t n,
                   const TR *a, ptrdiff_t lda,
                   ptrdiff_t offset, TR *b, bool unit)
{
    ptrdiff_t i, ii, j, jj;
    TR data01 = 0.0L, data02, data03 = 0.0L, data04 = 0.0L;
    const TR *a1, *a2;

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
                b[0] = unit ? 1.0L : (1.0L / data01);
                b[1] = data03;
                b[3] = unit ? 1.0L : (1.0L / data04);
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
                b[0] = unit ? 1.0L : (1.0L / data01);
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
                b[0] = unit ? 1.0L : (1.0L / data01);
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

/* ── UPPER, !TRANS (backward) / IUTCOPY ─────────────────────────────── */
void etrsm_iutcopy(ptrdiff_t m, ptrdiff_t n,
                   const TR *a, ptrdiff_t lda,
                   ptrdiff_t offset, TR *b, bool unit)
{
    ptrdiff_t i, ii, j, jj;
    TR data01 = 0.0L, data02 = 0.0L, data03 = 0.0L, data04 = 0.0L;
    const TR *a1, *a2;

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
                b[0] = unit ? 1.0L : (1.0L / data01);
                b[2] = data03;
                b[3] = unit ? 1.0L : (1.0L / data04);
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
                b[0] = unit ? 1.0L : (1.0L / data01);
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
                b[0] = unit ? 1.0L : (1.0L / data01);
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
