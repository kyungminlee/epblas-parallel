/*
 * xtbsv — kind16 port of OpenBLAS ztbsv.  Complex banded triangular solve.
 *
 *   x := inv(op(A)) * x   (op = A, A^T, or A^H).  Band storage matches qtbsv.
 *
 * Fortran ABI:  subroutine xtbsv(uplo, trans, diag, n, k, a, lda, x, incx)
 */

#include <stddef.h>
#include <quadmath.h>
#include <complex.h>
#include <ctype.h>

typedef __complex128 C;

void xtbsv_(const char *UPLO, const char *TRANS, const char *DIAG,
            const int *N, const int *K, const C *a, const int *LDA,
            C *x, const int *INCX)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t k    = (ptrdiff_t)(*K);
    ptrdiff_t lda  = (ptrdiff_t)(*LDA);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);

    if (n == 0) return;

    int upper  = (toupper((unsigned char)*UPLO)  == 'U');
    char trc   = (char)toupper((unsigned char)*TRANS);
    int trans  = (trc == 'T' || trc == 'C') ? 1 : 0;
    int noconj = (trc == 'T') ? 1 : 0;
    int nounit = (toupper((unsigned char)*DIAG) == 'N');

    ptrdiff_t kx;
    if (incx <= 0) kx = -(n - 1) * incx;
    else           kx = 0;

#define CONJIF(z) (noconj ? (z) : conjq(z))

    if (!trans) {
        if (upper) {
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != 0.0Q) {
                        const C *col = &a[(size_t)j * lda];
                        ptrdiff_t off = k - j;
                        if (nounit) x[j] /= col[k];
                        C temp = x[j];
                        ptrdiff_t i_lo = (j > k) ? j - k : 0;
                        for (ptrdiff_t i = j - 1; i >= i_lo; --i) x[i] -= temp * col[off + i];
                    }
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    kx -= incx;
                    if (x[jx] != 0.0Q) {
                        ptrdiff_t ix = kx;
                        const C *col = &a[(size_t)j * lda];
                        ptrdiff_t off = k - j;
                        if (nounit) x[jx] /= col[k];
                        C temp = x[jx];
                        ptrdiff_t i_lo = (j > k) ? j - k : 0;
                        for (ptrdiff_t i = j - 1; i >= i_lo; --i) { x[ix] -= temp * col[off + i]; ix -= incx; }
                    }
                    jx -= incx;
                }
            }
        } else {
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != 0.0Q) {
                        const C *col = &a[(size_t)j * lda];
                        ptrdiff_t off = -j;
                        if (nounit) x[j] /= col[0];
                        C temp = x[j];
                        ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                        for (ptrdiff_t i = j + 1; i <= i_hi; ++i) x[i] -= temp * col[off + i];
                    }
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    kx += incx;
                    if (x[jx] != 0.0Q) {
                        ptrdiff_t ix = kx;
                        const C *col = &a[(size_t)j * lda];
                        ptrdiff_t off = -j;
                        if (nounit) x[jx] /= col[0];
                        C temp = x[jx];
                        ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                        for (ptrdiff_t i = j + 1; i <= i_hi; ++i) { x[ix] -= temp * col[off + i]; ix += incx; }
                    }
                    jx += incx;
                }
            }
        }
    } else {
        if (upper) {
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    C temp = x[j];
                    const C *col = &a[(size_t)j * lda];
                    ptrdiff_t off = k - j;
                    ptrdiff_t i_lo = (j > k) ? j - k : 0;
                    for (ptrdiff_t i = i_lo; i < j; ++i) temp -= CONJIF(col[off + i]) * x[i];
                    if (nounit) temp /= CONJIF(col[k]);
                    x[j] = temp;
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    C temp = x[jx];
                    ptrdiff_t ix = kx;
                    const C *col = &a[(size_t)j * lda];
                    ptrdiff_t off = k - j;
                    ptrdiff_t i_lo = (j > k) ? j - k : 0;
                    for (ptrdiff_t i = i_lo; i < j; ++i) { temp -= CONJIF(col[off + i]) * x[ix]; ix += incx; }
                    if (nounit) temp /= CONJIF(col[k]);
                    x[jx] = temp;
                    jx += incx;
                    if (j >= k) kx += incx;
                }
            }
        } else {
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    C temp = x[j];
                    const C *col = &a[(size_t)j * lda];
                    ptrdiff_t off = -j;
                    ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                    for (ptrdiff_t i = i_hi; i > j; --i) temp -= CONJIF(col[off + i]) * x[i];
                    if (nounit) temp /= CONJIF(col[0]);
                    x[j] = temp;
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    C temp = x[jx];
                    ptrdiff_t ix = kx;
                    const C *col = &a[(size_t)j * lda];
                    ptrdiff_t off = -j;
                    ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                    for (ptrdiff_t i = i_hi; i > j; --i) { temp -= CONJIF(col[off + i]) * x[ix]; ix -= incx; }
                    if (nounit) temp /= CONJIF(col[0]);
                    x[jx] = temp;
                    jx -= incx;
                    if (n - 1 - j >= k) kx -= incx;
                }
            }
        }
    }
#undef CONJIF
}
