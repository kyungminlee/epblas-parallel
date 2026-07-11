/*
 * etbsv — kind10 port of OpenBLAS dtbsv.  Banded triangular solve.
 *
 *   x := inv(op(A)) * x   where A is N x N triangular banded with K extra diagonals
 *
 * Band storage matches etbmv. Loop directions match Fortran reference.
 *
 * Fortran ABI:  subroutine etbsv(uplo, trans, diag, n, k, a, lda, x, incx)
 */

#include <stddef.h>
#include <ctype.h>

typedef long double T;

void etbsv_(const char *UPLO, const char *TRANS, const char *DIAG,
            const int *N, const int *K, const T *a, const int *LDA,
            T *x, const int *INCX)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t k    = (ptrdiff_t)(*K);
    ptrdiff_t lda  = (ptrdiff_t)(*LDA);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);

    if (n == 0) return;

    int upper  = (toupper((unsigned char)*UPLO)  == 'U');
    char trc   = (char)toupper((unsigned char)*TRANS);
    int trans  = (trc == 'T' || trc == 'C') ? 1 : 0;
    int nounit = (toupper((unsigned char)*DIAG) == 'N');

    ptrdiff_t kx;
    if (incx <= 0) kx = -(n - 1) * incx;
    else           kx = 0;

    if (!trans) {
        if (upper) {
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != 0.0L) {
                        const T *col = &a[(size_t)j * lda];
                        ptrdiff_t off = k - j;
                        if (nounit) x[j] /= col[k];
                        T temp = x[j];
                        ptrdiff_t i_lo = (j > k) ? j - k : 0;
                        for (ptrdiff_t i = j - 1; i >= i_lo; --i)
                            x[i] -= temp * col[off + i];
                    }
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    kx -= incx;
                    if (x[jx] != 0.0L) {
                        ptrdiff_t ix = kx;
                        const T *col = &a[(size_t)j * lda];
                        ptrdiff_t off = k - j;
                        if (nounit) x[jx] /= col[k];
                        T temp = x[jx];
                        ptrdiff_t i_lo = (j > k) ? j - k : 0;
                        for (ptrdiff_t i = j - 1; i >= i_lo; --i) {
                            x[ix] -= temp * col[off + i];
                            ix -= incx;
                        }
                    }
                    jx -= incx;
                }
            }
        } else {
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != 0.0L) {
                        const T *col = &a[(size_t)j * lda];
                        ptrdiff_t off = -j;
                        if (nounit) x[j] /= col[0];
                        T temp = x[j];
                        ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                        for (ptrdiff_t i = j + 1; i <= i_hi; ++i)
                            x[i] -= temp * col[off + i];
                    }
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    kx += incx;
                    if (x[jx] != 0.0L) {
                        ptrdiff_t ix = kx;
                        const T *col = &a[(size_t)j * lda];
                        ptrdiff_t off = -j;
                        if (nounit) x[jx] /= col[0];
                        T temp = x[jx];
                        ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                        for (ptrdiff_t i = j + 1; i <= i_hi; ++i) {
                            x[ix] -= temp * col[off + i];
                            ix += incx;
                        }
                    }
                    jx += incx;
                }
            }
        }
    } else {
        if (upper) {
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    T temp = x[j];
                    const T *col = &a[(size_t)j * lda];
                    ptrdiff_t off = k - j;
                    ptrdiff_t i_lo = (j > k) ? j - k : 0;
                    for (ptrdiff_t i = i_lo; i < j; ++i)
                        temp -= col[off + i] * x[i];
                    if (nounit) temp /= col[k];
                    x[j] = temp;
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    T temp = x[jx];
                    ptrdiff_t ix = kx;
                    const T *col = &a[(size_t)j * lda];
                    ptrdiff_t off = k - j;
                    ptrdiff_t i_lo = (j > k) ? j - k : 0;
                    for (ptrdiff_t i = i_lo; i < j; ++i) {
                        temp -= col[off + i] * x[ix];
                        ix += incx;
                    }
                    if (nounit) temp /= col[k];
                    x[jx] = temp;
                    jx += incx;
                    if (j >= k) kx += incx;
                }
            }
        } else {
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    T temp = x[j];
                    const T *col = &a[(size_t)j * lda];
                    ptrdiff_t off = -j;
                    ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                    for (ptrdiff_t i = i_hi; i > j; --i)
                        temp -= col[off + i] * x[i];
                    if (nounit) temp /= col[0];
                    x[j] = temp;
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    T temp = x[jx];
                    ptrdiff_t ix = kx;
                    const T *col = &a[(size_t)j * lda];
                    ptrdiff_t off = -j;
                    ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                    for (ptrdiff_t i = i_hi; i > j; --i) {
                        temp -= col[off + i] * x[ix];
                        ix -= incx;
                    }
                    if (nounit) temp /= col[0];
                    x[jx] = temp;
                    jx -= incx;
                    if (n - 1 - j >= k) kx -= incx;
                }
            }
        }
    }
}
