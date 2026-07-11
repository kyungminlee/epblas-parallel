/*
 * ytrsv — kind10 port of OpenBLAS ztrsv.  Complex triangular solve.
 *
 *   x := inv(op(A)) * x   where op(A) = A, A^T, or A^H
 *
 * Loop directions match Fortran reference ytrsv.f.
 *
 * Fortran ABI:  subroutine ytrsv(uplo, trans, diag, n, a, lda, x, incx)
 */

#include <stddef.h>
#include <complex.h>
#include <ctype.h>

typedef _Complex long double C;

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

void ytrsv_(const char *UPLO, const char *TRANS, const char *DIAG,
            const int *N, const C *a, const int *LDA,
            C *x, const int *INCX)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
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

    if (!trans) {
        if (upper) {
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != 0.0L) {
                        const C *aj = &A_(0, j);
                        if (nounit) x[j] /= aj[j];
                        C temp = x[j];
                        for (ptrdiff_t i = j - 1; i >= 0; --i) x[i] -= temp * aj[i];
                    }
                }
            } else {
                ptrdiff_t jx = kx + (n - 1) * incx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[jx] != 0.0L) {
                        const C *aj = &A_(0, j);
                        if (nounit) x[jx] /= aj[j];
                        C temp = x[jx];
                        ptrdiff_t ix = jx;
                        for (ptrdiff_t i = j - 1; i >= 0; --i) { ix -= incx; x[ix] -= temp * aj[i]; }
                    }
                    jx -= incx;
                }
            }
        } else {
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != 0.0L) {
                        const C *aj = &A_(0, j);
                        if (nounit) x[j] /= aj[j];
                        C temp = x[j];
                        for (ptrdiff_t i = j + 1; i < n; ++i) x[i] -= temp * aj[i];
                    }
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[jx] != 0.0L) {
                        const C *aj = &A_(0, j);
                        if (nounit) x[jx] /= aj[j];
                        C temp = x[jx];
                        ptrdiff_t ix = jx;
                        for (ptrdiff_t i = j + 1; i < n; ++i) { ix += incx; x[ix] -= temp * aj[i]; }
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
                    const C *aj = &A_(0, j);
                    if (noconj) {
                        for (ptrdiff_t i = 0; i < j; ++i) temp -= aj[i] * x[i];
                        if (nounit) temp /= aj[j];
                    } else {
                        for (ptrdiff_t i = 0; i < j; ++i) temp -= conjl(aj[i]) * x[i];
                        if (nounit) temp /= conjl(aj[j]);
                    }
                    x[j] = temp;
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    C temp = x[jx];
                    ptrdiff_t ix = kx;
                    const C *aj = &A_(0, j);
                    if (noconj) {
                        for (ptrdiff_t i = 0; i < j; ++i) { temp -= aj[i] * x[ix]; ix += incx; }
                        if (nounit) temp /= aj[j];
                    } else {
                        for (ptrdiff_t i = 0; i < j; ++i) { temp -= conjl(aj[i]) * x[ix]; ix += incx; }
                        if (nounit) temp /= conjl(aj[j]);
                    }
                    x[jx] = temp;
                    jx += incx;
                }
            }
        } else {
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    C temp = x[j];
                    const C *aj = &A_(0, j);
                    if (noconj) {
                        for (ptrdiff_t i = n - 1; i > j; --i) temp -= aj[i] * x[i];
                        if (nounit) temp /= aj[j];
                    } else {
                        for (ptrdiff_t i = n - 1; i > j; --i) temp -= conjl(aj[i]) * x[i];
                        if (nounit) temp /= conjl(aj[j]);
                    }
                    x[j] = temp;
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    C temp = x[jx];
                    ptrdiff_t ix = kx;
                    const C *aj = &A_(0, j);
                    if (noconj) {
                        for (ptrdiff_t i = n - 1; i > j; --i) { temp -= aj[i] * x[ix]; ix -= incx; }
                        if (nounit) temp /= aj[j];
                    } else {
                        for (ptrdiff_t i = n - 1; i > j; --i) { temp -= conjl(aj[i]) * x[ix]; ix -= incx; }
                        if (nounit) temp /= conjl(aj[j]);
                    }
                    x[jx] = temp;
                    jx -= incx;
                }
            }
        }
    }
}

#undef A_
