/*
 * qtrsv — kind16 port of OpenBLAS dtrsv.  Triangular solve.
 *
 *   x := inv(op(A)) * x      where op(A) = A or A^T
 *
 * Loop directions match Fortran reference qtrsv.f line-for-line.
 *
 * Fortran ABI:  subroutine qtrsv(uplo, trans, diag, n, a, lda, x, incx)
 */

#include <stddef.h>
#include <quadmath.h>
#include <ctype.h>

typedef __float128 T;

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

void qtrsv_(const char *UPLO, const char *TRANS, const char *DIAG,
            const int *N, const T *a, const int *LDA,
            T *x, const int *INCX,
            size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    ptrdiff_t n    = (ptrdiff_t)(*N);
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
                    if (x[j] != 0.0Q) {
                        const T *aj = &A_(0, j);
                        if (nounit) x[j] /= aj[j];
                        T temp = x[j];
                        for (ptrdiff_t i = j - 1; i >= 0; --i)
                            x[i] -= temp * aj[i];
                    }
                }
            } else {
                ptrdiff_t jx = kx + (n - 1) * incx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[jx] != 0.0Q) {
                        const T *aj = &A_(0, j);
                        if (nounit) x[jx] /= aj[j];
                        T temp = x[jx];
                        ptrdiff_t ix = jx;
                        for (ptrdiff_t i = j - 1; i >= 0; --i) {
                            ix -= incx;
                            x[ix] -= temp * aj[i];
                        }
                    }
                    jx -= incx;
                }
            }
        } else {
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != 0.0Q) {
                        const T *aj = &A_(0, j);
                        if (nounit) x[j] /= aj[j];
                        T temp = x[j];
                        for (ptrdiff_t i = j + 1; i < n; ++i)
                            x[i] -= temp * aj[i];
                    }
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[jx] != 0.0Q) {
                        const T *aj = &A_(0, j);
                        if (nounit) x[jx] /= aj[j];
                        T temp = x[jx];
                        ptrdiff_t ix = jx;
                        for (ptrdiff_t i = j + 1; i < n; ++i) {
                            ix += incx;
                            x[ix] -= temp * aj[i];
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
                    const T *aj = &A_(0, j);
                    for (ptrdiff_t i = 0; i < j; ++i)
                        temp -= aj[i] * x[i];
                    if (nounit) temp /= aj[j];
                    x[j] = temp;
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    T temp = x[jx];
                    ptrdiff_t ix = kx;
                    const T *aj = &A_(0, j);
                    for (ptrdiff_t i = 0; i < j; ++i) {
                        temp -= aj[i] * x[ix];
                        ix += incx;
                    }
                    if (nounit) temp /= aj[j];
                    x[jx] = temp;
                    jx += incx;
                }
            }
        } else {
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    T temp = x[j];
                    const T *aj = &A_(0, j);
                    for (ptrdiff_t i = n - 1; i > j; --i)
                        temp -= aj[i] * x[i];
                    if (nounit) temp /= aj[j];
                    x[j] = temp;
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    T temp = x[jx];
                    ptrdiff_t ix = kx;
                    const T *aj = &A_(0, j);
                    for (ptrdiff_t i = n - 1; i > j; --i) {
                        temp -= aj[i] * x[ix];
                        ix -= incx;
                    }
                    if (nounit) temp /= aj[j];
                    x[jx] = temp;
                    jx -= incx;
                }
            }
        }
    }
}

#undef A_
