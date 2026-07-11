/*
 * mtpsv — kind10 port of OpenBLAS dtpsv.  Packed triangular solve.
 *
 *   x := inv(op(A)) * x   where A is N x N triangular, packed.
 *
 * Packed layout: UPLO='U': column j at ap[j*(j+1)/2 .. j*(j+1)/2 + j]
 *                UPLO='L': column j at ap[j*(2n-j-1)/2 .. j*(2n-j-1)/2 + (n-1)]
 *
 * Loop directions and KK walk match Fortran reference mtpsv.f.
 *
 * Fortran ABI:  subroutine mtpsv(uplo, trans, diag, n, ap, x, incx)
 */

#include <stddef.h>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#include <ctype.h>

typedef multifloats::float64x2 T;

extern "C" void mtpsv_(const char *UPLO, const char *TRANS, const char *DIAG,
            const int *N, const T *ap,
            T *x, const int *INCX)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
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
            /* kk = index of diagonal element of column j  -> 0-based: kk = j + j*(j+1)/2 */
            ptrdiff_t kk = n * (n + 1) / 2 - 1;  /* diagonal of column n-1 */
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != 0.0) {
                        if (nounit) x[j] /= ap[kk];
                        T temp = x[j];
                        ptrdiff_t k = kk - 1;
                        for (ptrdiff_t i = j - 1; i >= 0; --i) {
                            x[i] -= temp * ap[k];
                            --k;
                        }
                    }
                    kk -= (j + 1);  /* move to diagonal of column j-1 */
                }
            } else {
                ptrdiff_t jx = kx + (n - 1) * incx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[jx] != 0.0) {
                        if (nounit) x[jx] /= ap[kk];
                        T temp = x[jx];
                        ptrdiff_t ix = jx;
                        ptrdiff_t k = kk - 1;
                        for (ptrdiff_t i = j - 1; i >= 0; --i) {
                            ix -= incx;
                            x[ix] -= temp * ap[k];
                            --k;
                        }
                    }
                    jx -= incx;
                    kk -= (j + 1);
                }
            }
        } else {
            /* kk = diagonal index of column 0 = 0 in 0-based */
            ptrdiff_t kk = 0;
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != 0.0) {
                        if (nounit) x[j] /= ap[kk];
                        T temp = x[j];
                        ptrdiff_t k = kk + 1;
                        for (ptrdiff_t i = j + 1; i < n; ++i) {
                            x[i] -= temp * ap[k];
                            ++k;
                        }
                    }
                    kk += (n - j);
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[jx] != 0.0) {
                        if (nounit) x[jx] /= ap[kk];
                        T temp = x[jx];
                        ptrdiff_t ix = jx;
                        ptrdiff_t k = kk + 1;
                        for (ptrdiff_t i = j + 1; i < n; ++i) {
                            ix += incx;
                            x[ix] -= temp * ap[k];
                            ++k;
                        }
                    }
                    jx += incx;
                    kk += (n - j);
                }
            }
        }
    } else {
        if (upper) {
            ptrdiff_t kk = 0;
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    T temp = x[j];
                    ptrdiff_t k = kk;
                    for (ptrdiff_t i = 0; i < j; ++i) {
                        temp -= ap[k] * x[i];
                        ++k;
                    }
                    if (nounit) temp /= ap[kk + j];
                    x[j] = temp;
                    kk += (j + 1);
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    T temp = x[jx];
                    ptrdiff_t ix = kx;
                    ptrdiff_t k = kk;
                    for (ptrdiff_t i = 0; i < j; ++i) {
                        temp -= ap[k] * x[ix];
                        ix += incx;
                        ++k;
                    }
                    if (nounit) temp /= ap[kk + j];
                    x[jx] = temp;
                    jx += incx;
                    kk += (j + 1);
                }
            }
        } else {
            ptrdiff_t kk = n * (n + 1) / 2 - 1;
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    T temp = x[j];
                    ptrdiff_t k = kk;
                    for (ptrdiff_t i = n - 1; i > j; --i) {
                        temp -= ap[k] * x[i];
                        --k;
                    }
                    if (nounit) temp /= ap[kk - (n - 1 - j)];
                    x[j] = temp;
                    kk -= (n - j);
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    T temp = x[jx];
                    ptrdiff_t ix = kx;
                    ptrdiff_t k = kk;
                    for (ptrdiff_t i = n - 1; i > j; --i) {
                        temp -= ap[k] * x[ix];
                        ix -= incx;
                        --k;
                    }
                    if (nounit) temp /= ap[kk - (n - 1 - j)];
                    x[jx] = temp;
                    jx -= incx;
                    kk -= (n - j);
                }
            }
        }
    }
}
