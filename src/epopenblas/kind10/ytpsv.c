/*
 * ytpsv — kind10 port of OpenBLAS ztpsv.  Complex packed triangular solve.
 *
 *   x := inv(op(A)) * x  (op = A, A^T, A^H).  Packed storage matches etpsv.
 *
 * Fortran ABI:  subroutine ytpsv(uplo, trans, diag, n, ap, x, incx)
 */

#include <stddef.h>
#include <complex.h>
#include <ctype.h>

typedef _Complex long double C;

void ytpsv_(const char *UPLO, const char *TRANS, const char *DIAG,
            const int *N, const C *ap,
            C *x, const int *INCX,
            size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    ptrdiff_t n    = (ptrdiff_t)(*N);
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

#define CONJIF(z) (noconj ? (z) : conjl(z))

    if (!trans) {
        if (upper) {
            ptrdiff_t kk = n * (n + 1) / 2 - 1;
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != 0.0L) {
                        if (nounit) x[j] /= ap[kk];
                        C temp = x[j];
                        ptrdiff_t k = kk - 1;
                        for (ptrdiff_t i = j - 1; i >= 0; --i) { x[i] -= temp * ap[k]; --k; }
                    }
                    kk -= (j + 1);
                }
            } else {
                ptrdiff_t jx = kx + (n - 1) * incx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[jx] != 0.0L) {
                        if (nounit) x[jx] /= ap[kk];
                        C temp = x[jx];
                        ptrdiff_t ix = jx;
                        ptrdiff_t k = kk - 1;
                        for (ptrdiff_t i = j - 1; i >= 0; --i) { ix -= incx; x[ix] -= temp * ap[k]; --k; }
                    }
                    jx -= incx;
                    kk -= (j + 1);
                }
            }
        } else {
            ptrdiff_t kk = 0;
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != 0.0L) {
                        if (nounit) x[j] /= ap[kk];
                        C temp = x[j];
                        ptrdiff_t k = kk + 1;
                        for (ptrdiff_t i = j + 1; i < n; ++i) { x[i] -= temp * ap[k]; ++k; }
                    }
                    kk += (n - j);
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[jx] != 0.0L) {
                        if (nounit) x[jx] /= ap[kk];
                        C temp = x[jx];
                        ptrdiff_t ix = jx;
                        ptrdiff_t k = kk + 1;
                        for (ptrdiff_t i = j + 1; i < n; ++i) { ix += incx; x[ix] -= temp * ap[k]; ++k; }
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
                    C temp = x[j];
                    ptrdiff_t k = kk;
                    for (ptrdiff_t i = 0; i < j; ++i) { temp -= CONJIF(ap[k]) * x[i]; ++k; }
                    if (nounit) temp /= CONJIF(ap[kk + j]);
                    x[j] = temp;
                    kk += (j + 1);
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    C temp = x[jx];
                    ptrdiff_t ix = kx;
                    ptrdiff_t k = kk;
                    for (ptrdiff_t i = 0; i < j; ++i) { temp -= CONJIF(ap[k]) * x[ix]; ix += incx; ++k; }
                    if (nounit) temp /= CONJIF(ap[kk + j]);
                    x[jx] = temp;
                    jx += incx;
                    kk += (j + 1);
                }
            }
        } else {
            ptrdiff_t kk = n * (n + 1) / 2 - 1;
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    C temp = x[j];
                    ptrdiff_t k = kk;
                    for (ptrdiff_t i = n - 1; i > j; --i) { temp -= CONJIF(ap[k]) * x[i]; --k; }
                    if (nounit) temp /= CONJIF(ap[kk - (n - 1 - j)]);
                    x[j] = temp;
                    kk -= (n - j);
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    C temp = x[jx];
                    ptrdiff_t ix = kx;
                    ptrdiff_t k = kk;
                    for (ptrdiff_t i = n - 1; i > j; --i) { temp -= CONJIF(ap[k]) * x[ix]; ix -= incx; --k; }
                    if (nounit) temp /= CONJIF(ap[kk - (n - 1 - j)]);
                    x[jx] = temp;
                    jx -= incx;
                    kk -= (n - j);
                }
            }
        }
    }
#undef CONJIF
}
