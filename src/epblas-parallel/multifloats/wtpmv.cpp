/* wtpmv — multifloats complex DD triangular packed matrix-vector. */

#include <cstddef>
#include <cctype>
#include <multifloats.h>

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
const R rzero{0.0, 0.0};
inline bool dd_iszero(const R &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool cdd_iszero(const T &x) { return dd_iszero(x.re) && dd_iszero(x.im); }
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }
}

extern "C" void wtpmv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *ap,
    T *x, const int *incx_,
    std::size_t uplo_len, std::size_t trans_len, std::size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const int N = *n_;
    const int incx = *incx_;
    const char UPLO = up(uplo);
    const char TR = up(trans);
    const int noconj = (TR == 'T');
    const int nounit = (up(diag) != 'U');

    if (N == 0) return;

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'U') {
                int kk = 0;
                for (int j = 0; j < N; ++j) {
                    if (!cdd_iszero(x[j])) {
                        const T tmp = x[j];
                        int k = kk;
                        for (int i = 0; i < j; ++i) { x[i] = cadd(x[i], cmul(tmp, ap[k])); ++k; }
                        if (nounit) x[j] = cmul(x[j], ap[kk + j]);
                    }
                    kk += j + 1;
                }
            } else {
                int kk = (N * (N + 1)) / 2 - 1;
                for (int j = N - 1; j >= 0; --j) {
                    if (!cdd_iszero(x[j])) {
                        const T tmp = x[j];
                        int k = kk;
                        for (int i = N - 1; i > j; --i) { x[i] = cadd(x[i], cmul(tmp, ap[k])); --k; }
                        if (nounit) x[j] = cmul(x[j], ap[kk - (N - 1 - j)]);
                    }
                    kk -= (N - j);
                }
            }
        } else {
            if (UPLO == 'U') {
                int kk = (N * (N + 1)) / 2 - 1;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    if (nounit) tmp = cmul(tmp, (noconj ? ap[kk] : cconj(ap[kk])));
                    int k = kk - 1;
                    if (noconj) for (int i = j - 1; i >= 0; --i) { tmp = cadd(tmp, cmul(ap[k], x[i])); --k; }
                    else        for (int i = j - 1; i >= 0; --i) { tmp = cadd(tmp, cmul(cconj(ap[k]), x[i])); --k; }
                    x[j] = tmp;
                    kk -= j + 1;
                }
            } else {
                int kk = 0;
                for (int j = 0; j < N; ++j) {
                    T tmp = x[j];
                    if (nounit) tmp = cmul(tmp, (noconj ? ap[kk] : cconj(ap[kk])));
                    int k = kk + 1;
                    if (noconj) for (int i = j + 1; i < N; ++i) { tmp = cadd(tmp, cmul(ap[k], x[i])); ++k; }
                    else        for (int i = j + 1; i < N; ++i) { tmp = cadd(tmp, cmul(cconj(ap[k]), x[i])); ++k; }
                    x[j] = tmp;
                    kk += N - j;
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'U') {
                int kk = 0;
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    if (!cdd_iszero(x[jx])) {
                        const T tmp = x[jx];
                        int ix = kx;
                        for (int k = kk; k < kk + j; ++k) {
                            x[ix] = cadd(x[ix], cmul(tmp, ap[k]));
                            ix += incx;
                        }
                        if (nounit) x[jx] = cmul(x[jx], ap[kk + j]);
                    }
                    jx += incx;
                    kk += j + 1;
                }
            } else {
                int kk = (N * (N + 1)) / 2 - 1;
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    if (!cdd_iszero(x[jx])) {
                        const T tmp = x[jx];
                        int ix = kx;
                        for (int k = kk; k > kk - (N - 1 - j); --k) {
                            x[ix] = cadd(x[ix], cmul(tmp, ap[k]));
                            ix -= incx;
                        }
                        if (nounit) x[jx] = cmul(x[jx], ap[kk - (N - 1 - j)]);
                    }
                    jx -= incx;
                    kk -= (N - j);
                }
            }
        } else {
            if (UPLO == 'U') {
                int kk = (N * (N + 1)) / 2 - 1;
                int jx = kx + (N - 1) * incx;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    int ix = jx;
                    if (nounit) tmp = cmul(tmp, (noconj ? ap[kk] : cconj(ap[kk])));
                    for (int k = kk - 1; k >= kk - j; --k) {
                        ix -= incx;
                        tmp = cadd(tmp, cmul((noconj ? ap[k] : cconj(ap[k])), x[ix]));
                    }
                    x[jx] = tmp;
                    jx -= incx;
                    kk -= j + 1;
                }
            } else {
                int kk = 0;
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    T tmp = x[jx];
                    int ix = jx;
                    if (nounit) tmp = cmul(tmp, (noconj ? ap[kk] : cconj(ap[kk])));
                    for (int k = kk + 1; k < kk + N - j; ++k) {
                        ix += incx;
                        tmp = cadd(tmp, cmul((noconj ? ap[k] : cconj(ap[k])), x[ix]));
                    }
                    x[jx] = tmp;
                    jx += incx;
                    kk += N - j;
                }
            }
        }
    }
}
