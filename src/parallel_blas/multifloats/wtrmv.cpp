/* wtrmv — multifloats complex DD triangular matrix-vector. */

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
const T zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };
inline bool cdd_iszero(const T &x) {
    return x.re.limbs[0] == 0.0 && x.re.limbs[1] == 0.0
        && x.im.limbs[0] == 0.0 && x.im.limbs[1] == 0.0;
}
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

extern "C" void wtrmv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *a, const int *lda_,
    T *x, const int *incx_,
    std::size_t uplo_len, std::size_t trans_len, std::size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const int N = *n_;
    const int lda = *lda_, incx = *incx_;
    const char UPLO = up(uplo);
    const char TR   = up(trans);
    const char DIAG = up(diag);
    const bool nounit = (DIAG != 'U');

    if (N == 0) return;

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (int j = N - 1; j >= 0; --j) {
                    const T temp = x[j];
                    if (!cdd_iszero(temp)) {
                        const T *aj = &A_(0, j);
                        for (int i = j + 1; i < N; ++i) x[i] = cadd(x[i], cmul(temp, aj[i]));
                    }
                    if (nounit) x[j] = cmul(x[j], A_(j, j));
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    const T temp = x[j];
                    if (!cdd_iszero(temp)) {
                        const T *aj = &A_(0, j);
                        for (int i = 0; i < j; ++i) x[i] = cadd(x[i], cmul(temp, aj[i]));
                    }
                    if (nounit) x[j] = cmul(x[j], A_(j, j));
                }
            }
        } else {
            const bool conj_a = (TR == 'C');
            if (UPLO == 'L') {
                for (int j = 0; j < N; ++j) {
                    T temp = x[j];
                    if (nounit) temp = cmul(temp, conj_a ? cconj(A_(j, j)) : A_(j, j));
                    const T *aj = &A_(0, j);
                    if (conj_a) {
                        for (int i = j + 1; i < N; ++i) temp = cadd(temp, cmul(cconj(aj[i]), x[i]));
                    } else {
                        for (int i = j + 1; i < N; ++i) temp = cadd(temp, cmul(aj[i], x[i]));
                    }
                    x[j] = temp;
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    T temp = x[j];
                    if (nounit) temp = cmul(temp, conj_a ? cconj(A_(j, j)) : A_(j, j));
                    const T *aj = &A_(0, j);
                    if (conj_a) {
                        for (int i = 0; i < j; ++i) temp = cadd(temp, cmul(cconj(aj[i]), x[i]));
                    } else {
                        for (int i = 0; i < j; ++i) temp = cadd(temp, cmul(aj[i], x[i]));
                    }
                    x[j] = temp;
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (int j = N - 1; j >= 0; --j) {
                    const T temp = x[kx + j * incx];
                    if (!cdd_iszero(temp))
                        for (int i = j + 1; i < N; ++i) x[kx + i * incx] = cadd(x[kx + i * incx], cmul(temp, A_(i, j)));
                    if (nounit) x[kx + j * incx] = cmul(x[kx + j * incx], A_(j, j));
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    const T temp = x[kx + j * incx];
                    if (!cdd_iszero(temp))
                        for (int i = 0; i < j; ++i) x[kx + i * incx] = cadd(x[kx + i * incx], cmul(temp, A_(i, j)));
                    if (nounit) x[kx + j * incx] = cmul(x[kx + j * incx], A_(j, j));
                }
            }
        } else {
            const bool conj_a = (TR == 'C');
            if (UPLO == 'L') {
                for (int j = 0; j < N; ++j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp = cmul(temp, conj_a ? cconj(A_(j, j)) : A_(j, j));
                    for (int i = j + 1; i < N; ++i) {
                        const T aij = conj_a ? cconj(A_(i, j)) : A_(i, j);
                        temp = cadd(temp, cmul(aij, x[kx + i * incx]));
                    }
                    x[kx + j * incx] = temp;
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp = cmul(temp, conj_a ? cconj(A_(j, j)) : A_(j, j));
                    for (int i = 0; i < j; ++i) {
                        const T aij = conj_a ? cconj(A_(i, j)) : A_(i, j);
                        temp = cadd(temp, cmul(aij, x[kx + i * incx]));
                    }
                    x[kx + j * incx] = temp;
                }
            }
        }
    }
}

#undef A_
