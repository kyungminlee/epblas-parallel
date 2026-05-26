/* mtrmv — multifloats real DD triangular matrix-vector. */

#include <cstddef>
#include <cctype>
#include <multifloats.h>

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
const T zero_dd{0.0, 0.0};
inline bool dd_iszero(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

extern "C" void mtrmv_(
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
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
    const char DIAG = up(diag);
    const bool nounit = (DIAG != 'U');

    if (N == 0) return;

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (int j = N - 1; j >= 0; --j) {
                    const T temp = x[j];
                    if (!dd_iszero(temp)) {
                        const T *aj = &A_(0, j);
                        for (int i = j + 1; i < N; ++i) x[i] = x[i] + temp * aj[i];
                    }
                    if (nounit) x[j] = x[j] * A_(j, j);
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    const T temp = x[j];
                    if (!dd_iszero(temp)) {
                        const T *aj = &A_(0, j);
                        for (int i = 0; i < j; ++i) x[i] = x[i] + temp * aj[i];
                    }
                    if (nounit) x[j] = x[j] * A_(j, j);
                }
            }
        } else {
            if (UPLO == 'L') {
                for (int j = 0; j < N; ++j) {
                    T temp = x[j];
                    if (nounit) temp = temp * A_(j, j);
                    const T *aj = &A_(0, j);
                    for (int i = j + 1; i < N; ++i) temp = temp + aj[i] * x[i];
                    x[j] = temp;
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    T temp = x[j];
                    if (nounit) temp = temp * A_(j, j);
                    const T *aj = &A_(0, j);
                    for (int i = 0; i < j; ++i) temp = temp + aj[i] * x[i];
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
                    if (!dd_iszero(temp))
                        for (int i = j + 1; i < N; ++i) x[kx + i * incx] = x[kx + i * incx] + temp * A_(i, j);
                    if (nounit) x[kx + j * incx] = x[kx + j * incx] * A_(j, j);
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    const T temp = x[kx + j * incx];
                    if (!dd_iszero(temp))
                        for (int i = 0; i < j; ++i) x[kx + i * incx] = x[kx + i * incx] + temp * A_(i, j);
                    if (nounit) x[kx + j * incx] = x[kx + j * incx] * A_(j, j);
                }
            }
        } else {
            if (UPLO == 'L') {
                for (int j = 0; j < N; ++j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp = temp * A_(j, j);
                    for (int i = j + 1; i < N; ++i) temp = temp + A_(i, j) * x[kx + i * incx];
                    x[kx + j * incx] = temp;
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp = temp * A_(j, j);
                    for (int i = 0; i < j; ++i) temp = temp + A_(i, j) * x[kx + i * incx];
                    x[kx + j * incx] = temp;
                }
            }
        }
    }
}

#undef A_
