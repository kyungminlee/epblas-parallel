/* msbmv — multifloats real DD symmetric band matrix-vector multiply.
 *   y := alpha*A*x + beta*y, A symmetric with K super-(or sub-)diagonals.
 *
 * Serial: each j-iteration touches both halves of A and updates y[j]
 * plus y[i] in two disjoint stripes, so column-parallel writes overlap.
 */

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
const T one_dd{1.0, 0.0};
inline bool dd_iszero(const T &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_isone (const T &x) { return x.limbs[0] == 1.0 && x.limbs[1] == 0.0; }
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

extern "C" void msbmv_(
    const char *uplo,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *x, const int *incx_,
    const T *beta_,
    T *y, const int *incy_,
    std::size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);

    if (N == 0 || (dd_iszero(alpha) && dd_isone(beta))) return;

    if (!dd_isone(beta)) {
        int iy = (incy < 0) ? -(N - 1) * incy : 0;
        if (dd_iszero(beta)) {
            for (int i = 0; i < N; ++i) { y[iy] = zero_dd; iy += incy; }
        } else {
            for (int i = 0; i < N; ++i) { y[iy] = beta * y[iy]; iy += incy; }
        }
    }
    if (dd_iszero(alpha)) return;

    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero_dd;
                const int L = K - j;
                const int i_lo = (j - K > 0) ? (j - K) : 0;
                for (int i = i_lo; i < j; ++i) {
                    y[i] = y[i] + t1 * A_(L + i, j);
                    t2 = t2 + A_(L + i, j) * x[i];
                }
                y[j] = y[j] + t1 * A_(K, j) + alpha * t2;
            }
        } else {
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero_dd;
                y[j] = y[j] + t1 * A_(0, j);
                const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                for (int i = j + 1; i < i_hi; ++i) {
                    y[i] = y[i] + t1 * A_(i - j, j);
                    t2 = t2 + A_(i - j, j) * x[i];
                }
                y[j] = y[j] + alpha * t2;
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        if (UPLO == 'U') {
            int jx = kx, jy = ky;
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero_dd;
                int ix = kx, iy = ky;
                const int L = K - j;
                const int i_lo = (j - K > 0) ? (j - K) : 0;
                for (int i = i_lo; i < j; ++i) {
                    y[iy] = y[iy] + t1 * A_(L + i, j);
                    t2 = t2 + A_(L + i, j) * x[ix];
                    ix += incx; iy += incy;
                }
                y[jy] = y[jy] + t1 * A_(K, j) + alpha * t2;
                jx += incx; jy += incy;
                if (j >= K) { kx += incx; ky += incy; }
            }
        } else {
            int jx = kx, jy = ky;
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero_dd;
                y[jy] = y[jy] + t1 * A_(0, j);
                int ix = jx, iy = jy;
                const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                for (int i = j + 1; i < i_hi; ++i) {
                    ix += incx; iy += incy;
                    y[iy] = y[iy] + t1 * A_(i - j, j);
                    t2 = t2 + A_(i - j, j) * x[ix];
                }
                y[jy] = y[jy] + alpha * t2;
                jx += incx; jy += incy;
            }
        }
    }
}

#undef A_
