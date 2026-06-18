/* wtbsv — multifloats complex DD triangular band solve. */

#include <cstddef>
#include <cctype>
#include <vector>
#include <multifloats.h>
#include "mf_tri_simd.h"

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
inline bool dd_iszero(const R &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool cdd_iszero(const T &x) { return dd_iszero(x.re) && dd_iszero(x.im); }
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T csub(T const &a, T const &b) { return T{ a.re - b.re, a.im - b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }
inline T cdiv(T const &a, T const &b) {
    const R d = b.re * b.re + b.im * b.im;
    const R inv_d = R{1.0, 0.0} / d;
    return T{ (a.re * b.re + a.im * b.im) * inv_d,
              (a.im * b.re - a.re * b.im) * inv_d };
}
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

/* Contiguous (incx==1) serial core. NoTrans band elimination is a column
 * AXPY-sub (caxpy_sub, bit-exact); Trans/ConjTrans is a banded-column complex
 * dot (cdot, reorders -> within fuzz tol). The cross-column recurrence stays
 * scalar. Strided callers gather x to a contiguous scratch around this. */
static void wtbsv_serial_contig(char UPLO, char TR, int noconj, int nounit,
                                int N, int K, const T *a, std::size_t lda, T *x)
{
    const bool conj = (noconj == 0);
    if (TR == 'N') {
        if (UPLO == 'U') {
            for (int j = N - 1; j >= 0; --j) {
                if (!cdd_iszero(x[j])) {
                    const int L = K - j;
                    if (nounit) x[j] = cdiv(x[j], A_(K, j));
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    mf_tri::caxpy_sub(j - i_lo, &x[i_lo], &A_(L + i_lo, j), x[j]);
                }
            }
        } else {
            for (int j = 0; j < N; ++j) {
                if (!cdd_iszero(x[j])) {
                    if (nounit) x[j] = cdiv(x[j], A_(0, j));
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    mf_tri::caxpy_sub(i_hi - (j + 1), &x[j + 1], &A_(1, j), x[j]);
                }
            }
        }
    } else {
        if (UPLO == 'U') {
            for (int j = 0; j < N; ++j) {
                const int L = K - j;
                const int i_lo = (j - K > 0) ? (j - K) : 0;
                T tmp = csub(x[j], mf_tri::cdot(j - i_lo, &A_(L + i_lo, j), &x[i_lo], conj));
                if (nounit) tmp = cdiv(tmp, (noconj ? A_(K, j) : cconj(A_(K, j))));
                x[j] = tmp;
            }
        } else {
            for (int j = N - 1; j >= 0; --j) {
                const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                T tmp = csub(x[j], mf_tri::cdot(i_hi - (j + 1), &A_(1, j), &x[j + 1], conj));
                if (nounit) tmp = cdiv(tmp, (noconj ? A_(0, j) : cconj(A_(0, j))));
                x[j] = tmp;
            }
        }
    }
}

extern "C" void wtbsv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_, const int *k_,
    const T *a, const int *lda_,
    T *x, const int *incx_,
    std::size_t uplo_len, std::size_t trans_len, std::size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, incx = *incx_;
    const char UPLO = up(uplo);
    const char TR = up(trans);
    const int noconj = (TR == 'T');
    const int nounit = (up(diag) != 'U');

    if (N == 0) return;

    if (incx == 1) {
        wtbsv_serial_contig(UPLO, TR, noconj, nounit, N, K, a, lda, x);
        return;
    }

    /* Strided: gather x to a contiguous scratch, run the SIMD core, scatter. */
    T *xbase = (incx < 0) ? x - (std::ptrdiff_t)(N - 1) * incx : x;
    std::vector<T> xs(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) xs[i] = xbase[(std::ptrdiff_t)i * incx];
    wtbsv_serial_contig(UPLO, TR, noconj, nounit, N, K, a, lda, xs.data());
    for (int i = 0; i < N; ++i) xbase[(std::ptrdiff_t)i * incx] = xs[i];
}

#undef A_
