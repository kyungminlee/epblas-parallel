/* wtbmv — multifloats complex DD triangular band matrix-vector. */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <cstdlib>
#include <omp.h>
#include "../common/blas_omp.h"
#define WTBMV_OMP_MIN 256
#define WTBMV_MAX_CPUS 256
#endif

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

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

#ifdef _OPENMP
/* Row-gather: x := A*x / A^T*x / A^H*x, in-place triangular band. Each output row
 * r is an independent dot once original x is copied off. NoTrans reads matrix row
 * r (anti-diagonal stride lda-1, elements direct); Trans/ConjTrans reads column r
 * (contiguous), conjugating elements when conj. Diagonal scales x[r] when
 * non-unit (conjugated when conj). xin is the contiguous copy, xout the
 * destination. Mirrors the serial accumulation order → within DD fuzz tol; the
 * serial path stays bit-exact. */
static void wtbmv_rowgather(bool upper, bool trans, bool conj, bool nounit,
                            int n, int k, int lo, int hi,
                            const T *a, std::size_t lda,
                            const T *xin, T *xout, int incx)
{
    const std::ptrdiff_t s1 = static_cast<std::ptrdiff_t>(lda) - 1;
    for (int r = lo; r < hi; ++r) {
        const T *base = &A_(0, r);
        T diagc = upper ? base[k] : base[0];
        if (conj) diagc = cconj(diagc);
        T s = nounit ? cmul(diagc, xin[r]) : xin[r];
        if (!trans) {
            if (upper) {
                const int rlen = (n - 1 - r < k) ? (n - 1 - r) : k;
                for (int d = 1; d <= rlen; ++d)
                    s = cadd(s, cmul(base[k + (std::ptrdiff_t)d * s1], xin[r + d]));
            } else {
                const int llen = (r < k) ? r : k;
                for (int d = 1; d <= llen; ++d)
                    s = cadd(s, cmul(base[-(std::ptrdiff_t)d * s1], xin[r - d]));
            }
        } else {
            if (upper) {
                const int llen = (r < k) ? r : k;
                for (int d = 1; d <= llen; ++d) {
                    T e = base[k - d]; if (conj) e = cconj(e);
                    s = cadd(s, cmul(e, xin[r - d]));
                }
            } else {
                const int rlen = (n - 1 - r < k) ? (n - 1 - r) : k;
                for (int d = 1; d <= rlen; ++d) {
                    T e = base[d]; if (conj) e = cconj(e);
                    s = cadd(s, cmul(e, xin[r + d]));
                }
            }
        }
        xout[(std::ptrdiff_t)r * incx] = s;
    }
}

/* Threaded in-place complex triangular band matvec. Copy x to contiguous,
 * partition the output rows, write back. Returns true if handled. */
__attribute__((noinline)) static bool wtbmv_omp(
    bool upper, bool trans, bool conj, bool nounit, int n, int k,
    const T *a, std::size_t lda, T *x, int incx)
{
    if (n < WTBMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > WTBMV_MAX_CPUS) nthreads = WTBMV_MAX_CPUS;

    T *xbase = (incx < 0) ? x - (std::ptrdiff_t)(n - 1) * incx : x;
    T *xbuf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
    if (!xbuf) return false;
    for (int i = 0; i < n; ++i) xbuf[i] = xbase[(std::ptrdiff_t)i * incx];

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int lo = (int)((long long)n * tid / nthreads);
        int hi = (int)((long long)n * (tid + 1) / nthreads);
        wtbmv_rowgather(upper, trans, conj, nounit, n, k, lo, hi, a, lda, xbuf, xbase, incx);
    }
    std::free(xbuf);
    return true;
}
#endif

extern "C" void wtbmv_(
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

#ifdef _OPENMP
    if (N >= WTBMV_OMP_MIN && blas_omp_max_threads() > 1
        && wtbmv_omp(UPLO == 'U', TR != 'N', TR == 'C', nounit != 0, N, K, a, lda, x, incx))
        return;
#endif

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'U') {
                for (int j = 0; j < N; ++j) {
                    if (!cdd_iszero(x[j])) {
                        const T tmp = x[j];
                        const int L = K - j;
                        const int i_lo = (j - K > 0) ? (j - K) : 0;
                        for (int i = i_lo; i < j; ++i) x[i] = cadd(x[i], cmul(tmp, A_(L + i, j)));
                        if (nounit) x[j] = cmul(x[j], A_(K, j));
                    }
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    if (!cdd_iszero(x[j])) {
                        const T tmp = x[j];
                        const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                        for (int i = i_hi - 1; i > j; --i) x[i] = cadd(x[i], cmul(tmp, A_(i - j, j)));
                        if (nounit) x[j] = cmul(x[j], A_(0, j));
                    }
                }
            }
        } else {
            if (UPLO == 'U') {
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    const int L = K - j;
                    if (nounit) tmp = cmul(tmp, (noconj ? A_(K, j) : cconj(A_(K, j))));
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    if (noconj) for (int i = j - 1; i >= i_lo; --i) tmp = cadd(tmp, cmul(A_(L + i, j), x[i]));
                    else        for (int i = j - 1; i >= i_lo; --i) tmp = cadd(tmp, cmul(cconj(A_(L + i, j)), x[i]));
                    x[j] = tmp;
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    T tmp = x[j];
                    if (nounit) tmp = cmul(tmp, (noconj ? A_(0, j) : cconj(A_(0, j))));
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    if (noconj) for (int i = j + 1; i < i_hi; ++i) tmp = cadd(tmp, cmul(A_(i - j, j), x[i]));
                    else        for (int i = j + 1; i < i_hi; ++i) tmp = cadd(tmp, cmul(cconj(A_(i - j, j)), x[i]));
                    x[j] = tmp;
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'U') {
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    if (!cdd_iszero(x[jx])) {
                        const T tmp = x[jx];
                        int ix = kx;
                        const int L = K - j;
                        const int i_lo = (j - K > 0) ? (j - K) : 0;
                        for (int i = i_lo; i < j; ++i) {
                            x[ix] = cadd(x[ix], cmul(tmp, A_(L + i, j)));
                            ix += incx;
                        }
                        if (nounit) x[jx] = cmul(x[jx], A_(K, j));
                    }
                    jx += incx;
                    if (j >= K) kx += incx;
                }
            } else {
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    if (!cdd_iszero(x[jx])) {
                        const T tmp = x[jx];
                        int ix = kx;
                        const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                        for (int i = i_hi - 1; i > j; --i) {
                            x[ix] = cadd(x[ix], cmul(tmp, A_(i - j, j)));
                            ix -= incx;
                        }
                        if (nounit) x[jx] = cmul(x[jx], A_(0, j));
                    }
                    jx -= incx;
                    if ((N - 1 - j) >= K) kx -= incx;
                }
            }
        } else {
            if (UPLO == 'U') {
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    kx -= incx;
                    int ix = kx;
                    const int L = K - j;
                    if (nounit) tmp = cmul(tmp, (noconj ? A_(K, j) : cconj(A_(K, j))));
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    for (int i = j - 1; i >= i_lo; --i) {
                        const T aij = noconj ? A_(L + i, j) : cconj(A_(L + i, j));
                        tmp = cadd(tmp, cmul(aij, x[ix]));
                        ix -= incx;
                    }
                    x[jx] = tmp;
                    jx -= incx;
                }
            } else {
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    T tmp = x[jx];
                    kx += incx;
                    int ix = kx;
                    if (nounit) tmp = cmul(tmp, (noconj ? A_(0, j) : cconj(A_(0, j))));
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (int i = j + 1; i < i_hi; ++i) {
                        const T aij = noconj ? A_(i - j, j) : cconj(A_(i - j, j));
                        tmp = cadd(tmp, cmul(aij, x[ix]));
                        ix += incx;
                    }
                    x[jx] = tmp;
                    jx += incx;
                }
            }
        }
    }
}

#undef A_
