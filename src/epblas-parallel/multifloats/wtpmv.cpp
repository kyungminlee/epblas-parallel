/* wtpmv — multifloats complex DD triangular packed matrix-vector. */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <cstdlib>
#include <omp.h>
#include "../common/blas_omp.h"
#define WTPMV_OMP_MIN 256
#define WTPMV_MAX_CPUS 256
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

#ifdef _OPENMP
/* Row-gather: x := A*x / A^T*x / A^H*x, in-place triangular packed. Each output
 * row r is an independent dot once original x is copied off. For UPLO='U' column
 * j starts at kk_j=j(j+1)/2 (diag at +j); for 'L' at kk_j=j*N-j(j-1)/2 (diag at
 * +0). NoTrans walks a column-jumping run (elements direct); Trans/ConjTrans
 * walks one contiguous run inside column r (conjugated when conj). Diagonal
 * scales x[r] when non-unit. xin is the contiguous copy, xout the destination.
 * Mirrors the serial accumulation order → within DD fuzz tol; serial bit-exact. */
static void wtpmv_rowgather(bool upper, bool trans, bool conj, bool nounit,
                            int n, int lo, int hi,
                            const T *ap, const T *xin, T *xout, int incx)
{
    for (int r = lo; r < hi; ++r) {
        if (upper) {
            const std::size_t kk_r = static_cast<std::size_t>(r) * (r + 1) / 2;
            T diagc = ap[kk_r + r]; if (conj) diagc = cconj(diagc);
            T s = nounit ? cmul(diagc, xin[r]) : xin[r];
            if (!trans) {                                    /* right: col-jump direct */
                std::size_t kk_c = kk_r + (r + 1);
                for (int c = r + 1; c < n; ++c) {
                    s = cadd(s, cmul(ap[kk_c + r], xin[c]));
                    kk_c += static_cast<std::size_t>(c + 1);
                }
            } else {                                         /* left: contiguous in col r */
                for (int c = 0; c < r; ++c) {
                    T e = ap[kk_r + c]; if (conj) e = cconj(e);
                    s = cadd(s, cmul(e, xin[c]));
                }
            }
            xout[(std::ptrdiff_t)r * incx] = s;
        } else {
            const std::size_t kk_r =
                static_cast<std::size_t>(r) * n - static_cast<std::size_t>(r) * (r - 1) / 2;
            T diagc = ap[kk_r]; if (conj) diagc = cconj(diagc);
            T s = nounit ? cmul(diagc, xin[r]) : xin[r];
            if (!trans) {                                    /* left: col-jump direct */
                std::size_t kk_c = 0;
                for (int c = 0; c < r; ++c) {
                    s = cadd(s, cmul(ap[kk_c + (r - c)], xin[c]));
                    kk_c += static_cast<std::size_t>(n - c);
                }
            } else {                                         /* right: contiguous in col r */
                for (int c = r + 1; c < n; ++c) {
                    T e = ap[kk_r + (c - r)]; if (conj) e = cconj(e);
                    s = cadd(s, cmul(e, xin[c]));
                }
            }
            xout[(std::ptrdiff_t)r * incx] = s;
        }
    }
}

/* Threaded in-place complex triangular packed matvec. Copy x to contiguous,
 * partition the output rows, write back. Returns true if handled. */
__attribute__((noinline)) static bool wtpmv_omp(
    bool upper, bool trans, bool conj, bool nounit, int n,
    const T *ap, T *x, int incx)
{
    if (n < WTPMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > WTPMV_MAX_CPUS) nthreads = WTPMV_MAX_CPUS;

    T *xbase = (incx < 0) ? x - (std::ptrdiff_t)(n - 1) * incx : x;
    T *xbuf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
    if (!xbuf) return false;
    for (int i = 0; i < n; ++i) xbuf[i] = xbase[(std::ptrdiff_t)i * incx];

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int lo = (int)((long long)n * tid / nthreads);
        int hi = (int)((long long)n * (tid + 1) / nthreads);
        wtpmv_rowgather(upper, trans, conj, nounit, n, lo, hi, ap, xbuf, xbase, incx);
    }
    std::free(xbuf);
    return true;
}
#endif

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

#ifdef _OPENMP
    if (N >= WTPMV_OMP_MIN && blas_omp_max_threads() > 1
        && wtpmv_omp(UPLO == 'U', TR != 'N', TR == 'C', nounit != 0, N, ap, x, incx))
        return;
#endif

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
