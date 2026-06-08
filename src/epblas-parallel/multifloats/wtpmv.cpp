/* wtpmv — multifloats complex DD triangular packed matrix-vector. */

#include <cstddef>
#include <cctype>
#include <vector>
#include <multifloats.h>
#ifdef _OPENMP
#include <cstdlib>
#include <omp.h>
#include "../common/blas_omp.h"
#define WTPMV_OMP_MIN 128
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

/* Bit-exact row-tiled upper-triangular packed NoTrans serial matvec (incx==1).
 * Packed upper column j stores rows 0..j-1 contiguously at ap[j(j+1)/2], so the
 * plain loop streams a growing x[0..j-1] run per column and thrashes the x cache
 * (the OpenBLAS clone tiles the output rows even at one thread and wins ~15-19%).
 * Sweep a 32-row output tile that stays cache-hot across the column passes,
 * processing within-tile then above-tile columns in strictly ascending order so
 * each output element's accumulation (diagonal first, then ascending column
 * index) is byte-identical to the untiled loop. */
inline std::size_t wtpmv_kk_upper(int j) {
    return static_cast<std::size_t>(j) * (j + 1) / 2;
}
static void wtpmv_serial_N_upper(bool nounit, int n, const T *ap, T *x) {
    const int RB = 32;
    std::vector<T> ybuf(static_cast<std::size_t>(n));
    T *y = ybuf.data();
    for (int ib = 0; ib < n; ib += RB) {
        const int ie = ib + RB < n ? ib + RB : n;
        for (int i = ib; i < ie; ++i)
            y[i] = nounit ? cmul(x[i], ap[wtpmv_kk_upper(i) + i]) : x[i];
        for (int j = ib; j < ie; ++j) {                 /* within-tile, ascending j */
            const T xj = x[j];
            if (cdd_iszero(xj)) continue;
            const T *col = &ap[wtpmv_kk_upper(j)];
            for (int i = ib; i < j; ++i) y[i] = cadd(y[i], cmul(xj, col[i]));
        }
        for (int j = ie; j < n; ++j) {                  /* above-tile columns, ascending j */
            const T xj = x[j];
            if (cdd_iszero(xj)) continue;
            const T *col = &ap[wtpmv_kk_upper(j)];
            for (int i = ib; i < ie; ++i) y[i] = cadd(y[i], cmul(xj, col[i]));
        }
    }
    for (int i = 0; i < n; ++i) x[i] = y[i];
}
}

#ifdef _OPENMP
namespace {
const T zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };
/* Base index of packed column j. Upper: kk=j(j+1)/2, diag at kk+j, off-diag rows
 * 0..j-1 at kk+0..j-1. Lower: kk=j*N-j(j-1)/2, diag at kk+0, rows j+1..N-1 at
 * kk+1..N-1-j. So &ap[kk] is column j contiguous — identical to the dense
 * (wtrmv) per-column kernel, just with packed offsets. */
inline std::size_t kk_upper(std::ptrdiff_t j) {
    return static_cast<std::size_t>(j) * (j + 1) / 2;
}
inline std::size_t kk_lower(std::ptrdiff_t j, std::ptrdiff_t n) {
    return static_cast<std::size_t>(j) * n - static_cast<std::size_t>(j) * (j - 1) / 2;
}
}

/* Threaded contiguous-x core, mirroring wtrmv_omp_contig with packed column
 * offsets. The earlier row-gather threaded poorly: NoTrans walked a column-
 * JUMPING run — cache-hostile — and the contiguous-block row partition load-
 * imbalanced the triangular work. This keeps packed-column access contiguous and
 * uses schedule(static,1) cyclic balancing:
 *   - Trans/ConjTrans: each x[j] is an independent contiguous-column dot
 *     (conjugated when conj; disjoint writes).
 *   - NoTrans: per-thread accumulator + column AXPY, reduced at the end.
 * DD addition reorders vs serial → within fuzz tol; serial stays bit-exact. */
static bool wtpmv_omp_contig(bool upper, bool trans, bool conj, bool nounit,
                             std::ptrdiff_t n, const T *ap, T *x, int nt)
{
    if (trans) {
        T *y_buf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
        if (!y_buf) return false;
        #pragma omp parallel num_threads(nt)
        {
            #pragma omp for schedule(static, 1)
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const T *aj = upper ? &ap[kk_upper(j)] : &ap[kk_lower(j, n)];
                T diagc = upper ? aj[j] : aj[0]; if (conj) diagc = cconj(diagc);
                T s = nounit ? cmul(diagc, x[j]) : x[j];
                if (upper) {
                    for (std::ptrdiff_t c = 0; c < j; ++c) {
                        T e = aj[c]; if (conj) e = cconj(e);
                        s = cadd(s, cmul(e, x[c]));
                    }
                } else {
                    for (std::ptrdiff_t c = j + 1; c < n; ++c) {
                        T e = aj[c - j]; if (conj) e = cconj(e);
                        s = cadd(s, cmul(e, x[c]));
                    }
                }
                y_buf[j] = s;
            }
            #pragma omp for schedule(static)
            for (std::ptrdiff_t i = 0; i < n; ++i) x[i] = y_buf[i];
        }
        std::free(y_buf);
        return true;
    } else {
        const T one_cdd{ R{1.0, 0.0}, R{0.0, 0.0} };
        T *y_priv_all = static_cast<T *>(
            std::calloc((std::size_t)nt * (std::size_t)n, sizeof(T)));
        if (!y_priv_all) return false;
        #pragma omp parallel num_threads(nt)
        {
            const std::ptrdiff_t tid = omp_get_thread_num();
            T *y_priv = &y_priv_all[(std::size_t)tid * n];  /* calloc-zeroed */
            #pragma omp for schedule(static, 1)
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const T xj = x[j];
                if (upper) {
                    const T *aj = &ap[kk_upper(j)];
                    for (std::ptrdiff_t i = 0; i < j; ++i)
                        y_priv[i] = cadd(y_priv[i], cmul(xj, aj[i]));
                    y_priv[j] = cadd(y_priv[j], cmul(xj, nounit ? aj[j] : one_cdd));
                } else {
                    const T *aj = &ap[kk_lower(j, n)];
                    y_priv[j] = cadd(y_priv[j], cmul(xj, nounit ? aj[0] : one_cdd));
                    for (std::ptrdiff_t i = j + 1; i < n; ++i)
                        y_priv[i] = cadd(y_priv[i], cmul(xj, aj[i - j]));
                }
            }
            #pragma omp for schedule(static)
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                T s = zero_cdd;
                for (std::ptrdiff_t t = 0; t < nt; ++t)
                    s = cadd(s, y_priv_all[(std::size_t)t * n + i]);
                x[i] = s;
            }
        }
        std::free(y_priv_all);
        return true;
    }
}

/* Threaded in-place complex triangular packed matvec. incx==1 drives the
 * contiguous core directly; strided gathers/scatters around it. */
__attribute__((noinline)) static bool wtpmv_omp(
    bool upper, bool trans, bool conj, bool nounit, int n,
    const T *ap, T *x, int incx)
{
    if (n < WTPMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > WTPMV_MAX_CPUS) nthreads = WTPMV_MAX_CPUS;

    if (incx == 1)
        return wtpmv_omp_contig(upper, trans, conj, nounit, n, ap, x, nthreads);

    T *xbase = (incx < 0) ? x - (std::ptrdiff_t)(n - 1) * incx : x;
    T *xbuf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
    if (!xbuf) return false;
    for (int i = 0; i < n; ++i) xbuf[i] = xbase[(std::ptrdiff_t)i * incx];
    bool ok = wtpmv_omp_contig(upper, trans, conj, nounit, n, ap, xbuf, nthreads);
    if (ok)
        for (int i = 0; i < n; ++i) xbase[(std::ptrdiff_t)i * incx] = xbuf[i];
    std::free(xbuf);
    return ok;
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
                if (N >= 128) {
                    wtpmv_serial_N_upper(nounit, N, ap, x);
                } else {
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
