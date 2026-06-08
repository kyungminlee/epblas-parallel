/* mtpmv — multifloats real DD triangular packed matrix-vector.
 *   x := A*x or A^T*x
 *
 * Serial — data dependencies in x.
 */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <cstdlib>
#include <omp.h>
#include "../common/blas_omp.h"
#define MTPMV_OMP_MIN 128
#define MTPMV_MAX_CPUS 256
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
inline bool dd_iszero(const T &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
}

#ifdef _OPENMP
namespace {
const T zero_dd{0.0, 0.0};
/* Base index of packed column j (its first stored element). Upper: kk=j(j+1)/2,
 * diag at kk+j, off-diag rows 0..j-1 at kk+0..j-1. Lower: kk=j*N-j(j-1)/2, diag
 * at kk+0, rows j+1..N-1 at kk+1..N-1-j. So &ap[kk] is column j contiguous —
 * identical access to the dense (mtrmv) per-column kernel, just with packed
 * offsets, so the threading matches mtrmv's contiguous-column scheme. */
inline std::size_t kk_upper(std::ptrdiff_t j) {
    return static_cast<std::size_t>(j) * (j + 1) / 2;
}
inline std::size_t kk_lower(std::ptrdiff_t j, std::ptrdiff_t n) {
    return static_cast<std::size_t>(j) * n - static_cast<std::size_t>(j) * (j - 1) / 2;
}
}

/* Threaded contiguous-x core, mirroring mtrmv_omp_contig but with packed column
 * offsets. The earlier row-gather threaded poorly: NoTrans walked a column-
 * JUMPING run (offset += c+1 per step) — cache-hostile — and the contiguous-
 * block row partition load-imbalanced the triangular work. This keeps packed-
 * column access contiguous and uses schedule(static,1) cyclic balancing:
 *   - Trans: each x[j] is an independent contiguous-column dot (disjoint writes).
 *   - NoTrans: per-thread accumulator + column AXPY, reduced at the end.
 * DD addition reorders vs serial → within fuzz tol; serial stays bit-exact.
 * Returns true on success, false if a scratch alloc failed. */
static bool mtpmv_omp_contig(bool upper, bool trans, bool nounit,
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
                T s = zero_dd;
                if (upper) {
                    T temp = nounit ? (aj[j] * x[j]) : x[j];
                    for (std::ptrdiff_t c = 0; c < j; ++c) s = s + aj[c] * x[c];
                    y_buf[j] = temp + s;
                } else {
                    T temp = nounit ? (aj[0] * x[j]) : x[j];
                    for (std::ptrdiff_t c = j + 1; c < n; ++c) s = s + aj[c - j] * x[c];
                    y_buf[j] = temp + s;
                }
            }
            #pragma omp for schedule(static)
            for (std::ptrdiff_t i = 0; i < n; ++i) x[i] = y_buf[i];
        }
        std::free(y_buf);
        return true;
    } else {
        const T one_dd{1.0, 0.0};
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
                        y_priv[i] = y_priv[i] + xj * aj[i];
                    y_priv[j] = y_priv[j] + xj * (nounit ? aj[j] : one_dd);
                } else {
                    const T *aj = &ap[kk_lower(j, n)];
                    y_priv[j] = y_priv[j] + xj * (nounit ? aj[0] : one_dd);
                    for (std::ptrdiff_t i = j + 1; i < n; ++i)
                        y_priv[i] = y_priv[i] + xj * aj[i - j];
                }
            }
            #pragma omp for schedule(static)
            for (std::ptrdiff_t i = 0; i < n; ++i) {
                T s = zero_dd;
                for (std::ptrdiff_t t = 0; t < nt; ++t)
                    s = s + y_priv_all[(std::size_t)t * n + i];
                x[i] = s;
            }
        }
        std::free(y_priv_all);
        return true;
    }
}

/* Threaded in-place triangular packed matvec. incx==1 drives the contiguous
 * core directly; strided gathers/scatters around it. Returns true if handled. */
__attribute__((noinline)) static bool mtpmv_omp(
    bool upper, bool trans, bool nounit, int n,
    const T *ap, T *x, int incx)
{
    if (n < MTPMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MTPMV_MAX_CPUS) nthreads = MTPMV_MAX_CPUS;

    if (incx == 1)
        return mtpmv_omp_contig(upper, trans, nounit, n, ap, x, nthreads);

    T *xbase = (incx < 0) ? x - (std::ptrdiff_t)(n - 1) * incx : x;
    T *xbuf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
    if (!xbuf) return false;
    for (int i = 0; i < n; ++i) xbuf[i] = xbase[(std::ptrdiff_t)i * incx];
    bool ok = mtpmv_omp_contig(upper, trans, nounit, n, ap, xbuf, nthreads);
    if (ok)
        for (int i = 0; i < n; ++i) xbase[(std::ptrdiff_t)i * incx] = xbuf[i];
    std::free(xbuf);
    return ok;
}
#endif

extern "C" void mtpmv_(
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
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
    const int nounit = (up(diag) != 'U');

    if (N == 0) return;

#ifdef _OPENMP
    if (N >= MTPMV_OMP_MIN && blas_omp_max_threads() > 1
        && mtpmv_omp(UPLO == 'U', TR != 'N', nounit != 0, N, ap, x, incx))
        return;
#endif

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'U') {
                int kk = 0;
                for (int j = 0; j < N; ++j) {
                    if (!dd_iszero(x[j])) {
                        const T tmp = x[j];
                        int k = kk;
                        for (int i = 0; i < j; ++i) { x[i] = x[i] + tmp * ap[k]; ++k; }
                        if (nounit) x[j] = x[j] * ap[kk + j];
                    }
                    kk += j + 1;
                }
            } else {
                int kk = (N * (N + 1)) / 2 - 1;
                for (int j = N - 1; j >= 0; --j) {
                    if (!dd_iszero(x[j])) {
                        const T tmp = x[j];
                        int k = kk;
                        for (int i = N - 1; i > j; --i) { x[i] = x[i] + tmp * ap[k]; --k; }
                        if (nounit) x[j] = x[j] * ap[kk - (N - 1 - j)];
                    }
                    kk -= (N - j);
                }
            }
        } else {
            if (UPLO == 'U') {
                int kk = (N * (N + 1)) / 2 - 1;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    if (nounit) tmp = tmp * ap[kk];
                    int k = kk - 1;
                    for (int i = j - 1; i >= 0; --i) { tmp = tmp + ap[k] * x[i]; --k; }
                    x[j] = tmp;
                    kk -= j + 1;
                }
            } else {
                int kk = 0;
                for (int j = 0; j < N; ++j) {
                    T tmp = x[j];
                    if (nounit) tmp = tmp * ap[kk];
                    int k = kk + 1;
                    for (int i = j + 1; i < N; ++i) { tmp = tmp + ap[k] * x[i]; ++k; }
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
                    if (!dd_iszero(x[jx])) {
                        const T tmp = x[jx];
                        int ix = kx;
                        for (int k = kk; k < kk + j; ++k) {
                            x[ix] = x[ix] + tmp * ap[k];
                            ix += incx;
                        }
                        if (nounit) x[jx] = x[jx] * ap[kk + j];
                    }
                    jx += incx;
                    kk += j + 1;
                }
            } else {
                int kk = (N * (N + 1)) / 2 - 1;
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    if (!dd_iszero(x[jx])) {
                        const T tmp = x[jx];
                        int ix = kx;
                        for (int k = kk; k > kk - (N - 1 - j); --k) {
                            x[ix] = x[ix] + tmp * ap[k];
                            ix -= incx;
                        }
                        if (nounit) x[jx] = x[jx] * ap[kk - (N - 1 - j)];
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
                    if (nounit) tmp = tmp * ap[kk];
                    for (int k = kk - 1; k >= kk - j; --k) {
                        ix -= incx;
                        tmp = tmp + ap[k] * x[ix];
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
                    if (nounit) tmp = tmp * ap[kk];
                    for (int k = kk + 1; k < kk + N - j; ++k) {
                        ix += incx;
                        tmp = tmp + ap[k] * x[ix];
                    }
                    x[jx] = tmp;
                    jx += incx;
                    kk += N - j;
                }
            }
        }
    }
}
