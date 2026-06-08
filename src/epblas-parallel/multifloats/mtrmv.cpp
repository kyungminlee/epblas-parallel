/* mtrmv — multifloats real DD triangular matrix-vector. */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <cstdlib>
#include <omp.h>
#include "../common/blas_omp.h"
#define MTRMV_OMP_MIN 128
#define MTRMV_MAX_CPUS 256
#endif

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

#ifdef _OPENMP
/* Threaded contiguous-x core, mirroring kind10 etrmv_omp_contig. The earlier
 * row-gather threaded poorly: NoTrans read the matrix by ROW (A_(r,c) strided
 * by lda across columns) — cache-hostile for column-major DD storage — and the
 * contiguous-block row partition load-imbalanced the triangular work, so par4
 * barely beat par1. This instead keeps matrix access COLUMN-contiguous and uses
 * schedule(static,1) cyclic distribution to balance the triangular work:
 *   - Trans: each x[j] is an independent dot over a contiguous column slice
 *     (disjoint writes → no reduction).
 *   - NoTrans: per-thread private accumulator + column AXPY (contiguous),
 *     reduced at the end (esymv pattern).
 * DD addition reorders vs the serial path → within fuzz tol; the serial path
 * stays bit-exact. Operates on a contiguous x; the strided dispatch gathers /
 * scatters around it. Returns true on success, false if a scratch alloc failed
 * (caller falls back to serial). */
static bool mtrmv_omp_contig(bool upper, bool trans, bool nounit,
                             std::ptrdiff_t n, const T *a, std::size_t lda,
                             T *x, int nt)
{
    if (trans) {
        T *y_buf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
        if (!y_buf) return false;
        #pragma omp parallel num_threads(nt)
        {
            if (!upper) {
                #pragma omp for schedule(static, 1)
                for (std::ptrdiff_t j = 0; j < n; ++j) {
                    T temp = nounit ? (x[j] * A_(j, j)) : x[j];
                    const T *aj = &A_(0, j);
                    T s0 = zero_dd, s1 = zero_dd;
                    std::ptrdiff_t i = j + 1;
                    for (; i + 1 < n; i += 2) {
                        s0 = s0 + aj[i]     * x[i];
                        s1 = s1 + aj[i + 1] * x[i + 1];
                    }
                    T s = s0 + s1;
                    for (; i < n; ++i) s = s + aj[i] * x[i];
                    y_buf[j] = temp + s;
                }
            } else {
                #pragma omp for schedule(static, 1)
                for (std::ptrdiff_t j = 0; j < n; ++j) {
                    T temp = nounit ? (x[j] * A_(j, j)) : x[j];
                    const T *aj = &A_(0, j);
                    T s0 = zero_dd, s1 = zero_dd;
                    std::ptrdiff_t i = j - 1;
                    for (; i - 1 >= 0; i -= 2) {
                        s0 = s0 + aj[i]     * x[i];
                        s1 = s1 + aj[i - 1] * x[i - 1];
                    }
                    T s = s0 + s1;
                    for (; i >= 0; --i) s = s + aj[i] * x[i];
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
            if (!upper) {
                #pragma omp for schedule(static, 1)
                for (std::ptrdiff_t j = 0; j < n; ++j) {
                    const T xj = x[j];
                    const T *aj = &A_(0, j);
                    y_priv[j] = y_priv[j] + xj * (nounit ? aj[j] : one_dd);
                    for (std::ptrdiff_t i = j + 1; i < n; ++i)
                        y_priv[i] = y_priv[i] + xj * aj[i];
                }
            } else {
                #pragma omp for schedule(static, 1)
                for (std::ptrdiff_t j = 0; j < n; ++j) {
                    const T xj = x[j];
                    const T *aj = &A_(0, j);
                    for (std::ptrdiff_t i = 0; i < j; ++i)
                        y_priv[i] = y_priv[i] + xj * aj[i];
                    y_priv[j] = y_priv[j] + xj * (nounit ? aj[j] : one_dd);
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

/* Threaded in-place dense triangular matvec. incx==1 drives the contiguous core
 * directly; strided gathers into a contiguous buffer, drives the core, scatters
 * back. Returns true if handled. */
__attribute__((noinline)) static bool mtrmv_omp(
    bool upper, bool trans, bool nounit, int n,
    const T *a, std::size_t lda, T *x, int incx)
{
    if (n < MTRMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MTRMV_MAX_CPUS) nthreads = MTRMV_MAX_CPUS;

    if (incx == 1)
        return mtrmv_omp_contig(upper, trans, nounit, n, a, lda, x, nthreads);

    T *xbase = (incx < 0) ? x - (std::ptrdiff_t)(n - 1) * incx : x;
    T *xbuf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
    if (!xbuf) return false;
    for (int i = 0; i < n; ++i) xbuf[i] = xbase[(std::ptrdiff_t)i * incx];
    bool ok = mtrmv_omp_contig(upper, trans, nounit, n, a, lda, xbuf, nthreads);
    if (ok)
        for (int i = 0; i < n; ++i) xbase[(std::ptrdiff_t)i * incx] = xbuf[i];
    std::free(xbuf);
    return ok;
}
#endif

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

#ifdef _OPENMP
    if (N >= MTRMV_OMP_MIN && blas_omp_max_threads() > 1
        && mtrmv_omp(UPLO == 'U', TR != 'N', nounit, N, a, lda, x, incx))
        return;
#endif

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
