/* mtbmv — multifloats real DD triangular band matrix-vector.
 *   x := A*x or A^T*x, A triangular band with K+1 diagonals.
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
#define MTBMV_OMP_MIN 256
#define MTBMV_MAX_CPUS 256
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
inline bool dd_iszero(const T &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

#ifdef _OPENMP
/* Row-gather for the Trans triangle: x := A^T*x is an in-place band matvec, so
 * each output row r is an independent dot once the original x is preserved in a
 * copy. Trans reads matrix column r CONTIGUOUSLY (base[k-d] upper / base[d]
 * lower), so this already scales well; the NoTrans triangle, whose row read is
 * anti-diagonal, uses the column-scatter below instead. Diagonal scales x[r]
 * when non-unit. Mirrors the serial accumulation order (bit-exact). */
static void mtbmv_rowgather_t(bool upper, bool nounit,
                              int n, int k, int lo, int hi,
                              const T *a, std::size_t lda,
                              const T *xin, T *y)
{
    for (int r = lo; r < hi; ++r) {
        const T *base = &A_(0, r);
        const T diagc = upper ? base[k] : base[0];
        T s = nounit ? diagc * xin[r] : xin[r];
        if (upper) {
            const int llen = (r < k) ? r : k;
            for (int d = 1; d <= llen; ++d) s = s + base[k - d] * xin[r - d];
        } else {
            const int rlen = (n - 1 - r < k) ? (n - 1 - r) : k;
            for (int d = 1; d <= rlen; ++d) s = s + base[d] * xin[r + d];
        }
        y[r] = s;
    }
}

/* Column-scatter for the NoTrans triangle: each thread owns output rows [lo,hi)
 * and walks the columns touching them, reading every column's band segment
 * CONTIGUOUSLY (stride 1 in the row index, vs the row-gather's anti-diagonal
 * lda-1 stride) and scattering only into its owned rows. Writes are disjoint
 * across threads -> no race, no O(nthreads*n) fold. Iterating columns ascending
 * (upper) / descending (lower) seeds each owned row at its diagonal column
 * first, then accumulates off-diagonals in column order -> identical per-row
 * association as the serial scatter (bit-exact). y[lo,hi) need not be pre-zeroed:
 * every owned row is assigned at its diagonal column before any += reaches it. */
static void mtbmv_colscatter(bool upper, bool nounit, int n, int k,
                             int lo, int hi, const T *a, std::size_t lda,
                             const T *xin, T *y)
{
    if (upper) {
        const int jmax = (hi + k < n) ? (hi + k) : n;
        for (int j = lo; j < jmax; ++j) {
            const T tmp = xin[j];
            const int L = k - j;                         /* A(i,j) = A_(L+i, j) */
            const int i_lo = (j - k > lo) ? (j - k) : lo;
            const int i_hi = (j < hi) ? j : hi;          /* off-diagonal rows < j */
            const T *col = &A_(L + i_lo, j);             /* contiguous in i */
            for (int i = i_lo; i < i_hi; ++i) y[i] = y[i] + tmp * (*col++);
            if (j >= lo && j < hi)                       /* diagonal seed */
                y[j] = nounit ? tmp * A_(k, j) : tmp;
        }
    } else {
        const int jmin = (lo - k > 0) ? (lo - k) : 0;
        for (int j = hi - 1; j >= jmin; --j) {
            const T tmp = xin[j];
            const int i_lo = (j + 1 > lo) ? (j + 1) : lo;
            const int i_hi = (j + k + 1 < hi) ? (j + k + 1) : hi;  /* rows > j */
            const T *col = &A_(i_lo - j, j);             /* A(i,j)=A_(i-j,j), contiguous */
            for (int i = i_lo; i < i_hi; ++i) y[i] = y[i] + tmp * (*col++);
            if (j >= lo && j < hi)                       /* diagonal seed */
                y[j] = nounit ? tmp * A_(0, j) : tmp;
        }
    }
}

/* Threaded in-place triangular band matvec. Each thread owns a disjoint output-
 * row range [lo,hi): it gathers y[lo,hi) reading x (never written here), a
 * barrier, then copies its own range back to x. NoTrans uses the contiguous
 * column-scatter above; Trans uses the row-gather (which reads its column
 * contiguously already). Unit-stride reads x directly — no serial input copy
 * (the old full x->xbuf memcpy was an O(n) Amdahl tax); only a strided x is
 * gathered to contiguous up front. Returns true if handled. */
__attribute__((noinline)) static bool mtbmv_omp(
    bool upper, bool trans, bool nounit, int n, int k,
    const T *a, std::size_t lda, T *x, int incx)
{
    if (n < MTBMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MTBMV_MAX_CPUS) nthreads = MTBMV_MAX_CPUS;

    if (incx < 0) x -= (std::ptrdiff_t)(n - 1) * incx;   /* x at logical 0 */

    const T *xptr = x;
    T *xbuf = NULL;
    if (incx != 1) {
        xbuf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
        if (!xbuf) return false;
        for (int i = 0; i < n; ++i) xbuf[i] = x[(std::ptrdiff_t)i * incx];
        xptr = xbuf;
    }
    T *y = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
    if (!y) { std::free(xbuf); return false; }

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int lo = (int)((long long)n * tid / nthreads);
        int hi = (int)((long long)n * (tid + 1) / nthreads);
        if (!trans) mtbmv_colscatter(upper, nounit, n, k, lo, hi, a, lda, xptr, y);
        else        mtbmv_rowgather_t(upper, nounit, n, k, lo, hi, a, lda, xptr, y);
        #pragma omp barrier              /* all reads of x done before any write-back */
        if (incx == 1) for (int i = lo; i < hi; ++i) x[i] = y[i];
        else           for (int i = lo; i < hi; ++i) x[(std::ptrdiff_t)i * incx] = y[i];
    }
    std::free(y); std::free(xbuf);
    return true;
}
#endif

extern "C" void mtbmv_(
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
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
    const int nounit = (up(diag) != 'U');

    if (N == 0) return;

#ifdef _OPENMP
    if (N >= MTBMV_OMP_MIN && blas_omp_max_threads() > 1
        && mtbmv_omp(UPLO == 'U', TR != 'N', nounit != 0, N, K, a, lda, x, incx))
        return;
#endif

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'U') {
                for (int j = 0; j < N; ++j) {
                    if (!dd_iszero(x[j])) {
                        const T tmp = x[j];
                        const int L = K - j;
                        const int i_lo = (j - K > 0) ? (j - K) : 0;
                        for (int i = i_lo; i < j; ++i) x[i] = x[i] + tmp * A_(L + i, j);
                        if (nounit) x[j] = x[j] * A_(K, j);
                    }
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    if (!dd_iszero(x[j])) {
                        const T tmp = x[j];
                        const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                        for (int i = i_hi - 1; i > j; --i) x[i] = x[i] + tmp * A_(i - j, j);
                        if (nounit) x[j] = x[j] * A_(0, j);
                    }
                }
            }
        } else {
            if (UPLO == 'U') {
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    const int L = K - j;
                    if (nounit) tmp = tmp * A_(K, j);
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    for (int i = j - 1; i >= i_lo; --i) tmp = tmp + A_(L + i, j) * x[i];
                    x[j] = tmp;
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    T tmp = x[j];
                    if (nounit) tmp = tmp * A_(0, j);
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (int i = j + 1; i < i_hi; ++i) tmp = tmp + A_(i - j, j) * x[i];
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
                    if (!dd_iszero(x[jx])) {
                        const T tmp = x[jx];
                        int ix = kx;
                        const int L = K - j;
                        const int i_lo = (j - K > 0) ? (j - K) : 0;
                        for (int i = i_lo; i < j; ++i) {
                            x[ix] = x[ix] + tmp * A_(L + i, j);
                            ix += incx;
                        }
                        if (nounit) x[jx] = x[jx] * A_(K, j);
                    }
                    jx += incx;
                    if (j >= K) kx += incx;
                }
            } else {
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    if (!dd_iszero(x[jx])) {
                        const T tmp = x[jx];
                        int ix = kx;
                        const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                        for (int i = i_hi - 1; i > j; --i) {
                            x[ix] = x[ix] + tmp * A_(i - j, j);
                            ix -= incx;
                        }
                        if (nounit) x[jx] = x[jx] * A_(0, j);
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
                    if (nounit) tmp = tmp * A_(K, j);
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    for (int i = j - 1; i >= i_lo; --i) {
                        tmp = tmp + A_(L + i, j) * x[ix];
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
                    if (nounit) tmp = tmp * A_(0, j);
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (int i = j + 1; i < i_hi; ++i) {
                        tmp = tmp + A_(i - j, j) * x[ix];
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
