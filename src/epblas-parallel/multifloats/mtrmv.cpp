/* mtrmv — multifloats real DD triangular matrix-vector. */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <cstdlib>
#include <omp.h>
#include "../common/blas_omp.h"
#define MTRMV_OMP_MIN 256
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
/* Row-gather: x := A*x (or A^T*x) in-place dense triangular matvec. Each output
 * row r is an independent dot once the original x is copied off. NoTrans reads
 * matrix row r (A_(r,c), strided by lda across c); Trans reads column r
 * (A_(c,r), contiguous). Diagonal scales x[r] when non-unit. xin is the
 * contiguous copy, xout the destination. Mirrors the serial accumulation order
 * → within DD fuzz tol; the serial path stays bit-exact. */
static void mtrmv_rowgather(bool upper, bool trans, bool nounit,
                            int n, int lo, int hi,
                            const T *a, std::size_t lda,
                            const T *xin, T *xout, int incx)
{
    for (int r = lo; r < hi; ++r) {
        T s = nounit ? A_(r, r) * xin[r] : xin[r];
        if (!trans) {
            if (upper) for (int c = r + 1; c < n; ++c) s = s + A_(r, c) * xin[c];
            else       for (int c = 0; c < r; ++c)     s = s + A_(r, c) * xin[c];
        } else {
            const T *aj = &A_(0, r);
            if (upper) for (int c = 0; c < r; ++c)     s = s + aj[c] * xin[c];
            else       for (int c = r + 1; c < n; ++c) s = s + aj[c] * xin[c];
        }
        xout[(std::ptrdiff_t)r * incx] = s;
    }
}

/* Threaded in-place dense triangular matvec. Copy x to contiguous, partition the
 * output rows, write back. Returns true if handled. */
__attribute__((noinline)) static bool mtrmv_omp(
    bool upper, bool trans, bool nounit, int n,
    const T *a, std::size_t lda, T *x, int incx)
{
    if (n < MTRMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MTRMV_MAX_CPUS) nthreads = MTRMV_MAX_CPUS;

    T *xbase = (incx < 0) ? x - (std::ptrdiff_t)(n - 1) * incx : x;
    T *xbuf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
    if (!xbuf) return false;
    for (int i = 0; i < n; ++i) xbuf[i] = xbase[(std::ptrdiff_t)i * incx];

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int lo = (int)((long long)n * tid / nthreads);
        int hi = (int)((long long)n * (tid + 1) / nthreads);
        mtrmv_rowgather(upper, trans, nounit, n, lo, hi, a, lda, xbuf, xbase, incx);
    }
    std::free(xbuf);
    return true;
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
