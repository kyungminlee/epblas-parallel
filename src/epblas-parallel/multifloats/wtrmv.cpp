/* wtrmv — multifloats complex DD triangular matrix-vector. */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include "../common/blas_omp.h"
#define WTRMV_OMP_MIN 128
#define WTRMV_MAX_CPUS 256
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
const T zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };
inline bool cdd_iszero(const T &x) {
    return x.re.limbs[0] == 0.0 && x.re.limbs[1] == 0.0
        && x.im.limbs[0] == 0.0 && x.im.limbs[1] == 0.0;
}
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

#ifdef _OPENMP
/* sqrt-balanced contiguous row-block partition (OpenBLAS trmv_thread.c). Each
 * thread owns a contiguous output-row range; widths shrink/grow so the per-thread
 * triangular work (≈ width·(rows above/below)) is equal. range[0..nt]. */
static void wtrmv_partition(bool upper, std::ptrdiff_t n, int nthreads,
                            std::ptrdiff_t *range)
{
    const std::ptrdiff_t mask = 7;
    const double dnum = (double)n * (double)n / (double)nthreads;
    if (!upper) {
        range[0] = 0;
        std::ptrdiff_t i = 0; int num_cpu = 0;
        while (i < n && num_cpu < nthreads) {
            std::ptrdiff_t width;
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                width = (di * di - dnum > 0.0)
                    ? (((std::ptrdiff_t)(-std::sqrt(di * di - dnum) + di) + mask) & ~mask)
                    : (n - i);
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else width = n - i;
            range[num_cpu + 1] = range[num_cpu] + width;
            num_cpu++; i += width;
        }
        for (int t = num_cpu + 1; t <= nthreads; ++t) range[t] = range[num_cpu];
    } else {
        range[nthreads] = n;
        std::ptrdiff_t i = 0; int num_cpu = 0;
        while (i < n && num_cpu < nthreads) {
            std::ptrdiff_t width;
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                width = (di * di - dnum > 0.0)
                    ? (((std::ptrdiff_t)(-std::sqrt(di * di - dnum) + di) + mask) & ~mask)
                    : (n - i);
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else width = n - i;
            range[nthreads - num_cpu - 1] = range[nthreads - num_cpu] - width;
            num_cpu++; i += width;
        }
        for (int t = 0; t < nthreads - num_cpu; ++t) range[t] = range[nthreads - num_cpu];
    }
}

/* NoTrans tiled column-AXPY kernel (OpenBLAS trmv_thread.c kernel_N). Reads only
 * the contiguous matrix column block [m_from,m_to); writes its own rows directly
 * plus the off-block "spill" rows into the thread's private y slot (merged by a
 * bounded reduction). DTB tiling keeps the active y-tile hot. */
static void wtrmv_kernel_N(bool upper, bool nounit, std::ptrdiff_t n,
                           std::ptrdiff_t m_from, std::ptrdiff_t m_to,
                           const T *a, std::size_t lda, const T *x, T *y)
{
    const std::ptrdiff_t TB = 32;
    for (std::ptrdiff_t is = m_from; is < m_to; is += TB) {
        std::ptrdiff_t min_i = (m_to - is < TB) ? m_to - is : TB;
        if (upper && is > 0) {
            for (std::ptrdiff_t j = is; j < is + min_i; ++j) {
                const T xj = x[j];
                const T *col = &A_(0, j);
                for (std::ptrdiff_t i = 0; i < is; ++i) y[i] = cadd(y[i], cmul(col[i], xj));
            }
        }
        for (std::ptrdiff_t i = is; i < is + min_i; ++i) {
            if (upper && i > is) {
                const T xi = x[i];
                const T *col = &A_(0, i);
                for (std::ptrdiff_t k = is; k < i; ++k) y[k] = cadd(y[k], cmul(col[k], xi));
            }
            y[i] = cadd(y[i], nounit ? cmul(A_(i, i), x[i]) : x[i]);
            if (!upper && i + 1 < is + min_i) {
                const T xi = x[i];
                const T *col = &A_(0, i);
                for (std::ptrdiff_t k = i + 1; k < is + min_i; ++k) y[k] = cadd(y[k], cmul(col[k], xi));
            }
        }
        if (!upper && is + min_i < n) {
            for (std::ptrdiff_t j = is; j < is + min_i; ++j) {
                const T xj = x[j];
                const T *col = &A_(0, j);
                for (std::ptrdiff_t i = is + min_i; i < n; ++i) y[i] = cadd(y[i], cmul(col[i], xj));
            }
        }
    }
}

/* Threaded contiguous-x core, mirroring mtrmv_omp_contig with complex DD math.
 * The earlier row-gather threaded poorly: NoTrans read the matrix by ROW
 * (A_(r,c) strided by lda) — cache-hostile for column-major storage — and the
 * contiguous-block row partition load-imbalanced the triangular work. This keeps
 * matrix access COLUMN-contiguous and uses schedule(static,1) cyclic balancing:
 *   - Trans/ConjTrans: each x[j] is an independent contiguous-column dot
 *     (conjugated when conj; disjoint writes → no reduction).
 *   - NoTrans: per-thread accumulator + column AXPY, reduced at the end.
 * DD addition reorders vs serial → within fuzz tol; serial stays bit-exact.
 * Returns true on success, false if a scratch alloc failed. */
static bool wtrmv_omp_contig(bool upper, bool trans, bool conj, bool nounit,
                             std::ptrdiff_t n, const T *a, std::size_t lda,
                             T *x, int nt)
{
    if (trans) {
        T *y_buf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
        if (!y_buf) return false;
        #pragma omp parallel num_threads(nt)
        {
            #pragma omp for schedule(static, 1)
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                T diagc = A_(j, j); if (conj) diagc = cconj(diagc);
                T s = nounit ? cmul(diagc, x[j]) : x[j];
                const T *aj = &A_(0, j);
                if (upper) {
                    for (std::ptrdiff_t c = 0; c < j; ++c) {
                        T e = aj[c]; if (conj) e = cconj(e);
                        s = cadd(s, cmul(e, x[c]));
                    }
                } else {
                    for (std::ptrdiff_t c = j + 1; c < n; ++c) {
                        T e = aj[c]; if (conj) e = cconj(e);
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
        /* NoTrans: OpenBLAS contiguous row-block scheme. Each thread reads only
         * its matrix column block (good cache locality vs cyclic) and merges its
         * bounded spill rows — beats the full per-thread accumulator + O(nt·n)
         * reduction at large n. */
        T *buf_all = static_cast<T *>(
            std::calloc((std::size_t)nt * (std::size_t)n, sizeof(T)));
        if (!buf_all) return false;
        std::ptrdiff_t *range = static_cast<std::ptrdiff_t *>(
            std::malloc((std::size_t)(nt + 1) * sizeof(std::ptrdiff_t)));
        if (!range) { std::free(buf_all); return false; }
        wtrmv_partition(upper, n, nt, range);
        #pragma omp parallel num_threads(nt)
        {
            const int tid = omp_get_thread_num();
            T *y = &buf_all[(std::size_t)tid * n];  /* calloc-zeroed */
            std::ptrdiff_t m_from, m_to;
            if (upper) { m_from = range[nt - tid - 1]; m_to = range[nt - tid]; }
            else       { m_from = range[tid];          m_to = range[tid + 1]; }
            if (m_from < m_to)
                wtrmv_kernel_N(upper, nounit, n, m_from, m_to, a, lda, x, y);
        }
        /* Bounded reduction: merge each thread's spill rows into slot 0. */
        if (upper) {
            for (int t = 1; t < nt; ++t) {
                std::ptrdiff_t m_to_t = range[nt - t];
                const T *slot = &buf_all[(std::size_t)t * n];
                for (std::ptrdiff_t i = 0; i < m_to_t; ++i)
                    buf_all[i] = cadd(buf_all[i], slot[i]);
            }
        } else {
            for (int t = 1; t < nt; ++t) {
                std::ptrdiff_t m_from_t = range[t];
                const T *slot = &buf_all[(std::size_t)t * n];
                for (std::ptrdiff_t i = m_from_t; i < n; ++i)
                    buf_all[i] = cadd(buf_all[i], slot[i]);
            }
        }
        for (std::ptrdiff_t i = 0; i < n; ++i) x[i] = buf_all[i];
        std::free(buf_all); std::free(range);
        return true;
    }
}

/* Threaded in-place complex dense triangular matvec. incx==1 drives the
 * contiguous core directly; strided gathers/scatters around it. */
__attribute__((noinline)) static bool wtrmv_omp(
    bool upper, bool trans, bool conj, bool nounit, int n,
    const T *a, std::size_t lda, T *x, int incx)
{
    if (n < WTRMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > WTRMV_MAX_CPUS) nthreads = WTRMV_MAX_CPUS;

    if (incx == 1)
        return wtrmv_omp_contig(upper, trans, conj, nounit, n, a, lda, x, nthreads);

    T *xbase = (incx < 0) ? x - (std::ptrdiff_t)(n - 1) * incx : x;
    T *xbuf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
    if (!xbuf) return false;
    for (int i = 0; i < n; ++i) xbuf[i] = xbase[(std::ptrdiff_t)i * incx];
    bool ok = wtrmv_omp_contig(upper, trans, conj, nounit, n, a, lda, xbuf, nthreads);
    if (ok)
        for (int i = 0; i < n; ++i) xbase[(std::ptrdiff_t)i * incx] = xbuf[i];
    std::free(xbuf);
    return ok;
}
#endif

extern "C" void wtrmv_(
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
    const char TR   = up(trans);
    const char DIAG = up(diag);
    const bool nounit = (DIAG != 'U');

    if (N == 0) return;

#ifdef _OPENMP
    if (N >= WTRMV_OMP_MIN && blas_omp_max_threads() > 1
        && wtrmv_omp(UPLO == 'U', TR != 'N', TR == 'C', nounit, N, a, lda, x, incx))
        return;
#endif

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (int j = N - 1; j >= 0; --j) {
                    const T temp = x[j];
                    if (!cdd_iszero(temp)) {
                        const T *aj = &A_(0, j);
                        for (int i = j + 1; i < N; ++i) x[i] = cadd(x[i], cmul(temp, aj[i]));
                    }
                    if (nounit) x[j] = cmul(x[j], A_(j, j));
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    const T temp = x[j];
                    if (!cdd_iszero(temp)) {
                        const T *aj = &A_(0, j);
                        for (int i = 0; i < j; ++i) x[i] = cadd(x[i], cmul(temp, aj[i]));
                    }
                    if (nounit) x[j] = cmul(x[j], A_(j, j));
                }
            }
        } else {
            const bool conj_a = (TR == 'C');
            if (UPLO == 'L') {
                for (int j = 0; j < N; ++j) {
                    T temp = x[j];
                    if (nounit) temp = cmul(temp, conj_a ? cconj(A_(j, j)) : A_(j, j));
                    const T *aj = &A_(0, j);
                    if (conj_a) {
                        for (int i = j + 1; i < N; ++i) temp = cadd(temp, cmul(cconj(aj[i]), x[i]));
                    } else {
                        for (int i = j + 1; i < N; ++i) temp = cadd(temp, cmul(aj[i], x[i]));
                    }
                    x[j] = temp;
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    T temp = x[j];
                    if (nounit) temp = cmul(temp, conj_a ? cconj(A_(j, j)) : A_(j, j));
                    const T *aj = &A_(0, j);
                    if (conj_a) {
                        for (int i = 0; i < j; ++i) temp = cadd(temp, cmul(cconj(aj[i]), x[i]));
                    } else {
                        for (int i = 0; i < j; ++i) temp = cadd(temp, cmul(aj[i], x[i]));
                    }
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
                    if (!cdd_iszero(temp))
                        for (int i = j + 1; i < N; ++i) x[kx + i * incx] = cadd(x[kx + i * incx], cmul(temp, A_(i, j)));
                    if (nounit) x[kx + j * incx] = cmul(x[kx + j * incx], A_(j, j));
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    const T temp = x[kx + j * incx];
                    if (!cdd_iszero(temp))
                        for (int i = 0; i < j; ++i) x[kx + i * incx] = cadd(x[kx + i * incx], cmul(temp, A_(i, j)));
                    if (nounit) x[kx + j * incx] = cmul(x[kx + j * incx], A_(j, j));
                }
            }
        } else {
            const bool conj_a = (TR == 'C');
            if (UPLO == 'L') {
                for (int j = 0; j < N; ++j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp = cmul(temp, conj_a ? cconj(A_(j, j)) : A_(j, j));
                    for (int i = j + 1; i < N; ++i) {
                        const T aij = conj_a ? cconj(A_(i, j)) : A_(i, j);
                        temp = cadd(temp, cmul(aij, x[kx + i * incx]));
                    }
                    x[kx + j * incx] = temp;
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp = cmul(temp, conj_a ? cconj(A_(j, j)) : A_(j, j));
                    for (int i = 0; i < j; ++i) {
                        const T aij = conj_a ? cconj(A_(i, j)) : A_(i, j);
                        temp = cadd(temp, cmul(aij, x[kx + i * incx]));
                    }
                    x[kx + j * incx] = temp;
                }
            }
        }
    }
}

#undef A_
