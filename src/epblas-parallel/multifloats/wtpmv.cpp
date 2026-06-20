/* wtpmv — multifloats complex DD triangular packed matrix-vector. */

#include <cstddef>
#include <cctype>
#include <vector>
#include <multifloats.h>
#include "mf_util.h"
#include "mf_pred.h"
#include "mf_kernels.h"
#include "mf_packed.h"
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


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::cconj;

/* Bit-exact row-tiled upper-triangular packed NoTrans serial matvec (incx==1).
 * Packed upper column j stores rows 0..j-1 contiguously at ap[j(j+1)/2], so the
 * plain loop streams a growing x[0..j-1] run per column and thrashes the x cache
 * (the OpenBLAS clone tiles the output rows even at one thread and wins ~15-19%).
 * Sweep a 32-row output tile that stays cache-hot across the column passes,
 * processing within-tile then above-tile columns in strictly ascending order so
 * each output element's accumulation (diagonal first, then ascending column
 * index) is byte-identical to the untiled loop. */
using mf_packed::kk_upper;
static void wtpmv_serial_N_upper(bool nounit, int n, const T *ap, T *x) {
    const int RB = 32;
    std::vector<T> ybuf(static_cast<std::size_t>(n));
    T *y = ybuf.data();
    for (int ib = 0; ib < n; ib += RB) {
        const int ie = ib + RB < n ? ib + RB : n;
        for (int i = ib; i < ie; ++i)
            y[i] = nounit ? cmul(x[i], ap[kk_upper(i) + i]) : x[i];
        for (int j = ib; j < ie; ++j) {                 /* within-tile, ascending j */
            const T xj = x[j];
            if (ceq0(xj)) continue;
            const T *col = &ap[kk_upper(j)];
            mf_kernels::caxpy_add(j - ib, &y[ib], &col[ib], xj);
        }
        for (int j = ie; j < n; ++j) {                  /* above-tile columns, ascending j */
            const T xj = x[j];
            if (ceq0(xj)) continue;
            const T *col = &ap[kk_upper(j)];
            mf_kernels::caxpy_add(ie - ib, &y[ib], &col[ib], xj);
        }
    }
    for (int i = 0; i < n; ++i) x[i] = y[i];
}

/* Contiguous (incx==1) serial core. NoTrans is a column AXPY (caxpy_add, bit-
 * exact); Trans/ConjTrans is a packed-column complex dot via the AVX2 wide
 * wdotu/wdotc kernel (reorders → within fuzz tol). Strided callers gather x to
 * a contiguous scratch, run this, and scatter back. */
static void wtpmv_serial_contig(bool upper, bool trans, bool noconj,
                                bool nounit, int N, const T *ap, T *x) {
    if (!trans) {
        if (upper) {
            if (N >= 128) {
                wtpmv_serial_N_upper(nounit, N, ap, x);
            } else {
                int kk = 0;
                for (int j = 0; j < N; ++j) {
                    if (!ceq0(x[j])) {
                        const T tmp = x[j];
                        mf_kernels::caxpy_add(j, &x[0], &ap[kk], tmp);
                        if (nounit) x[j] = cmul(x[j], ap[kk + j]);
                    }
                    kk += j + 1;
                }
            }
        } else {
            int kk = (N * (N + 1)) / 2 - 1;
            for (int j = N - 1; j >= 0; --j) {
                if (!ceq0(x[j])) {
                    const T tmp = x[j];
                    mf_kernels::caxpy_add(N - 1 - j, &x[j + 1], &ap[kk - (N - 2 - j)], tmp);
                    if (nounit) x[j] = cmul(x[j], ap[kk - (N - 1 - j)]);
                }
                kk -= (N - j);
            }
        }
    } else {
        if (upper) {
            int kk = (N * (N + 1)) / 2 - 1;              /* diag of column j */
            for (int j = N - 1; j >= 0; --j) {
                T dot = noconj ? mf_kernels::wdotu_unit(j, &ap[kk - j], x)
                               : mf_kernels::wdotc_unit(j, &ap[kk - j], x);
                T r = nounit ? cmul(x[j], (noconj ? ap[kk] : cconj(ap[kk]))) : x[j];
                x[j] = cadd(r, dot);
                kk -= j + 1;
            }
        } else {
            int kk = 0;                                  /* diag of column j */
            for (int j = 0; j < N; ++j) {
                const int len = N - 1 - j;
                T dot = noconj ? mf_kernels::wdotu_unit(len, &ap[kk + 1], &x[j + 1])
                               : mf_kernels::wdotc_unit(len, &ap[kk + 1], &x[j + 1]);
                T r = nounit ? cmul(x[j], (noconj ? ap[kk] : cconj(ap[kk]))) : x[j];
                x[j] = cadd(r, dot);
                kk += N - j;
            }
        }
    }
}
}

#ifdef _OPENMP
namespace {
const T zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };
using mf_packed::kk_upper;
using mf_packed::kk_lower;
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
                T dot = upper
                    ? (conj ? mf_kernels::wdotc_unit(j, &aj[0], &x[0])
                            : mf_kernels::wdotu_unit(j, &aj[0], &x[0]))
                    : (conj ? mf_kernels::wdotc_unit(n - j - 1, &aj[1], &x[j + 1])
                            : mf_kernels::wdotu_unit(n - j - 1, &aj[1], &x[j + 1]));
                y_buf[j] = cadd(s, dot);
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
                    if (!ceq0(xj)) mf_kernels::caxpy_add(j, &y_priv[0], &aj[0], xj);
                    y_priv[j] = cadd(y_priv[j], cmul(xj, nounit ? aj[j] : one_cdd));
                } else {
                    const T *aj = &ap[kk_lower(j, n)];
                    y_priv[j] = cadd(y_priv[j], cmul(xj, nounit ? aj[0] : one_cdd));
                    if (!ceq0(xj)) mf_kernels::caxpy_add(n - j - 1, &y_priv[j + 1], &aj[1], xj);
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
        wtpmv_serial_contig(UPLO == 'U', TR != 'N', noconj != 0, nounit != 0, N, ap, x);
        return;
    }

    /* Strided: gather x to a contiguous scratch, run the (SIMD) contiguous core,
     * scatter back. O(N) gather/scatter vs the O(N^2) packed sweep. */
    T *xbase = (incx < 0) ? x - (std::ptrdiff_t)(N - 1) * incx : x;
    std::vector<T> xs(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) xs[i] = xbase[(std::ptrdiff_t)i * incx];
    wtpmv_serial_contig(UPLO == 'U', TR != 'N', noconj != 0, nounit != 0, N, ap, xs.data());
    for (int i = 0; i < N; ++i) xbase[(std::ptrdiff_t)i * incx] = xs[i];
}
