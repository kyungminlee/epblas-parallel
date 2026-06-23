/* wtpsv — multifloats complex DD triangular packed solve. */

#include <cstddef>
#include <cctype>
#include <vector>
#include <multifloats.h>
#include "mf_util.h"
#include "mf_pred.h"
#include "mf_kernels.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define WTPSV_OMP_MIN  256   /* below this, run the bit-exact serial path */
#define WTPSV_BLK      128   /* diagonal-block size for the blocked solve */
#define WTPSV_MAX_CPUS 256
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using R = mf::float64x2;
using TC = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
const R rzero{0.0, 0.0};
using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::csub;
using mf_kernels::cconj;
inline TC cdiv(TC const &a, TC const &b) {
    /* a / b = a·conj(b) / |b|², direct DD divide (canonical form shared with
     * wtbsv/wtrsv/wtrsm_serial — see F2, simd_audit). */
    const R denom = b.re * b.re + b.im * b.im;
    return TC{ (a.re * b.re + a.im * b.im) / denom,
              (a.im * b.re - a.re * b.im) / denom };
}
const TC czero{rzero, rzero};

#ifdef _OPENMP
/* Column base offsets into the packed array (column-major triangle).
 *   Lower: column j starts at its diagonal (row j); element (i,j) i>=j at base+(i-j).
 *   Upper: column j starts at row 0;            element (i,j) i<=j at base+i.       */
inline std::size_t cbL(std::ptrdiff_t j, std::ptrdiff_t n) {
    return static_cast<std::size_t>(j) * static_cast<std::size_t>(n)
         - static_cast<std::size_t>(j) * static_cast<std::size_t>(j - 1) / 2;
}
inline std::size_t cbU(std::ptrdiff_t j) {
    return static_cast<std::size_t>(j) * static_cast<std::size_t>(j + 1) / 2;
}
inline TC melem(const TC &a, bool noconj) { return noconj ? a : cconj(a); }
#endif
}

/* Contiguous (incx==1) serial core. NoTrans elimination is a column AXPY-sub
 * (caxpy_sub, bit-exact); Trans/ConjTrans is a packed-column complex dot
 * (cdot, reorders -> within fuzz tol). The cross-column recurrence (divide by
 * the diagonal, feed the result forward) stays scalar here. */
static void wtpsv_serial_contig(char UPLO, char TRANS, bool noconj, bool nounit,
                                std::ptrdiff_t n, const TC *ap, TC *x)
{
    if (TRANS == 'N') {
        if (UPLO == 'U') {
            std::ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                if (!ceq0(x[j])) {
                    if (nounit) x[j] = cdiv(x[j], ap[kk]);
                    mf_kernels::caxpy_sub(j, &x[0], &ap[kk - j], x[j]);
                }
                kk -= j + 1;
            }
        } else {
            std::ptrdiff_t kk = 0;
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                if (!ceq0(x[j])) {
                    if (nounit) x[j] = cdiv(x[j], ap[kk]);
                    mf_kernels::caxpy_sub(n - 1 - j, &x[j + 1], &ap[kk + 1], x[j]);
                }
                kk += n - j;
            }
        }
    } else {
        const bool conj = (noconj == 0);
        if (UPLO == 'U') {
            std::ptrdiff_t kk = 0;
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                TC tmp = csub(x[j], mf_kernels::cdot(j, &ap[kk], &x[0], conj));
                if (nounit) tmp = cdiv(tmp, (noconj ? ap[kk + j] : cconj(ap[kk + j])));
                x[j] = tmp;
                kk += j + 1;
            }
        } else {
            std::ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                TC tmp = csub(x[j], mf_kernels::cdot(n - 1 - j, &ap[kk - (n - 2 - j)], &x[j + 1], conj));
                if (nounit) tmp = cdiv(tmp, (noconj ? ap[kk - (n - 1 - j)] : cconj(ap[kk - (n - 1 - j)])));
                x[j] = tmp;
                kk -= (n - j);
            }
        }
    }
}

/* Serial entry / <threshold / strided fallback. Strided gathers x to a
 * contiguous scratch, runs the SIMD core, and scatters back. */
static void wtpsv_serial(char UPLO, char TRANS, bool noconj, bool nounit,
                         std::ptrdiff_t n, const TC *ap, TC *x, std::ptrdiff_t incx)
{
    if (incx == 1) {
        wtpsv_serial_contig(UPLO, TRANS, noconj, nounit, n, ap, x);
        return;
    }
    TC *xbase = (incx < 0) ? x - (std::ptrdiff_t)(n - 1) * incx : x;
    std::vector<TC> xs(static_cast<std::size_t>(n));
    for (std::ptrdiff_t i = 0; i < n; ++i) xs[i] = xbase[(std::ptrdiff_t)i * incx];
    wtpsv_serial_contig(UPLO, TRANS, noconj, nounit, n, ap, xs.data());
    for (std::ptrdiff_t i = 0; i < n; ++i) xbase[(std::ptrdiff_t)i * incx] = xs[i];
}

#ifdef _OPENMP
/* Solve a single diagonal block [j0,j1) in packed storage (within-block coupling
 * only). Threaded path need only match serial within DD fuzz tol. */
static void wtpsv_block(char UPLO, char TRANS, bool noconj, bool nounit,
                        std::ptrdiff_t j0, std::ptrdiff_t j1, std::ptrdiff_t n, const TC *ap, TC *x)
{
    const bool lower = (UPLO == 'L');
    const bool conj = (noconj == 0);
    if (TRANS == 'N') {
        if (!lower) {                                   /* Upper: backward */
            for (std::ptrdiff_t j = j1 - 1; j >= j0; --j) {
                if (ceq0(x[j])) continue;
                const std::size_t b = cbU(j);
                if (nounit) x[j] = cdiv(x[j], ap[b + j]);
                mf_kernels::caxpy_sub(j - j0, &x[j0], &ap[b + j0], x[j]);
            }
        } else {                                        /* Lower: forward */
            for (std::ptrdiff_t j = j0; j < j1; ++j) {
                if (ceq0(x[j])) continue;
                const std::size_t b = cbL(j, n);
                if (nounit) x[j] = cdiv(x[j], ap[b]);
                mf_kernels::caxpy_sub(j1 - (j + 1), &x[j + 1], &ap[b + 1], x[j]);
            }
        }
    } else {
        if (!lower) {                                   /* Upper^(T/C): forward, k<j */
            for (std::ptrdiff_t j = j0; j < j1; ++j) {
                const std::size_t b = cbU(j);
                TC tmp = csub(x[j], mf_kernels::cdot(j - j0, &ap[b + j0], &x[j0], conj));
                if (nounit) tmp = cdiv(tmp, melem(ap[b + j], noconj));
                x[j] = tmp;
            }
        } else {                                        /* Lower^(T/C): backward, k>j */
            for (std::ptrdiff_t j = j1 - 1; j >= j0; --j) {
                const std::size_t b = cbL(j, n);
                TC tmp = csub(x[j], mf_kernels::cdot(j1 - (j + 1), &ap[b + 1], &x[j + 1], conj));
                if (nounit) tmp = cdiv(tmp, melem(ap[b], noconj));
                x[j] = tmp;
            }
        }
    }
}

/* Blocked threaded packed solve, incx==1 only. Loop-carried dependence confined
 * to small WTPSV_BLK diagonal blocks; bulk O(N^2) off-diagonal coupling threaded
 * over disjoint output rows. Returns true if it handled the call. */
__attribute__((noinline)) static bool wtpsv_omp(
    char UPLO, char TRANS, bool noconj, bool nounit, std::ptrdiff_t n, const TC *ap, TC *x)
{
    if (n < WTPSV_OMP_MIN || !blas_omp_should_thread())
        return false;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > WTPSV_MAX_CPUS) nthreads = WTPSV_MAX_CPUS;
    const bool lower = (UPLO == 'L');
    const bool trans = (TRANS != 'N');
    const bool conj = (noconj == 0);

    if (!trans) {
        if (lower) {
            for (std::ptrdiff_t j0 = 0; j0 < n; j0 += WTPSV_BLK) {
                std::ptrdiff_t j1 = j0 + WTPSV_BLK; if (j1 > n) j1 = n;
                wtpsv_block(UPLO, TRANS, noconj, nounit, j0, j1, n, ap, x);
                if (j1 >= n) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    std::ptrdiff_t tid = omp_get_thread_num();
                    std::ptrdiff_t rlo = j1 + blas_part_bound(n - j1, tid, nthreads);
                    std::ptrdiff_t rhi = j1 + blas_part_bound(n - j1, tid + 1, nthreads);
                    for (std::ptrdiff_t i = j0; i < j1; ++i) {
                        const TC xi = x[i];
                        if (ceq0(xi)) continue;
                        const TC *col = &ap[cbL(i, n)];      /* col[k-i] = A(k,i) */
                        mf_kernels::caxpy_sub(rhi - rlo, &x[rlo], &col[rlo - i], xi);
                    }
                }
            }
        } else {
            for (std::ptrdiff_t j1 = n; j1 > 0; j1 -= WTPSV_BLK) {
                std::ptrdiff_t j0 = j1 - WTPSV_BLK; if (j0 < 0) j0 = 0;
                wtpsv_block(UPLO, TRANS, noconj, nounit, j0, j1, n, ap, x);
                if (j0 <= 0) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    std::ptrdiff_t tid = omp_get_thread_num();
                    std::ptrdiff_t rlo = blas_part_bound(j0, tid, nthreads);
                    std::ptrdiff_t rhi = blas_part_bound(j0, tid + 1, nthreads);
                    for (std::ptrdiff_t i = j0; i < j1; ++i) {
                        const TC xi = x[i];
                        if (ceq0(xi)) continue;
                        const TC *col = &ap[cbU(i)];         /* col[k] = A(k,i) */
                        mf_kernels::caxpy_sub(rhi - rlo, &x[rlo], &col[rlo], xi);
                    }
                }
            }
        }
    } else {
        if (lower) {                                       /* backward, k > j */
            for (std::ptrdiff_t j1 = n; j1 > 0; j1 -= WTPSV_BLK) {
                std::ptrdiff_t j0 = j1 - WTPSV_BLK; if (j0 < 0) j0 = 0;
                if (j1 < n) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        std::ptrdiff_t tid = omp_get_thread_num();
                        std::ptrdiff_t ilo = j0 + blas_part_bound(j1 - j0, tid, nthreads);
                        std::ptrdiff_t ihi = j0 + blas_part_bound(j1 - j0, tid + 1, nthreads);
                        for (std::ptrdiff_t i = ilo; i < ihi; ++i) {
                            const TC *col = &ap[cbL(i, n)];
                            x[i] = csub(x[i], mf_kernels::cdot(n - j1, &col[j1 - i], &x[j1], conj));
                        }
                    }
                }
                wtpsv_block(UPLO, TRANS, noconj, nounit, j0, j1, n, ap, x);
            }
        } else {                                           /* forward, k < j */
            for (std::ptrdiff_t j0 = 0; j0 < n; j0 += WTPSV_BLK) {
                std::ptrdiff_t j1 = j0 + WTPSV_BLK; if (j1 > n) j1 = n;
                if (j0 > 0) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        std::ptrdiff_t tid = omp_get_thread_num();
                        std::ptrdiff_t ilo = j0 + blas_part_bound(j1 - j0, tid, nthreads);
                        std::ptrdiff_t ihi = j0 + blas_part_bound(j1 - j0, tid + 1, nthreads);
                        for (std::ptrdiff_t i = ilo; i < ihi; ++i) {
                            const TC *col = &ap[cbU(i)];
                            x[i] = csub(x[i], mf_kernels::cdot(j0, &col[0], &x[0], conj));
                        }
                    }
                }
                wtpsv_block(UPLO, TRANS, noconj, nounit, j0, j1, n, ap, x);
            }
        }
    }
    return true;
}
#endif

static void wtpsv_core(
    char uplo, char trans, char diag,
    std::ptrdiff_t n,
    const TC *ap,
    TC *x, std::ptrdiff_t incx)
{
    const char UPLO = up(&uplo);
    const char TRANS = up(&trans);
    const bool noconj = (TRANS == 'T');
    const bool nounit = (up(&diag) != 'U');

    if (n == 0) return;

#ifdef _OPENMP
    if (incx == 1 && n >= WTPSV_OMP_MIN && blas_omp_available()
        && wtpsv_omp(UPLO, TRANS, noconj, nounit, n, ap, x))
        return;
#endif

    wtpsv_serial(UPLO, TRANS, noconj, nounit, n, ap, x, incx);
}

extern "C" {
EPBLAS_FACADE_TPMV(wtpsv, TC)
}
