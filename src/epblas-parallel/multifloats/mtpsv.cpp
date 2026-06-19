/* mtpsv — multifloats real DD triangular packed solve.
 *   x := inv(A)*x or inv(A^T)*x
 */

#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <multifloats.h>
#include "mf_tri_simd.h"   /* mf_tri::axpy_sub / dot — shared SoA loop kernels */
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define MTPSV_OMP_MIN  256   /* below this, run the bit-exact serial path */
#define MTPSV_BLK      128   /* diagonal-block size for the blocked solve */
#define MTPSV_MAX_CPUS 256
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
inline bool dd_iszero(const T &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
const T zero_dd{0.0, 0.0};

/* Column base offsets into the packed array (column-major triangle).
 *   Lower: column j starts at its diagonal (row j); element (i,j) i>=j at base+(i-j).
 *   Upper: column j starts at row 0;            element (i,j) i<=j at base+i.       */
inline std::size_t cbL(int j, int N) {
    return static_cast<std::size_t>(j) * static_cast<std::size_t>(N)
         - static_cast<std::size_t>(j) * static_cast<std::size_t>(j - 1) / 2;
}
inline std::size_t cbU(int j) {
    return static_cast<std::size_t>(j) * static_cast<std::size_t>(j + 1) / 2;
}
}

/* In-place contiguous (incx==1) packed triangular solve. Per column j, &ap[cb*]
 * is a contiguous run, so this is the packed twin of mtbsv_contig: NoTrans is a
 * band AXPY of the solved x[j] into the trailing/leading rows (mf_tri::axpy_sub,
 * order-free -> bit-exact); Trans is a column dot against the already-solved x
 * (mf_tri::dot, vector accumulate + hreduce -> within tolerance). The diagonal
 * divide and cross-column recurrence stay scalar/exact. The strided entry
 * gathers into contiguous scratch and reuses this. */
static void mtpsv_serial_contig(char UPLO, char TR, int nounit,
                                std::ptrdiff_t n, const T *ap, T *x)
{
    if (TR == 'N') {
        if (UPLO == 'U') {
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                if (!dd_iszero(x[j])) {
                    const T *aj = &ap[cbU(j)];
                    if (nounit) x[j] = x[j] / aj[j];
                    mf_tri::axpy_sub(j, &x[0], &aj[0], x[j]);
                }
            }
        } else {
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                if (!dd_iszero(x[j])) {
                    const T *aj = &ap[cbL((int)j, (int)n)];
                    if (nounit) x[j] = x[j] / aj[0];
                    mf_tri::axpy_sub(n - 1 - j, &x[j + 1], &aj[1], x[j]);
                }
            }
        }
    } else {
        if (UPLO == 'U') {
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const T *aj = &ap[cbU(j)];
                T tmp = x[j] - mf_tri::dot(j, &aj[0], &x[0]);
                if (nounit) tmp = tmp / aj[j];
                x[j] = tmp;
            }
        } else {
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                const T *aj = &ap[cbL((int)j, (int)n)];
                T tmp = x[j] - mf_tri::dot(n - 1 - j, &aj[1], &x[j + 1]);
                if (nounit) tmp = tmp / aj[0];
                x[j] = tmp;
            }
        }
    }
}

/* Bit-exact serial path (verbatim reference). Also reused as the <threshold /
 * incx!=1 fallback. TR is already normalized ('C' folded to 'T' by the caller). */
static void mtpsv_serial(char UPLO, char TR, int nounit,
                         int N, const T *ap, T *x, int incx)
{
    if (incx == 1) {
        mtpsv_serial_contig(UPLO, TR, nounit, N, ap, x);
        return;
    }

    /* Strided: gather x into contiguous scratch, run the stride-1 solve (which
     * beats the ob clone's indexed strided walk), scatter back. The loop-carried
     * dependence lives in the contiguous core; gather/scatter only re-lays-out x,
     * bit-identical. Stack scratch for small N, heap past it; direct strided walk
     * kept only as a heap-alloc-failure fallback. */
    {
        const std::ptrdiff_t n = N, sx = incx;
        const std::ptrdiff_t kx0 = (sx < 0) ? -(n - 1) * sx : 0;
        T stackbuf[512];
        T *heap = nullptr;
        T *xc = (n <= 512) ? stackbuf
                           : (heap = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T))));
        if (xc) {
            std::ptrdiff_t ix = kx0;
            for (std::ptrdiff_t k = 0; k < n; ++k) { xc[k] = x[ix]; ix += sx; }
            mtpsv_serial_contig(UPLO, TR, nounit, n, ap, xc);
            ix = kx0;
            for (std::ptrdiff_t k = 0; k < n; ++k) { x[ix] = xc[k]; ix += sx; }
            std::free(heap);
            return;
        }
        std::free(heap);
    }

    {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'U') {
                int kk = (N * (N + 1)) / 2 - 1;
                int jx = kx + (N - 1) * incx;
                for (int j = N - 1; j >= 0; --j) {
                    if (!dd_iszero(x[jx])) {
                        if (nounit) x[jx] = x[jx] / ap[kk];
                        const T tmp = x[jx];
                        int ix = jx;
                        for (int k = kk - 1; k >= kk - j; --k) {
                            ix -= incx;
                            x[ix] = x[ix] - tmp * ap[k];
                        }
                    }
                    jx -= incx;
                    kk -= j + 1;
                }
            } else {
                int kk = 0;
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    if (!dd_iszero(x[jx])) {
                        if (nounit) x[jx] = x[jx] / ap[kk];
                        const T tmp = x[jx];
                        int ix = jx;
                        for (int k = kk + 1; k < kk + N - j; ++k) {
                            ix += incx;
                            x[ix] = x[ix] - tmp * ap[k];
                        }
                    }
                    jx += incx;
                    kk += N - j;
                }
            }
        } else {
            if (UPLO == 'U') {
                int kk = 0;
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    T tmp = x[jx];
                    int ix = kx;
                    for (int k = kk; k < kk + j; ++k) {
                        tmp = tmp - ap[k] * x[ix];
                        ix += incx;
                    }
                    if (nounit) tmp = tmp / ap[kk + j];
                    x[jx] = tmp;
                    jx += incx;
                    kk += j + 1;
                }
            } else {
                int kk = (N * (N + 1)) / 2 - 1;
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    int ix = kx;
                    for (int k = kk; k > kk - (N - 1 - j); --k) {
                        tmp = tmp - ap[k] * x[ix];
                        ix -= incx;
                    }
                    if (nounit) tmp = tmp / ap[kk - (N - 1 - j)];
                    x[jx] = tmp;
                    jx -= incx;
                    kk -= (N - j);
                }
            }
        }
    }
}

#ifdef _OPENMP
/* Solve a single diagonal block [j0,j1) in packed storage (scalar, within-block
 * coupling only). Used by the threaded path; need only match serial within DD
 * fuzz tol (the threaded result is regrouped anyway). */
static void mtpsv_block(char UPLO, char TR, int nounit,
                        int j0, int j1, int N, const T *ap, T *x)
{
    const bool lower = (UPLO == 'L');
    if (TR == 'N') {
        if (!lower) {                                   /* Upper: backward */
            for (int j = j1 - 1; j >= j0; --j) {
                if (dd_iszero(x[j])) continue;
                const std::size_t b = cbU(j);
                if (nounit) x[j] = x[j] / ap[b + j];
                mf_tri::axpy_sub(j - j0, &x[j0], &ap[b + j0], x[j]);
            }
        } else {                                        /* Lower: forward */
            for (int j = j0; j < j1; ++j) {
                if (dd_iszero(x[j])) continue;
                const std::size_t b = cbL(j, N);
                if (nounit) x[j] = x[j] / ap[b];
                mf_tri::axpy_sub(j1 - (j + 1), &x[j + 1], &ap[b + 1], x[j]);
            }
        }
    } else {
        if (!lower) {                                   /* Upper^T: forward, k<j */
            for (int j = j0; j < j1; ++j) {
                const std::size_t b = cbU(j);
                T tmp = x[j] - mf_tri::dot(j - j0, &ap[b + j0], &x[j0]);
                if (nounit) tmp = tmp / ap[b + j];
                x[j] = tmp;
            }
        } else {                                        /* Lower^T: backward, k>j */
            for (int j = j1 - 1; j >= j0; --j) {
                const std::size_t b = cbL(j, N);
                T tmp = x[j] - mf_tri::dot(j1 - (j + 1), &ap[b + 1], &x[j + 1]);
                if (nounit) tmp = tmp / ap[b];
                x[j] = tmp;
            }
        }
    }
}

/* Blocked threaded packed solve, incx==1 only. Loop-carried dependence confined
 * to small MTPSV_BLK diagonal blocks (solved serially); the bulk O(N^2)
 * off-diagonal coupling is threaded over disjoint output rows. Returns true if
 * it handled the call. */
__attribute__((noinline)) static bool mtpsv_omp(
    char UPLO, char TR, int nounit, int N, const T *ap, T *x)
{
    if (N < MTPSV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MTPSV_MAX_CPUS) nthreads = MTPSV_MAX_CPUS;
    const bool lower = (UPLO == 'L');
    const bool trans = (TR != 'N');

    if (!trans) {
        /* NoTrans: axpy form. Solve a diagonal block, then propagate its solved
         * columns into the not-yet-solved rows (trailing for L, leading for U). */
        if (lower) {
            for (int j0 = 0; j0 < N; j0 += MTPSV_BLK) {
                int j1 = j0 + MTPSV_BLK; if (j1 > N) j1 = N;
                mtpsv_block(UPLO, TR, nounit, j0, j1, N, ap, x);
                if (j1 >= N) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    int tid = omp_get_thread_num();
                    int rlo = j1 + (int)((long long)(N - j1) * tid / nthreads);
                    int rhi = j1 + (int)((long long)(N - j1) * (tid + 1) / nthreads);
                    for (int i = j0; i < j1; ++i) {
                        const T xi = x[i];
                        if (dd_iszero(xi)) continue;
                        const T *col = &ap[cbL(i, N)];      /* col[k-i] = A(k,i) */
                        mf_tri::axpy_sub(rhi - rlo, &x[rlo], &col[rlo - i], xi);
                    }
                }
            }
        } else {
            for (int j1 = N; j1 > 0; j1 -= MTPSV_BLK) {
                int j0 = j1 - MTPSV_BLK; if (j0 < 0) j0 = 0;
                mtpsv_block(UPLO, TR, nounit, j0, j1, N, ap, x);
                if (j0 <= 0) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    int tid = omp_get_thread_num();
                    int rlo = (int)((long long)j0 * tid / nthreads);
                    int rhi = (int)((long long)j0 * (tid + 1) / nthreads);
                    for (int i = j0; i < j1; ++i) {
                        const T xi = x[i];
                        if (dd_iszero(xi)) continue;
                        const T *col = &ap[cbU(i)];         /* col[k] = A(k,i) */
                        mf_tri::axpy_sub(rhi - rlo, &x[rlo], &col[rlo], xi);
                    }
                }
            }
        }
    } else {
        /* Trans: dot form. Fold the already-solved out-of-block tail/head into
         * the block rows (threaded, disjoint rows), then solve the diagonal
         * block serially (within-block coupling + divide). */
        if (lower) {                                       /* backward, k > j */
            for (int j1 = N; j1 > 0; j1 -= MTPSV_BLK) {
                int j0 = j1 - MTPSV_BLK; if (j0 < 0) j0 = 0;
                if (j1 < N) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();
                        int ilo = j0 + (int)((long long)(j1 - j0) * tid / nthreads);
                        int ihi = j0 + (int)((long long)(j1 - j0) * (tid + 1) / nthreads);
                        for (int i = ilo; i < ihi; ++i) {
                            const T *col = &ap[cbL(i, N)];
                            x[i] = x[i] - mf_tri::dot(N - j1, &col[j1 - i], &x[j1]);
                        }
                    }
                }
                mtpsv_block(UPLO, TR, nounit, j0, j1, N, ap, x);
            }
        } else {                                           /* forward, k < j */
            for (int j0 = 0; j0 < N; j0 += MTPSV_BLK) {
                int j1 = j0 + MTPSV_BLK; if (j1 > N) j1 = N;
                if (j0 > 0) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();
                        int ilo = j0 + (int)((long long)(j1 - j0) * tid / nthreads);
                        int ihi = j0 + (int)((long long)(j1 - j0) * (tid + 1) / nthreads);
                        for (int i = ilo; i < ihi; ++i) {
                            const T *col = &ap[cbU(i)];
                            x[i] = x[i] - mf_tri::dot(j0, &col[0], &x[0]);
                        }
                    }
                }
                mtpsv_block(UPLO, TR, nounit, j0, j1, N, ap, x);
            }
        }
    }
    return true;
}
#endif

extern "C" void mtpsv_(
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
    if (incx == 1 && N >= MTPSV_OMP_MIN && blas_omp_max_threads() > 1
        && mtpsv_omp(UPLO, TR, nounit, N, ap, x))
        return;
#endif

    mtpsv_serial(UPLO, TR, nounit, N, ap, x, incx);
}
