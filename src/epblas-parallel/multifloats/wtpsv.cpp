/* wtpsv — multifloats complex DD triangular packed solve. */

#include <cstddef>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define WTPSV_OMP_MIN  256   /* below this, run the bit-exact serial path */
#define WTPSV_BLK      128   /* diagonal-block size for the blocked solve */
#define WTPSV_MAX_CPUS 256
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
inline T csub(T const &a, T const &b) { return T{ a.re - b.re, a.im - b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }
inline T cdiv(T const &a, T const &b) {
    const R d = b.re * b.re + b.im * b.im;
    const R inv_d = R{1.0, 0.0} / d;
    return T{ (a.re * b.re + a.im * b.im) * inv_d,
              (a.im * b.re - a.re * b.im) * inv_d };
}
const T czero{rzero, rzero};

#ifdef _OPENMP
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
inline T melem(const T &a, int noconj) { return noconj ? a : cconj(a); }
#endif
}

/* Bit-exact serial path (verbatim reference). Also reused as the <threshold /
 * incx!=1 fallback. */
static void wtpsv_serial(char UPLO, char TR, int noconj, int nounit,
                         int N, const T *ap, T *x, int incx)
{
    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'U') {
                int kk = (N * (N + 1)) / 2 - 1;
                for (int j = N - 1; j >= 0; --j) {
                    if (!cdd_iszero(x[j])) {
                        if (nounit) x[j] = cdiv(x[j], ap[kk]);
                        const T tmp = x[j];
                        int k = kk - 1;
                        for (int i = j - 1; i >= 0; --i) { x[i] = csub(x[i], cmul(tmp, ap[k])); --k; }
                    }
                    kk -= j + 1;
                }
            } else {
                int kk = 0;
                for (int j = 0; j < N; ++j) {
                    if (!cdd_iszero(x[j])) {
                        if (nounit) x[j] = cdiv(x[j], ap[kk]);
                        const T tmp = x[j];
                        int k = kk + 1;
                        for (int i = j + 1; i < N; ++i) { x[i] = csub(x[i], cmul(tmp, ap[k])); ++k; }
                    }
                    kk += N - j;
                }
            }
        } else {
            if (UPLO == 'U') {
                int kk = 0;
                for (int j = 0; j < N; ++j) {
                    T tmp = x[j];
                    int k = kk;
                    if (noconj) for (int i = 0; i < j; ++i) { tmp = csub(tmp, cmul(ap[k], x[i])); ++k; }
                    else        for (int i = 0; i < j; ++i) { tmp = csub(tmp, cmul(cconj(ap[k]), x[i])); ++k; }
                    if (nounit) tmp = cdiv(tmp, (noconj ? ap[kk + j] : cconj(ap[kk + j])));
                    x[j] = tmp;
                    kk += j + 1;
                }
            } else {
                int kk = (N * (N + 1)) / 2 - 1;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    int k = kk;
                    if (noconj) for (int i = N - 1; i > j; --i) { tmp = csub(tmp, cmul(ap[k], x[i])); --k; }
                    else        for (int i = N - 1; i > j; --i) { tmp = csub(tmp, cmul(cconj(ap[k]), x[i])); --k; }
                    if (nounit) tmp = cdiv(tmp, (noconj ? ap[kk - (N - 1 - j)] : cconj(ap[kk - (N - 1 - j)])));
                    x[j] = tmp;
                    kk -= (N - j);
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'U') {
                int kk = (N * (N + 1)) / 2 - 1;
                int jx = kx + (N - 1) * incx;
                for (int j = N - 1; j >= 0; --j) {
                    if (!cdd_iszero(x[jx])) {
                        if (nounit) x[jx] = cdiv(x[jx], ap[kk]);
                        const T tmp = x[jx];
                        int ix = jx;
                        for (int k = kk - 1; k >= kk - j; --k) {
                            ix -= incx;
                            x[ix] = csub(x[ix], cmul(tmp, ap[k]));
                        }
                    }
                    jx -= incx;
                    kk -= j + 1;
                }
            } else {
                int kk = 0;
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    if (!cdd_iszero(x[jx])) {
                        if (nounit) x[jx] = cdiv(x[jx], ap[kk]);
                        const T tmp = x[jx];
                        int ix = jx;
                        for (int k = kk + 1; k < kk + N - j; ++k) {
                            ix += incx;
                            x[ix] = csub(x[ix], cmul(tmp, ap[k]));
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
                        tmp = csub(tmp, cmul((noconj ? ap[k] : cconj(ap[k])), x[ix]));
                        ix += incx;
                    }
                    if (nounit) tmp = cdiv(tmp, (noconj ? ap[kk + j] : cconj(ap[kk + j])));
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
                        tmp = csub(tmp, cmul((noconj ? ap[k] : cconj(ap[k])), x[ix]));
                        ix -= incx;
                    }
                    if (nounit) tmp = cdiv(tmp, (noconj ? ap[kk - (N - 1 - j)] : cconj(ap[kk - (N - 1 - j)])));
                    x[jx] = tmp;
                    jx -= incx;
                    kk -= (N - j);
                }
            }
        }
    }
}

#ifdef _OPENMP
/* Solve a single diagonal block [j0,j1) in packed storage (within-block coupling
 * only). Threaded path need only match serial within DD fuzz tol. */
static void wtpsv_block(char UPLO, char TR, int noconj, int nounit,
                        int j0, int j1, int N, const T *ap, T *x)
{
    const bool lower = (UPLO == 'L');
    if (TR == 'N') {
        if (!lower) {                                   /* Upper: backward */
            for (int j = j1 - 1; j >= j0; --j) {
                if (cdd_iszero(x[j])) continue;
                const std::size_t b = cbU(j);
                if (nounit) x[j] = cdiv(x[j], ap[b + j]);
                const T tmp = x[j];
                for (int i = j0; i < j; ++i) x[i] = csub(x[i], cmul(tmp, ap[b + i]));
            }
        } else {                                        /* Lower: forward */
            for (int j = j0; j < j1; ++j) {
                if (cdd_iszero(x[j])) continue;
                const std::size_t b = cbL(j, N);
                if (nounit) x[j] = cdiv(x[j], ap[b]);
                const T tmp = x[j];
                for (int i = j + 1; i < j1; ++i) x[i] = csub(x[i], cmul(tmp, ap[b + (i - j)]));
            }
        }
    } else {
        if (!lower) {                                   /* Upper^(T/C): forward, k<j */
            for (int j = j0; j < j1; ++j) {
                const std::size_t b = cbU(j);
                T tmp = x[j];
                for (int i = j0; i < j; ++i) tmp = csub(tmp, cmul(melem(ap[b + i], noconj), x[i]));
                if (nounit) tmp = cdiv(tmp, melem(ap[b + j], noconj));
                x[j] = tmp;
            }
        } else {                                        /* Lower^(T/C): backward, k>j */
            for (int j = j1 - 1; j >= j0; --j) {
                const std::size_t b = cbL(j, N);
                T tmp = x[j];
                for (int i = j + 1; i < j1; ++i) tmp = csub(tmp, cmul(melem(ap[b + (i - j)], noconj), x[i]));
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
    char UPLO, char TR, int noconj, int nounit, int N, const T *ap, T *x)
{
    if (N < WTPSV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > WTPSV_MAX_CPUS) nthreads = WTPSV_MAX_CPUS;
    const bool lower = (UPLO == 'L');
    const bool trans = (TR != 'N');

    if (!trans) {
        if (lower) {
            for (int j0 = 0; j0 < N; j0 += WTPSV_BLK) {
                int j1 = j0 + WTPSV_BLK; if (j1 > N) j1 = N;
                wtpsv_block(UPLO, TR, noconj, nounit, j0, j1, N, ap, x);
                if (j1 >= N) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    int tid = omp_get_thread_num();
                    int rlo = j1 + (int)((long long)(N - j1) * tid / nthreads);
                    int rhi = j1 + (int)((long long)(N - j1) * (tid + 1) / nthreads);
                    for (int i = j0; i < j1; ++i) {
                        const T xi = x[i];
                        if (cdd_iszero(xi)) continue;
                        const T *col = &ap[cbL(i, N)];      /* col[k-i] = A(k,i) */
                        for (int k = rlo; k < rhi; ++k) x[k] = csub(x[k], cmul(xi, col[k - i]));
                    }
                }
            }
        } else {
            for (int j1 = N; j1 > 0; j1 -= WTPSV_BLK) {
                int j0 = j1 - WTPSV_BLK; if (j0 < 0) j0 = 0;
                wtpsv_block(UPLO, TR, noconj, nounit, j0, j1, N, ap, x);
                if (j0 <= 0) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    int tid = omp_get_thread_num();
                    int rlo = (int)((long long)j0 * tid / nthreads);
                    int rhi = (int)((long long)j0 * (tid + 1) / nthreads);
                    for (int i = j0; i < j1; ++i) {
                        const T xi = x[i];
                        if (cdd_iszero(xi)) continue;
                        const T *col = &ap[cbU(i)];         /* col[k] = A(k,i) */
                        for (int k = rlo; k < rhi; ++k) x[k] = csub(x[k], cmul(xi, col[k]));
                    }
                }
            }
        }
    } else {
        if (lower) {                                       /* backward, k > j */
            for (int j1 = N; j1 > 0; j1 -= WTPSV_BLK) {
                int j0 = j1 - WTPSV_BLK; if (j0 < 0) j0 = 0;
                if (j1 < N) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();
                        int ilo = j0 + (int)((long long)(j1 - j0) * tid / nthreads);
                        int ihi = j0 + (int)((long long)(j1 - j0) * (tid + 1) / nthreads);
                        for (int i = ilo; i < ihi; ++i) {
                            const T *col = &ap[cbL(i, N)];
                            T s = czero;
                            for (int k = j1; k < N; ++k) s = cadd(s, cmul(melem(col[k - i], noconj), x[k]));
                            x[i] = csub(x[i], s);
                        }
                    }
                }
                wtpsv_block(UPLO, TR, noconj, nounit, j0, j1, N, ap, x);
            }
        } else {                                           /* forward, k < j */
            for (int j0 = 0; j0 < N; j0 += WTPSV_BLK) {
                int j1 = j0 + WTPSV_BLK; if (j1 > N) j1 = N;
                if (j0 > 0) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();
                        int ilo = j0 + (int)((long long)(j1 - j0) * tid / nthreads);
                        int ihi = j0 + (int)((long long)(j1 - j0) * (tid + 1) / nthreads);
                        for (int i = ilo; i < ihi; ++i) {
                            const T *col = &ap[cbU(i)];
                            T s = czero;
                            for (int k = 0; k < j0; ++k) s = cadd(s, cmul(melem(col[k], noconj), x[k]));
                            x[i] = csub(x[i], s);
                        }
                    }
                }
                wtpsv_block(UPLO, TR, noconj, nounit, j0, j1, N, ap, x);
            }
        }
    }
    return true;
}
#endif

extern "C" void wtpsv_(
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
    if (incx == 1 && N >= WTPSV_OMP_MIN && blas_omp_max_threads() > 1
        && wtpsv_omp(UPLO, TR, noconj, nounit, N, ap, x))
        return;
#endif

    wtpsv_serial(UPLO, TR, noconj, nounit, N, ap, x, incx);
}
