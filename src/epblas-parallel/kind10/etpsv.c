/*
 * etpsv — kind10 (long double) triangular packed solve.
 *   x := inv(A)*x or inv(A^T)*x
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define ETPSV_OMP_MIN  256   /* below this, run the bit-exact serial path */
#define ETPSV_BLK      128   /* diagonal-block size for the blocked solve */
#define ETPSV_MAX_CPUS 256
#endif

#include "../common/epblas_facade.h"

typedef long double T;

static inline char up(char c) {
    return (char)toupper((unsigned char)c);
}

#ifdef _OPENMP
/* Column base offsets into the packed array (column-major triangle).
 *   Lower: column j starts at its diagonal (row j); element (i,j) i>=j at base+(i-j).
 *   Upper: column j starts at row 0;            element (i,j) i<=j at base+i.       */
static inline size_t cbL(ptrdiff_t j, ptrdiff_t N) {
    return (size_t)j * (size_t)N - (size_t)j * (size_t)(j - 1) / 2;
}
static inline size_t cbU(ptrdiff_t j) {
    return (size_t)j * (size_t)(j + 1) / 2;
}
#endif

/* Bit-exact serial path (verbatim reference). Also reused as the <threshold /
 * incx!=1 fallback. TR is already normalized ('C' folded to 'T' by the caller). */
static void etpsv_serial(char UPLO, char TR, ptrdiff_t nounit,
                         ptrdiff_t N, const T *restrict ap, T *restrict x, ptrdiff_t incx)
{
    const T zero = 0.0L;
    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'U') {
                ptrdiff_t kk = (N * (N + 1)) / 2 - 1;
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        if (nounit) x[j] /= ap[kk];
                        const T tmp = x[j];
                        ptrdiff_t k = kk - 1;
                        for (ptrdiff_t i = j - 1; i >= 0; --i) { x[i] -= tmp * ap[k]; --k; }
                    }
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < N; ++j) {
                    if (x[j] != zero) {
                        if (nounit) x[j] /= ap[kk];
                        const T tmp = x[j];
                        ptrdiff_t k = kk + 1;
                        for (ptrdiff_t i = j + 1; i < N; ++i) { x[i] -= tmp * ap[k]; ++k; }
                    }
                    kk += N - j;
                }
            }
        } else {
            if (UPLO == 'U') {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < N; ++j) {
                    /* Single-acc x87 dot — split into two parallel chains
                     * (Rule 22 / Addendum 21). */
                    T t0 = x[j], t1 = zero;
                    ptrdiff_t k = kk;
                    ptrdiff_t i = 0;
                    for (; i + 1 < j; i += 2) {
                        t0 -= ap[k]     * x[i];
                        t1 -= ap[k + 1] * x[i + 1];
                        k += 2;
                    }
                    if (i < j) { t0 -= ap[k] * x[i]; }
                    T tmp = t0 + t1;
                    if (nounit) tmp /= ap[kk + j];
                    x[j] = tmp;
                    kk += j + 1;
                }
            } else {
                ptrdiff_t kk = (N * (N + 1)) / 2 - 1;
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    ptrdiff_t k = kk;
                    for (ptrdiff_t i = N - 1; i > j; --i) { tmp -= ap[k] * x[i]; --k; }
                    if (nounit) tmp /= ap[kk - (N - 1 - j)];
                    x[j] = tmp;
                    kk -= (N - j);
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'U') {
                ptrdiff_t kk = (N * (N + 1)) / 2 - 1;
                ptrdiff_t jx = kx + (N - 1) * incx;
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    if (x[jx] != zero) {
                        if (nounit) x[jx] /= ap[kk];
                        const T tmp = x[jx];
                        ptrdiff_t ix = jx;
                        for (ptrdiff_t k = kk - 1; k >= kk - j; --k) {
                            ix -= incx;
                            x[ix] -= tmp * ap[k];
                        }
                    }
                    jx -= incx;
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t kk = 0;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < N; ++j) {
                    if (x[jx] != zero) {
                        if (nounit) x[jx] /= ap[kk];
                        const T tmp = x[jx];
                        ptrdiff_t ix = jx;
                        for (ptrdiff_t k = kk + 1; k < kk + N - j; ++k) {
                            ix += incx;
                            x[ix] -= tmp * ap[k];
                        }
                    }
                    jx += incx;
                    kk += N - j;
                }
            }
        } else {
            if (UPLO == 'U') {
                ptrdiff_t kk = 0;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T tmp = x[jx];
                    ptrdiff_t ix = kx;
                    for (ptrdiff_t k = kk; k < kk + j; ++k) {
                        tmp -= ap[k] * x[ix];
                        ix += incx;
                    }
                    if (nounit) tmp /= ap[kk + j];
                    x[jx] = tmp;
                    jx += incx;
                    kk += j + 1;
                }
            } else {
                ptrdiff_t kk = (N * (N + 1)) / 2 - 1;
                kx += (N - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    ptrdiff_t ix = kx;
                    for (ptrdiff_t k = kk; k > kk - (N - 1 - j); --k) {
                        tmp -= ap[k] * x[ix];
                        ix -= incx;
                    }
                    if (nounit) tmp /= ap[kk - (N - 1 - j)];
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
 * only). Threaded path need only match serial within fp80 fuzz tol. */
static void etpsv_block(char UPLO, char TR, ptrdiff_t nounit,
                        ptrdiff_t j0, ptrdiff_t j1, ptrdiff_t N,
                        const T *restrict ap, T *restrict x)
{
    const T zero = 0.0L;
    const int lower = (UPLO == 'L');
    if (TR == 'N') {
        if (!lower) {                                   /* Upper: backward */
            for (ptrdiff_t j = j1 - 1; j >= j0; --j) {
                if (x[j] == zero) continue;
                const size_t b = cbU(j);
                if (nounit) x[j] /= ap[b + j];
                const T tmp = x[j];
                for (ptrdiff_t i = j0; i < j; ++i) x[i] -= tmp * ap[b + i];
            }
        } else {                                        /* Lower: forward */
            for (ptrdiff_t j = j0; j < j1; ++j) {
                if (x[j] == zero) continue;
                const size_t b = cbL(j, N);
                if (nounit) x[j] /= ap[b];
                const T tmp = x[j];
                for (ptrdiff_t i = j + 1; i < j1; ++i) x[i] -= tmp * ap[b + (i - j)];
            }
        }
    } else {
        if (!lower) {                                   /* Upper^T: forward, k<j */
            for (ptrdiff_t j = j0; j < j1; ++j) {
                const size_t b = cbU(j);
                T tmp = x[j];
                for (ptrdiff_t i = j0; i < j; ++i) tmp -= ap[b + i] * x[i];
                if (nounit) tmp /= ap[b + j];
                x[j] = tmp;
            }
        } else {                                        /* Lower^T: backward, k>j */
            for (ptrdiff_t j = j1 - 1; j >= j0; --j) {
                const size_t b = cbL(j, N);
                T tmp = x[j];
                for (ptrdiff_t i = j + 1; i < j1; ++i) tmp -= ap[b + (i - j)] * x[i];
                if (nounit) tmp /= ap[b];
                x[j] = tmp;
            }
        }
    }
}

/* Blocked threaded packed solve, incx==1 only. Loop-carried dependence confined
 * to small ETPSV_BLK diagonal blocks (solved serially); the bulk O(N^2)
 * off-diagonal coupling is threaded over disjoint output rows. Returns 1 if it
 * handled the call. */
__attribute__((noinline)) static int etpsv_omp(
    char UPLO, char TR, ptrdiff_t nounit, ptrdiff_t N, const T *restrict ap, T *restrict x)
{
    if (N < ETPSV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > ETPSV_MAX_CPUS) nthreads = ETPSV_MAX_CPUS;
    const T zero = 0.0L;
    const int lower = (UPLO == 'L');
    const int trans = (TR != 'N');

    if (!trans) {
        if (lower) {
            for (ptrdiff_t j0 = 0; j0 < N; j0 += ETPSV_BLK) {
                ptrdiff_t j1 = j0 + ETPSV_BLK; if (j1 > N) j1 = N;
                etpsv_block(UPLO, TR, nounit, j0, j1, N, ap, x);
                if (j1 >= N) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    int tid = omp_get_thread_num();
                    ptrdiff_t rlo = j1 + blas_part_bound((N - j1), tid, nthreads);
                    ptrdiff_t rhi = j1 + blas_part_bound((N - j1), tid + 1, nthreads);
                    for (ptrdiff_t i = j0; i < j1; ++i) {
                        const T xi = x[i];
                        if (xi == zero) continue;
                        const T *restrict col = &ap[cbL(i, N)];     /* col[k-i] = A(k,i) */
                        for (ptrdiff_t k = rlo; k < rhi; ++k) x[k] -= xi * col[k - i];
                    }
                }
            }
        } else {
            for (ptrdiff_t j1 = N; j1 > 0; j1 -= ETPSV_BLK) {
                ptrdiff_t j0 = j1 - ETPSV_BLK; if (j0 < 0) j0 = 0;
                etpsv_block(UPLO, TR, nounit, j0, j1, N, ap, x);
                if (j0 <= 0) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    int tid = omp_get_thread_num();
                    ptrdiff_t rlo = blas_part_bound(j0, tid, nthreads);
                    ptrdiff_t rhi = blas_part_bound(j0, tid + 1, nthreads);
                    for (ptrdiff_t i = j0; i < j1; ++i) {
                        const T xi = x[i];
                        if (xi == zero) continue;
                        const T *restrict col = &ap[cbU(i)];        /* col[k] = A(k,i) */
                        for (ptrdiff_t k = rlo; k < rhi; ++k) x[k] -= xi * col[k];
                    }
                }
            }
        }
    } else {
        if (lower) {                                       /* backward, k > j */
            for (ptrdiff_t j1 = N; j1 > 0; j1 -= ETPSV_BLK) {
                ptrdiff_t j0 = j1 - ETPSV_BLK; if (j0 < 0) j0 = 0;
                if (j1 < N) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();
                        ptrdiff_t ilo = j0 + blas_part_bound((j1 - j0), tid, nthreads);
                        ptrdiff_t ihi = j0 + blas_part_bound((j1 - j0), tid + 1, nthreads);
                        for (ptrdiff_t i = ilo; i < ihi; ++i) {
                            const T *restrict col = &ap[cbL(i, N)];
                            T s = zero;
                            for (ptrdiff_t k = j1; k < N; ++k) s += col[k - i] * x[k];
                            x[i] -= s;
                        }
                    }
                }
                etpsv_block(UPLO, TR, nounit, j0, j1, N, ap, x);
            }
        } else {                                           /* forward, k < j */
            for (ptrdiff_t j0 = 0; j0 < N; j0 += ETPSV_BLK) {
                ptrdiff_t j1 = j0 + ETPSV_BLK; if (j1 > N) j1 = N;
                if (j0 > 0) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();
                        ptrdiff_t ilo = j0 + blas_part_bound((j1 - j0), tid, nthreads);
                        ptrdiff_t ihi = j0 + blas_part_bound((j1 - j0), tid + 1, nthreads);
                        for (ptrdiff_t i = ilo; i < ihi; ++i) {
                            const T *restrict col = &ap[cbU(i)];
                            T s = zero;
                            for (ptrdiff_t k = 0; k < j0; ++k) s += col[k] * x[k];
                            x[i] -= s;
                        }
                    }
                }
                etpsv_block(UPLO, TR, nounit, j0, j1, N, ap, x);
            }
        }
    }
    return 1;
}
#endif

static void etpsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t N,
    const T *restrict ap,
    T *restrict x, ptrdiff_t incx)
{
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
    const ptrdiff_t nounit = (up(diag) != 'U');

    if (N == 0) return;

#ifdef _OPENMP
    if (incx == 1 && N >= ETPSV_OMP_MIN && blas_omp_max_threads() > 1
        && etpsv_omp(UPLO, TR, nounit, N, ap, x))
        return;
#endif

    etpsv_serial(UPLO, TR, nounit, N, ap, x, incx);
}

EPBLAS_FACADE_TPMV(etpsv, T)
