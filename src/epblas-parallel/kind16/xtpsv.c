/*
 * xtpsv — kind16 complex (__complex128) triangular packed solve.
 *   x := inv(A)*x, inv(A^T)*x, or inv(A^H)*x
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define XTPSV_OMP_MIN  256   /* below this, run the bit-exact serial path */
#define XTPSV_BLK      128   /* diagonal-block size for the blocked solve */
#define XTPSV_MAX_CPUS 256
#endif

typedef __complex128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

/* Matrix element with optional conjugation ('C' ⇒ conjugate, 'T' ⇒ as-is). */
static inline T xelem(T a, int noconj) {
    return noconj ? a : conjq(a);
}

#ifdef _OPENMP
/* Column base offsets into the packed array (column-major triangle).
 *   Lower: column j starts at its diagonal (row j); element (i,j) i>=j at base+(i-j).
 *   Upper: column j starts at row 0;            element (i,j) i<=j at base+i.       */
static inline size_t cbL(int j, int N) {
    return (size_t)j * (size_t)N - (size_t)j * (size_t)(j - 1) / 2;
}
static inline size_t cbU(int j) {
    return (size_t)j * (size_t)(j + 1) / 2;
}
#endif

/* Bit-exact serial path (verbatim reference). Also reused as the <threshold /
 * incx!=1 fallback. noconj = (TR=='T'); NoTrans never conjugates. */
static void xtpsv_serial(char UPLO, char TR, int noconj, int nounit,
                         int N, const T *restrict ap, T *restrict x, int incx)
{
    const T zero = 0.0Q + 0.0Qi;
    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'U') {
                int kk = (N * (N + 1)) / 2 - 1;
                for (int j = N - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        if (nounit) x[j] /= ap[kk];
                        const T tmp = x[j];
                        int k = kk - 1;
                        for (int i = j - 1; i >= 0; --i) { x[i] -= tmp * ap[k]; --k; }
                    }
                    kk -= j + 1;
                }
            } else {
                int kk = 0;
                for (int j = 0; j < N; ++j) {
                    if (x[j] != zero) {
                        if (nounit) x[j] /= ap[kk];
                        const T tmp = x[j];
                        int k = kk + 1;
                        for (int i = j + 1; i < N; ++i) { x[i] -= tmp * ap[k]; ++k; }
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
                    if (noconj) for (int i = 0; i < j; ++i) { tmp -= ap[k] * x[i]; ++k; }
                    else        for (int i = 0; i < j; ++i) { tmp -= conjq(ap[k]) * x[i]; ++k; }
                    if (nounit) tmp /= (noconj ? ap[kk + j] : conjq(ap[kk + j]));
                    x[j] = tmp;
                    kk += j + 1;
                }
            } else {
                int kk = (N * (N + 1)) / 2 - 1;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    int k = kk;
                    if (noconj) for (int i = N - 1; i > j; --i) { tmp -= ap[k] * x[i]; --k; }
                    else        for (int i = N - 1; i > j; --i) { tmp -= conjq(ap[k]) * x[i]; --k; }
                    if (nounit) tmp /= (noconj ? ap[kk - (N - 1 - j)] : conjq(ap[kk - (N - 1 - j)]));
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
                    if (x[jx] != zero) {
                        if (nounit) x[jx] /= ap[kk];
                        const T tmp = x[jx];
                        int ix = jx;
                        for (int k = kk - 1; k >= kk - j; --k) {
                            ix -= incx;
                            x[ix] -= tmp * ap[k];
                        }
                    }
                    jx -= incx;
                    kk -= j + 1;
                }
            } else {
                int kk = 0;
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    if (x[jx] != zero) {
                        if (nounit) x[jx] /= ap[kk];
                        const T tmp = x[jx];
                        int ix = jx;
                        for (int k = kk + 1; k < kk + N - j; ++k) {
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
                int kk = 0;
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    T tmp = x[jx];
                    int ix = kx;
                    for (int k = kk; k < kk + j; ++k) {
                        tmp -= (noconj ? ap[k] : conjq(ap[k])) * x[ix];
                        ix += incx;
                    }
                    if (nounit) tmp /= (noconj ? ap[kk + j] : conjq(ap[kk + j]));
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
                        tmp -= (noconj ? ap[k] : conjq(ap[k])) * x[ix];
                        ix -= incx;
                    }
                    if (nounit) tmp /= (noconj ? ap[kk - (N - 1 - j)] : conjq(ap[kk - (N - 1 - j)]));
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
 * only). Threaded path need only match serial within fp128 fuzz tol. */
static void xtpsv_block(char UPLO, char TR, int noconj, int nounit,
                        int j0, int j1, int N, const T *restrict ap, T *restrict x)
{
    const T zero = 0.0Q + 0.0Qi;
    const int lower = (UPLO == 'L');
    if (TR == 'N') {
        if (!lower) {                                   /* Upper: backward */
            for (int j = j1 - 1; j >= j0; --j) {
                if (x[j] == zero) continue;
                const size_t b = cbU(j);
                if (nounit) x[j] /= ap[b + j];
                const T tmp = x[j];
                for (int i = j0; i < j; ++i) x[i] -= tmp * ap[b + i];
            }
        } else {                                        /* Lower: forward */
            for (int j = j0; j < j1; ++j) {
                if (x[j] == zero) continue;
                const size_t b = cbL(j, N);
                if (nounit) x[j] /= ap[b];
                const T tmp = x[j];
                for (int i = j + 1; i < j1; ++i) x[i] -= tmp * ap[b + (i - j)];
            }
        }
    } else {
        if (!lower) {                                   /* Upper^T/H: forward, k<j */
            for (int j = j0; j < j1; ++j) {
                const size_t b = cbU(j);
                T tmp = x[j];
                for (int i = j0; i < j; ++i) tmp -= xelem(ap[b + i], noconj) * x[i];
                if (nounit) tmp /= xelem(ap[b + j], noconj);
                x[j] = tmp;
            }
        } else {                                        /* Lower^T/H: backward, k>j */
            for (int j = j1 - 1; j >= j0; --j) {
                const size_t b = cbL(j, N);
                T tmp = x[j];
                for (int i = j + 1; i < j1; ++i) tmp -= xelem(ap[b + (i - j)], noconj) * x[i];
                if (nounit) tmp /= xelem(ap[b], noconj);
                x[j] = tmp;
            }
        }
    }
}

/* Blocked threaded packed solve, incx==1 only. Loop-carried dependence confined
 * to small XTPSV_BLK diagonal blocks (solved serially); the bulk O(N^2)
 * off-diagonal coupling is threaded over disjoint output rows. Returns 1 if it
 * handled the call. */
__attribute__((noinline)) static int xtpsv_omp(
    char UPLO, char TR, int noconj, int nounit, int N,
    const T *restrict ap, T *restrict x)
{
    if (N < XTPSV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > XTPSV_MAX_CPUS) nthreads = XTPSV_MAX_CPUS;
    const T zero = 0.0Q + 0.0Qi;
    const int lower = (UPLO == 'L');
    const int trans = (TR != 'N');

    if (!trans) {
        if (lower) {
            for (int j0 = 0; j0 < N; j0 += XTPSV_BLK) {
                int j1 = j0 + XTPSV_BLK; if (j1 > N) j1 = N;
                xtpsv_block(UPLO, TR, noconj, nounit, j0, j1, N, ap, x);
                if (j1 >= N) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    int tid = omp_get_thread_num();
                    int rlo = j1 + (int)((long long)(N - j1) * tid / nthreads);
                    int rhi = j1 + (int)((long long)(N - j1) * (tid + 1) / nthreads);
                    for (int i = j0; i < j1; ++i) {
                        const T xi = x[i];
                        if (xi == zero) continue;
                        const T *restrict col = &ap[cbL(i, N)];     /* col[k-i] = A(k,i) */
                        for (int k = rlo; k < rhi; ++k) x[k] -= xi * col[k - i];
                    }
                }
            }
        } else {
            for (int j1 = N; j1 > 0; j1 -= XTPSV_BLK) {
                int j0 = j1 - XTPSV_BLK; if (j0 < 0) j0 = 0;
                xtpsv_block(UPLO, TR, noconj, nounit, j0, j1, N, ap, x);
                if (j0 <= 0) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    int tid = omp_get_thread_num();
                    int rlo = (int)((long long)j0 * tid / nthreads);
                    int rhi = (int)((long long)j0 * (tid + 1) / nthreads);
                    for (int i = j0; i < j1; ++i) {
                        const T xi = x[i];
                        if (xi == zero) continue;
                        const T *restrict col = &ap[cbU(i)];        /* col[k] = A(k,i) */
                        for (int k = rlo; k < rhi; ++k) x[k] -= xi * col[k];
                    }
                }
            }
        }
    } else {
        if (lower) {                                       /* backward, k > j */
            for (int j1 = N; j1 > 0; j1 -= XTPSV_BLK) {
                int j0 = j1 - XTPSV_BLK; if (j0 < 0) j0 = 0;
                if (j1 < N) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();
                        int ilo = j0 + (int)((long long)(j1 - j0) * tid / nthreads);
                        int ihi = j0 + (int)((long long)(j1 - j0) * (tid + 1) / nthreads);
                        for (int i = ilo; i < ihi; ++i) {
                            const T *restrict col = &ap[cbL(i, N)];
                            T s = zero;
                            for (int k = j1; k < N; ++k) s += xelem(col[k - i], noconj) * x[k];
                            x[i] -= s;
                        }
                    }
                }
                xtpsv_block(UPLO, TR, noconj, nounit, j0, j1, N, ap, x);
            }
        } else {                                           /* forward, k < j */
            for (int j0 = 0; j0 < N; j0 += XTPSV_BLK) {
                int j1 = j0 + XTPSV_BLK; if (j1 > N) j1 = N;
                if (j0 > 0) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();
                        int ilo = j0 + (int)((long long)(j1 - j0) * tid / nthreads);
                        int ihi = j0 + (int)((long long)(j1 - j0) * (tid + 1) / nthreads);
                        for (int i = ilo; i < ihi; ++i) {
                            const T *restrict col = &ap[cbU(i)];
                            T s = zero;
                            for (int k = 0; k < j0; ++k) s += xelem(col[k], noconj) * x[k];
                            x[i] -= s;
                        }
                    }
                }
                xtpsv_block(UPLO, TR, noconj, nounit, j0, j1, N, ap, x);
            }
        }
    }
    return 1;
}
#endif

void xtpsv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict ap,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
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
    if (incx == 1 && N >= XTPSV_OMP_MIN && blas_omp_max_threads() > 1
        && xtpsv_omp(UPLO, TR, noconj, nounit, N, ap, x))
        return;
#endif

    xtpsv_serial(UPLO, TR, noconj, nounit, N, ap, x, incx);
}
