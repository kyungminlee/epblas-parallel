/*
 * ytpsv — kind10 complex triangular packed solve.
 *   x := inv(A)*x, inv(A^T)*x, or inv(A^H)*x
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include "../common/epblas_facade.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define YTPSV_OMP_MIN  256   /* below this, run the bit-exact serial path */
#define YTPSV_BLK      128   /* diagonal-block size for the blocked solve */
#define YTPSV_MAX_CPUS 256
#endif

typedef _Complex long double T;
static inline T cconj(T z) { return ~z; }


/* Matrix element with optional conjugation ('C' ⇒ conjugate, 'T' ⇒ as-is). */
static inline T yelem(T a, bool noconj) {
    return noconj ? a : cconj(a);
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
 * incx!=1 fallback. noconj = (TR=='T'); NoTrans never conjugates. */
static void ytpsv_serial(char UPLO, char TR, bool noconj, bool nounit,
                         ptrdiff_t N, const T *restrict ap, T *restrict x, ptrdiff_t incx)
{
    const T zero = 0.0L + 0.0Li;
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
                    T tmp = x[j];
                    ptrdiff_t k = kk;
                    if (noconj) for (ptrdiff_t i = 0; i < j; ++i) { tmp -= ap[k] * x[i]; ++k; }
                    else        for (ptrdiff_t i = 0; i < j; ++i) { tmp -= cconj(ap[k]) * x[i]; ++k; }
                    if (nounit) tmp /= (noconj ? ap[kk + j] : cconj(ap[kk + j]));
                    x[j] = tmp;
                    kk += j + 1;
                }
            } else {
                ptrdiff_t kk = (N * (N + 1)) / 2 - 1;
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    ptrdiff_t k = kk;
                    if (noconj) for (ptrdiff_t i = N - 1; i > j; --i) { tmp -= ap[k] * x[i]; --k; }
                    else        for (ptrdiff_t i = N - 1; i > j; --i) { tmp -= cconj(ap[k]) * x[i]; --k; }
                    if (nounit) tmp /= (noconj ? ap[kk - (N - 1 - j)] : cconj(ap[kk - (N - 1 - j)]));
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
                        tmp -= (noconj ? ap[k] : cconj(ap[k])) * x[ix];
                        ix += incx;
                    }
                    if (nounit) tmp /= (noconj ? ap[kk + j] : cconj(ap[kk + j]));
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
                        tmp -= (noconj ? ap[k] : cconj(ap[k])) * x[ix];
                        ix -= incx;
                    }
                    if (nounit) tmp /= (noconj ? ap[kk - (N - 1 - j)] : cconj(ap[kk - (N - 1 - j)]));
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
static void ytpsv_block(char UPLO, char TR, bool noconj, bool nounit,
                        ptrdiff_t j0, ptrdiff_t j1, ptrdiff_t N,
                        const T *restrict ap, T *restrict x)
{
    const T zero = 0.0L + 0.0Li;
    const bool lower = (UPLO == 'L');
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
        if (!lower) {                                   /* Upper^T/H: forward, k<j */
            for (ptrdiff_t j = j0; j < j1; ++j) {
                const size_t b = cbU(j);
                T tmp = x[j];
                for (ptrdiff_t i = j0; i < j; ++i) tmp -= yelem(ap[b + i], noconj) * x[i];
                if (nounit) tmp /= yelem(ap[b + j], noconj);
                x[j] = tmp;
            }
        } else {                                        /* Lower^T/H: backward, k>j */
            for (ptrdiff_t j = j1 - 1; j >= j0; --j) {
                const size_t b = cbL(j, N);
                T tmp = x[j];
                for (ptrdiff_t i = j + 1; i < j1; ++i) tmp -= yelem(ap[b + (i - j)], noconj) * x[i];
                if (nounit) tmp /= yelem(ap[b], noconj);
                x[j] = tmp;
            }
        }
    }
}

/* Blocked threaded packed solve, incx==1 only. Loop-carried dependence confined
 * to small YTPSV_BLK diagonal blocks (solved serially); the bulk O(N^2)
 * off-diagonal coupling is threaded over disjoint output rows. Returns 1 if it
 * handled the call. */
__attribute__((noinline)) static bool ytpsv_omp(
    char UPLO, char TR, bool noconj, bool nounit, ptrdiff_t N,
    const T *restrict ap, T *restrict x)
{
    if (N < YTPSV_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > YTPSV_MAX_CPUS) nthreads = YTPSV_MAX_CPUS;
    const T zero = 0.0L + 0.0Li;
    const bool lower = (UPLO == 'L');
    const bool trans = (TR != 'N');

    if (!trans) {
        if (lower) {
            for (ptrdiff_t j0 = 0; j0 < N; j0 += YTPSV_BLK) {
                ptrdiff_t j1 = j0 + YTPSV_BLK; if (j1 > N) j1 = N;
                ytpsv_block(UPLO, TR, noconj, nounit, j0, j1, N, ap, x);
                if (j1 >= N) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    ptrdiff_t tid = omp_get_thread_num();
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
            for (ptrdiff_t j1 = N; j1 > 0; j1 -= YTPSV_BLK) {
                ptrdiff_t j0 = j1 - YTPSV_BLK; if (j0 < 0) j0 = 0;
                ytpsv_block(UPLO, TR, noconj, nounit, j0, j1, N, ap, x);
                if (j0 <= 0) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    ptrdiff_t tid = omp_get_thread_num();
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
            for (ptrdiff_t j1 = N; j1 > 0; j1 -= YTPSV_BLK) {
                ptrdiff_t j0 = j1 - YTPSV_BLK; if (j0 < 0) j0 = 0;
                if (j1 < N) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        ptrdiff_t tid = omp_get_thread_num();
                        ptrdiff_t ilo = j0 + blas_part_bound((j1 - j0), tid, nthreads);
                        ptrdiff_t ihi = j0 + blas_part_bound((j1 - j0), tid + 1, nthreads);
                        for (ptrdiff_t i = ilo; i < ihi; ++i) {
                            const T *restrict col = &ap[cbL(i, N)];
                            T s = zero;
                            for (ptrdiff_t k = j1; k < N; ++k) s += yelem(col[k - i], noconj) * x[k];
                            x[i] -= s;
                        }
                    }
                }
                ytpsv_block(UPLO, TR, noconj, nounit, j0, j1, N, ap, x);
            }
        } else {                                           /* forward, k < j */
            for (ptrdiff_t j0 = 0; j0 < N; j0 += YTPSV_BLK) {
                ptrdiff_t j1 = j0 + YTPSV_BLK; if (j1 > N) j1 = N;
                if (j0 > 0) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        ptrdiff_t tid = omp_get_thread_num();
                        ptrdiff_t ilo = j0 + blas_part_bound((j1 - j0), tid, nthreads);
                        ptrdiff_t ihi = j0 + blas_part_bound((j1 - j0), tid + 1, nthreads);
                        for (ptrdiff_t i = ilo; i < ihi; ++i) {
                            const T *restrict col = &ap[cbU(i)];
                            T s = zero;
                            for (ptrdiff_t k = 0; k < j0; ++k) s += yelem(col[k], noconj) * x[k];
                            x[i] -= s;
                        }
                    }
                }
                ytpsv_block(UPLO, TR, noconj, nounit, j0, j1, N, ap, x);
            }
        }
    }
    return 1;
}
#endif

static void ytpsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t N,
    const T *restrict ap,
    T *restrict x, ptrdiff_t incx)
{
    const char UPLO = blas_up(uplo);
    const char TR = blas_up(trans);
    const bool noconj = (TR == 'T');
    const bool nounit = (blas_up(diag) != 'U');

    if (N == 0) return;

#ifdef _OPENMP
    if (incx == 1 && N >= YTPSV_OMP_MIN && blas_omp_max_threads() > 1
        && ytpsv_omp(UPLO, TR, noconj, nounit, N, ap, x))
        return;
#endif

    ytpsv_serial(UPLO, TR, noconj, nounit, N, ap, x, incx);
}

EPBLAS_FACADE_TPMV(ytpsv, T)
