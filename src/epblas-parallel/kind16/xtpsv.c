/*
 * xtpsv — kind16 complex (__complex128) triangular packed solve.
 *   x := inv(A)*x, inv(A^T)*x, or inv(A^H)*x
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define XTPSV_OMP_MIN  256   /* below this, run the bit-exact serial path */
#define XTPSV_BLK      128   /* diagonal-block size for the blocked solve */
#define XTPSV_MAX_CPUS 256
#endif
#include "../common/epblas_facade.h"

typedef __complex128 TC;



static inline TC cconj(TC z) { return conjq(z); }
/* Matrix element with optional conjugation ('C' ⇒ conjugate, 'T' ⇒ as-is). */
static inline TC xelem(TC a, bool noconj) {
    return noconj ? a : cconj(a);
}

#ifdef _OPENMP
/* Column base offsets into the packed array (column-major triangle).
 *   Lower: column j starts at its diagonal (row j); element (i,j) i>=j at base+(i-j).
 *   Upper: column j starts at row 0;            element (i,j) i<=j at base+i.       */
static inline size_t cbL(ptrdiff_t j, ptrdiff_t n) {
    return (size_t)j * (size_t)n - (size_t)j * (size_t)(j - 1) / 2;
}
static inline size_t cbU(ptrdiff_t j) {
    return (size_t)j * (size_t)(j + 1) / 2;
}
#endif

/* Bit-exact serial path (verbatim reference). Also reused as the <threshold /
 * incx!=1 fallback. noconj = (TRANS=='T'); NoTrans never conjugates. */
static void xtpsv_serial(char UPLO, char TRANS, bool noconj, bool nounit,
                         ptrdiff_t n, const TC *restrict ap, TC *restrict x, ptrdiff_t incx)
{
    const TC zero = 0.0Q + 0.0Qi;
    if (incx == 1) {
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        if (nounit) x[j] /= ap[kk];
                        const TC tmp = x[j];
                        ptrdiff_t k = kk - 1;
                        for (ptrdiff_t i = j - 1; i >= 0; --i) { x[i] -= tmp * ap[k]; --k; }
                    }
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != zero) {
                        if (nounit) x[j] /= ap[kk];
                        const TC tmp = x[j];
                        ptrdiff_t k = kk + 1;
                        for (ptrdiff_t i = j + 1; i < n; ++i) { x[i] -= tmp * ap[k]; ++k; }
                    }
                    kk += n - j;
                }
            }
        } else {
            if (UPLO == 'U') {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TC tmp = x[j];
                    ptrdiff_t k = kk;
                    if (noconj) for (ptrdiff_t i = 0; i < j; ++i) { tmp -= ap[k] * x[i]; ++k; }
                    else        for (ptrdiff_t i = 0; i < j; ++i) { tmp -= cconj(ap[k]) * x[i]; ++k; }
                    if (nounit) tmp /= (noconj ? ap[kk + j] : cconj(ap[kk + j]));
                    x[j] = tmp;
                    kk += j + 1;
                }
            } else {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TC tmp = x[j];
                    ptrdiff_t k = kk;
                    if (noconj) for (ptrdiff_t i = n - 1; i > j; --i) { tmp -= ap[k] * x[i]; --k; }
                    else        for (ptrdiff_t i = n - 1; i > j; --i) { tmp -= cconj(ap[k]) * x[i]; --k; }
                    if (nounit) tmp /= (noconj ? ap[kk - (n - 1 - j)] : cconj(ap[kk - (n - 1 - j)]));
                    x[j] = tmp;
                    kk -= (n - j);
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                ptrdiff_t jx = kx + (n - 1) * incx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[jx] != zero) {
                        if (nounit) x[jx] /= ap[kk];
                        const TC tmp = x[jx];
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
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[jx] != zero) {
                        if (nounit) x[jx] /= ap[kk];
                        const TC tmp = x[jx];
                        ptrdiff_t ix = jx;
                        for (ptrdiff_t k = kk + 1; k < kk + n - j; ++k) {
                            ix += incx;
                            x[ix] -= tmp * ap[k];
                        }
                    }
                    jx += incx;
                    kk += n - j;
                }
            }
        } else {
            if (UPLO == 'U') {
                ptrdiff_t kk = 0;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TC tmp = x[jx];
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
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TC tmp = x[jx];
                    ptrdiff_t ix = kx;
                    for (ptrdiff_t k = kk; k > kk - (n - 1 - j); --k) {
                        tmp -= (noconj ? ap[k] : cconj(ap[k])) * x[ix];
                        ix -= incx;
                    }
                    if (nounit) tmp /= (noconj ? ap[kk - (n - 1 - j)] : cconj(ap[kk - (n - 1 - j)]));
                    x[jx] = tmp;
                    jx -= incx;
                    kk -= (n - j);
                }
            }
        }
    }
}

#ifdef _OPENMP
/* Solve a single diagonal block [j0,j1) in packed storage (within-block coupling
 * only). Threaded path need only match serial within fp128 fuzz tol. */
static void xtpsv_block(char UPLO, char TRANS, bool noconj, bool nounit,
                        ptrdiff_t j0, ptrdiff_t j1, ptrdiff_t n, const TC *restrict ap, TC *restrict x)
{
    const TC zero = 0.0Q + 0.0Qi;
    const bool lower = (UPLO == 'L');
    if (TRANS == 'N') {
        if (!lower) {                                   /* Upper: backward */
            for (ptrdiff_t j = j1 - 1; j >= j0; --j) {
                if (x[j] == zero) continue;
                const size_t b = cbU(j);
                if (nounit) x[j] /= ap[b + j];
                const TC tmp = x[j];
                for (ptrdiff_t i = j0; i < j; ++i) x[i] -= tmp * ap[b + i];
            }
        } else {                                        /* Lower: forward */
            for (ptrdiff_t j = j0; j < j1; ++j) {
                if (x[j] == zero) continue;
                const size_t b = cbL(j, n);
                if (nounit) x[j] /= ap[b];
                const TC tmp = x[j];
                for (ptrdiff_t i = j + 1; i < j1; ++i) x[i] -= tmp * ap[b + (i - j)];
            }
        }
    } else {
        if (!lower) {                                   /* Upper^T/H: forward, k<j */
            for (ptrdiff_t j = j0; j < j1; ++j) {
                const size_t b = cbU(j);
                TC tmp = x[j];
                for (ptrdiff_t i = j0; i < j; ++i) tmp -= xelem(ap[b + i], noconj) * x[i];
                if (nounit) tmp /= xelem(ap[b + j], noconj);
                x[j] = tmp;
            }
        } else {                                        /* Lower^T/H: backward, k>j */
            for (ptrdiff_t j = j1 - 1; j >= j0; --j) {
                const size_t b = cbL(j, n);
                TC tmp = x[j];
                for (ptrdiff_t i = j + 1; i < j1; ++i) tmp -= xelem(ap[b + (i - j)], noconj) * x[i];
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
__attribute__((noinline)) static bool xtpsv_omp(
    char UPLO, char TRANS, bool noconj, bool nounit, ptrdiff_t n,
    const TC *restrict ap, TC *restrict x)
{
    if (n < XTPSV_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > XTPSV_MAX_CPUS) nthreads = XTPSV_MAX_CPUS;
    const TC zero = 0.0Q + 0.0Qi;
    const bool lower = (UPLO == 'L');
    const bool trans = (TRANS != 'N');

    if (!trans) {
        if (lower) {
            for (ptrdiff_t j0 = 0; j0 < n; j0 += XTPSV_BLK) {
                ptrdiff_t j1 = j0 + XTPSV_BLK; if (j1 > n) j1 = n;
                xtpsv_block(UPLO, TRANS, noconj, nounit, j0, j1, n, ap, x);
                if (j1 >= n) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    ptrdiff_t tid = omp_get_thread_num();
                    ptrdiff_t rlo = j1 + blas_part_bound(n - j1, tid, nthreads);
                    ptrdiff_t rhi = j1 + blas_part_bound(n - j1, tid + 1, nthreads);
                    for (ptrdiff_t i = j0; i < j1; ++i) {
                        const TC xi = x[i];
                        if (xi == zero) continue;
                        const TC *restrict col = &ap[cbL(i, n)];     /* col[k-i] = A(k,i) */
                        for (ptrdiff_t k = rlo; k < rhi; ++k) x[k] -= xi * col[k - i];
                    }
                }
            }
        } else {
            for (ptrdiff_t j1 = n; j1 > 0; j1 -= XTPSV_BLK) {
                ptrdiff_t j0 = j1 - XTPSV_BLK; if (j0 < 0) j0 = 0;
                xtpsv_block(UPLO, TRANS, noconj, nounit, j0, j1, n, ap, x);
                if (j0 <= 0) break;
                #pragma omp parallel num_threads(nthreads)
                {
                    ptrdiff_t tid = omp_get_thread_num();
                    ptrdiff_t rlo = blas_part_bound(j0, tid, nthreads);
                    ptrdiff_t rhi = blas_part_bound(j0, tid + 1, nthreads);
                    for (ptrdiff_t i = j0; i < j1; ++i) {
                        const TC xi = x[i];
                        if (xi == zero) continue;
                        const TC *restrict col = &ap[cbU(i)];        /* col[k] = A(k,i) */
                        for (ptrdiff_t k = rlo; k < rhi; ++k) x[k] -= xi * col[k];
                    }
                }
            }
        }
    } else {
        if (lower) {                                       /* backward, k > j */
            for (ptrdiff_t j1 = n; j1 > 0; j1 -= XTPSV_BLK) {
                ptrdiff_t j0 = j1 - XTPSV_BLK; if (j0 < 0) j0 = 0;
                if (j1 < n) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        ptrdiff_t tid = omp_get_thread_num();
                        ptrdiff_t ilo = j0 + blas_part_bound(j1 - j0, tid, nthreads);
                        ptrdiff_t ihi = j0 + blas_part_bound(j1 - j0, tid + 1, nthreads);
                        for (ptrdiff_t i = ilo; i < ihi; ++i) {
                            const TC *restrict col = &ap[cbL(i, n)];
                            TC s = zero;
                            for (ptrdiff_t k = j1; k < n; ++k) s += xelem(col[k - i], noconj) * x[k];
                            x[i] -= s;
                        }
                    }
                }
                xtpsv_block(UPLO, TRANS, noconj, nounit, j0, j1, n, ap, x);
            }
        } else {                                           /* forward, k < j */
            for (ptrdiff_t j0 = 0; j0 < n; j0 += XTPSV_BLK) {
                ptrdiff_t j1 = j0 + XTPSV_BLK; if (j1 > n) j1 = n;
                if (j0 > 0) {
                    #pragma omp parallel num_threads(nthreads)
                    {
                        ptrdiff_t tid = omp_get_thread_num();
                        ptrdiff_t ilo = j0 + blas_part_bound(j1 - j0, tid, nthreads);
                        ptrdiff_t ihi = j0 + blas_part_bound(j1 - j0, tid + 1, nthreads);
                        for (ptrdiff_t i = ilo; i < ihi; ++i) {
                            const TC *restrict col = &ap[cbU(i)];
                            TC s = zero;
                            for (ptrdiff_t k = 0; k < j0; ++k) s += xelem(col[k], noconj) * x[k];
                            x[i] -= s;
                        }
                    }
                }
                xtpsv_block(UPLO, TRANS, noconj, nounit, j0, j1, n, ap, x);
            }
        }
    }
    return 1;
}
#endif

void xtpsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n,
    const TC *restrict ap,
    TC *restrict x, ptrdiff_t incx)
{
    const char UPLO = blas_up(uplo);
    const char TRANS = blas_up(trans);
    const bool noconj = (TRANS == 'T');
    const bool nounit = (blas_up(diag) != 'U');

    if (n == 0) return;

#ifdef _OPENMP
    if (incx == 1 && n >= XTPSV_OMP_MIN && blas_omp_max_threads() > 1
        && xtpsv_omp(UPLO, TRANS, noconj, nounit, n, ap, x))
        return;
#endif

    xtpsv_serial(UPLO, TRANS, noconj, nounit, n, ap, x, incx);
}

EPBLAS_FACADE_TPMV(xtpsv, TC)
