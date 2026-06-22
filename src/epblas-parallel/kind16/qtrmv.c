/*
 * qtrmv — kind16 (__float128) triangular matrix-vector.
 *   x := A · x  (TRANS='N') or Aᵀ · x  (TRANS='T'/'C')
 *
 * Serial reference is the in-place Netlib column sweep (sequential). The
 * threaded path (incx==1, large N) dissolves the in-place dependency with an
 * external output buffer, mirroring kind10 etrmv / esymv (Addendum 36):
 *   - TR='T': y[j] = dot(column j, x), disjoint per thread -> shared buffer, no
 *     reduce, copy back.
 *   - TR='N': column j scatters into x[i] (i>j for L, i<j for U), so cross-
 *     thread j-ranges write overlapping i ranges -> per-thread y_priv + reduce.
 * Quad is compute-bound under libquadmath, so the per-column work amortizes the
 * fork/buffer almost immediately. Serial reference stays byte-for-byte. */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

typedef __float128 T;

static inline char up(char c) {
    return (char)toupper((unsigned char)c);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#ifndef QTRMV_OMP_MIN
#define QTRMV_OMP_MIN 128
#endif

#ifdef _OPENMP
/* Threaded out-of-place path (incx==1). Returns 1 if handled, 0 to fall back to
 * the serial reference. noinline so the serial loops compile in a clean register
 * context. */
__attribute__((noinline))
static int qtrmv_omp(int upper, int trans_t, int nounit, ptrdiff_t N, ptrdiff_t lda,
                     const T *restrict a, T *restrict x)
{
    const ptrdiff_t nt = blas_omp_max_threads();
    if (N < QTRMV_OMP_MIN || nt <= 1 || omp_in_parallel()) return 0;
    const T zero = 0.0Q;

    if (trans_t) {
        /* TR='T': each j writes a single x[j] (dot of column j with x). All
         * threads read x then write disjoint y_buf[j] — own j, no overlap. */
        T *y_buf = (T *)malloc((size_t)N * sizeof(T));
        if (!y_buf) return 0;
        #pragma omp parallel num_threads(nt)
        {
            if (!upper) {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T temp = nounit ? (x[j] * A_(j, j)) : x[j];
                    const T *aj = &A_(0, j);
                    T s = zero;
                    for (ptrdiff_t i = j + 1; i < N; ++i) s += aj[i] * x[i];
                    y_buf[j] = temp + s;
                }
            } else {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T temp = nounit ? (x[j] * A_(j, j)) : x[j];
                    const T *aj = &A_(0, j);
                    T s = zero;
                    for (ptrdiff_t i = 0; i < j; ++i) s += aj[i] * x[i];
                    y_buf[j] = temp + s;
                }
            }
            #pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < N; ++i) x[i] = y_buf[i];
        }
        free(y_buf);
        return 1;
    } else {
        /* TR='N': per-thread y_priv + reduction (cross-thread overlapping writes). */
        T *y_priv_all = (T *)calloc((size_t)nt * (size_t)N, sizeof(T));
        if (!y_priv_all) return 0;
        #pragma omp parallel num_threads(nt)
        {
            const ptrdiff_t tid = omp_get_thread_num();
            T *y_priv = &y_priv_all[(size_t)tid * N];  /* calloc-zeroed */
            if (!upper) {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < N; ++j) {
                    const T xj = x[j];
                    const T *aj = &A_(0, j);
                    y_priv[j] += xj * (nounit ? aj[j] : (T)1.0Q);
                    for (ptrdiff_t i = j + 1; i < N; ++i) y_priv[i] += xj * aj[i];
                }
            } else {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < N; ++j) {
                    const T xj = x[j];
                    const T *aj = &A_(0, j);
                    for (ptrdiff_t i = 0; i < j; ++i) y_priv[i] += xj * aj[i];
                    y_priv[j] += xj * (nounit ? aj[j] : (T)1.0Q);
                }
            }
            #pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < N; ++i) {
                T s = zero;
                for (ptrdiff_t t = 0; t < nt; ++t) s += y_priv_all[(size_t)t * N + i];
                x[i] = s;
            }
        }
        free(y_priv_all);
        return 1;
    }
}
#endif /* _OPENMP */

void qtrmv_core(
    char uplo, char trans, char diag,
    ptrdiff_t N,
    const T *restrict a, ptrdiff_t lda,
    T *restrict x, ptrdiff_t incx)
{
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
    const char DIAG = up(diag);
    const ptrdiff_t nounit = (DIAG != 'U');

    if (N == 0) return;
    const T zero = 0.0Q;

    if (incx == 1) {
#ifdef _OPENMP
        if (qtrmv_omp(UPLO == 'U', TR == 'T', nounit, N, lda, a, x)) return;
#endif
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    const T temp = x[j];
                    if (temp != zero) {
                        const T *aj = &A_(0, j);
                        for (ptrdiff_t i = j + 1; i < N; ++i) x[i] += temp * aj[i];
                    }
                    if (nounit) x[j] *= A_(j, j);
                }
            } else {
                for (ptrdiff_t j = 0; j < N; ++j) {
                    const T temp = x[j];
                    if (temp != zero) {
                        const T *aj = &A_(0, j);
                        for (ptrdiff_t i = 0; i < j; ++i) x[i] += temp * aj[i];
                    }
                    if (nounit) x[j] *= A_(j, j);
                }
            }
        } else {
            if (UPLO == 'L') {
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T temp = x[j];
                    if (nounit) temp *= A_(j, j);
                    const T *aj = &A_(0, j);
                    for (ptrdiff_t i = j + 1; i < N; ++i) temp += aj[i] * x[i];
                    x[j] = temp;
                }
            } else {
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T temp = x[j];
                    if (nounit) temp *= A_(j, j);
                    const T *aj = &A_(0, j);
                    for (ptrdiff_t i = 0; i < j; ++i) temp += aj[i] * x[i];
                    x[j] = temp;
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
#ifdef _OPENMP
        /* Thread the strided path by gathering x into a contiguous buffer,
         * driving the shared OMP core, and scattering back — the threading
         * lives in one place (qtrmv_omp) and the serial strided code below
         * stays byte-for-byte unchanged. */
        if (N >= QTRMV_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel()) {
            T *xc = (T *)malloc((size_t)N * sizeof(T));
            if (xc) {
                for (ptrdiff_t i = 0; i < N; ++i) xc[i] = x[kx + i * incx];
                if (qtrmv_omp(UPLO == 'U', TR == 'T', nounit, N, lda, a, xc)) {
                    for (ptrdiff_t i = 0; i < N; ++i) x[kx + i * incx] = xc[i];
                    free(xc);
                    return;
                }
                free(xc);
            }
        }
#endif
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    const T temp = x[kx + j * incx];
                    if (temp != zero)
                        for (ptrdiff_t i = j + 1; i < N; ++i) x[kx + i * incx] += temp * A_(i, j);
                    if (nounit) x[kx + j * incx] *= A_(j, j);
                }
            } else {
                for (ptrdiff_t j = 0; j < N; ++j) {
                    const T temp = x[kx + j * incx];
                    if (temp != zero)
                        for (ptrdiff_t i = 0; i < j; ++i) x[kx + i * incx] += temp * A_(i, j);
                    if (nounit) x[kx + j * incx] *= A_(j, j);
                }
            }
        } else {
            if (UPLO == 'L') {
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp *= A_(j, j);
                    for (ptrdiff_t i = j + 1; i < N; ++i) temp += A_(i, j) * x[kx + i * incx];
                    x[kx + j * incx] = temp;
                }
            } else {
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp *= A_(j, j);
                    for (ptrdiff_t i = 0; i < j; ++i) temp += A_(i, j) * x[kx + i * incx];
                    x[kx + j * incx] = temp;
                }
            }
        }
    }
}

EPBLAS_FACADE_TRMV(qtrmv, T)

#undef A_
