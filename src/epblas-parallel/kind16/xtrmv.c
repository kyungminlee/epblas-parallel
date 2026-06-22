/*
 * xtrmv — kind16 complex triangular matrix-vector.
 *   x := A·x (TRANS='N'), Aᵀ·x (TRANS='T'), or Aᴴ·x (TRANS='C')
 *
 * Serial reference is the in-place Netlib column sweep (sequential). The
 * threaded path (incx==1, large N) dissolves the in-place dependency with an
 * external output buffer, mirroring kind10 ytrmv (Addendum 36):
 *   - TR='N': column j scatters into x[i] (overlapping across threads) ->
 *     per-thread y_priv + reduce.
 *   - TR='T'/'C': y[j] = dot(op(column j), x), disjoint per thread -> shared
 *     buffer, no reduce, copy back. ConjTrans conjugates band + diagonal.
 * Quad complex arithmetic is heavy under libquadmath, so per-column work
 * amortizes the fork/buffer almost immediately. Serial reference byte-for-byte. */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

typedef __complex128 T;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#ifndef XTRMV_OMP_MIN
#define XTRMV_OMP_MIN 128
#endif

#ifdef _OPENMP
/* Threaded out-of-place path (incx==1). Returns 1 if handled, 0 to fall back to
 * the serial reference. noinline so the serial loops compile in a clean register
 * context. trans_t = (TR != 'N'), conj_a = (TR == 'C'). */
__attribute__((noinline))
static bool xtrmv_omp(bool upper, bool trans_t, bool conj_a, bool nounit, ptrdiff_t N, ptrdiff_t lda,
                     const T *restrict a, T *restrict x)
{
    const ptrdiff_t nthreads = blas_omp_max_threads();
    if (N < XTRMV_OMP_MIN || !blas_omp_should_thread()) return 0;
    const T zero = 0.0Q + 0.0Qi;
    const T one  = 1.0Q + 0.0Qi;

    if (!trans_t) {
        /* TR='N': per-thread y_priv + reduction (cross-thread overlapping writes). */
        T *y_priv_all = (T *)calloc((size_t)nthreads * (size_t)N, sizeof(T));
        if (!y_priv_all) return 0;
        #pragma omp parallel num_threads(nthreads)
        {
            const ptrdiff_t tid = omp_get_thread_num();
            T *y_priv = &y_priv_all[(size_t)tid * N];  /* calloc-zeroed */
            if (!upper) {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < N; ++j) {
                    const T xj = x[j];
                    const T *aj = &A_(0, j);
                    y_priv[j] += xj * (nounit ? aj[j] : one);
                    for (ptrdiff_t i = j + 1; i < N; ++i) y_priv[i] += xj * aj[i];
                }
            } else {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < N; ++j) {
                    const T xj = x[j];
                    const T *aj = &A_(0, j);
                    for (ptrdiff_t i = 0; i < j; ++i) y_priv[i] += xj * aj[i];
                    y_priv[j] += xj * (nounit ? aj[j] : one);
                }
            }
            #pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < N; ++i) {
                T s = zero;
                for (ptrdiff_t t = 0; t < nthreads; ++t) s += y_priv_all[(size_t)t * N + i];
                x[i] = s;
            }
        }
        free(y_priv_all);
        return 1;
    } else {
        /* TR='T'/'C': each j writes its own output slot. */
        T *y_buf = (T *)malloc((size_t)N * sizeof(T));
        if (!y_buf) return 0;
        #pragma omp parallel num_threads(nthreads)
        {
            if (!upper) {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T temp = x[j];
                    if (nounit) temp *= (conj_a ? conjq(A_(j, j)) : A_(j, j));
                    const T *aj = &A_(0, j);
                    if (conj_a) for (ptrdiff_t i = j + 1; i < N; ++i) temp += conjq(aj[i]) * x[i];
                    else        for (ptrdiff_t i = j + 1; i < N; ++i) temp += aj[i] * x[i];
                    y_buf[j] = temp;
                }
            } else {
                #pragma omp for schedule(static, 1)
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T temp = x[j];
                    if (nounit) temp *= (conj_a ? conjq(A_(j, j)) : A_(j, j));
                    const T *aj = &A_(0, j);
                    if (conj_a) for (ptrdiff_t i = 0; i < j; ++i) temp += conjq(aj[i]) * x[i];
                    else        for (ptrdiff_t i = 0; i < j; ++i) temp += aj[i] * x[i];
                    y_buf[j] = temp;
                }
            }
            #pragma omp for schedule(static)
            for (ptrdiff_t i = 0; i < N; ++i) x[i] = y_buf[i];
        }
        free(y_buf);
        return 1;
    }
}
#endif /* _OPENMP */

void xtrmv_core(
    char uplo, char trans, char diag,
    ptrdiff_t N,
    const T *restrict a, ptrdiff_t lda,
    T *restrict x, ptrdiff_t incx)
{
    const char UPLO = blas_up(uplo);
    const char TR   = blas_up(trans);
    const char DIAG = blas_up(diag);
    const bool nounit = (DIAG != 'U');

    if (N == 0) return;
    const T zero = 0.0Q + 0.0Qi;

    if (incx == 1) {
#ifdef _OPENMP
        if (xtrmv_omp(UPLO == 'U', TR != 'N', TR == 'C', nounit, N, lda, a, x)) return;
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
            const bool conj_a = (TR == 'C');
            if (UPLO == 'L') {
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T temp = x[j];
                    if (nounit) temp *= (conj_a ? conjq(A_(j, j)) : A_(j, j));
                    const T *aj = &A_(0, j);
                    if (conj_a) {
                        for (ptrdiff_t i = j + 1; i < N; ++i) temp += conjq(aj[i]) * x[i];
                    } else {
                        for (ptrdiff_t i = j + 1; i < N; ++i) temp += aj[i] * x[i];
                    }
                    x[j] = temp;
                }
            } else {
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T temp = x[j];
                    if (nounit) temp *= (conj_a ? conjq(A_(j, j)) : A_(j, j));
                    const T *aj = &A_(0, j);
                    if (conj_a) {
                        for (ptrdiff_t i = 0; i < j; ++i) temp += conjq(aj[i]) * x[i];
                    } else {
                        for (ptrdiff_t i = 0; i < j; ++i) temp += aj[i] * x[i];
                    }
                    x[j] = temp;
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
#ifdef _OPENMP
        /* Thread the strided path by gathering x into a contiguous buffer,
         * driving the shared OMP core, and scattering back — the threading
         * lives in one place (xtrmv_omp) and the serial strided code below
         * stays byte-for-byte unchanged. */
        if (N >= XTRMV_OMP_MIN && blas_omp_should_thread()) {
            T *xc = (T *)malloc((size_t)N * sizeof(T));
            if (xc) {
                for (ptrdiff_t i = 0; i < N; ++i) xc[i] = x[kx + i * incx];
                if (xtrmv_omp(UPLO == 'U', TR != 'N', TR == 'C', nounit, N, lda, a, xc)) {
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
            const bool conj_a = (TR == 'C');
            if (UPLO == 'L') {
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp *= (conj_a ? conjq(A_(j, j)) : A_(j, j));
                    for (ptrdiff_t i = j + 1; i < N; ++i) {
                        const T aij = conj_a ? conjq(A_(i, j)) : A_(i, j);
                        temp += aij * x[kx + i * incx];
                    }
                    x[kx + j * incx] = temp;
                }
            } else {
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp *= (conj_a ? conjq(A_(j, j)) : A_(j, j));
                    for (ptrdiff_t i = 0; i < j; ++i) {
                        const T aij = conj_a ? conjq(A_(i, j)) : A_(i, j);
                        temp += aij * x[kx + i * incx];
                    }
                    x[kx + j * incx] = temp;
                }
            }
        }
    }
}

EPBLAS_FACADE_TRMV(xtrmv, T)

#undef A_
