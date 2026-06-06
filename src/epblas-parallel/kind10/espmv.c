/*
 * espmv — kind10 (long double) symmetric packed matrix-vector multiply.
 *   y := alpha*A*x + beta*y
 *
 * The serial reference sweep is column-oriented with cross-column writes
 * (column j updates y[0..j] / y[j..n-1] *and* accumulates into y[j]), so it
 * cannot be parallelized by a bare `omp parallel for` over columns — threads
 * would race on y[i]. The threaded path (contiguous case, large N) mirrors
 * OpenBLAS spmv_thread: a sqrt-balanced contiguous column partition, each
 * thread accumulating the *bare* A*x for its columns into a private size-N
 * slot, then a range-limited controller reduction applies alpha into y. The
 * per-thread kernel keeps this overlay's faster *sequential* packed-index
 * carry (running `k`) rather than recomputing j*(j+1)/2 per column.
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef long double T;

/* Thread the contiguous path once n*n exceeds this (matches OpenBLAS dspmv's
 * MULTI_THREAD_MINIMAL): below it the serial sweep — faster than ob's serial
 * here — wins outright, and ob also stays serial, so par keeps the lead. */
#define ESPMV_OMP_MIN 16384
#define ESPMV_MAX_CPUS 256

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

/* Contiguous (stride-1) two-pass run shared by the serial and threaded
 * paths: y[i] += t1·ap[i] for i in [0, cnt), returning sum ap[i]·x[i].
 * Carved out noinline so the fp80 hot loop compiles in a clean x87 register
 * context — inlined amid the routine's OMP/beta/strided scaffolding GCC
 * spills and RELOADS ap[i] every element (4 fldt/elt vs 3); in isolation it
 * keeps ap[i] resident, matching the migrated Fortran reference. Callers pass
 * pointers at the run start and carry the packed index. Bit-identical. */
__attribute__((noinline)) static
T espmv_axpydot(ptrdiff_t cnt, T t1,
                const T *restrict ap, const T *restrict x, T *restrict y)
{
    T t2 = 0.0L;
    for (ptrdiff_t i = 0; i < cnt; ++i) {
        const T a = ap[i];
        y[i] += t1 * a;
        t2   += a * x[i];
    }
    return t2;
}

/* Strided twin of espmv_axpydot: ap walked contiguously, x/y by stride.
 * Same noinline rationale, applied to the general-stride fallback. ix/iy
 * carry the running strided indices; bit-identical to the inline form. */
__attribute__((noinline)) static
T espmv_axpydot_strided(ptrdiff_t cnt, T t1, const T *restrict ap,
                        const T *restrict x, ptrdiff_t incx, ptrdiff_t ix,
                        T *restrict y, ptrdiff_t incy, ptrdiff_t iy)
{
    T t2 = 0.0L;
    for (ptrdiff_t i = 0; i < cnt; ++i) {
        const T a = ap[i];
        y[iy] += t1 * a;
        t2    += a * x[ix];
        ix += incx; iy += incy;
    }
    return t2;
}

#ifdef _OPENMP
/* Sqrt-balanced contiguous column partition (OpenBLAS symv_partition, mask=3,
 * min_width=4). Per-column work grows with j for UPPER (length j+1) and shrinks
 * for LOWER (length n-j), so widths shrink / grow to equalize triangle area. */
static ptrdiff_t espmv_partition(ptrdiff_t upper, ptrdiff_t n, ptrdiff_t nthreads, ptrdiff_t *range)
{
    const ptrdiff_t mask = 3, min_width = 4;
    const double dnum = (double)n * (double)n / (double)nthreads;
    ptrdiff_t num_cpu = 0;
    range[0] = 0;
    ptrdiff_t i = 0;
    while (i < n) {
        ptrdiff_t width;
        if (nthreads - num_cpu > 1) {
            if (upper) {
                double di = (double)i;
                width = (ptrdiff_t)(sqrt(di * di + dnum) - di);
            } else {
                double di = (double)(n - i);
                double rad = di * di - dnum;
                width = (rad > 0.0) ? (ptrdiff_t)(-sqrt(rad) + di) : (n - i);
            }
            width = (width + mask) & ~(ptrdiff_t)mask;
            if (width < min_width) width = min_width;
            if (width > n - i)     width = n - i;
        } else {
            width = n - i;
        }
        range[num_cpu + 1] = range[num_cpu] + width;
        num_cpu++;
        i += width;
        if (num_cpu >= ESPMV_MAX_CPUS) break;
    }
    return num_cpu;
}
#endif

void espmv_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *restrict ap,
    const T *restrict x, const int *incx_,
    const T *beta_,
    T *restrict y, const int *incy_,
    size_t uplo_len)
{
    (void)uplo_len;
    const ptrdiff_t N = *n_;
    const ptrdiff_t incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const T zero = 0.0L, one = 1.0L;
    const char UPLO = up(uplo);

    if (N == 0 || (alpha == zero && beta == one)) return;

    if (beta != one) {
        ptrdiff_t iy = (incy < 0) ? -(N - 1) * incy : 0;
        if (beta == zero) {
            for (ptrdiff_t i = 0; i < N; ++i) { y[iy] = zero; iy += incy; }
        } else {
            for (ptrdiff_t i = 0; i < N; ++i) { y[iy] = beta * y[iy]; iy += incy; }
        }
    }
    if (alpha == zero) return;

    ptrdiff_t kk = 0;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if ((size_t)N * (size_t)N > ESPMV_OMP_MIN
            && blas_omp_max_threads() > 1 && !omp_in_parallel()) {
            ptrdiff_t nthreads = blas_omp_max_threads();
            if (nthreads > ESPMV_MAX_CPUS) nthreads = ESPMV_MAX_CPUS;
            ptrdiff_t range[ESPMV_MAX_CPUS + 1];
            ptrdiff_t num_cpu = espmv_partition(UPLO == 'U', N, nthreads, range);
            T *buf = (num_cpu > 1)
                ? (T *)calloc((size_t)num_cpu * (size_t)N, sizeof(T)) : NULL;
            if (buf) {
                const ptrdiff_t n = N;
                #pragma omp parallel num_threads(num_cpu)
                {
                    ptrdiff_t t = omp_get_thread_num();
                    ptrdiff_t m_from = range[t], m_to = range[t + 1];
                    T *restrict slot = buf + (size_t)t * (size_t)n;
                    if (UPLO == 'U') {
                        size_t k = (size_t)m_from * (size_t)(m_from + 1) / 2;
                        for (ptrdiff_t j = m_from; j < m_to; ++j) {
                            const T t1 = x[j];
                            const T t2 = espmv_axpydot(j, t1, &ap[k], x, slot);
                            k += (size_t)j;
                            slot[j] += t1 * ap[k] + t2;   /* diagonal */
                            ++k;
                        }
                    } else {
                        size_t k = (size_t)m_from * (size_t)(2 * n - m_from + 1) / 2;
                        for (ptrdiff_t j = m_from; j < m_to; ++j) {
                            const T t1 = x[j];
                            slot[j] += t1 * ap[k];        /* diagonal */
                            ++k;
                            const T t2 = espmv_axpydot(n - 1 - j, t1, &ap[k], &x[j + 1], &slot[j + 1]);
                            k += (size_t)(n - 1 - j);
                            slot[j] += t2;
                        }
                    }
                }
                /* Range-limited reduction: each UPPER thread touched [0,range[t+1]),
                 * each LOWER thread [range[t],n). Fold into one slot, then alpha-AXPY. */
                if (UPLO == 'U') {
                    T *restrict target = buf + (size_t)(num_cpu - 1) * (size_t)n;
                    for (ptrdiff_t t = 0; t < num_cpu - 1; ++t) {
                        const T *restrict src = buf + (size_t)t * (size_t)n;
                        ptrdiff_t len = range[t + 1];
                        for (ptrdiff_t k = 0; k < len; ++k) target[k] += src[k];
                    }
                    for (ptrdiff_t k = 0; k < n; ++k) y[k] += alpha * target[k];
                } else {
                    T *restrict target = buf;
                    for (ptrdiff_t t = 1; t < num_cpu; ++t) {
                        const T *restrict src = buf + (size_t)t * (size_t)n;
                        for (ptrdiff_t k = range[t]; k < n; ++k) target[k] += src[k];
                    }
                    for (ptrdiff_t k = 0; k < n; ++k) y[k] += alpha * target[k];
                }
                free(buf);
                return;
            }
            free(buf);
        }
#endif
        if (UPLO == 'U') {
            for (ptrdiff_t j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                const T t2 = espmv_axpydot(j, t1, &ap[kk], x, y);
                y[j] += t1 * ap[kk + j] + alpha * t2;
                kk += j + 1;
            }
        } else {
            for (ptrdiff_t j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                y[j] += t1 * ap[kk];
                const T t2 = espmv_axpydot(N - 1 - j, t1, &ap[kk + 1], &x[j + 1], &y[j + 1]);
                y[j] += alpha * t2;
                kk += N - j;
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(N - 1) * incy : 0;
        if (UPLO == 'U') {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                const T t2 = espmv_axpydot_strided(
                    j, t1, &ap[kk], x, incx, kx, y, incy, ky);
                y[jy] += t1 * ap[kk + j] + alpha * t2;
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                y[jy] += t1 * ap[kk];
                const T t2 = espmv_axpydot_strided(
                    N - j - 1, t1, &ap[kk + 1],
                    x, incx, jx + incx, y, incy, jy + incy);
                y[jy] += alpha * t2;
                jx += incx; jy += incy;
                kk += N - j;
            }
        }
    }
}
