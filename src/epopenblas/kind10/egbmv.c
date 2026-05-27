/*
 * egbmv — kind10 port of OpenBLAS dgbmv (interface/gbmv.c +
 * driver/level2/gbmv_thread.c).  General banded matrix-vector.
 *
 *   y := alpha * A   * x + beta * y    (TRANS='N')   A is M x N
 *   y := alpha * A^T * x + beta * y    (TRANS='T'/'C')
 *
 * Band storage with KL sub-diagonals and KU super-diagonals (lda >= KL+KU+1):
 *   ab[(ku - j + i) + j*lda] = A(i,j) for max(0, j-ku) <= i <= min(m-1, j+kl)
 *
 * Threading mirrors OpenBLAS dgbmv_thread:
 *   - Serial path if m*n < 250000 || kl+ku < 15 (OpenBLAS threshold).
 *   - Otherwise: even-partition columns across threads via OpenBLAS's
 *     width = ceil(remaining/(nthreads-num_cpu)), min=4 formula.
 *     Each thread accumulates into a private y_priv buffer of length leny
 *     (without alpha applied).  Final controller AXPY: y += alpha * sum(y_priv).
 *   - For TRANSA with strided x, OpenBLAS copies x into a contiguous
 *     scratch buffer first; we mirror that.
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef long double T;

/* Partition n columns across nthreads using OpenBLAS's gbmv_thread width
 * recipe: width = ceil(remaining / (nthreads - num_cpu)), clamped to >=4. */
static int gbmv_partition(ptrdiff_t n, int nthreads, ptrdiff_t *range_n)
{
    range_n[0] = 0;
    ptrdiff_t i = n;
    int num_cpu = 0;
    while (i > 0 && num_cpu < nthreads) {
        ptrdiff_t width = (i + nthreads - num_cpu - 1) / (nthreads - num_cpu);
        if (width < 4) width = 4;
        if (i < width) width = i;
        range_n[num_cpu + 1] = range_n[num_cpu] + width;
        num_cpu++;
        i -= width;
    }
    for (int t = num_cpu + 1; t <= nthreads; ++t) range_n[t] = range_n[num_cpu];
    return num_cpu;
}

/* Per-thread kernel — NoTrans, contiguous y_priv (length m).  Computes
 * y_priv[i] += A(i,j)*x[j] for j in [j_lo, j_hi), i in band slice.
 * x walked with stride incx; y_priv has stride 1. */
static void gbmv_kernel_N(ptrdiff_t j_lo, ptrdiff_t j_hi, ptrdiff_t m,
                          ptrdiff_t kl, ptrdiff_t ku,
                          const T *a, ptrdiff_t lda,
                          const T *x, ptrdiff_t incx, T *y_priv)
{
    ptrdiff_t jx = j_lo * incx;
    for (ptrdiff_t j = j_lo; j < j_hi; ++j) {
        T xj = x[jx];
        ptrdiff_t i_lo = (j - ku > 0) ? j - ku : 0;
        ptrdiff_t i_hi = (j + kl < m - 1) ? j + kl : m - 1;
        const T *col = &a[(size_t)j * lda];
        ptrdiff_t row_off = ku - j;
        for (ptrdiff_t i = i_lo; i <= i_hi; ++i)
            y_priv[i] += xj * col[row_off + i];
        jx += incx;
    }
}

/* Per-thread kernel — Trans, contiguous y_priv (length n).  Computes
 * y_priv[j] = sum_i A(i,j) * x[i] for j in [j_lo, j_hi).
 * x is contiguous (caller-copied if incx != 1). */
static void gbmv_kernel_T(ptrdiff_t j_lo, ptrdiff_t j_hi, ptrdiff_t m,
                          ptrdiff_t kl, ptrdiff_t ku,
                          const T *a, ptrdiff_t lda,
                          const T *x, T *y_priv)
{
    for (ptrdiff_t j = j_lo; j < j_hi; ++j) {
        T s = 0.0L;
        ptrdiff_t i_lo = (j - ku > 0) ? j - ku : 0;
        ptrdiff_t i_hi = (j + kl < m - 1) ? j + kl : m - 1;
        const T *col = &a[(size_t)j * lda];
        ptrdiff_t row_off = ku - j;
        for (ptrdiff_t i = i_lo; i <= i_hi; ++i)
            s += col[row_off + i] * x[i];
        y_priv[j] += s;
    }
}

void egbmv_(const char *TRANS, const int *M, const int *N,
            const int *KL, const int *KU, const T *ALPHA,
            const T *a, const int *LDA,
            const T *x, const int *INCX,
            const T *BETA, T *y, const int *INCY,
            size_t trans_len)
{
    (void)trans_len;
    ptrdiff_t m    = (ptrdiff_t)(*M);
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t kl   = (ptrdiff_t)(*KL);
    ptrdiff_t ku   = (ptrdiff_t)(*KU);
    ptrdiff_t lda  = (ptrdiff_t)(*LDA);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    T alpha = *ALPHA;
    T beta  = *BETA;

    if (m == 0 || n == 0 || (alpha == 0.0L && beta == 1.0L)) return;

    char tr = (char)toupper((unsigned char)*TRANS);
    int trans = (tr == 'T' || tr == 'C') ? 1 : 0;

    ptrdiff_t lenx = trans ? m : n;
    ptrdiff_t leny = trans ? n : m;

    if (beta != 1.0L) {
        ptrdiff_t absy = incy < 0 ? -incy : incy;
        ptrdiff_t iy = 0;
        if (beta == 0.0L) for (ptrdiff_t i = 0; i < leny; ++i) { y[iy] = 0.0L; iy += absy; }
        else              for (ptrdiff_t i = 0; i < leny; ++i) { y[iy] *= beta;  iy += absy; }
    }

    if (alpha == 0.0L) return;

    if (incx < 0) x -= (lenx - 1) * incx;
    if (incy < 0) y -= (leny - 1) * incy;

#ifdef _OPENMP
    int nthreads = 1;
    if (m * n >= 250000 && kl + ku >= 15) nthreads = omp_get_max_threads();
    if (nthreads > 1) {
        ptrdiff_t *range_n = (ptrdiff_t *)malloc((size_t)(nthreads + 1) * sizeof(ptrdiff_t));
        T *y_priv_all = (T *)calloc((size_t)nthreads * (size_t)leny, sizeof(T));
        T *xbuf = NULL;
        const T *xptr = x;
        if (trans && incx != 1) {
            xbuf = (T *)malloc((size_t)lenx * sizeof(T));
            if (xbuf) {
                for (ptrdiff_t i = 0; i < lenx; ++i) xbuf[i] = x[i * incx];
                xptr = xbuf;
            }
        }
        if (range_n && y_priv_all && (!trans || incx == 1 || xbuf)) {
            int num_cpu = gbmv_partition(n, nthreads, range_n);
            (void)num_cpu;
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                T *y_priv = &y_priv_all[(size_t)tid * (size_t)leny];
                ptrdiff_t j_lo = range_n[tid];
                ptrdiff_t j_hi = range_n[tid + 1];
                if (j_lo < j_hi) {
                    if (!trans) gbmv_kernel_N(j_lo, j_hi, m, kl, ku, a, lda, x, incx, y_priv);
                    else        gbmv_kernel_T(j_lo, j_hi, m, kl, ku, a, lda, xptr,       y_priv);
                }
            }
            /* Controller AXPY: y += alpha * sum_t(y_priv[t]). */
            ptrdiff_t iy = 0;
            for (ptrdiff_t i = 0; i < leny; ++i) {
                T s = 0.0L;
                for (int t = 0; t < nthreads; ++t)
                    s += y_priv_all[(size_t)t * (size_t)leny + (size_t)i];
                y[iy] += alpha * s;
                iy += incy;
            }
            free(y_priv_all); free(range_n);
            if (xbuf) free(xbuf);
            return;
        }
        free(y_priv_all); free(range_n);
        if (xbuf) free(xbuf);
    }
#endif

    /* Serial path — same algorithm but no per-thread buffer. */
    if (!trans) {
        if (incy == 1) {
            ptrdiff_t jx = 0;
            for (ptrdiff_t j = 0; j < n; ++j) {
                T temp = alpha * x[jx];
                ptrdiff_t i_lo = (j - ku > 0) ? j - ku : 0;
                ptrdiff_t i_hi = (j + kl < m - 1) ? j + kl : m - 1;
                const T *col = &a[(size_t)j * lda];
                ptrdiff_t row_off = ku - j;
                for (ptrdiff_t i = i_lo; i <= i_hi; ++i)
                    y[i] += temp * col[row_off + i];
                jx += incx;
            }
        } else {
            ptrdiff_t jx = 0, ky = 0;
            for (ptrdiff_t j = 0; j < n; ++j) {
                T temp = alpha * x[jx];
                ptrdiff_t i_lo = (j - ku > 0) ? j - ku : 0;
                ptrdiff_t i_hi = (j + kl < m - 1) ? j + kl : m - 1;
                const T *col = &a[(size_t)j * lda];
                ptrdiff_t row_off = ku - j;
                ptrdiff_t iy = ky;
                for (ptrdiff_t i = i_lo; i <= i_hi; ++i) {
                    y[iy] += temp * col[row_off + i];
                    iy += incy;
                }
                jx += incx;
                if (j >= ku) ky += incy;
            }
        }
    } else {
        if (incx == 1) {
            ptrdiff_t jy = 0;
            for (ptrdiff_t j = 0; j < n; ++j) {
                T temp = 0.0L;
                ptrdiff_t i_lo = (j - ku > 0) ? j - ku : 0;
                ptrdiff_t i_hi = (j + kl < m - 1) ? j + kl : m - 1;
                const T *col = &a[(size_t)j * lda];
                ptrdiff_t row_off = ku - j;
                for (ptrdiff_t i = i_lo; i <= i_hi; ++i)
                    temp += col[row_off + i] * x[i];
                y[jy] += alpha * temp;
                jy += incy;
            }
        } else {
            ptrdiff_t jy = 0, kx = 0;
            for (ptrdiff_t j = 0; j < n; ++j) {
                T temp = 0.0L;
                ptrdiff_t i_lo = (j - ku > 0) ? j - ku : 0;
                ptrdiff_t i_hi = (j + kl < m - 1) ? j + kl : m - 1;
                const T *col = &a[(size_t)j * lda];
                ptrdiff_t row_off = ku - j;
                ptrdiff_t ix = kx;
                for (ptrdiff_t i = i_lo; i <= i_hi; ++i) {
                    temp += col[row_off + i] * x[ix];
                    ix += incx;
                }
                y[jy] += alpha * temp;
                jy += incy;
                if (j >= ku) kx += incx;
            }
        }
    }
}
