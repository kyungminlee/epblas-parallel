/*
 * wgbmv — kind10 port of OpenBLAS zgbmv (interface/zgbmv.c +
 * driver/level2/gbmv_thread.c, mode=N/T/C).  Complex general banded MV.
 *
 *   y := alpha * op(A) * x + beta * y
 *
 * op(A) = A (TRANS='N'), A^T ('T'), or A^H ('C', conjugate-transpose).
 * Band storage with KL sub-diagonals and KU super-diagonals.
 *
 * Threading mirrors OpenBLAS zgbmv_thread:
 *   - Serial path if m*n < 125000 || kl+ku < 15.
 *   - Otherwise: even-partition columns; per-thread y_priv buffer of
 *     length leny.  Final controller AXPY: y += alpha * sum(y_priv).
 *   - For TRANS='T'/'C' with strided x, copy x into scratch first.
 */

#include <stddef.h>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#include <stdlib.h>
#include <complex>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using C = std::complex<multifloats::float64x2>;
typedef std::complex<multifloats::float64x2> C;

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

static void gbmv_kernel_N(ptrdiff_t j_lo, ptrdiff_t j_hi, ptrdiff_t m,
                          ptrdiff_t kl, ptrdiff_t ku,
                          const C *a, ptrdiff_t lda,
                          const C *x, ptrdiff_t incx, C *y_priv)
{
    ptrdiff_t jx = j_lo * incx;
    for (ptrdiff_t j = j_lo; j < j_hi; ++j) {
        C xj = x[jx];
        ptrdiff_t i_lo = (j - ku > 0) ? j - ku : 0;
        ptrdiff_t i_hi = (j + kl < m - 1) ? j + kl : m - 1;
        const C *col = &a[(size_t)j * lda];
        ptrdiff_t row_off = ku - j;
        for (ptrdiff_t i = i_lo; i <= i_hi; ++i)
            y_priv[i] += xj * col[row_off + i];
        jx += incx;
    }
}

static void gbmv_kernel_T(ptrdiff_t j_lo, ptrdiff_t j_hi, ptrdiff_t m,
                          ptrdiff_t kl, ptrdiff_t ku,
                          const C *a, ptrdiff_t lda,
                          const C *x, C *y_priv)
{
    for (ptrdiff_t j = j_lo; j < j_hi; ++j) {
        C s = C(0.0, 0.0);
        ptrdiff_t i_lo = (j - ku > 0) ? j - ku : 0;
        ptrdiff_t i_hi = (j + kl < m - 1) ? j + kl : m - 1;
        const C *col = &a[(size_t)j * lda];
        ptrdiff_t row_off = ku - j;
        for (ptrdiff_t i = i_lo; i <= i_hi; ++i)
            s += col[row_off + i] * x[i];
        y_priv[j] += s;
    }
}

static void gbmv_kernel_C(ptrdiff_t j_lo, ptrdiff_t j_hi, ptrdiff_t m,
                          ptrdiff_t kl, ptrdiff_t ku,
                          const C *a, ptrdiff_t lda,
                          const C *x, C *y_priv)
{
    for (ptrdiff_t j = j_lo; j < j_hi; ++j) {
        C s = C(0.0, 0.0);
        ptrdiff_t i_lo = (j - ku > 0) ? j - ku : 0;
        ptrdiff_t i_hi = (j + kl < m - 1) ? j + kl : m - 1;
        const C *col = &a[(size_t)j * lda];
        ptrdiff_t row_off = ku - j;
        for (ptrdiff_t i = i_lo; i <= i_hi; ++i)
            s += std::conj(col[row_off + i]) * x[i];
        y_priv[j] += s;
    }
}

extern "C" void wgbmv_(const char *TRANS, const int *M, const int *N,
            const int *KL, const int *KU, const C *ALPHA,
            const C *a, const int *LDA,
            const C *x, const int *INCX,
            const C *BETA, C *y, const int *INCY,
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
    C alpha = *ALPHA;
    C beta  = *BETA;

    if (m == 0 || n == 0 || (alpha == C(0.0, 0.0) && beta == C(1.0, 0.0))) return;

    char tr = (char)toupper((unsigned char)*TRANS);
    int trans = (tr == 'T' || tr == 'C') ? 1 : 0;
    int noconj = (tr == 'T') ? 1 : 0;

    ptrdiff_t lenx = trans ? m : n;
    ptrdiff_t leny = trans ? n : m;

    if (beta != C(1.0, 0.0)) {
        ptrdiff_t absy = incy < 0 ? -incy : incy;
        ptrdiff_t iy = 0;
        if (beta == C(0.0, 0.0)) for (ptrdiff_t i = 0; i < leny; ++i) { y[iy] = C(0.0, 0.0); iy += absy; }
        else              for (ptrdiff_t i = 0; i < leny; ++i) { y[iy] *= beta;  iy += absy; }
    }

    if (alpha == C(0.0, 0.0)) return;

    if (incx < 0) x -= (lenx - 1) * incx;
    if (incy < 0) y -= (leny - 1) * incy;

#ifdef _OPENMP
    int nthreads = 1;
    if (m * n >= 125000 && kl + ku >= 15) nthreads = omp_get_max_threads();
    if (nthreads > 1) {
        ptrdiff_t *range_n = (ptrdiff_t *)malloc((size_t)(nthreads + 1) * sizeof(ptrdiff_t));
        C *y_priv_all = (C *)calloc((size_t)nthreads * (size_t)leny, sizeof(C));
        C *xbuf = NULL;
        const C *xptr = x;
        if (trans && incx != 1) {
            xbuf = (C *)malloc((size_t)lenx * sizeof(C));
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
                C *y_priv = &y_priv_all[(size_t)tid * (size_t)leny];
                ptrdiff_t j_lo = range_n[tid];
                ptrdiff_t j_hi = range_n[tid + 1];
                if (j_lo < j_hi) {
                    if (!trans)         gbmv_kernel_N(j_lo, j_hi, m, kl, ku, a, lda, x,    incx, y_priv);
                    else if (noconj)    gbmv_kernel_T(j_lo, j_hi, m, kl, ku, a, lda, xptr,       y_priv);
                    else                gbmv_kernel_C(j_lo, j_hi, m, kl, ku, a, lda, xptr,       y_priv);
                }
            }
            ptrdiff_t iy = 0;
            for (ptrdiff_t i = 0; i < leny; ++i) {
                C s = C(0.0, 0.0);
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

    /* Serial path. */
    if (!trans) {
        if (incy == 1) {
            ptrdiff_t jx = 0;
            for (ptrdiff_t j = 0; j < n; ++j) {
                C temp = alpha * x[jx];
                ptrdiff_t i_lo = (j - ku > 0) ? j - ku : 0;
                ptrdiff_t i_hi = (j + kl < m - 1) ? j + kl : m - 1;
                const C *col = &a[(size_t)j * lda];
                ptrdiff_t off = ku - j;
                for (ptrdiff_t i = i_lo; i <= i_hi; ++i)
                    y[i] += temp * col[off + i];
                jx += incx;
            }
        } else {
            ptrdiff_t jx = 0, ky = 0;
            for (ptrdiff_t j = 0; j < n; ++j) {
                C temp = alpha * x[jx];
                ptrdiff_t i_lo = (j - ku > 0) ? j - ku : 0;
                ptrdiff_t i_hi = (j + kl < m - 1) ? j + kl : m - 1;
                const C *col = &a[(size_t)j * lda];
                ptrdiff_t off = ku - j;
                ptrdiff_t iy = ky;
                for (ptrdiff_t i = i_lo; i <= i_hi; ++i) {
                    y[iy] += temp * col[off + i];
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
                C temp = C(0.0, 0.0);
                ptrdiff_t i_lo = (j - ku > 0) ? j - ku : 0;
                ptrdiff_t i_hi = (j + kl < m - 1) ? j + kl : m - 1;
                const C *col = &a[(size_t)j * lda];
                ptrdiff_t off = ku - j;
                if (noconj) {
                    for (ptrdiff_t i = i_lo; i <= i_hi; ++i) temp += col[off + i] * x[i];
                } else {
                    for (ptrdiff_t i = i_lo; i <= i_hi; ++i) temp += std::conj(col[off + i]) * x[i];
                }
                y[jy] += alpha * temp;
                jy += incy;
            }
        } else {
            ptrdiff_t jy = 0, kx = 0;
            for (ptrdiff_t j = 0; j < n; ++j) {
                C temp = C(0.0, 0.0);
                ptrdiff_t i_lo = (j - ku > 0) ? j - ku : 0;
                ptrdiff_t i_hi = (j + kl < m - 1) ? j + kl : m - 1;
                const C *col = &a[(size_t)j * lda];
                ptrdiff_t off = ku - j;
                ptrdiff_t ix = kx;
                if (noconj) {
                    for (ptrdiff_t i = i_lo; i <= i_hi; ++i) {
                        temp += col[off + i] * x[ix]; ix += incx;
                    }
                } else {
                    for (ptrdiff_t i = i_lo; i <= i_hi; ++i) {
                        temp += std::conj(col[off + i]) * x[ix]; ix += incx;
                    }
                }
                y[jy] += alpha * temp;
                jy += incy;
                if (j >= ku) kx += incx;
            }
        }
    }
}
