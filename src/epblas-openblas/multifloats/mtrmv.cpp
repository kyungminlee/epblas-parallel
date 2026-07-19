/*
 * mtrmv — kind10 port of OpenBLAS dtrmv (interface/trmv.c +
 * driver/level2/trmv_thread.c).  Triangular matrix-vector.
 *
 *   x := op(A) * x      where op(A) = A or A^T  (A is N x N triangular)
 *
 * Serial path mirrors Fortran reference dtrmv.f — the algorithm is the
 * same up-to summation reorder; the OpenBLAS serial path uses tiled GEMV
 * but produces the same result.  At small N (n < 50) we stay serial,
 * matching the OpenBLAS threshold.
 *
 * Parallel path mirrors trmv_thread.c:
 *   - Each thread has a private accumulator slot of length n (out-of-place).
 *     For UPPER, thread t (tid in OMP order) covers the HIGHEST column
 *     range first (sqrt-partition picks small width at the top tip,
 *     bigger near m-1).  For LOWER, thread 0 covers [0, range_m[1]) and
 *     so on.
 *   - Inside the kernel, columns of the thread's range are processed in
 *     DTB_ENTRIES-wide tiles.  Each tile fires (a) an off-tile GEMV
 *     against the long off-diagonal block, then (b) a tight scalar loop
 *     across the diagonal block.  Direction matches Fortran reference.
 *   - TRANS path: all threads write disjoint y[m_from..m_to) slices of
 *     the same slot[0], so no inter-thread reduction is needed.
 *   - NoTrans path: each thread fills its private slot independently;
 *     controller AXPY-reduces slot[t] into slot[0] over the touched range.
 *   - Final copy: x = buffer[0..n) with stride.
 *
 * Fortran ABI:  subroutine mtrmv(uplo, trans, diag, n, a, lda, x, incx)
 */

#include <stddef.h>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef multifloats::float64x2 T;

#define DTB_ENTRIES  32  /* inherited unchanged from the kind10 tuning (same 16-byte element) */

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

/* Sqrt-partition mirror of trmv_thread.c.  Both LOWER and UPPER are
 * computed forward; UPPER reverses the resulting ranges so thread 0
 * gets the highest indices (i.e., the column range with the most
 * off-diagonal work for upper-triangular trmv). */
static void trmv_partition(int upper, ptrdiff_t n, int nthreads, ptrdiff_t *range)
{
    const int mask = 7;
    double dnum = (double)n * (double)n / (double)nthreads;
    if (!upper) {
        range[0] = 0;
        ptrdiff_t i = 0;
        int num_cpu = 0;
        while (i < n && num_cpu < nthreads) {
            ptrdiff_t width;
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                if (di * di - dnum > 0.0)
                    width = ((ptrdiff_t)(-sqrt(di * di - dnum) + di) + mask) & ~(ptrdiff_t)mask;
                else
                    width = n - i;
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else {
                width = n - i;
            }
            range[num_cpu + 1] = range[num_cpu] + width;
            num_cpu++;
            i += width;
        }
        for (int t = num_cpu + 1; t <= nthreads; ++t) range[t] = range[num_cpu];
    } else {
        range[nthreads] = n;
        ptrdiff_t i = 0;
        int num_cpu = 0;
        while (i < n && num_cpu < nthreads) {
            ptrdiff_t width;
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                if (di * di - dnum > 0.0)
                    width = ((ptrdiff_t)(-sqrt(di * di - dnum) + di) + mask) & ~(ptrdiff_t)mask;
                else
                    width = n - i;
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else {
                width = n - i;
            }
            range[nthreads - num_cpu - 1] = range[nthreads - num_cpu] - width;
            num_cpu++;
            i += width;
        }
        /* Fill any unused thread slots harmlessly (range stays in [0, n]). */
        for (int t = 0; t < nthreads - num_cpu; ++t) range[t] = range[nthreads - num_cpu];
    }
}

/* Per-thread kernel: tiled DTB_ENTRIES algorithm.  y is the thread's
 * accumulator slot (zero-initialized by caller).  Computes contributions
 * for columns [m_from, m_to). */
static void trmv_kernel_N(int upper, int nounit, ptrdiff_t n,
                          ptrdiff_t m_from, ptrdiff_t m_to,
                          const T *a, ptrdiff_t lda, const T *x, T *y)
{
    const ptrdiff_t TB = DTB_ENTRIES;
    for (ptrdiff_t is = m_from; is < m_to; is += TB) {
        ptrdiff_t min_i = (m_to - is < TB) ? m_to - is : TB;
        if (upper && is > 0) {
            /* Off-diagonal block above the tile: y[0..is) += A[0..is, is..is+min_i] * x[is..is+min_i]. */
            for (ptrdiff_t j = is; j < is + min_i; ++j) {
                T xj = x[j];
                const T *col = &A_(0, j);
                for (ptrdiff_t i = 0; i < is; ++i) y[i] += col[i] * xj;
            }
        }
        /* Diagonal tile. */
        for (ptrdiff_t i = is; i < is + min_i; ++i) {
            if (upper && i > is) {
                /* y[is..i) += x[i] * A[is..i, i]. */
                T xi = x[i];
                const T *col = &A_(0, i);
                for (ptrdiff_t k = is; k < i; ++k) y[k] += col[k] * xi;
            }
            /* Diagonal element. */
            if (nounit) y[i] += A_(i, i) * x[i];
            else        y[i] += x[i];
            if (!upper && i + 1 < is + min_i) {
                /* y[i+1..is+min_i) += x[i] * A[i+1..is+min_i, i]. */
                T xi = x[i];
                const T *col = &A_(0, i);
                for (ptrdiff_t k = i + 1; k < is + min_i; ++k) y[k] += col[k] * xi;
            }
        }
        if (!upper && is + min_i < n) {
            /* Off-diagonal block below: y[is+min_i..n) += A[is+min_i..n, is..is+min_i] * x[is..is+min_i]. */
            for (ptrdiff_t j = is; j < is + min_i; ++j) {
                T xj = x[j];
                const T *col = &A_(0, j);
                for (ptrdiff_t i = is + min_i; i < n; ++i) y[i] += col[i] * xj;
            }
        }
    }
}

static void trmv_kernel_T(int upper, int nounit, ptrdiff_t n,
                          ptrdiff_t m_from, ptrdiff_t m_to,
                          const T *a, ptrdiff_t lda, const T *x, T *y)
{
    const ptrdiff_t TB = DTB_ENTRIES;
    for (ptrdiff_t is = m_from; is < m_to; is += TB) {
        ptrdiff_t min_i = (m_to - is < TB) ? m_to - is : TB;
        if (upper && is > 0) {
            /* y[is..is+min_i) += A[0..is, is..is+min_i]^T * x[0..is). */
            for (ptrdiff_t j = is; j < is + min_i; ++j) {
                T s = 0.0;
                const T *col = &A_(0, j);
                for (ptrdiff_t i = 0; i < is; ++i) s += col[i] * x[i];
                y[j] += s;
            }
        }
        for (ptrdiff_t i = is; i < is + min_i; ++i) {
            if (upper && i > is) {
                /* y[i] += sum_{k=is..i} A[k, i] * x[k]. */
                T s = 0.0;
                const T *col = &A_(0, i);
                for (ptrdiff_t k = is; k < i; ++k) s += col[k] * x[k];
                y[i] += s;
            }
            if (nounit) y[i] += A_(i, i) * x[i];
            else        y[i] += x[i];
            if (!upper && i + 1 < is + min_i) {
                /* y[i] += sum_{k=i+1..is+min_i} A[k, i] * x[k]. */
                T s = 0.0;
                const T *col = &A_(0, i);
                for (ptrdiff_t k = i + 1; k < is + min_i; ++k) s += col[k] * x[k];
                y[i] += s;
            }
        }
        if (!upper && is + min_i < n) {
            /* y[is..is+min_i) += A[is+min_i..n, is..is+min_i]^T * x[is+min_i..n). */
            for (ptrdiff_t j = is; j < is + min_i; ++j) {
                T s = 0.0;
                const T *col = &A_(0, j);
                for (ptrdiff_t i = is + min_i; i < n; ++i) s += col[i] * x[i];
                y[j] += s;
            }
        }
    }
}

extern "C" void mtrmv_(const char *UPLO, const char *TRANS, const char *DIAG,
            const int *N, const T *a, const int *LDA,
            T *x, const int *INCX)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t lda  = (ptrdiff_t)(*LDA);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);

    if (n == 0) return;

    int upper  = (toupper((unsigned char)*UPLO)  == 'U');
    char trc   = (char)toupper((unsigned char)*TRANS);
    int trans  = (trc == 'T' || trc == 'C') ? 1 : 0;
    int nounit = (toupper((unsigned char)*DIAG) == 'N');

    if (incx < 0) x -= (n - 1) * incx;

#ifdef _OPENMP
    int nthreads = 1;
    if (n >= 50) {
        nthreads = omp_get_max_threads();
        if (n < 500 && nthreads > 2) nthreads = 2;
    }
    if (nthreads > 1) {
        T *buf_all = (T *)calloc((size_t)nthreads * (size_t)n, sizeof(T));
        ptrdiff_t *range_m = (ptrdiff_t *)malloc((size_t)(nthreads + 1) * sizeof(ptrdiff_t));
        T *xbuf = NULL;
        const T *xptr = x;
        if (incx != 1) {
            xbuf = (T *)malloc((size_t)n * sizeof(T));
            if (xbuf) {
                for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[i * incx];
                xptr = xbuf;
            }
        }
        if (buf_all && range_m && (incx == 1 || xbuf)) {
            trmv_partition(upper, n, nthreads, range_m);

            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                T *y = trans ? buf_all : &buf_all[(size_t)tid * (size_t)n];
                ptrdiff_t m_from, m_to;
                if (upper) {
                    m_from = range_m[nthreads - tid - 1];
                    m_to   = range_m[nthreads - tid];
                } else {
                    m_from = range_m[tid];
                    m_to   = range_m[tid + 1];
                }
                if (m_from < m_to) {
                    if (!trans) trmv_kernel_N(upper, nounit, n, m_from, m_to, a, lda, xptr, y);
                    else        trmv_kernel_T(upper, nounit, n, m_from, m_to, a, lda, xptr, y);
                }
            }

            /* For NoTrans: reduce slot[t] into slot[0] over the touched range. */
            if (!trans) {
                if (upper) {
                    for (int t = 1; t < nthreads; ++t) {
                        ptrdiff_t m_to_t = range_m[nthreads - t];
                        T *slot = &buf_all[(size_t)t * (size_t)n];
                        for (ptrdiff_t i = 0; i < m_to_t; ++i) buf_all[i] += slot[i];
                    }
                } else {
                    for (int t = 1; t < nthreads; ++t) {
                        ptrdiff_t m_from_t = range_m[t];
                        T *slot = &buf_all[(size_t)t * (size_t)n];
                        for (ptrdiff_t i = m_from_t; i < n; ++i) buf_all[i] += slot[i];
                    }
                }
            }

            /* Copy buf_all[0..n) back to x. */
            if (incx == 1) {
                for (ptrdiff_t i = 0; i < n; ++i) x[i] = buf_all[i];
            } else {
                for (ptrdiff_t i = 0; i < n; ++i) x[i * incx] = buf_all[i];
            }

            free(buf_all); free(range_m);
            if (xbuf) free(xbuf);
            return;
        }
        free(buf_all); free(range_m);
        if (xbuf) free(xbuf);
    }
#endif

    /* Serial Fortran-reference path — in-place, no scratch. */
    ptrdiff_t kx = 0;

    if (!trans) {
        if (upper) {
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != 0.0) {
                        T temp = x[j];
                        const T *aj = &A_(0, j);
                        for (ptrdiff_t i = 0; i < j; ++i) x[i] += temp * aj[i];
                        if (nounit) x[j] *= aj[j];
                    }
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[jx] != 0.0) {
                        T temp = x[jx];
                        ptrdiff_t ix = kx;
                        const T *aj = &A_(0, j);
                        for (ptrdiff_t i = 0; i < j; ++i) { x[ix] += temp * aj[i]; ix += incx; }
                        if (nounit) x[jx] *= aj[j];
                    }
                    jx += incx;
                }
            }
        } else {
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != 0.0) {
                        T temp = x[j];
                        const T *aj = &A_(0, j);
                        for (ptrdiff_t i = n - 1; i > j; --i) x[i] += temp * aj[i];
                        if (nounit) x[j] *= aj[j];
                    }
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[jx] != 0.0) {
                        T temp = x[jx];
                        ptrdiff_t ix = kx;
                        const T *aj = &A_(0, j);
                        for (ptrdiff_t i = n - 1; i > j; --i) { x[ix] += temp * aj[i]; ix -= incx; }
                        if (nounit) x[jx] *= aj[j];
                    }
                    jx -= incx;
                }
            }
        }
    } else {
        if (upper) {
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    T temp = x[j];
                    const T *aj = &A_(0, j);
                    if (nounit) temp *= aj[j];
                    for (ptrdiff_t i = j - 1; i >= 0; --i) temp += aj[i] * x[i];
                    x[j] = temp;
                }
            } else {
                ptrdiff_t jx = kx + (n - 1) * incx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    T temp = x[jx];
                    ptrdiff_t ix = jx;
                    const T *aj = &A_(0, j);
                    if (nounit) temp *= aj[j];
                    for (ptrdiff_t i = j - 1; i >= 0; --i) { ix -= incx; temp += aj[i] * x[ix]; }
                    x[jx] = temp;
                    jx -= incx;
                }
            }
        } else {
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    T temp = x[j];
                    const T *aj = &A_(0, j);
                    if (nounit) temp *= aj[j];
                    for (ptrdiff_t i = j + 1; i < n; ++i) temp += aj[i] * x[i];
                    x[j] = temp;
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    T temp = x[jx];
                    ptrdiff_t ix = jx;
                    const T *aj = &A_(0, j);
                    if (nounit) temp *= aj[j];
                    for (ptrdiff_t i = j + 1; i < n; ++i) { ix += incx; temp += aj[i] * x[ix]; }
                    x[jx] = temp;
                    jx += incx;
                }
            }
        }
    }
}

#undef A_
