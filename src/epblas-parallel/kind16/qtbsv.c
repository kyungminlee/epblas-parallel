/*
 * qtbsv — kind16 (__float128) triangular band solve.
 *   x := inv(A)*x or inv(A^T)*x, A triangular band with K+1 diagonals.
 */

#include <stddef.h>
#include <stdbool.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>
#include "../common/epblas_facade.h"

typedef __float128 TR;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* Contiguous Trans/ConjTrans-Upper solve, phase-split: initial triangle
 * (j <= k, dot always starts at x[0]) then steady band (fixed k-length
 * dot), both pure pointer walks.  The band dot is only k iterations, so
 * a generic column loop that recomputes max(j-k,0) and rebuilds both
 * base pointers per column doesn't amortize (~2% vs ob at N=128).
 * Ascending i order kept => bit-exact.  noinline keeps this leaf's
 * layout independent of the dispatch body. */
__attribute__((noinline, noclone))
static void qtbsv_trans_upper_contig(
    ptrdiff_t n, ptrdiff_t k, bool nounit,
    const TR *restrict a, ptrdiff_t lda, TR *restrict x)
{
    const ptrdiff_t jt = (k + 1 < n) ? (k + 1) : n;
    const TR *c = a + k;                 /* c[i] = A_(k-j+i, j); c[j] = diag */
    for (ptrdiff_t j = 0; j < jt; ++j, c += lda - 1) {
        TR tmp = x[j];
        for (ptrdiff_t i = 0; i < j; ++i) tmp -= c[i] * x[i];
        if (nounit) tmp /= c[j];
        x[j] = tmp;
    }
    const TR *b = a + (size_t)jt * lda;  /* &A_(0, j) for j = k+1.. */
    for (ptrdiff_t j = jt; j < n; ++j, b += lda) {
        const TR *xi = x + (j - k);
        TR tmp = x[j];
        for (ptrdiff_t i = 0; i < k; ++i) tmp -= b[i] * xi[i];
        if (nounit) tmp /= b[k];
        x[j] = tmp;
    }
}

/* Contiguous NoTrans-Lower solve, same phase-split treatment: steady band
 * (fixed k-length AXPY off the column pointer) then final triangle
 * (shrinking AXPY), both pure pointer walks.  Ascending i order kept =>
 * bit-exact.  noinline keeps this leaf's layout independent of the
 * dispatch body. */
__attribute__((noinline, noclone))
static void qtbsv_notrans_lower_contig(
    ptrdiff_t n, ptrdiff_t k, bool nounit,
    const TR *restrict a, ptrdiff_t lda, TR *restrict x)
{
    const ptrdiff_t js = (n - k > 0) ? (n - k) : 0;  /* j < js: full k-length AXPY */
    const TR *col = a;                               /* col[t] = A_(t, j); col[0] = diag */
    ptrdiff_t j = 0;
    for (; j < js; ++j, col += lda) {
        if (x[j] != 0.0Q) {
            if (nounit) x[j] /= col[0];
            const TR tmp = x[j];
            TR *xj = x + j;
            for (ptrdiff_t t = 1; t <= k; ++t) xj[t] -= tmp * col[t];
        }
    }
    for (; j < n; ++j, col += lda) {
        if (x[j] != 0.0Q) {
            if (nounit) x[j] /= col[0];
            const TR tmp = x[j];
            TR *xj = x + j;
            const ptrdiff_t len = n - 1 - j;
            for (ptrdiff_t t = 1; t <= len; ++t) xj[t] -= tmp * col[t];
        }
    }
}

/* Contiguous NoTrans-Upper solve, phase-split like its NoTrans-Lower twin:
 * steady band (fixed k-length AXPY walking backward from the diagonal at
 * c[0] = A_(k,j)) for j >= k, then final triangle (shrinking AXPY) for
 * j < k.  Element order matches the generic arm (t = j-i ascending 1..len,
 * i.e. i descending from j-1) => bit-exact.  noinline+noclone keeps this
 * leaf's layout independent of the dispatch body. */
__attribute__((noinline, noclone))
static void qtbsv_notrans_upper_contig(
    ptrdiff_t n, ptrdiff_t k, bool nounit,
    const TR *restrict a, ptrdiff_t lda, TR *restrict x)
{
    const ptrdiff_t jt = (k < n) ? k : n;          /* j < jt: shrinking AXPY */
    const TR *c = a + (size_t)(n - 1) * lda + k;   /* c[0] = A_(k,j) diag; c[-t] = A_(k-t,j) */
    ptrdiff_t j = n - 1;
    for (; j >= jt; --j, c -= lda) {
        if (x[j] != 0.0Q) {
            if (nounit) x[j] /= c[0];
            const TR tmp = x[j];
            TR *xj = x + j;
            for (ptrdiff_t t = 1; t <= k; ++t) xj[-t] -= tmp * c[-t];
        }
    }
    for (; j >= 0; --j, c -= lda) {
        if (x[j] != 0.0Q) {
            if (nounit) x[j] /= c[0];
            const TR tmp = x[j];
            TR *xj = x + j;
            for (ptrdiff_t t = 1; t <= j; ++t) xj[-t] -= tmp * c[-t];
        }
    }
}

/* Contiguous Trans/ConjTrans-Lower solve, phase-split like its Trans-Upper
 * twin: initial triangle (shrinking dot, j walking down from n-1) then
 * steady band (fixed k-length dot), both pure pointer walks off the column
 * base c[0] = A_(0,j).  Element order matches the generic arm (t = i-j
 * DESCENDING len..1, i.e. i descending from i_hi-1) => bit-exact. */
__attribute__((noinline, noclone))
static void qtbsv_trans_lower_contig(
    ptrdiff_t n, ptrdiff_t k, bool nounit,
    const TR *restrict a, ptrdiff_t lda, TR *restrict x)
{
    const ptrdiff_t js = (n - k > 0) ? (n - k) : 0;  /* j >= js: shortened dot */
    const TR *c = a + (size_t)(n - 1) * lda;         /* c[t] = A_(t, j); c[0] = diag */
    ptrdiff_t j = n - 1;
    for (; j >= js; --j, c -= lda) {
        TR tmp = x[j];
        const TR *xj = x + j;
        const ptrdiff_t len = n - 1 - j;
        for (ptrdiff_t t = len; t >= 1; --t) tmp -= c[t] * xj[t];
        if (nounit) tmp /= c[0];
        x[j] = tmp;
    }
    for (; j >= 0; --j, c -= lda) {
        TR tmp = x[j];
        const TR *xj = x + j;
        for (ptrdiff_t t = k; t >= 1; --t) tmp -= c[t] * xj[t];
        if (nounit) tmp /= c[0];
        x[j] = tmp;
    }
}

/* Strided Trans/ConjTrans-Upper solve, quarantined VERBATIM from the
 * dispatch body: shrinking the dispatch (contiguous arms became leaf
 * calls) re-laid-out this untouched arm and cost it ~1% systematically
 * (UTN/UTU strided, all sizes).  Its own noinline+noclone leaf pins the
 * layout independent of future dispatch edits.  Body unchanged =>
 * bit-exact. */
__attribute__((noinline, noclone))
static void qtbsv_trans_upper_strided(
    ptrdiff_t n, ptrdiff_t k, bool nounit,
    const TR *restrict a, ptrdiff_t lda,
    TR *restrict x, ptrdiff_t incx, ptrdiff_t kx)
{
    ptrdiff_t jx = kx;
    for (ptrdiff_t j = 0; j < n; ++j) {
        TR tmp = x[jx];
        ptrdiff_t ix = kx;
        const ptrdiff_t L = k - j;
        const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
        for (ptrdiff_t i = i_lo; i < j; ++i) {
            tmp -= A_(L + i, j) * x[ix];
            ix += incx;
        }
        if (nounit) tmp /= A_(k, j);
        x[jx] = tmp;
        jx += incx;
        if (j >= k) kx += incx;
    }
}

void qtbsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n, ptrdiff_t k,
    const TR *restrict a, ptrdiff_t lda,
    TR *restrict x, ptrdiff_t incx)
{
    const TR zero = 0.0Q;
    const char UPLO = blas_up(uplo);
    char TRANS = blas_up(trans);
    if (TRANS == 'C') TRANS = 'T';
    const bool nounit = (blas_up(diag) != 'U');

    if (n == 0) return;

    if (incx == 1) {
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                qtbsv_notrans_upper_contig(n, k, nounit, a, lda, x);
            } else {
                qtbsv_notrans_lower_contig(n, k, nounit, a, lda, x);
            }
        } else {
            if (UPLO == 'U') {
                qtbsv_trans_upper_contig(n, k, nounit, a, lda, x);
            } else {
                qtbsv_trans_lower_contig(n, k, nounit, a, lda, x);
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    kx -= incx;
                    if (x[jx] != zero) {
                        ptrdiff_t ix = kx;
                        const ptrdiff_t L = k - j;
                        if (nounit) x[jx] /= A_(k, j);
                        const TR tmp = x[jx];
                        const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                        for (ptrdiff_t i = j - 1; i >= i_lo; --i) {
                            x[ix] -= tmp * A_(L + i, j);
                            ix -= incx;
                        }
                    }
                    jx -= incx;
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    kx += incx;
                    if (x[jx] != zero) {
                        ptrdiff_t ix = kx;
                        if (nounit) x[jx] /= A_(0, j);
                        const TR tmp = x[jx];
                        const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                        for (ptrdiff_t i = j + 1; i < i_hi; ++i) {
                            x[ix] -= tmp * A_(i - j, j);
                            ix += incx;
                        }
                    }
                    jx += incx;
                }
            }
        } else {
            if (UPLO == 'U') {
                qtbsv_trans_upper_strided(n, k, nounit, a, lda, x, incx, kx);
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TR tmp = x[jx];
                    ptrdiff_t ix = kx;
                    const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                    for (ptrdiff_t i = i_hi - 1; i > j; --i) {
                        tmp -= A_(i - j, j) * x[ix];
                        ix -= incx;
                    }
                    if (nounit) tmp /= A_(0, j);
                    x[jx] = tmp;
                    jx -= incx;
                    if ((n - 1 - j) >= k) kx -= incx;
                }
            }
        }
    }
}

EPBLAS_FACADE_TBMV(qtbsv, TR)

#undef A_
