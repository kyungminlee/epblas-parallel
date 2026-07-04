/*
 * xtbsv — kind16 complex triangular band solve.
 *   x := inv(op(A)) * x   (op = A, A^T, or A^H), A triangular band, K+1 diags.
 *
 * Serial — back/forward substitution; O(N*K) with a K-deep loop-carried
 * recurrence, so there is no parallel path. The four CONTIGUOUS arms are
 * phase-split pointer walks (initial/final triangle + steady fixed-k band),
 * ported from the qtbsv split arms: the band loop is only ~k iterations, so
 * a generic column loop that recomputes the min/max clamp and rebuilds the
 * column base pointer per column doesn't amortize. Conj is folded into
 * CONJIF (operand pick only — element order unchanged, as in ytbsv) instead
 * of duplicated loop pairs. Element order matches the netlib transcription
 * arm-for-arm => bit-exact. The strided Trans/ConjTrans arms are quarantined
 * VERBATIM in noinline+noclone leaves: shrinking the dispatch re-laid-out
 * the untouched arms in the qtbsv/ytbsv twins and cost them ~1-4%
 * systematically; body unchanged => bit-exact.
 */

#include <stddef.h>
#include <stdbool.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>
#include "../common/epblas_facade.h"

typedef __complex128 TC;



static inline TC cconj(TC z) { return conjq(z); }
#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define CONJIF(z) (noconj ? (z) : cconj(z))

/* Contiguous NoTrans-Upper solve, phase-split: steady band (fixed k-length
 * AXPY walking backward from the diagonal at c[0] = A_(k,j)) for j >= k,
 * then final triangle (shrinking AXPY) for j < k.  Element order matches
 * the generic arm (t = j-i ascending 1..len, i.e. i descending from j-1)
 * => bit-exact.  noinline+noclone keeps this leaf's layout independent of
 * the dispatch body. */
__attribute__((noinline, noclone))
static void xtbsv_notrans_upper_contig(
    ptrdiff_t n, ptrdiff_t k, bool nounit,
    const TC *restrict a, ptrdiff_t lda, TC *restrict x)
{
    const TC zero = 0.0Q + 0.0Qi;
    const ptrdiff_t jt = (k < n) ? k : n;          /* j < jt: shrinking AXPY */
    const TC *c = a + (size_t)(n - 1) * lda + k;   /* c[0] = A_(k,j) diag; c[-t] = A_(k-t,j) */
    ptrdiff_t j = n - 1;
    for (; j >= jt; --j, c -= lda) {
        if (x[j] != zero) {
            if (nounit) x[j] /= c[0];
            const TC tmp = x[j];
            TC *xj = x + j;
            for (ptrdiff_t t = 1; t <= k; ++t) xj[-t] -= tmp * c[-t];
        }
    }
    for (; j >= 0; --j, c -= lda) {
        if (x[j] != zero) {
            if (nounit) x[j] /= c[0];
            const TC tmp = x[j];
            TC *xj = x + j;
            for (ptrdiff_t t = 1; t <= j; ++t) xj[-t] -= tmp * c[-t];
        }
    }
}

/* Contiguous NoTrans-Lower solve, phase-split: steady band (fixed k-length
 * AXPY off the column pointer) then final triangle (shrinking AXPY), both
 * pure pointer walks.  Ascending i order kept => bit-exact. */
__attribute__((noinline, noclone))
static void xtbsv_notrans_lower_contig(
    ptrdiff_t n, ptrdiff_t k, bool nounit,
    const TC *restrict a, ptrdiff_t lda, TC *restrict x)
{
    const TC zero = 0.0Q + 0.0Qi;
    const ptrdiff_t js = (n - k > 0) ? (n - k) : 0;  /* j < js: full k-length AXPY */
    const TC *col = a;                               /* col[t] = A_(t, j); col[0] = diag */
    ptrdiff_t j = 0;
    for (; j < js; ++j, col += lda) {
        if (x[j] != zero) {
            if (nounit) x[j] /= col[0];
            const TC tmp = x[j];
            TC *xj = x + j;
            for (ptrdiff_t t = 1; t <= k; ++t) xj[t] -= tmp * col[t];
        }
    }
    for (; j < n; ++j, col += lda) {
        if (x[j] != zero) {
            if (nounit) x[j] /= col[0];
            const TC tmp = x[j];
            TC *xj = x + j;
            const ptrdiff_t len = n - 1 - j;
            for (ptrdiff_t t = 1; t <= len; ++t) xj[t] -= tmp * col[t];
        }
    }
}

/* Contiguous Trans/ConjTrans-Upper solve, phase-split: initial triangle
 * (j <= k, dot always starts at x[0]) then steady band (fixed k-length
 * dot), both pure pointer walks.  Ascending i order kept, conj picked per
 * operand by CONJIF => bit-exact. */
__attribute__((noinline, noclone))
static void xtbsv_trans_upper_contig(
    ptrdiff_t n, ptrdiff_t k, bool noconj, bool nounit,
    const TC *restrict a, ptrdiff_t lda, TC *restrict x)
{
    const ptrdiff_t jt = (k + 1 < n) ? (k + 1) : n;
    const TC *c = a + k;                 /* c[i] = A_(k-j+i, j); c[j] = diag */
    for (ptrdiff_t j = 0; j < jt; ++j, c += lda - 1) {
        TC tmp = x[j];
        for (ptrdiff_t i = 0; i < j; ++i) tmp -= CONJIF(c[i]) * x[i];
        if (nounit) tmp /= CONJIF(c[j]);
        x[j] = tmp;
    }
    const TC *b = a + (size_t)jt * lda;  /* &A_(0, j) for j = k+1.. */
    for (ptrdiff_t j = jt; j < n; ++j, b += lda) {
        const TC *xi = x + (j - k);
        TC tmp = x[j];
        for (ptrdiff_t i = 0; i < k; ++i) tmp -= CONJIF(b[i]) * xi[i];
        if (nounit) tmp /= CONJIF(b[k]);
        x[j] = tmp;
    }
}

/* Contiguous Trans/ConjTrans-Lower solve, phase-split: initial triangle
 * (shrinking dot, j walking down from n-1) then steady band (fixed
 * k-length dot), both pure pointer walks off the column base
 * c[0] = A_(0,j).  Element order matches the generic arm (t = i-j
 * DESCENDING len..1, i.e. i descending from i_hi-1), conj picked per
 * operand by CONJIF => bit-exact. */
__attribute__((noinline, noclone))
static void xtbsv_trans_lower_contig(
    ptrdiff_t n, ptrdiff_t k, bool noconj, bool nounit,
    const TC *restrict a, ptrdiff_t lda, TC *restrict x)
{
    const ptrdiff_t js = (n - k > 0) ? (n - k) : 0;  /* j >= js: shortened dot */
    const TC *c = a + (size_t)(n - 1) * lda;         /* c[t] = A_(t, j); c[0] = diag */
    ptrdiff_t j = n - 1;
    for (; j >= js; --j, c -= lda) {
        TC tmp = x[j];
        const TC *xj = x + j;
        const ptrdiff_t len = n - 1 - j;
        for (ptrdiff_t t = len; t >= 1; --t) tmp -= CONJIF(c[t]) * xj[t];
        if (nounit) tmp /= CONJIF(c[0]);
        x[j] = tmp;
    }
    for (; j >= 0; --j, c -= lda) {
        TC tmp = x[j];
        const TC *xj = x + j;
        for (ptrdiff_t t = k; t >= 1; --t) tmp -= CONJIF(c[t]) * xj[t];
        if (nounit) tmp /= CONJIF(c[0]);
        x[j] = tmp;
    }
}

/* Strided Trans/ConjTrans-Upper solve, body VERBATIM from the old dispatch
 * loop; noinline+noclone quarantines its layout against dispatch edits. */
__attribute__((noinline, noclone))
static void xtbsv_trans_upper_strided(
    ptrdiff_t n, ptrdiff_t k, bool noconj, bool nounit,
    const TC *restrict a, ptrdiff_t lda,
    TC *restrict x, ptrdiff_t incx, ptrdiff_t kx)
{
    ptrdiff_t jx = kx;
    for (ptrdiff_t j = 0; j < n; ++j) {
        TC tmp = x[jx];
        ptrdiff_t ix = kx;
        const ptrdiff_t L = k - j;
        const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
        for (ptrdiff_t i = i_lo; i < j; ++i) {
            const TC aij = noconj ? A_(L + i, j) : cconj(A_(L + i, j));
            tmp -= aij * x[ix];
            ix += incx;
        }
        if (nounit) tmp /= (noconj ? A_(k, j) : cconj(A_(k, j)));
        x[jx] = tmp;
        jx += incx;
        if (j >= k) kx += incx;
    }
}

/* Strided Trans/ConjTrans-Lower solve, body VERBATIM from the old dispatch
 * loop; noinline+noclone quarantines its layout against dispatch edits. */
__attribute__((noinline, noclone))
static void xtbsv_trans_lower_strided(
    ptrdiff_t n, ptrdiff_t k, bool noconj, bool nounit,
    const TC *restrict a, ptrdiff_t lda,
    TC *restrict x, ptrdiff_t incx, ptrdiff_t kx)
{
    kx += (n - 1) * incx;
    ptrdiff_t jx = kx;
    for (ptrdiff_t j = n - 1; j >= 0; --j) {
        TC tmp = x[jx];
        ptrdiff_t ix = kx;
        const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
        for (ptrdiff_t i = i_hi - 1; i > j; --i) {
            const TC aij = noconj ? A_(i - j, j) : cconj(A_(i - j, j));
            tmp -= aij * x[ix];
            ix -= incx;
        }
        if (nounit) tmp /= (noconj ? A_(0, j) : cconj(A_(0, j)));
        x[jx] = tmp;
        jx -= incx;
        if ((n - 1 - j) >= k) kx -= incx;
    }
}

void xtbsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n, ptrdiff_t k,
    const TC *restrict a, ptrdiff_t lda,
    TC *restrict x, ptrdiff_t incx)
{
    const TC zero = 0.0Q + 0.0Qi;
    const char UPLO = blas_up(uplo);
    const char TRANS = blas_up(trans);
    const bool noconj = (TRANS == 'T');
    const bool nounit = (blas_up(diag) != 'U');

    if (n == 0) return;

    if (incx == 1) {
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                xtbsv_notrans_upper_contig(n, k, nounit, a, lda, x);
            } else {
                xtbsv_notrans_lower_contig(n, k, nounit, a, lda, x);
            }
        } else {
            if (UPLO == 'U') {
                xtbsv_trans_upper_contig(n, k, noconj, nounit, a, lda, x);
            } else {
                xtbsv_trans_lower_contig(n, k, noconj, nounit, a, lda, x);
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
                        const TC tmp = x[jx];
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
                        const TC tmp = x[jx];
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
                xtbsv_trans_upper_strided(n, k, noconj, nounit, a, lda, x, incx, kx);
            } else {
                xtbsv_trans_lower_strided(n, k, noconj, nounit, a, lda, x, incx, kx);
            }
        }
    }
}

EPBLAS_FACADE_TBMV(xtbsv, TC)

#undef A_
#undef CONJIF
