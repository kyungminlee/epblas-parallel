/*
 * etbsv — kind10 (long double) triangular band solve.
 *   x := inv(A)*x or inv(A^T)*x, A triangular band with K+1 diagonals.
 */

#include <stddef.h>
#include <ctype.h>

typedef long double T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* x[lo..hi) -= tmp * cb[lo..hi); 8x unrolled (K=16 → exactly two iterations).
 * The contiguous NoTrans solve store-loops amortize loop-control overhead
 * across 8 independent stores (fp80 has no SIMD; this is the one lever).
 * Bit-exact: each x[i] -= tmp*cb[i] is an independent store, no reassociation. */
static inline void band_msub(T *restrict x, const T *restrict cb, T tmp,
                             ptrdiff_t lo, ptrdiff_t hi) {
    ptrdiff_t i = lo;
    for (; i + 8 <= hi; i += 8) {
        x[i]   -= tmp * cb[i];   x[i+1] -= tmp * cb[i+1];
        x[i+2] -= tmp * cb[i+2]; x[i+3] -= tmp * cb[i+3];
        x[i+4] -= tmp * cb[i+4]; x[i+5] -= tmp * cb[i+5];
        x[i+6] -= tmp * cb[i+6]; x[i+7] -= tmp * cb[i+7];
    }
    for (; i < hi; ++i) x[i] -= tmp * cb[i];
}

void etbsv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_, const int *k_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const ptrdiff_t N = *n_, K = *k_;
    const ptrdiff_t lda = *lda_, incx = *incx_;
    const T zero = 0.0L;
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
    const ptrdiff_t nounit = (up(diag) != 'U');

    if (N == 0) return;

    if (incx == 1) {
        /* incx==1 hot path: hoist the column base pointer so the two array
         * walks (band column + x) collapse onto a SINGLE byte-offset induction
         * (col[off+i] and x[i] share i). The strided branch below cannot
         * collapse (its ix induction is independent of the band index) so it
         * stays on the macro form.
         *
         * Codegen of these K-deep inner loops is delicate: the i_lo guard form
         * and the loop-bound form decide whether GCC fuses the per-column base
         * addresses into running inductions or recomputes them each column.
         * The right idiom differs per shape (store-loop vs accumulate-loop) and
         * is pinned by measurement at each loop, not unified. The NoTrans store
         * loops go through band_msub (8x manual unroll) — at K=16 the inner
         * trip count is short, so loop-control overhead dominates and GCC's own
         * rolling left par ~14% behind the netlib reference; unrolling closes
         * it. par is now parity-or-better than both ob and the netlib-fortran
         * reference (mig) on every incx==1 cell. */
        if (TR == 'N') {
            if (UPLO == 'U') {
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        const T *restrict col = &a[(size_t)j * lda];
                        const ptrdiff_t off = K - j;
                        if (nounit) x[j] /= col[K];
                        const T tmp = x[j];
                        const ptrdiff_t i_lo = (j - K > 0) ? (j - K) : 0;
                        band_msub(x, col + off, tmp, i_lo, j);
                    }
                }
            } else {
                for (ptrdiff_t j = 0; j < N; ++j) {
                    if (x[j] != zero) {
                        const T *restrict col = &a[(size_t)j * lda];
                        const ptrdiff_t off = -j;
                        if (nounit) x[j] /= col[0];
                        const T tmp = x[j];
                        const ptrdiff_t i_hi = (j + K < N - 1) ? (j + K) : (N - 1);
                        band_msub(x, col + off, tmp, j + 1, i_hi + 1);
                    }
                }
            }
        } else {
            if (UPLO == 'U') {
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T tmp = x[j];
                    const T *restrict col = &a[(size_t)j * lda];
                    const ptrdiff_t off = K - j;
                    /* (j > K), not the equivalent (j - K > 0): the unconditional
                     * j-K subtraction spawns an extra induction variable that
                     * blocks GCC from fusing &a[j*lda]+(K-j) into the single
                     * stride-(lda-1) band-diagonal pointer ob walks, forcing a
                     * per-column address recompute. The compare-only form lets
                     * the fusion happen → UTN par/ob 1.05->1.01, UTU ->0.97.
                     * (Opposite for the NoTrans store-loop above, which prefers
                     * the subtraction form — measured per-shape, do not unify.) */
                    const ptrdiff_t i_lo = (j > K) ? (j - K) : 0;
                    for (ptrdiff_t i = i_lo; i < j; ++i) tmp -= col[off + i] * x[i];
                    if (nounit) tmp /= col[K];
                    x[j] = tmp;
                }
            } else {
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    const T *restrict col = &a[(size_t)j * lda];
                    const ptrdiff_t off = -j;
                    const ptrdiff_t i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (ptrdiff_t i = i_hi - 1; i > j; --i) tmp -= col[off + i] * x[i];
                    if (nounit) tmp /= col[0];
                    x[j] = tmp;
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'U') {
                kx += (N - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    kx -= incx;
                    if (x[jx] != zero) {
                        ptrdiff_t ix = kx;
                        const ptrdiff_t L = K - j;
                        if (nounit) x[jx] /= A_(K, j);
                        const T tmp = x[jx];
                        const ptrdiff_t i_lo = (j - K > 0) ? (j - K) : 0;
                        for (ptrdiff_t i = j - 1; i >= i_lo; --i) {
                            x[ix] -= tmp * A_(L + i, j);
                            ix -= incx;
                        }
                    }
                    jx -= incx;
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < N; ++j) {
                    kx += incx;
                    if (x[jx] != zero) {
                        ptrdiff_t ix = kx;
                        if (nounit) x[jx] /= A_(0, j);
                        const T tmp = x[jx];
                        const ptrdiff_t i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
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
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T tmp = x[jx];
                    ptrdiff_t ix = kx;
                    const ptrdiff_t L = K - j;
                    const ptrdiff_t i_lo = (j - K > 0) ? (j - K) : 0;
                    for (ptrdiff_t i = i_lo; i < j; ++i) {
                        tmp -= A_(L + i, j) * x[ix];
                        ix += incx;
                    }
                    if (nounit) tmp /= A_(K, j);
                    x[jx] = tmp;
                    jx += incx;
                    if (j >= K) kx += incx;
                }
            } else {
                kx += (N - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    ptrdiff_t ix = kx;
                    const ptrdiff_t i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (ptrdiff_t i = i_hi - 1; i > j; --i) {
                        tmp -= A_(i - j, j) * x[ix];
                        ix -= incx;
                    }
                    if (nounit) tmp /= A_(0, j);
                    x[jx] = tmp;
                    jx -= incx;
                    if ((N - 1 - j) >= K) kx -= incx;
                }
            }
        }
    }
}

#undef A_
