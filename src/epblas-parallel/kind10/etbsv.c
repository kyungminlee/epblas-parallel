/*
 * etbsv — kind10 (long double) triangular band solve.
 *   x := inv(A)*x or inv(A^T)*x, A triangular band with K+1 diagonals.
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include "../common/epblas_facade.h"

typedef long double T;


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

static void etbsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t N, ptrdiff_t K,
    const T *restrict a, ptrdiff_t lda,
    T *restrict x, ptrdiff_t incx)
{
    const T zero = 0.0L;
    const char UPLO = blas_up(uplo);
    char TR = blas_up(trans);
    if (TR == 'C') TR = 'T';
    const bool nounit = (blas_up(diag) != 'U');

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
        /* Strided (incx != 1) branch. The ix induction is independent of the
         * band index, so it cannot collapse onto a single byte-offset walk the
         * way the incx==1 paths do. Instead hoist the column base pointer
         * (col = &a[j*lda]) ONCE per column and walk col[off+i] — faithful to
         * the ob/netlib form. With K=16 the inner trip count is short, so the
         * per-column addressing setup is a meaningful fraction; expressing the
         * band element through the A_(i,j) macro re-derived (size_t)j*lda at
         * each access and left par ~4% behind ob on the strided Upper cells.
         * Hoisting the pointer brings every strided cell to parity-or-better. */
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'U') {
                /* Upper NoTrans: the band x-window for column j ends just above
                 * the diagonal at row j-1, i.e. x[jx - incx], descending. ix is
                 * derived as jx - incx (drops the redundant running kx). The
                 * per-column `if (nounit)` divide is HOISTED out of the j-loop:
                 * left inside, GCC keeps the diag char live in a register to
                 * re-test it every column and spills the two hot loop-invariants
                 * (col[K] byte offset + x base ptr) to the stack — reloaded each
                 * nounit column (the UNN strided gap; the unit path skips the
                 * divide block, so only nounit was hit). Specializing per diag
                 * removes the per-column test so both pointers stay resident. */
                T *restrict xj = &x[kx + (N - 1) * incx];
                if (nounit) {
                    for (ptrdiff_t j = N - 1; j >= 0; --j) {
                        if (*xj != zero) {
                            const T *restrict col = &a[(size_t)j * lda];
                            const ptrdiff_t off = K - j;
                            *xj /= col[K];
                            const T tmp = *xj;
                            const ptrdiff_t i_lo = (j > K) ? (j - K) : 0;
                            T *restrict xi = xj - incx;
                            for (ptrdiff_t i = j - 1; i >= i_lo; --i) {
                                *xi -= tmp * col[off + i];
                                xi -= incx;
                            }
                        }
                        xj -= incx;
                    }
                } else {
                    for (ptrdiff_t j = N - 1; j >= 0; --j) {
                        if (*xj != zero) {
                            const T *restrict col = &a[(size_t)j * lda];
                            const ptrdiff_t off = K - j;
                            const T tmp = *xj;
                            const ptrdiff_t i_lo = (j > K) ? (j - K) : 0;
                            T *restrict xi = xj - incx;
                            for (ptrdiff_t i = j - 1; i >= i_lo; --i) {
                                *xi -= tmp * col[off + i];
                                xi -= incx;
                            }
                        }
                        xj -= incx;
                    }
                }
            } else {
                /* Lower NoTrans: the band x-window for column j starts at row
                 * j+1, i.e. x[jx + incx]. The faithful-ob form maintains a
                 * separate running kx == jx + incx — pure redundancy that costs
                 * an extra live induction. Deriving ix = jx + incx drops it,
                 * which lets the column bound stay in a register instead of
                 * spilling to the stack each column (the LNU unit-diag gap). */
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < N; ++j) {
                    if (x[jx] != zero) {
                        ptrdiff_t ix = jx + incx;
                        const T *restrict col = &a[(size_t)j * lda];
                        const ptrdiff_t off = -j;
                        if (nounit) x[jx] /= col[0];
                        const T tmp = x[jx];
                        const ptrdiff_t i_hi = (j + K < N - 1) ? (j + K) : (N - 1);
                        for (ptrdiff_t i = j + 1; i <= i_hi; ++i) {
                            x[ix] -= tmp * col[off + i];
                            ix += incx;
                        }
                    }
                    jx += incx;
                }
            }
        } else {
            if (UPLO == 'U') {
                /* Upper Trans: ix walks i_lo..j-1. The faithful-ob form keeps a
                 * running kx slid by a per-column conditional (if j>=K kx+=incx),
                 * which forces a cmov AND spills the loop-invariant column bound
                 * to the stack (reloaded every column — the UTU unit-diag gap,
                 * exposed once the per-column divide is absent). Deriving the
                 * start ix = jx - min(j,K)*incx from the diagonal index jx drops
                 * both the running kx and the cmov; the bound stays in a
                 * register. (i ascends for bit-exact accumulation order.) */
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T tmp = x[jx];
                    const T *restrict col = &a[(size_t)j * lda];
                    const ptrdiff_t off = K - j;
                    const ptrdiff_t i_lo = (j > K) ? (j - K) : 0;
                    ptrdiff_t ix = jx - (j - i_lo) * incx;
                    for (ptrdiff_t i = i_lo; i < j; ++i) {
                        tmp -= col[off + i] * x[ix];
                        ix += incx;
                    }
                    if (nounit) tmp /= col[K];
                    x[jx] = tmp;
                    jx += incx;
                }
            } else {
                kx += (N - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    ptrdiff_t ix = kx;
                    const T *restrict col = &a[(size_t)j * lda];
                    const ptrdiff_t off = -j;
                    const ptrdiff_t i_hi = (j + K < N - 1) ? (j + K) : (N - 1);
                    for (ptrdiff_t i = i_hi; i > j; --i) {
                        tmp -= col[off + i] * x[ix];
                        ix -= incx;
                    }
                    if (nounit) tmp /= col[0];
                    x[jx] = tmp;
                    jx -= incx;
                    if ((N - 1 - j) >= K) kx -= incx;
                }
            }
        }
    }
}

EPBLAS_FACADE_TBMV(etbsv, T)

#undef A_
