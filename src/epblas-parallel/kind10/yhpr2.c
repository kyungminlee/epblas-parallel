/*
 * yhpr2 — kind10 complex Hermitian packed rank-2 update.
 *   A := alpha*x*y^H + conj(alpha)*y*x^H + A
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define YHPR2_OMP_MIN 64

typedef _Complex long double T;
typedef long double TR;
static inline T cconj(T z) { return ~z; }

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

/* Per-column rank-2 updates, carved out as their own functions so the inner
 * loop compiles with clean x87 register allocation. Inlined into the
 * `omp parallel for` body, the upper-triangle loop loses ~10% (the outlined
 * region spills the kept-resident operands); keeping it a separate noinline
 * function restores parity with the reference and lets both the serial and
 * threaded paths share one tight loop. The Hermitian diagonal is forced real
 * here: the off-diagonal run plus the single real diagonal write. */
__attribute__((noinline))
static void yhpr2_col_upper(int j, T t1, T t2,
                            const T *restrict x, const T *restrict y, T *restrict ap) {
    T *restrict c = ap + (size_t)j * (j + 1) / 2;
    for (int i = 0; i < j; ++i) c[i] += x[i] * t1 + y[i] * t2;
    c[j] = (TR)__real__ c[j] + (TR)__real__ (x[j] * t1 + y[j] * t2);
}

__attribute__((noinline))
static void yhpr2_col_lower(int j, int N, T t1, T t2,
                            const T *restrict x, const T *restrict y, T *restrict ap) {
    /* Pre-advance the off-diagonal bases so the loop runs 0-based over a single
     * induction variable indexing three pointers — the exact tight form gcc
     * picks for the upper helper. A loop that starts at i=1 (or indexes the
     * original arrays by the absolute j+1..N-1) instead makes gcc walk three
     * separate pointers with an extra increment per iteration (~7% on the
     * lower triangle). Diagonal last so the loop compiles on a clean x87 stack. */
    const int mo = N - j - 1;
    T *restrict c0 = ap + ((size_t)j * N - (size_t)j * (j - 1) / 2);
    T *restrict c = c0 + 1;
    const T *restrict xc = x + j + 1, *restrict yc = y + j + 1;
    for (int i = 0; i < mo; ++i) c[i] += xc[i] * t1 + yc[i] * t2;
    c0[0] = (TR)__real__ c0[0] + (TR)__real__ (x[j] * t1 + y[j] * t2);
}

void yhpr2_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *restrict x, const int *incx_,
    const T *restrict y, const int *incy_,
    T *restrict ap,
    size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, incy = *incy_;
    const T alpha = *alpha_;
    const T zero = 0.0L + 0.0Li;
    const char UPLO = up(uplo);

    if (N == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
        /* schedule(static,1): column j touches j (upper) or N-1-j (lower)
         * off-diagonal packed elements, so a contiguous static block hands one
         * thread the heavy triangle end and starves the rest (par caps at ~2x
         * on 4 cores). Cyclic static,1 interleaves short and long columns
         * across the team, balancing the skew symmetrically for both UPLO. The
         * Hermitian diagonal is forced real every column — including the
         * skipped (x[j]==y[j]==0) ones — so the else branch still writes it. */
        if (UPLO == 'U') {
#ifdef _OPENMP
            const int use_omp = (N >= YHPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
            for (int j = 0; j < N; ++j) {
                if (x[j] != zero || y[j] != zero)
                    yhpr2_col_upper(j, alpha * cconj(y[j]), cconj(alpha * x[j]), x, y, ap);
                else {
                    const size_t kk = (size_t)j * (j + 1) / 2;
                    ap[kk + j] = (TR)__real__ ap[kk + j];
                }
            }
        } else {
#ifdef _OPENMP
            const int use_omp = (N >= YHPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
            for (int j = 0; j < N; ++j) {
                if (x[j] != zero || y[j] != zero)
                    yhpr2_col_lower(j, N, alpha * cconj(y[j]), cconj(alpha * x[j]), x, y, ap);
                else {
                    const size_t kk = (size_t)j * N - (size_t)j * (j - 1) / 2;
                    ap[kk] = (TR)__real__ ap[kk];
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        int kk = 0;
        int jx = kx, jy = ky;
        if (UPLO == 'U') {
            for (int j = 0; j < N; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const T t1 = alpha * cconj(y[jy]);
                    const T t2 = cconj(alpha * x[jx]);
                    int ix = kx, iy = ky;
                    for (int k = kk; k < kk + j; ++k) {
                        ap[k] += x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                    ap[kk + j] = (TR)__real__ ap[kk + j] + (TR)__real__ (x[jx] * t1 + y[jy] * t2);
                } else {
                    ap[kk + j] = (TR)__real__ ap[kk + j];
                }
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            for (int j = 0; j < N; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const T t1 = alpha * cconj(y[jy]);
                    const T t2 = cconj(alpha * x[jx]);
                    ap[kk] = (TR)__real__ ap[kk] + (TR)__real__ (x[jx] * t1 + y[jy] * t2);
                    int ix = jx, iy = jy;
                    for (int k = kk + 1; k < kk + N - j; ++k) {
                        ix += incx; iy += incy;
                        ap[k] += x[ix] * t1 + y[iy] * t2;
                    }
                } else {
                    ap[kk] = (TR)__real__ ap[kk];
                }
                jx += incx; jy += incy;
                kk += N - j;
            }
        }
    }
}
