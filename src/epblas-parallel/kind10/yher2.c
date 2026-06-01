/*
 * yher2 — kind10 complex Hermitian rank-2 update.
 *   A := alpha · x · yᴴ + conj(alpha) · y · xᴴ + A
 * alpha is complex. Diagonal of A stays real on output.
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define YHER2_OMP_MIN 64

typedef _Complex long double T;
typedef long double R;
static const T ZERO = 0.0L + 0.0Li;
static inline T cconj(T z) { return ~z; }

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* Contiguous (incx=incy=1) Hermitian rank-2 update over the column range
 * [j0,j1).  Per Netlib ZHER2: TEMP1 = ALPHA·conj(Y(J)), TEMP2 = conj(ALPHA·X(J)),
 * A(i,j) += X(i)·TEMP1 + Y(i)·TEMP2; the diagonal entry stays real.
 *
 * UPPER and LOWER are SEPARATE (non-inlined) functions — not one branchy body
 * — for two reasons:
 *   1. Isolation/regalloc — each is compiled away from the other and from the
 *      strided/OMP scaffolding, so gcc keeps the complex temporaries on the
 *      8-deep x87 register stack.  A shared body forces a compromise schedule
 *      that suits one triangle and costs the other ~4%; epblas-openblas emits
 *      two distinct loops for exactly this reason.
 *   2. The serial path calls one ONCE over [0,N) — its hot loop is inlined with
 *      zero per-column overhead; the OMP path calls it per column, where that
 *      overhead is hidden by parallelism.
 * Each column streams sequentially — UPPER writes 0..j-1 then j, LOWER writes
 * j then j+1..N-1.  The diagonal's real contribution is reduced to a SCALAR
 * (dadd) BEFORE the off-diagonal loop, consuming x[j]/y[j] up front; only that
 * one long double is carried across the loop (instead of the two complex
 * endpoints), which keeps t1/t2 register-resident (fld=9/iter).  Diagonal and
 * off-diagonal entries are disjoint, so the order is bit-identical. */
__attribute__((noinline))
static void yher2_contig_U(int j0, int j1, T alpha,
                           const T *restrict x, const T *restrict y,
                           T *restrict a, int lda)
{
    for (int j = j0; j < j1; ++j) {
        T *aj = &A_(0, j);
        if (x[j] != ZERO || y[j] != ZERO) {
            const T t1 = alpha * cconj(y[j]);
            const T t2 = cconj(alpha * x[j]);
            const R dadd = __real__ (x[j] * t1 + y[j] * t2);
            for (int i = 0; i < j; ++i) aj[i] += x[i] * t1 + y[i] * t2;
            aj[j] = __real__ aj[j] + dadd;
        }
    }
}

__attribute__((noinline))
static void yher2_contig_L(int j0, int j1, int N, T alpha,
                           const T *restrict x, const T *restrict y,
                           T *restrict a, int lda)
{
    for (int j = j0; j < j1; ++j) {
        T *aj = &A_(0, j);
        if (x[j] != ZERO || y[j] != ZERO) {
            const T t1 = alpha * cconj(y[j]);
            const T t2 = cconj(alpha * x[j]);
            aj[j] = __real__ aj[j] + __real__ (x[j] * t1 + y[j] * t2);
            for (int i = j + 1; i < N; ++i) aj[i] += x[i] * t1 + y[i] * t2;
        }
    }
}

void yher2_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *restrict x, const int *incx_,
    const T *restrict y, const int *incy_,
    T *restrict a, const int *lda_,
    size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, incy = *incy_, lda = *lda_;
    const T alpha = *alpha_;
    const char UPLO = up(uplo);

    if (N == 0 || alpha == ZERO) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= YHER2_OMP_MIN && blas_omp_max_threads() > 1
                             && !omp_in_parallel());
#else
        const int use_omp = 0;
#endif
        /* All column work runs through yher2_contig_{U,L}().  The serial path
         * issues ONE call over [0,N) — the hot loop is inlined there with no
         * per-column overhead.  The OMP path round-robins single columns:
         * schedule(static, 1) balances the triangular work (linear in (N-j) for
         * L, j for U) — Rule 49; plain schedule(static) hands thread 0 the
         * heaviest block.  Branching on use_omp at C source level (Add-16)
         * avoids `if(use_omp)` outlining the region and paying GOMP_parallel at
         * OMP=1; the per-column call cost is hidden by parallelism. */
        if (use_omp) {
#ifdef _OPENMP
            if (UPLO == 'L') {
                #pragma omp parallel for schedule(static, 1)
                for (int j = 0; j < N; ++j)
                    yher2_contig_L(j, j + 1, N, alpha, x, y, a, lda);
            } else {
                #pragma omp parallel for schedule(static, 1)
                for (int j = 0; j < N; ++j)
                    yher2_contig_U(j, j + 1, alpha, x, y, a, lda);
            }
#endif
        } else if (UPLO == 'L') {
            yher2_contig_L(0, N, N, alpha, x, y, a, lda);
        } else {
            yher2_contig_U(0, N, alpha, x, y, a, lda);
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        for (int j = 0; j < N; ++j) {
            const T xj = x[kx + j * incx];
            const T yj = y[ky + j * incy];
            if (xj != ZERO || yj != ZERO) {
                const T temp1 = alpha * cconj(yj);
                const T temp2 = cconj(alpha * xj);
                if (UPLO == 'L') {
                    for (int i = j + 1; i < N; ++i)
                        A_(i, j) += x[kx + i * incx] * temp1 + y[ky + i * incy] * temp2;
                    A_(j, j) = __real__ A_(j, j) + __real__ (xj * temp1 + yj * temp2);
                } else {
                    for (int i = 0; i < j; ++i)
                        A_(i, j) += x[kx + i * incx] * temp1 + y[ky + i * incy] * temp2;
                    A_(j, j) = __real__ A_(j, j) + __real__ (xj * temp1 + yj * temp2);
                }
            }
        }
    }
}

#undef A_
