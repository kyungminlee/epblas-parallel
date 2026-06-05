/*
 * qsyrk_serial.c — kind16 (REAL(KIND=16) / __float128) symmetric rank-k
 * update, serial core.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the
 * uplo/trans decode and the per-column compute core (declared in
 * qsyrk_kernel.h), plus the public `qsyrk_serial_` Fortran entry. No
 * OpenMP anywhere on this call path — safe to invoke from inside another
 * function's `#pragma omp parallel` region.
 *
 * Both qsyrk_serial_ and the parallel qsyrk_ drive numerics through
 * qsyrk_core, so the two paths are bitwise-identical.
 */

#include "qsyrk_kernel.h"
#include <ctype.h>
#include <quadmath.h>

typedef qsyrk_T T;

char qsyrk_uplo(const char *p) {
    return (char)toupper((unsigned char)*p);
}

int qsyrk_trans_code(const char *p, size_t len) {
    (void)len;
    char c = (char)toupper((unsigned char)*p);
    return (c == 'C') ? 'T' : c;
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

void qsyrk_core(
    int j0, int j1,
    char UPLO, int TR, int N, int K,
    T alpha, T beta,
    const T *a, int lda,
    T *c, int ldc)
{
    const T zero = 0.0Q, one = 1.0Q;

    for (int j = j0; j < j1; ++j) {
        const int i_lo = (UPLO == 'L') ? j : 0;
        const int i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + (size_t)j * ldc;

        /* beta scale of the UPLO slice of column j. */
        if (beta == zero)      for (int i = i_lo; i < i_hi; ++i) cj[i]  = zero;
        else if (beta != one)  for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;

        if (TR == 'N') {
            /* Rank-k via column outer products: C[:,j] += alpha · A[j,l] · A[:,l]. */
            for (int l = 0; l < K; ++l) {
                const T ajl = A_(j, l);
                if (ajl != zero) {
                    const T t = alpha * ajl;
                    for (int i = i_lo; i < i_hi; ++i) cj[i] += t * A_(i, l);
                }
            }
        } else {  /* TR == 'T' */
            /* Inner-product form on Aᵀ A: C(i,j) += alpha · Σ_l A(l,i) · A(l,j). */
            const T *Aj = a + (size_t)j * lda;
            for (int i = i_lo; i < i_hi; ++i) {
                const T *Ai = a + (size_t)i * lda;
                T s = zero;
                for (int l = 0; l < K; ++l) s += Ai[l] * Aj[l];
                cj[i] += alpha * s;
            }
        }
    }
}

void qsyrk_serial_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = qsyrk_uplo(uplo);
    const int TR = qsyrk_trans_code(trans, trans_len);

    if (N == 0) return;

    const T zero = 0.0Q, one = 1.0Q;

    /* alpha == 0 or K == 0 quick return — just scale the UPLO triangle of C. */
    if (alpha == zero || K == 0) {
        if (beta == one) return;
        for (int j = 0; j < N; ++j) {
            const int i_lo = (UPLO == 'L') ? j : 0;
            const int i_hi = (UPLO == 'L') ? N : j + 1;
            T *cj = c + (size_t)j * ldc;
            if (beta == zero) for (int i = i_lo; i < i_hi; ++i) cj[i] = zero;
            else              for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;
        }
        return;
    }

    qsyrk_core(0, N, UPLO, TR, N, K, alpha, beta, a, lda, c, ldc);
}

#undef A_
#undef C_
