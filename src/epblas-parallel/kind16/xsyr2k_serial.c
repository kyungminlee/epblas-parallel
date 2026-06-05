/*
 * xsyr2k_serial.c — kind16 complex (__complex128) symmetric rank-2k, serial
 * core.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the uplo
 * decode and the per-column compute core (declared in xsyr2k_kernel.h), plus
 * the public `xsyr2k_serial_` Fortran entry. No OpenMP anywhere on this call
 * path — safe to invoke from inside another function's `#pragma omp parallel`
 * region.
 *
 * Both xsyr2k_serial_ and the parallel xsyr2k_ drive numerics through
 * xsyr2k_core, so the two paths are bitwise-identical.
 */

#include "xsyr2k_kernel.h"
#include <ctype.h>
#include <quadmath.h>

typedef xsyr2k_T T;

char xsyr2k_uplo(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

void xsyr2k_core(
    int j0, int j1, int N, int K,
    char UPLO, char TR,
    T alpha, T beta,
    const T *a, int lda,
    const T *b, int ldb,
    T *c, int ldc)
{
    const T zero = 0.0Q + 0.0Qi, one = 1.0Q + 0.0Qi;

    for (int j = j0; j < j1; ++j) {
        const int i_lo = (UPLO == 'L') ? j : 0;
        const int i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + (size_t)j * ldc;

        if (beta == zero)      for (int i = i_lo; i < i_hi; ++i) cj[i]  = zero;
        else if (beta != one)  for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;

        if (TR == 'N') {
            for (int l = 0; l < K; ++l) {
                const T t1 = alpha * A_(j, l);
                const T t2 = alpha * B_(j, l);
                for (int i = i_lo; i < i_hi; ++i) {
                    cj[i] += B_(i, l) * t1 + A_(i, l) * t2;
                }
            }
        } else {
            const T *Aj = a + (size_t)j * lda;
            const T *Bj = b + (size_t)j * ldb;
            for (int i = i_lo; i < i_hi; ++i) {
                const T *Ai = a + (size_t)i * lda;
                const T *Bi = b + (size_t)i * ldb;
                T s = zero;
                for (int l = 0; l < K; ++l) s += Ai[l] * Bj[l] + Bi[l] * Aj[l];
                cj[i] += alpha * s;
            }
        }
    }
}

void xsyr2k_serial_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict b, const int *ldb_,
    const T *beta_,
    T *restrict c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = xsyr2k_uplo(uplo);
    char TR = xsyr2k_uplo(trans);
    if (TR == 'C') TR = 'T';

    if (N == 0) return;

    const T zero = 0.0Q + 0.0Qi, one = 1.0Q + 0.0Qi;

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

    xsyr2k_core(0, N, N, K, UPLO, TR, alpha, beta, a, lda, b, ldb, c, ldc);
}

#undef A_
#undef B_
#undef C_
