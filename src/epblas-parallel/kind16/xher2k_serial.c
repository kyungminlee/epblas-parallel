/*
 * xher2k_serial.c — kind16 complex (__complex128) Hermitian rank-2k, serial
 * core.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the uplo
 * decode and the per-column compute core (declared in xher2k_kernel.h), plus
 * the public `xher2k_serial_` Fortran entry. No OpenMP anywhere on this call
 * path — safe to invoke from inside another function's `#pragma omp parallel`
 * region.
 *
 * Both xher2k_serial_ and the parallel xher2k_ drive numerics through
 * xher2k_core, so the two paths are bitwise-identical. alpha is COMPLEX,
 * beta is REAL, and the Hermitian diagonal is kept real at every touch.
 */

#include "xher2k_kernel.h"
#include <ctype.h>
#include <quadmath.h>

typedef xher2k_TC TC;
typedef xher2k_TR TR;

char xher2k_uplo(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

void xher2k_core(
    int j0, int j1, int N, int K,
    char UPLO, char TR_c,
    TC alpha, TC alpha_conj, TR beta,
    const TC *a, int lda,
    const TC *b, int ldb,
    TC *c, int ldc)
{
    const TR rzero = 0.0Q, rone = 1.0Q;
    const TC czero = 0.0Q + 0.0Qi;

    for (int j = j0; j < j1; ++j) {
        const int i_lo = (UPLO == 'L') ? j : 0;
        const int i_hi = (UPLO == 'L') ? N : j + 1;
        TC *cj = c + (size_t)j * ldc;

        if (beta == rzero) {
            for (int i = i_lo; i < i_hi; ++i) cj[i] = czero;
        } else if (beta != rone) {
            for (int i = i_lo; i < i_hi; ++i) {
                if (i == j) cj[i] = beta * crealq(cj[i]);
                else        cj[i] = beta * cj[i];
            }
        } else {
            cj[j] = crealq(cj[j]);
        }

        if (TR_c == 'N') {
            /* C(I,J) += α A(I,l) conj(B(J,l)) + conj(α) B(I,l) conj(A(J,l)) */
            for (int l = 0; l < K; ++l) {
                const TC t1 = alpha      * conjq(B_(j, l));
                const TC t2 = alpha_conj * conjq(A_(j, l));
                for (int i = i_lo; i < i_hi; ++i) {
                    if (i == j) cj[i] += crealq(A_(i, l) * t1 + B_(i, l) * t2);
                    else        cj[i] += A_(i, l) * t1 + B_(i, l) * t2;
                }
            }
        } else {
            /* C(I,J) += α conj(A(l,I))·B(l,J) + conj(α) conj(B(l,I))·A(l,J) */
            const TC *Aj = a + (size_t)j * lda;
            const TC *Bj = b + (size_t)j * ldb;
            for (int i = i_lo; i < i_hi; ++i) {
                const TC *Ai = a + (size_t)i * lda;
                const TC *Bi = b + (size_t)i * ldb;
                TC s1 = czero, s2 = czero;
                for (int l = 0; l < K; ++l) {
                    s1 += conjq(Ai[l]) * Bj[l];
                    s2 += conjq(Bi[l]) * Aj[l];
                }
                if (i == j) cj[i] += crealq(alpha * s1 + alpha_conj * s2);
                else        cj[i] += alpha * s1 + alpha_conj * s2;
            }
        }
    }
}

void xher2k_serial_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const TC *alpha_,
    const TC *restrict a, const int *lda_,
    const TC *restrict b, const int *ldb_,
    const TR *beta_,
    TC *restrict c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const TC alpha = *alpha_;
    const TR beta  = *beta_;
    const char UPLO = xher2k_uplo(uplo);
    const char TR_c = xher2k_uplo(trans);

    if (N == 0) return;

    const TR rzero = 0.0Q, rone = 1.0Q;
    const TC czero = 0.0Q + 0.0Qi;
    const TC alpha_conj = conjq(alpha);

    if (alpha == czero || K == 0) {
        if (beta == rone) {
            for (int j = 0; j < N; ++j) c[(size_t)j * ldc + j] = crealq(c[(size_t)j * ldc + j]);
            return;
        }
        for (int j = 0; j < N; ++j) {
            const int i_lo = (UPLO == 'L') ? j : 0;
            const int i_hi = (UPLO == 'L') ? N : j + 1;
            TC *cj = c + (size_t)j * ldc;
            if (beta == rzero) {
                for (int i = i_lo; i < i_hi; ++i) cj[i] = czero;
            } else {
                for (int i = i_lo; i < i_hi; ++i) {
                    if (i == j) cj[i] = beta * crealq(cj[i]);
                    else        cj[i] = beta * cj[i];
                }
            }
        }
        return;
    }

    xher2k_core(0, N, N, K, UPLO, TR_c, alpha, alpha_conj, beta,
                a, lda, b, ldb, c, ldc);
}

#undef A_
#undef B_
