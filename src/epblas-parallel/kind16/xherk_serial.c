/*
 * xherk_serial.c — kind16 complex (__complex128) Hermitian rank-k update,
 * serial core.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the
 * uplo/trans decode and the per-column compute core (declared in
 * xherk_kernel.h), plus the public `xherk_serial_` Fortran entry. No
 * OpenMP anywhere on this call path — safe to invoke from inside another
 * function's `#pragma omp parallel` region.
 *
 * Both xherk_serial_ and the parallel xherk_ drive numerics through
 * xherk_core, so the two paths are bitwise-identical. The Hermitian
 * real-diagonal rule (the imag part of cj[j] is zeroed) is preserved inside
 * the core, exactly matching Netlib ZHERK.
 */

#include "xherk_kernel.h"
#include <ctype.h>
#include <quadmath.h>

typedef xherk_TC TC;
typedef xherk_TR TR;

char xherk_uplo(const char *p) {
    return (char)toupper((unsigned char)*p);
}

char xherk_trans(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void xherk_core(
    int j0, int j1,
    char UPLO, char TR_c, int N, int K,
    TR alpha, TR beta,
    const TC *a, int lda,
    TC *c, int ldc)
{
    const TR rzero = 0.0Q, rone = 1.0Q;
    const TC czero = 0.0Q + 0.0Qi;

    for (int j = j0; j < j1; ++j) {
        const int i_lo = (UPLO == 'L') ? j : 0;
        const int i_hi = (UPLO == 'L') ? N : j + 1;
        TC *cj = c + (size_t)j * ldc;

        /* beta scale of the UPLO slice of column j (diag stays real). */
        if (beta == rzero) {
            for (int i = i_lo; i < i_hi; ++i) cj[i] = czero;
        } else if (beta != rone) {
            for (int i = i_lo; i < i_hi; ++i) {
                if (i == j) cj[i] = beta * crealq(cj[i]);
                else        cj[i] = beta * cj[i];
            }
        } else {
            /* beta == 1: still zero diagonal imag. */
            cj[j] = crealq(cj[j]);
        }

        if (TR_c == 'N') {
            /* C := alpha · A · Aᴴ + (already-scaled) C.
             * Column outer product: C[:,j] += alpha · conj(A[j,l]) · A[:,l]
             * (the conj is on the j-th row because we're computing
             *  C[i,j] = Σ A[i,l] · conj(A[j,l]).) */
            for (int l = 0; l < K; ++l) {
                const TC ajl = A_(j, l);
                if (ajl != czero) {
                    const TC t = alpha * conjq(ajl);
                    for (int i = i_lo; i < i_hi; ++i) {
                        if (i == j) cj[i] += crealq(t * A_(i, l));
                        else        cj[i] += t * A_(i, l);
                    }
                }
            }
        } else {  /* TRANS = 'C': C := alpha · Aᴴ · A + C
                   * Inner product: C[i,j] = Σ_l conj(A[l,i]) · A[l,j]. */
            const TC *Aj = a + (size_t)j * lda;
            for (int i = i_lo; i < i_hi; ++i) {
                const TC *Ai = a + (size_t)i * lda;
                TC s = czero;
                for (int l = 0; l < K; ++l) s += conjq(Ai[l]) * Aj[l];
                if (i == j) cj[i] += alpha * crealq(s);
                else        cj[i] += alpha * s;
            }
        }
    }
}

void xherk_serial_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const TR *alpha_,
    const TC *a, const int *lda_,
    const TR *beta_,
    TC *c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldc = *ldc_;
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO = xherk_uplo(uplo);
    const char TR_c = xherk_trans(trans);

    if (N == 0) return;

    const TR rzero = 0.0Q, rone = 1.0Q;
    const TC czero = 0.0Q + 0.0Qi;

    /* Quick return when only beta scaling is needed. */
    if (alpha == rzero || K == 0) {
        if (beta == rone) {
            /* Even for beta=1, ZHERK strictly zeros the imag part of
             * the diagonal so it stays real. */
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

    xherk_core(0, N, UPLO, TR_c, N, K, alpha, beta, a, lda, c, ldc);
}

#undef A_
