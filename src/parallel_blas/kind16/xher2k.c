/*
 * xher2k — kind16 complex (__complex128) Hermitian rank-2k update.
 *
 *   C := alpha · A · Bᴴ + conj(alpha) · B · Aᴴ + beta · C  (TRANS='N')
 *   C := alpha · Aᴴ · B + conj(alpha) · Bᴴ · A + beta · C  (TRANS='C')
 *
 * alpha is COMPLEX, beta is REAL. Diagonal of C stays real.
 * Unblocked reference + omp-parallel-for over j.
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define XHER2K_OMP_MIN 32

typedef __complex128 TC;
typedef __float128   TR;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

void xher2k_(
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
    const char UPLO = up(uplo);
    const char TR_c = up(trans);

    if (N == 0) return;

    const TR rzero = 0.0Q, rone = 1.0Q;
    const TC czero = 0.0Q + 0.0Qi;
    const TC alpha_conj = conjq(alpha);

    if (alpha == czero || K == 0) {
        if (beta == rone) {
            for (int j = 0; j < N; ++j) c[(size_t)j * ldc + j] = crealq(c[(size_t)j * ldc + j]);
            return;
        }
#ifdef _OPENMP
        const int use_omp = (N >= XHER2K_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
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

#ifdef _OPENMP
    const int use_omp = (N >= XHER2K_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (int j = 0; j < N; ++j) {
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

#undef A_
#undef B_
