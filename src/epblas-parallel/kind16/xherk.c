/*
 * xherk — kind16 complex (__complex128) Hermitian rank-k update.
 *
 * Computes:
 *   C := alpha · A · Aᴴ + beta · C          (TRANS='N')
 *   C := alpha · Aᴴ · A + beta · C          (TRANS='C')
 *
 * where C is N×N Hermitian; only the UPLO triangle is referenced /
 * written and its diagonal is forced real (the imag part is zeroed
 * during the beta-scale step, matching Netlib ZHERK).
 *
 * alpha and beta are REAL (kind=16) — the Hermitian property keeps
 * the diagonal real, so the scaling factors stay real-valued.
 *
 * Unblocked Netlib reference with omp-parallel-for over columns of C.
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define XHERK_OMP_MIN 32

typedef __complex128 TC;
typedef __float128   TR;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void xherk_(
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
    const char UPLO = up(uplo);
    const char TR_c = up(trans);

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
#ifdef _OPENMP
        const int use_omp = (N >= XHERK_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
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
    const int use_omp = (N >= XHERK_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (int j = 0; j < N; ++j) {
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

#undef A_
