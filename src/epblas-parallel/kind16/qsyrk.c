/*
 * qsyrk — kind16 (REAL(KIND=16) / `__float128`) symmetric rank-k update.
 *
 * Computes one of:
 *   C := alpha · A · Aᵀ + beta · C          (TRANS='N')
 *   C := alpha · Aᵀ · A + beta · C          (TRANS='T'/'C')
 *
 * where C is N×N symmetric; only the UPLO triangle is referenced /
 * written. A is N×K (TRANS='N') or K×N (TRANS='T').
 *
 * Unblocked Netlib reference (DSYRK shape) with one omp-parallel-for
 * over columns of C. Each column j touches a different UPLO-slice of
 * C and the same A, so columns are fully independent across threads.
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define QSYRK_OMP_MIN 32

typedef __float128 T;

static int trans_code(const char *p, size_t len) {
    (void)len;
    char c = (char)toupper((unsigned char)*p);
    return (c == 'C') ? 'T' : c;
}

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

void qsyrk_(
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
    const char UPLO = up(uplo);
    const int TR = trans_code(trans, trans_len);

    if (N == 0) return;

    const T zero = 0.0Q, one = 1.0Q;

    /* alpha == 0 or K == 0 quick return — just scale the UPLO triangle of C. */
    if (alpha == zero || K == 0) {
        if (beta == one) return;
#ifdef _OPENMP
        const int use_omp = (N >= QSYRK_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const int i_lo = (UPLO == 'L') ? j : 0;
            const int i_hi = (UPLO == 'L') ? N : j + 1;
            T *cj = c + (size_t)j * ldc;
            if (beta == zero) for (int i = i_lo; i < i_hi; ++i) cj[i] = zero;
            else              for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;
        }
        return;
    }

#ifdef _OPENMP
    const int use_omp = (N >= QSYRK_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (int j = 0; j < N; ++j) {
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

#undef A_
#undef C_
