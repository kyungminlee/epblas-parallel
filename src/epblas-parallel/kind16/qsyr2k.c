/*
 * qsyr2k — kind16 (REAL(KIND=16) / `__float128`) symmetric rank-2k.
 *
 *   C := alpha · (A · Bᵀ + B · Aᵀ) + beta · C         (TRANS='N')
 *   C := alpha · (Aᵀ · B + Bᵀ · A) + beta · C         (TRANS='T'/'C')
 *
 * Unblocked Netlib DSYR2K shape with one omp-parallel-for over the
 * columns of C. libquadmath dispatch dominates the per-op time, so
 * no blocking or packing — coarse-grain parallelism over j scales
 * nearly linearly.
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define QSYR2K_OMP_MIN 32

typedef __float128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

void qsyr2k_(
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
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';

    if (N == 0) return;

    const T zero = 0.0Q, one = 1.0Q;

    if (alpha == zero || K == 0) {
        if (beta == one) return;
#ifdef _OPENMP
        const int use_omp = (N >= QSYR2K_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
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
    const int use_omp = (N >= QSYR2K_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (int j = 0; j < N; ++j) {
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

#undef A_
#undef B_
#undef C_
