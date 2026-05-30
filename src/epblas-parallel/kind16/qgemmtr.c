/*
 * qgemmtr — kind16 real triangular GEMM update.
 *   C := alpha · op(A) · op(B) + beta · C   (only UPLO triangle of C)
 * Reference DGEMMTR algorithm, kind16. libquadmath bound; arithmetic
 * dominates so OpenMP across j scales near-linearly.
 */
#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define QGEMMTR_OMP_MIN 32

typedef __float128 T;

static int trans_code(const char *p) {
    char c = (char)toupper((unsigned char)*p);
    return (c == 'C') ? 'T' : c;
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

void qgemmtr_(const char *uplo, const char *transa, const char *transb,
              const int *n_, const int *k_,
              const T *alpha_,
              const T *a, const int *lda_,
              const T *b, const int *ldb_,
              const T *beta_,
              T *c, const int *ldc_,
              size_t uplo_len, size_t ta_len, size_t tb_len)
{
    (void)uplo_len; (void)ta_len; (void)tb_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const int upper = ((char)toupper((unsigned char)*uplo) == 'U');
    const int ta = trans_code(transa);
    const int tb = trans_code(transb);

    if (N <= 0) return;
    const T zero = 0.0Q, one = 1.0Q;

    if (alpha == zero || K == 0) {
        if (beta == one) return;
#ifdef _OPENMP
        const int use_omp0 = (N >= QGEMMTR_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
        #pragma omp parallel for if(use_omp0) schedule(static, 1)
#endif
        for (int j = 0; j < N; ++j) {
            const int is = upper ? 0 : j;
            const int ie = upper ? (j + 1) : N;
            T *cj = &C_(0, j);
            if (beta == zero)      for (int i = is; i < ie; ++i) cj[i]  = zero;
            else                   for (int i = is; i < ie; ++i) cj[i] *= beta;
        }
        return;
    }

#ifdef _OPENMP
    const int use_omp = (N >= QGEMMTR_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
    #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
    for (int j = 0; j < N; ++j) {
        const int is = upper ? 0 : j;
        const int ie = upper ? (j + 1) : N;
        T *cj = &C_(0, j);

        if (ta == 'N') {
            if (beta == zero)      for (int i = is; i < ie; ++i) cj[i]  = zero;
            else if (beta != one)  for (int i = is; i < ie; ++i) cj[i] *= beta;
            if (tb == 'N') {
                for (int k = 0; k < K; ++k) {
                    const T bkj = B_(k, j);
                    if (bkj != zero) {
                        const T t = alpha * bkj;
                        const T *ak = &A_(0, k);
                        for (int i = is; i < ie; ++i) cj[i] += t * ak[i];
                    }
                }
            } else {
                for (int k = 0; k < K; ++k) {
                    const T bjk = B_(j, k);
                    if (bjk != zero) {
                        const T t = alpha * bjk;
                        const T *ak = &A_(0, k);
                        for (int i = is; i < ie; ++i) cj[i] += t * ak[i];
                    }
                }
            }
        } else {
            if (tb == 'N') {
                for (int i = is; i < ie; ++i) {
                    T s = zero;
                    for (int k = 0; k < K; ++k) s += A_(k, i) * B_(k, j);
                    cj[i] = (beta == zero) ? alpha * s : alpha * s + beta * cj[i];
                }
            } else {
                for (int i = is; i < ie; ++i) {
                    T s = zero;
                    for (int k = 0; k < K; ++k) s += A_(k, i) * B_(j, k);
                    cj[i] = (beta == zero) ? alpha * s : alpha * s + beta * cj[i];
                }
            }
        }
    }
}

#undef A_
#undef B_
#undef C_
