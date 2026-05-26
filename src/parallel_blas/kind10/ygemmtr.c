/*
 * ygemmtr — kind10 complex (_Complex long double) triangular GEMM update.
 *   C := alpha · op(A) · op(B) + beta · C   (only UPLO triangle of C)
 * Reference algorithm, OpenMP across j. `~z` is GCC-extension conjugate.
 */
#include <stddef.h>
#include <ctype.h>
#include <complex.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define YGEMMTR_OMP_MIN 32

typedef _Complex long double T;

static int trans_code(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

void ygemmtr_(const char *uplo, const char *transa, const char *transb,
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
    const T zero = 0.0L + 0.0iL;
    const T one  = 1.0L + 0.0iL;

    const int conj_a = (ta == 'C');
    const int conj_b = (tb == 'C');
    const int trans_a = (ta != 'N');
    const int trans_b = (tb != 'N');

    if (alpha == zero || K == 0) {
        if (beta == one) return;
#ifdef _OPENMP
        const int use_omp0 = (N >= YGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
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
    const int use_omp = (N >= YGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
    for (int j = 0; j < N; ++j) {
        const int is = upper ? 0 : j;
        const int ie = upper ? (j + 1) : N;
        T *cj = &C_(0, j);

        if (!trans_a) {
            if (beta == zero)      for (int i = is; i < ie; ++i) cj[i]  = zero;
            else if (beta != one)  for (int i = is; i < ie; ++i) cj[i] *= beta;
            /* K-unroll by 2 — expose two independent FMA chains per i to
             * mask x87 fmul latency. Same trick as ygemm's NN/NT paths
             * (findings doc Addendum 1 §kind10 complex). Conj_b is hoisted
             * out of the hot loop; a runtime branch inside the K-unrolled
             * body defeats gcc's scheduling for this kind10 complex pattern. */
            int l = 0;
            if (!trans_b) {
                for (; l + 1 < K; l += 2) {
                    const T t0 = alpha * B_(l,     j);
                    const T t1 = alpha * B_(l + 1, j);
                    const T *al0 = &A_(0, l);
                    const T *al1 = &A_(0, l + 1);
                    for (int i = is; i < ie; ++i)
                        cj[i] += t0 * al0[i] + t1 * al1[i];
                }
            } else if (!conj_b) {
                for (; l + 1 < K; l += 2) {
                    const T t0 = alpha * B_(j, l);
                    const T t1 = alpha * B_(j, l + 1);
                    const T *al0 = &A_(0, l);
                    const T *al1 = &A_(0, l + 1);
                    for (int i = is; i < ie; ++i)
                        cj[i] += t0 * al0[i] + t1 * al1[i];
                }
            } else {
                for (; l + 1 < K; l += 2) {
                    const T t0 = alpha * ~B_(j, l);
                    const T t1 = alpha * ~B_(j, l + 1);
                    const T *al0 = &A_(0, l);
                    const T *al1 = &A_(0, l + 1);
                    for (int i = is; i < ie; ++i)
                        cj[i] += t0 * al0[i] + t1 * al1[i];
                }
            }
            for (; l < K; ++l) {
                T bl;
                if (!trans_b)      bl = B_(l, j);
                else if (!conj_b)  bl = B_(j, l);
                else               bl = ~B_(j, l);
                const T t = alpha * bl;
                const T *al = &A_(0, l);
                for (int i = is; i < ie; ++i) cj[i] += t * al[i];
            }
        } else {
            for (int i = is; i < ie; ++i) {
                T s = zero;
                if (!trans_b) {
                    if (!conj_a) for (int l = 0; l < K; ++l) s += A_(l, i) * B_(l, j);
                    else         for (int l = 0; l < K; ++l) s += ~A_(l, i) * B_(l, j);
                } else if (!conj_b) {
                    if (!conj_a) for (int l = 0; l < K; ++l) s += A_(l, i) * B_(j, l);
                    else         for (int l = 0; l < K; ++l) s += ~A_(l, i) * B_(j, l);
                } else {
                    if (!conj_a) for (int l = 0; l < K; ++l) s += A_(l, i) * ~B_(j, l);
                    else         for (int l = 0; l < K; ++l) s += ~A_(l, i) * ~B_(j, l);
                }
                cj[i] = (beta == zero) ? alpha * s : alpha * s + beta * cj[i];
            }
        }
    }
}

#undef A_
#undef B_
#undef C_
