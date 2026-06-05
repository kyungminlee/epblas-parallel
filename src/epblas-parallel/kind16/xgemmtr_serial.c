/*
 * xgemmtr_serial.c — kind16 (COMPLEX(KIND=16) / __complex128) triangular GEMM
 * update, serial core.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the
 * trans-char decode and the two per-column cores (a beta-only pass and the
 * full compute pass, declared in xgemmtr_kernel.h), plus the public
 * `xgemmtr_serial_` Fortran entry. No OpenMP anywhere on this call path —
 * safe to invoke from inside another function's `#pragma omp parallel` region.
 *
 * Both xgemmtr_serial_ and the parallel xgemmtr_ drive numerics through these
 * cores, so the two paths are bitwise-identical.
 */

#include "xgemmtr_kernel.h"
#include <ctype.h>
#include <quadmath.h>

typedef xgemmtr_T T;

int xgemmtr_trans_code(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

void xgemmtr_beta_core(
    int j0, int j1, int N, int upper,
    T beta,
    T *c, int ldc)
{
    const T zero = 0.0Q + 0.0Qi;
    for (int j = j0; j < j1; ++j) {
        const int is = upper ? 0 : j;
        const int ie = upper ? (j + 1) : N;
        T *cj = &C_(0, j);
        if (beta == zero)      for (int i = is; i < ie; ++i) cj[i]  = zero;
        else                   for (int i = is; i < ie; ++i) cj[i] *= beta;
    }
}

void xgemmtr_compute_core(
    int j0, int j1, int N, int upper, int K,
    int trans_a, int trans_b, int conj_a, int conj_b,
    T alpha, T beta,
    const T *a, int lda,
    const T *b, int ldb,
    T *c, int ldc)
{
    const T zero = 0.0Q + 0.0Qi;
    const T one  = 1.0Q + 0.0Qi;

    for (int j = j0; j < j1; ++j) {
        const int is = upper ? 0 : j;
        const int ie = upper ? (j + 1) : N;
        T *cj = &C_(0, j);

        if (!trans_a) {
            /* axpy form */
            if (beta == zero)      for (int i = is; i < ie; ++i) cj[i]  = zero;
            else if (beta != one)  for (int i = is; i < ie; ++i) cj[i] *= beta;
            for (int l = 0; l < K; ++l) {
                T bl;
                if (!trans_b)      bl = B_(l, j);
                else if (!conj_b)  bl = B_(j, l);
                else               bl = conjq(B_(j, l));
                if (bl != zero) {
                    const T t = alpha * bl;
                    const T *al = &A_(0, l);
                    for (int i = is; i < ie; ++i) cj[i] += t * al[i];
                }
            }
        } else {
            /* inner-product form */
            for (int i = is; i < ie; ++i) {
                T s = zero;
                if (!trans_b) {
                    if (!conj_a) for (int l = 0; l < K; ++l) s += A_(l, i)        * B_(l, j);
                    else         for (int l = 0; l < K; ++l) s += conjq(A_(l, i)) * B_(l, j);
                } else if (!conj_b) {
                    if (!conj_a) for (int l = 0; l < K; ++l) s += A_(l, i)        * B_(j, l);
                    else         for (int l = 0; l < K; ++l) s += conjq(A_(l, i)) * B_(j, l);
                } else {
                    if (!conj_a) for (int l = 0; l < K; ++l) s += A_(l, i)        * conjq(B_(j, l));
                    else         for (int l = 0; l < K; ++l) s += conjq(A_(l, i)) * conjq(B_(j, l));
                }
                cj[i] = (beta == zero) ? alpha * s : alpha * s + beta * cj[i];
            }
        }
    }
}

void xgemmtr_serial_(const char *uplo, const char *transa, const char *transb,
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
    const int ta = xgemmtr_trans_code(transa);
    const int tb = xgemmtr_trans_code(transb);

    if (N <= 0) return;
    const T zero = 0.0Q + 0.0Qi;
    const T one  = 1.0Q + 0.0Qi;

    const int conj_a = (ta == 'C');
    const int conj_b = (tb == 'C');
    const int trans_a = (ta != 'N');
    const int trans_b = (tb != 'N');

    if (alpha == zero || K == 0) {
        if (beta == one) return;
        xgemmtr_beta_core(0, N, N, upper, beta, c, ldc);
        return;
    }

    xgemmtr_compute_core(0, N, N, upper, K,
                         trans_a, trans_b, conj_a, conj_b,
                         alpha, beta, a, lda, b, ldb, c, ldc);
}

#undef A_
#undef B_
#undef C_
