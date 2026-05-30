/*
 * xsymm — kind16 complex (`__complex128`) symmetric matrix multiply.
 * NOT Hermitian — no conjugate. For Hermitian see xhemm.
 *
 *   C := alpha · A · B + beta · C          (SIDE='L', A is M×M sym)
 *   C := alpha · B · A + beta · C          (SIDE='R', A is N×N sym)
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

#define XSYMM_OMP_MIN 32

typedef __complex128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static const T ZERO = 0.0Q + 0.0Qi;
static const T ONE  = 1.0Q + 0.0Qi;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

void xsymm_(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t side_len, size_t uplo_len)
{
    (void)side_len; (void)uplo_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char SIDE = up(side);
    const char UPLO = up(uplo);

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        if (beta == ONE) return;
#ifdef _OPENMP
        const int use_omp = (N >= XSYMM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            T *cj = c + (size_t)j * ldc;
            if (beta == ZERO) for (int i = 0; i < M; ++i) cj[i] = ZERO;
            else              for (int i = 0; i < M; ++i) cj[i] *= beta;
        }
        return;
    }

#ifdef _OPENMP
    const int use_omp = (N >= XSYMM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel());
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (int j = 0; j < N; ++j) {
        T *cj = c + (size_t)j * ldc;

        if (beta == ZERO)      for (int i = 0; i < M; ++i) cj[i]  = ZERO;
        else if (beta != ONE)  for (int i = 0; i < M; ++i) cj[i] *= beta;

        if (SIDE == 'L') {
            if (UPLO == 'L') {
                for (int i = 0; i < M; ++i) {
                    T temp1 = alpha * B_(i, j);
                    T temp2 = ZERO;
                    for (int k = 0; k < i; ++k) {
                        cj[k]  += temp1 * A_(i, k);
                        temp2  += B_(k, j) * A_(i, k);
                    }
                    cj[i] += temp1 * A_(i, i) + alpha * temp2;
                }
            } else {
                for (int i = M - 1; i >= 0; --i) {
                    T temp1 = alpha * B_(i, j);
                    T temp2 = ZERO;
                    for (int k = i + 1; k < M; ++k) {
                        cj[k]  += temp1 * A_(i, k);
                        temp2  += B_(k, j) * A_(i, k);
                    }
                    cj[i] += temp1 * A_(i, i) + alpha * temp2;
                }
            }
        } else {  /* SIDE = 'R' */
            {
                const T t = alpha * A_(j, j);
                for (int i = 0; i < M; ++i) cj[i] += t * B_(i, j);
            }
            if (UPLO == 'L') {
                for (int k = 0; k < j; ++k) {
                    const T ajk = A_(j, k);
                    if (ajk != ZERO) {
                        const T t = alpha * ajk;
                        for (int i = 0; i < M; ++i) cj[i] += t * B_(i, k);
                    }
                }
                for (int k = j + 1; k < N; ++k) {
                    const T akj = A_(k, j);
                    if (akj != ZERO) {
                        const T t = alpha * akj;
                        for (int i = 0; i < M; ++i) cj[i] += t * B_(i, k);
                    }
                }
            } else {
                for (int k = 0; k < j; ++k) {
                    const T akj = A_(k, j);
                    if (akj != ZERO) {
                        const T t = alpha * akj;
                        for (int i = 0; i < M; ++i) cj[i] += t * B_(i, k);
                    }
                }
                for (int k = j + 1; k < N; ++k) {
                    const T ajk = A_(j, k);
                    if (ajk != ZERO) {
                        const T t = alpha * ajk;
                        for (int i = 0; i < M; ++i) cj[i] += t * B_(i, k);
                    }
                }
            }
        }
    }
}

#undef A_
#undef B_
#undef C_
